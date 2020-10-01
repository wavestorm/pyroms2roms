# %run make_pycfsr2frc_one-hourly.py

'''
===========================================================================
This file is part of py-roms2roms

    py-roms2roms is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    py-roms2roms is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with py-roms2roms.  If not, see <http://www.gnu.org/licenses/>.

Version 1.0.1

Copyright (c) 2014-2018 by Evan Mason, IMEDEA
Email: emason@imedea.uib-csic.es

---------------------------------------------------------------------------
Ported to processing ERA5 reanalysis files into bulk
Dylan F. Bailey 2020

+ Handles ERA5 single level climate model data
+ Handles monthly files
+ Handles dewpoint to humidity conversions
* Moved across classes from pycfsr2frc.py to remove dependence on that file
* AirSea functions moved into wind class, since it is the only class using 
  those functions
- Removed forecast averaging functions since we are only dealing with 
  reanalysis


===========================================================================

Create a forcing file based on hourly ERA data

===========================================================================
'''
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import scipy.spatial as sp
import pylab as plb
import numpy as np
import numexpr as ne
import time as time
import glob as glob
from collections import OrderedDict
from datetime import datetime
from py_roms2roms import ROMS, horizInterp, RomsGrid, RomsData

class RomsGrid(RomsGrid):
    '''
    Modify the RomsGrid class
    '''
    
    def create_frc_nc(self, frcfile, nr, cl, madeby, bulk):
        '''
        Create a new forcing file based on dimensions from grdobj
            frcfile : path and name of new frc file
            sd      : start date2num
            ed      : end date
            nr      : no. of records
            cl      : cycle length
            madeby  : name of this file
        '''
        if bulk:
            try:
                frcfile = frcfile.replace('frc_', 'blk_')
            except Exception:
                pass
        else:
            try:
                frcfile = frcfile.replace('blk_', 'frc_')
            except Exception:
                pass
                
        self.frcfile = frcfile
        print('Creating new ERA forcing file:', frcfile)
        # Global attributes
        ''' The best choice should be format='NETCDF4', but it will not work with
        Sasha's 2008 code (not tested with Roms-Agrif).  Therefore I use
        format='NETCDF3_64BIT; the drawback is that it is very slow'
        '''
        #nc = netcdf.Dataset(frcfile, 'w', format='NETCDF3_64BIT')
        #nc = netcdf.Dataset(frcfile, 'w', format='NETCDF3_CLASSIC')
        nc = netcdf.Dataset(frcfile, 'w', format='NETCDF4')
        nc.created  = dt.datetime.datetime.now().isoformat()
        nc.type = 'ROMS interannual forcing file produced by %s.py' %madeby
        nc.grd_file = self.romsfile
        
        # Dimensions
        nc.createDimension('xi_rho', self.lon().shape[1])
        nc.createDimension('xi_u', self.lon().shape[1] - 1)
        nc.createDimension('eta_rho', self.lon().shape[0])
        nc.createDimension('eta_v', self.lon().shape[0] - 1)
        if bulk:
            nc.createDimension('bulk_time', nr)
        else:
            nc.createDimension('sms_time', nr)
            nc.createDimension('shf_time', nr)
            nc.createDimension('swf_time', nr)
            nc.createDimension('sst_time', nr)
            nc.createDimension('srf_time', nr)
            nc.createDimension('sss_time', nr)
            nc.createDimension('one', 1)
        

        # Dictionary for the variables
        frc_vars = OrderedDict()
        
        if bulk:
            frc_vars['bulk_time'] = ['time',
                                     'bulk_time',
                                     'bulk formulation execution time',
                                     'days']
        else:
            frc_vars['sms_time'] = ['time',
                                    'sms_time',
                                    'surface momentum stress time',
                                    'days']
            frc_vars['shf_time'] = ['time',
                                    'shf_time',
                                    'surface heat flux time',
                                    'days']
            frc_vars['swf_time'] = ['time',
                                    'swf_time',
                                    'surface freshwater flux time',
                                    'days']
            frc_vars['sst_time'] = ['time',
                                    'sst_time',
                                    'sea surface temperature time',
                                    'days']
            frc_vars['sss_time'] = ['time',
                                    'sss_time',
                                    'sea surface salinity time',
                                    'days']
            frc_vars['srf_time'] = ['time',
                                    'srf_time',
                                    'solar shortwave radiation time',
                                    'days']
        
            # Note this could be problematic if scale_cfsr_coads.py adjusts variables
            # with different frequencies...
            frc_vars['month']    = ['time',
                                    'sst_time',
                                    'not used by ROMS; useful for scale_cfsr_coads.py',
                                    'month of year']
        
        # Bulk formaulation
        if bulk:
            frc_vars['tair'] =   ['rho',
                                  'bulk_time',
                                  'surface air temperature',
                                  'Celsius']
            frc_vars['rhum'] =   ['rho',
                                  'bulk_time',
                                  'relative humidity',
                                  'fraction']
            frc_vars['prate'] =   ['rho',
                                  'bulk_time',
                                  'precipitation rate',
                                  'cm day-1']
            frc_vars['wspd'] =   ['rho',
                                  'bulk_time',
                                  'wind speed 10m',
                                  'm s-1']
            frc_vars['radlw'] =   ['rho',
                                   'bulk_time',
                                   'net outgoing longwave radiation',
                                   'Watts meter-2',
                                   'upward flux, cooling water']
            frc_vars['radlw_in'] = ['rho',
                                    'bulk_time',
                                    'downward longwave radiation',
                                    'Watts meter-2',
                                    'downward flux, warming water']
            frc_vars['radsw'] =    ['rho',
                                    'bulk_time',
                                    'shortwave radiation',
                                    'Watts meter-2',
                                    'downward flux, warming water']
            frc_vars['sustr'] =     ['u',
                                    'bulk_time',
                                    'surface u-momentum stress',
                                    'Newton meter-2']
            frc_vars['svstr'] =     ['v',
                                    'bulk_time',
                                    'surface v-momentum stress',
                                    'Newton meter-2']
            frc_vars['uwnd'] =     ['u',
                                    'bulk_time',
                                    '10m u-wind',
                                    'm s-1']
            frc_vars['vwnd'] =     ['v',
                                    'bulk_time',
                                    '10m v-wind',
                                    'm s-1']
            
        # dQdSST
        else:            
            frc_vars['sustr'] =  ['u',
                                  'sms_time',
                                  'surface u-momentum stress',
                                  'Newton meter-2']
            frc_vars['svstr'] =  ['v',
                                  'sms_time',
                                  'surface v-momentum stress',
                                  'Newton meter-2']   
            frc_vars['shflux'] = ['rho',
                                  'shf_time',
                                  'surface net heat flux',
                                  'Watts meter-2']
            frc_vars['swflux'] = ['rho',
                                  'swf_time',
                                  'surface freshwater flux (E-P)',
                                  'centimeters day-1',
                                  'net evaporation',
                                  'net precipitation']
            frc_vars['SST'] =    ['rho',
                                  'sst_time',
                                  'sea surface temperature',
                                  'Celsius']
            frc_vars['SSS'] =    ['rho',
                                  'sss_time',
                                  'sea surface salinity',
                                  'PSU']
            frc_vars['dQdSST'] = ['rho',
                                  'sst_time',
                                  'surface net heat flux sensitivity to SST',
                                  'Watts meter-2 Celsius-1']
            frc_vars['swrad'] =  ['rho',
                                  'srf_time',
                                  'solar shortwave radiation',
                                  'Watts meter-2',
                                  'downward flux, heating',
                                  'upward flux, cooling']
        
        
        for key, value in zip(frc_vars.keys(), frc_vars.values()):

            #print key, value

            if 'time' in value[0]:
                dims = (value[1])

            elif 'rho' in value[0]:
                dims = (value[1], 'eta_rho', 'xi_rho')

            elif 'u' in value[0]:
                dims = (value[1], 'eta_rho', 'xi_u')
                    
            elif 'v' in value[0]:
                dims = (value[1], 'eta_v', 'xi_rho')
                
            else:
                error
            
            #print 'key, dims = ',key, dims
            nc.createVariable(key, 'f4', dims, zlib=True)
            nc.variables[key].long_name = value[2]
            nc.variables[key].units = value[3]
            
            if 'time' in key and nr is not None:
                nc.variables[key].cycle_length = cl
            
            if 'swrad' in key:
                nc.variables[key].positive = value[4]
                nc.variables[key].negative = value[5]
                
            if 'swflux' in key:
                nc.variables[key].positive = value[4]
                nc.variables[key].negative = value[5]
                
            if 'radlw' in key:
                nc.variables[key].positive = value[4]
                
            if 'radlw_in' in key:
                nc.variables[key].positive = value[4]
                
            if 'radsw' in key:
                nc.variables[key].positive = value[4]

        nc.close()

    def gc_dist(self, lon1, lat1, lon2, lat2):
        '''
        Use Haversine formula to calculate distance
        between one point and another
        !! lat and lon in radians !!
        '''
        r_earth = self.r_earth # Mean earth radius in metres (from scalars.h)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        dang = 2. * np.arcsin(np.sqrt(np.power(np.sin(0.5 * dlat), 2) + \
                    np.cos(lat2) * np.cos(lat1) * np.power(np.sin(0.5 * dlon), 2)))
        return r_earth * dang # distance
            
    #def get_runoff(self, swflux_data, dai_file, mon):
    def get_runoff(self, dai_file, mon):
        '''
        This needs to be speeded up.  Separate into 2 parts:
          1 def runoff_setup() to do one-time tasks
          2 get_runoff() to do variable tasks
        '''
        mon -= 1
        area = 1.
        area /= (self.pm() * self.pn())
        dx = 1.
        dx /= np.mean(self.pm())
        
        roms_dir = self.romsfile.replace(self.romsfile.split('/')[-1], '')
        cdist = io.loadmat(roms_dir + 'coast_distances.mat')
        mask = self.maskr()
        
        # Exponential decay, runoff forced towards coast.  75km ?
        #mult = np.exp(-cdist['cdist'] / 150.e4)
        mult = np.exp(-cdist['cdist'] / 60.e4)
        np.place(mult, mask > 1., 0)

        # Read in river data and set data trim
        lon0 = np.round(self.lon().min() - 1.0)
        lon1 = np.round(self.lon().max() + 1.0)
        lat0 = np.round(self.lat().min() - 1.0)
        lat1 = np.round(self.lat().max() + 1.0)

        nc = netcdf.Dataset(dai_file)
        lon_runoff = nc.variables['lon'][:]
        lat_runoff = nc.variables['lat'][:]

        i0 = np.nonzero(lon_runoff > lon0)[0].min()
        i1 = np.nonzero(lon_runoff < lon1)[0].max() + 1
        j0 = np.nonzero(lat_runoff > lat0)[0].min()
        j1 = np.nonzero(lat_runoff < lat1)[0].max() + 1

        flux_unit = 1.1574e-07 # (cm/day)

        surf_ro = np.zeros(self.lon().shape)

        lon_runoff = lon_runoff[i0:i1]
        lat_runoff = lat_runoff[j0:j1]

        runoff = nc.variables['runoff'][mon, j0:j1, i0:i1]
        nc.close()
        
        lon_runoff, lat_runoff = np.meshgrid(lon_runoff, lat_runoff)
        lon_runoff = lon_runoff[runoff >= 1e-5]
        lat_runoff = lat_runoff[runoff >= 1e-5]
        runoff = runoff[runoff >= 1e-5]
        n_runoff = runoff.size
   
        for idx in np.arange(n_runoff):
      
            #print idx, n_runoff

            # Find suitable unmasked grid points to distribute run-off
            dist = self.gc_dist(np.deg2rad(self.lon()),
                                np.deg2rad(self.lat()),
                                np.deg2rad(lon_runoff[idx]),
                                np.deg2rad(lat_runoff[idx]))
        
            if dist.min() <= 5. * dx:
                #scale = 150. * 1e3 # scale of Dai data is at 1 degree
                scale = 60. * 1e3 # scale of Dai data is at 1 degree
                bool_mask = (dist < scale) * (mask > 0)
                int_are = area[bool_mask].sum()
                ave_wgt = mult[bool_mask].mean()

                surface_flux = 1e6 * runoff[idx]
                #print 'surface_flux, int_are', surface_flux, int_are
                surface_flux /= np.array([int_are, np.spacing(2)]).max()
                #print 'surface_flux',surface_flux
                surf_ro[bool_mask] = (surf_ro[bool_mask] + surface_flux /
                                      flux_unit * mult[bool_mask] / ave_wgt)

        #return (swflux_data * mask) - (surf_ro * mask) comment Sep 2013
        return surf_ro * mask

    def get_runoff_index_weights(self, dtnum):
        '''
        Get indices and weights for Dai river climatology
        Input: dt - datenum for current month
        Output: dai_ind_min
                dai_ind_max
                weights
        '''
        dtdate = dt.num2date(dtnum)
        day_plus_hr = np.float(dtdate.day + dtdate.hour / 24.)
        x_monrange = ca.monthrange(dtdate.year, dtdate.month)[-1]
        x_half_monrange = 0.5 * x_monrange
        if dtdate.day > x_half_monrange:
            # Second half of month, use present and next months
            dai_ind_min = dtdate.month - 1
            dai_ind_max = dtdate.month
            if dai_ind_max == 12:
                dai_ind_max = 0
                y_monrange = ca.monthrange(dtdate.year + 1, 1)[-1]
            else:
                y_monrange = ca.monthrange(dtdate.year, dtdate.month)[-1]
            x = day_plus_hr - x_half_monrange            
            y = 0.5 * y_monrange
            y += x_monrange
            y -= day_plus_hr
            w1, w2 = y, x
        else:
            # First half of month, use previous and present months
            dai_ind_min = dtdate.month - 2
            dai_ind_max = dtdate.month - 1
            if dai_ind_min < 0:
                dai_ind_min = 11
                y_monrange = ca.monthrange(dtdate.year - 1, 12)[-1]
            else:
                y_monrange = ca.monthrange(dtdate.year, dtdate.month - 1)[-1]
            x = x_half_monrange - day_plus_hr
            y = 0.5 * y_monrange
            y += day_plus_hr
            w1, w2 = x, y
        xpy = x + y
        weights = [w1 / xpy, w2 / xpy]
        return dai_ind_min, dai_ind_max, weights


class ERAGrid(ROMS):
    '''
    ERA grid class (inherits from RomsGrid class)
    '''
    def __init__(self, filename, model_type):
        '''
        
        '''
        super(ERAGrid, self).__init__(filename, model_type)
        print('Initialising ERAGrid', filename)
        self._lon = self.read_nc('lon', indices='[:]')
        self._lat = self.read_nc('lat', indices='[:]')
        self._lon, self._lat = np.meshgrid(self._lon,
                                           self._lat[::-1])
       
    def lon(self):
        return self._lon
    
    def lat(self):
        return self._lat
   
    def metrics(self):
        '''
        Return array of metrics unique to this grid
          (lonmin, lonmax, lon_res, latmin, latmax, lat_res)
          where 'res' is resolution in degrees
        '''
        self_shape = self.lon().shape
        lon_range = self.read_nc_att('lon', 'valid_range')
        lon_mean = self.lon().mean().round(2)
        lat_mean = self.lat().mean().round(2)
        res = np.diff(lon_range) / self.lon().shape[1]
        met = np.hstack((self_shape[0], self_shape[1], lon_mean, lat_mean, res))
        return met
        
    def resolution(self):
        '''
        Return the resolution of the data in degrees
        '''
        return self.metrics()[-1]
   
    def assert_resolution(self, eragrd_tmp, key):
        '''
        '''
        assert self.metrics()[2] <= eragrd_tmp.metrics()[2], \
            'Resolution of %s is lower than previous grids, move up in dict "era_files"' %key


class ERAData(RomsData):
    '''
    ERA data class (inherits from RomsData class)
    '''
    def __init__(self, filename, varname, model_type, romsgrd, masks=None):
        '''
        
        '''
        super(ERAData, self).__init__(filename, model_type)
        
        #print 'bbbbbbbbbbbbb'
        #classname = str(self.__class__).partition('.')[-1].strip("'>")
        #print '\n--- Instantiating *%s* instance from' %classname, filename
        self.varname = varname
        #self._check_product_description()
        self.needs_time_averaging = False
        #if 'mask' not in (str(type(self)).lower()):
        #    self._set_averaging_weights()
        #self._set_start_end_dates('ref_date_time')
        self._lon = self.read_nc('longitude', indices='[:]')
        self._lat = self.read_nc('latitude', indices='[:]')
        self._lon, self._lat = np.meshgrid(self._lon,
                                           self._lat[::-1])
        self._get_metrics()
        if masks is not None:
            self._select_mask(masks)
            self._maskr = self.eramsk._maskr
            self._maskr[self._maskr>0] = 1
            self.fillmask_cof = self.eramsk.fillmask_cof
        else:
            self.eramsk = None
            self.fillmask_cof = None
        self.romsgrd = romsgrd
        self.datain = np.ma.empty(self.lon().shape)
        self.datatmp = np.ma.empty(self.lon().shape)
        self.dataout = np.ma.empty(self.romsgrd.lon().size)
        self._datenum = self.read_nc('time', indices='[:]') #self._get_time_series()
        self.tind = None
        self.tri = None
        self.tri_all = None
        self.dt = None
        self.to_be_downscaled = False
        self.needs_all_point_tri = False
    
    def lon(self):
        return self._lon
    
    def lat(self):
        return self._lat
    
    def maskr(self):
        ''' Returns mask on rho points
        '''
        return self._maskr
    
    def _select_mask(self, masks):
        '''Loop over list of masks to find which one
           has same grid dimensions as self
        '''
        for each_mask in masks:
            if np.alltrue(each_mask.metrics == self.metrics):
                self.eramsk = each_mask
                #self._maskr = each_mask._get_landmask()
                return self
        return None
    
    
    def fillmask(self):
        '''Fill missing values in an array with an average of nearest  
           neighbours
           From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        Order:  call after self.get_fillmask_cof()
        '''
        dist, iquery, igood, ibad = self.fillmask_cof
        weight = dist / (dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
        np.place(weight, weight > 1., 0.)
        xfill = weight * self.datain[igood[:,0][iquery], igood[:,1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)
        self.datain[ibad[:,0], ibad[:,1]] = xfill
        return self
    
    
    def _set_start_end_dates(self, varname):
        '''
        '''
        #print 'varname', varname
        self._start_date = self.read_nc(varname, '[0]')
        #print 'fff',self._start_date.dtype
        #print self._start_date
        #aaaa
        try:
            self._end_date = self.read_nc(varname, '[-1]')
        except Exception:
            self._end_date = self._start_date
        return self
        
        
    def datenum(self):
        return self._datenum

    def _date2num(self, date):
        '''
        Convert ERA 'valid_date_time' to datenum
        Input: date : ndarray (e.g., "'2' '0' '1' '0' '1' '2' '3' '1' '1' '8'")
        '''
        #print 'dateee',date, date.size
        assert (isinstance(date, np.ndarray) and
                date.size == 10), 'date must be size 10 ndarray'
        return dt.date2num(dt.datetime.datetime(np.int(date.tostring()[:4]),
                                                  np.int(date.tostring()[4:6]),
                                                  np.int(date.tostring()[6:8]),
                                                  np.int(date.tostring()[8:10])))

    def _get_time_series(self):
        '''
        '''
        #print 'self._start_date', self._start_date
        date0 = self._date2num(self._start_date)
        date1 = self._date2num(self._end_date)
        datelen = self.read_dim_size('time')
        datenum = np.linspace(date0, date1, datelen)
        return datenum
        #if np.unique(np.diff(datenum)).size == 1:
            #return datenum
        #else:
            #raise 'dsdsdsd'




    def get_delaunay_tri(self):
        '''
        '''
        self.points_all = np.copy(self.points)
        self.tri_all = sp.Delaunay(self.points_all)
        self.points = np.array([self.points[:,0].flat[self.eramsk.era_ball],
                                self.points[:,1].flat[self.eramsk.era_ball]]).T            
        self.tri = sp.Delaunay(self.points)
        return self
    
    def reshape2roms(self):
        '''
        Following interpolation with horizInterp() we need to
        include land points and reshape
        '''
        self.dataout = self.dataout.reshape(self.romsgrd.lon().shape)
        return self
    
    def _check_for_nans(self):
        '''
        '''
        flat_mask = self.romsgrd.maskr().ravel()
        assert not np.any(np.isnan(self.dataout[np.nonzero(flat_mask)])
                          ), 'Nans in self.dataout sea points'
        self.dataout[:] = np.nan_to_num(self.dataout)
        return self
    
    
    def _interp2romsgrd(self):
        '''
        '''
        ball = self.eramsk.era_ball
        interp = horizInterp(self.tri, self.datain.flat[ball])
        self.dataout[self.romsgrd.idata()] = interp(self.romsgrd.points)
        return self
    
    def interp2romsgrd(self, fillmask=False):
        '''
        '''
        if fillmask:
            self.fillmask()
        self._interp2romsgrd()
        self._check_for_nans()
        return self
    
    
    def check_interp(self):
        '''
        '''
        fac = 5
        cmap = plt.cm.gist_ncar
        rlonmin, rlonmax = self.romsgrd.lon().min(), self.romsgrd.lon().max()
        rlatmin, rlatmax = self.romsgrd.lat().min(), self.romsgrd.lat().max()
        erai0, erai1 = (np.argmin(np.abs(self.lon()[0] - rlonmin)) - fac,
                          np.argmin(np.abs(self.lon()[0] - rlonmax)) + fac)
        eraj0, eraj1 = (np.argmin(np.abs(self.lat()[:,0] - rlatmin)) - fac,
                          np.argmin(np.abs(self.lat()[:,0] - rlatmax)) + fac)
        era = np.ma.masked_where(self.maskr() == 0, self.datain.copy())
        try:
            roms = np.ma.masked_where(self.romsgrd.maskr() == 0,
                                      self.dataout.copy())
        except Exception:
            
            roms = np.ma.masked_where(self.romsgrd.maskr() == 0,
                                      self.dataout.reshape(self.romsgrd.maskr().shape))
        cmin, cmax = roms.min(), roms.max()
        plt.figure()
        plt.pcolormesh(self.lon()[eraj0:eraj1, erai0:erai1],
                       self.lat()[eraj0:eraj1, erai0:erai1],
                             era[eraj0:eraj1, erai0:erai1], cmap=cmap, edgecolors='w')
        plt.clim(cmin, cmax)
        plt.pcolormesh(self.romsgrd.lon(), self.romsgrd.lat(), roms, cmap=cmap)
        plt.clim(cmin, cmax)
        plt.axis('image')
        plt.colorbar()
        plt.show()
        
        
    def get_date_index(self, other, ind):
        '''
        '''
        return np.nonzero(self.datenum() == other.datenum()[ind])[0][0]
        


    def _get_metrics(self):
        '''Return array of metrics unique to this grid
           (lonmin, lonmax, lon_res, latmin, latmax, lat_res)
           where 'res' is resolution in degrees
        '''
        self_shape = self._lon.shape
        lon_range = [self._lon[0][0],self._lon[0][-1]]#self.read_nc_att('longitude', 'valid_range')
        lon_mean, lat_mean = (self._lon.mean().round(2),
                              self._lat.mean().round(2))
        res = np.diff(lon_range) / self._lon.shape[1]
        #print(self_shape,self._lon[0][0],self._lon[0][-1])
        #exit()
        #print(self_shape[0], self_shape[1], lon_mean, lat_mean, res)
        self.metrics = np.hstack((self_shape[0], self_shape[1], lon_mean, lat_mean, res))
        return self



    def _read_era_frc(self, var, ind):
        ''' Read ERA forcing variable (var) at record (ind)
        '''
        return self.read_nc(var, '[' + str(ind) + ']')[::-1]
        
    
    
    def _get_era_data(self, varname):
        ''' Get ERA data with explicit variable name
        '''
        # Fill the mask
        #outvar = self.fillmask(outvar, self.outmask, flags[era_key])
        return self._read_era_frc(varname, self.tind)
    
    
    def _get_era_datatmp(self):
        ''' Get ERA data with implicit variable name
        '''
        self.datatmp[:] = self._read_era_frc(self.varname, self.tind-1)
        return self
    
    
    def get_era_data(self):
        ''' Get ERA data with implicit variable name
        '''
        self.datain[:] = self._read_era_frc(self.varname, self.tind)
        if self.needs_time_averaging:
            self._get_era_datatmp()
            self._get_data_time_average()
        return self
    
    
    def set_date_index(self, dt):
        try:
            self.tind = np.nonzero(self.datenum() == dt)[0][0]
        except Exception:
            raise # dt out of range in ERA file
        else:
            return self
        
    
    #def vd_time(self):
    #    '''
    #    Valid date and time as YYYYMMDDHH
    #    '''
    #    return self.read_nc('valid_date_time')
        
        
    def time(self):
        '''
        Hours since 1900-01-01 00:00:00.0 +0:00
        '''
        return self.read_nc('time')
        
   
    
    def check_vars_for_downscaling(self, var_instances):
        '''Loop over list of grids to find which have
           dimensions different from self
        '''       
        for ind, each_grid in enumerate(var_instances):
            print(each_grid.metrics,self.metrics)
            if np.any(each_grid.metrics != self.metrics):
                each_grid.to_be_downscaled = True
        return self
    
    
    
    def dQdSST(self, sst, sat, rho_air, U, qsea):
        '''
        Compute the kinematic surface net heat flux sensitivity to the
        the sea surface temperature: dQdSST.
        Q_model ~ Q + dQdSST * (T_model - SST)
        dQdSST = - 4 * eps * stef * T^3  - rho_air * Cp * CH * U
                 - rho_air * CE * L * U * 2353 * ln (10 * q_s / T^2)
    
        Input parameters:
        sst     : sea surface temperature (Celsius)
        sat     : sea surface atmospheric temperature (Celsius)
        rho_air : atmospheric density (kilogram meter-3) 
        U       : wind speed (meter s-1)
        qsea    : sea level specific humidity (kg/kg)
 
        Output:
        dqdsst  : kinematic surface net heat flux sensitivity to the
                  the sea surface temperature (Watts meter-2 Celsius-1)
        
        From Roms_tools of Penven etal
        '''
        # SST (Kelvin)
        sst = np.copy(sst)
        sst += self.Kelvin
        # Latent heat of vaporisation (J.kg-1)
        L = np.ones(sat.shape)
        L *= self.L1
        L -= (self.L2 * sat)
        
        # Infrared contribution
        q1 = -4.
        # Multiply by Stefan constant
        q1 *= self.Stefan
        q1 *= np.power(sst, 3)
        
        # Sensible heat contribution
        q2 = -rho_air
        q2 *= self.Cp
        q2 *= self.Ch
        q2 *= U
        
        # Latent heat contribution
        dqsdt = 2353.
        dqsdt *= np.log(10.)
        dqsdt *= qsea
        dqsdt /= np.power(sst, 2)
        q3 = -rho_air
        # Multiply by Ce (Latent heat transfer coefficient, stable condition)
        q3 *= self.Ce
        q3 *= L
        q3 *= U
        q3 *= dqsdt
        dQdSST = q1
        dQdSST += q2
        dQdSST += q3
        return dQdSST



class ERAMask(ERAData):
    '''Mask class (inherits from ERAData class)
    '''
    def __init__(self, mask_file, romsgrd, balldist):
        super(ERAMask, self).__init__(mask_file[0], mask_file[1], 'ERA', romsgrd)
        self.tind = 0
        self.get_era_data()
        self.balldist = balldist
        self.romsgrd = romsgrd
        self._maskr = np.abs(self.datain - 1)
        #self.maskr()
        self.proj2gnom(ignore_land_points=False, M=romsgrd.M)
        self.make_kdetree()
        self.has_ball = False
        self.get_kde_ball()
        self.eramsk = None
        self.get_fillmask_cof(self.maskr())
        
    def get_kde_ball(self):
        '''
        '''
        self.era_ball = self.kdetree.query_ball_tree(self.romsgrd.kdetree, self.balldist)
        self.era_ball = np.array(self.era_ball).nonzero()[0]
        self.has_ball = True
        return self
        

class ERADataHourly(RomsData):
    '''
    ERA data class (inherits from RomsData class)
    '''
    def __init__(self, filename, varname, model_type, romsgrd, masks=None):
        '''
        
        '''
        super(ERADataHourly, self).__init__(filename, model_type)
        self.varname = varname
         
        self.needs_averaging_to_hour = False
        self.needs_averaging_to_the_half_hour = False
             
        self._datenum = self._get_time_series()
        self._date_start = self._datenum[0]
        self._date_end = self._datenum[-1]
              
        self._start_averaging = False
        
        self._lon = self.read_nc('longitude', indices='[:]')
        self._lat = self.read_nc('latitude', indices='[:]')
        self._lon, self._lat = np.meshgrid(self._lon,
                                           self._lat[::-1])
        self._get_metrics()
        if masks is not None:
            self._select_mask(masks)
            self._maskr = self.eramsk._maskr
            self.fillmask_cof = self.eramsk.fillmask_cof
        else:
            self.eramsk = None
            #self.fillmask_cof = None
        self.romsgrd = romsgrd
        self.datain = np.ma.empty(self.lon().shape)
        if self.needs_averaging_to_hour:
            self.previous_in = np.ma.empty(self.lon().shape)
        elif self.needs_averaging_to_the_half_hour:
            self.datatmp = np.ma.empty(self.lon().shape)
        self.dataout = np.ma.empty(self.romsgrd.lon().size)
        self.tind = None
        self.tri = None
        self.tri_all = None
        self.dt = None
        self.to_be_downscaled = True
        self.needs_all_point_tri = False
    
    def _get_time_series(self):
        ERA_Time = self.read_nc('time', indices='[:]')
        ERA_Basetime = dt.date2num(dt.datetime.datetime(1900,1,1))
        time_series = ne.evaluate('(ERA_Time / 24) + ERA_Basetime')
        return time_series    
    
    def date_start(self):
        return self._date_start
        
    def date_end(self):
        return self._date_end
    
    def lon(self):
        return self._lon
    
    def lat(self):
        return self._lat
    
    def maskr(self):
        ''' Returns mask on rho points
        '''
        return self._maskr
    
    def _read_era_frc(self, var, ind):
        ''' Read ERA forcing variable (var) at record (ind)
        '''
        #print(ind)
        return self.read_nc(var, '[' + str(ind) + ']')[::-1]
    
    def _get_era_data(self, varname):
        ''' Get ERA data with explicit variable name
        '''
        return self._read_era_frc(varname, self.tind)
    
    def _get_era_datatmp(self):
        ''' Get ERA data with implicit variable name
        '''
        #print self.tind, self.tind-1
        self.datatmp[:] = self._read_era_frc(self.varname, self.tind-1)
        return self
    
    
    def get_era_data(self):
        ''' Get ERA data with implicit variable name
        '''
        self.datain[:] = self._read_era_frc(self.varname, self.tind)
        return self
    
        
    def get_era_data_previous(self):
        ''' Get ERA data with implicit variable name
        '''
        self.previous_in[:] = self._read_era_frc(self.varname, self.tind - 1)
        return self
    
    def era_data_subtract_averages(self):
        ''' Extract one hour average from accumulated averages
        '''
        self.datain *= self.forecast_hour()
        self.previous_in *= (self.forecast_hour() - 1)
        self.datain -= self.previous_in
        return self
        
    
    def _check_for_nans(self, message=None):
        '''
        '''
        flat_mask = self.romsgrd.maskr().ravel()
        assert not np.any(np.isnan(self.dataout[np.nonzero(flat_mask)])
                          ), 'Nans in self.dataout sea points; hints: %s' %message
        self.dataout[:] = np.nan_to_num(self.dataout)
        return self
    
    
    def _interp2romsgrd(self):
        '''
        '''
        ball = self.eramsk.era_ball
        interp = horizInterp(self.tri, self.datain.flat[ball])
        self.dataout[self.romsgrd.idata()] = interp(self.romsgrd.points)
        return self
    
    def interp2romsgrd(self, fillmask=False):
        '''
        '''
        if fillmask:
            self.fillmask()
        self._interp2romsgrd()
        self._check_for_nans()
        return self   
    
    def datenum(self):
        return self._datenum    
    
    def set_date_index(self, dt):
        tind = np.argsort(np.abs(self._datenum - dt))[0]
        if self.tind == tind:
            self.tind += 1
        else:
            self.tind = tind
        return self
        
    def _get_metrics(self):
        '''Return array of metrics unique to this grid
           (lonmin, lonmax, lon_res, latmin, latmax, lat_res)
           where 'res' is resolution in degrees
        '''
        
        self_shape = self._lon.shape
        lon_range = [self._lon[0][0],self._lon[0][-1]]#self.read_nc_att('longitude', 'valid_range')
        lon_mean, lat_mean = (self._lon.mean().round(2),
                              self._lat.mean().round(2))
        res = np.diff(lon_range) / self._lon.shape[1]
        #print(self_shape,self._lon[0][0],self._lon[0][-1])
        #exit()
        #print(self_shape[0], self_shape[1], lon_mean, lat_mean, res)
        self.metrics = np.hstack((self_shape[0], self_shape[1], lon_mean, lat_mean, res))
      
        return self

    def get_delaunay_tri(self):
        '''
        '''
        self.points_all = np.copy(self.points)
        self.tri_all = sp.Delaunay(self.points_all)
        self.points = np.array([self.points[:,0].flat[self.eramsk.era_ball],
                                self.points[:,1].flat[self.eramsk.era_ball]]).T            
        self.tri = sp.Delaunay(self.points)
        return self
    
    
    def _select_mask(self, masks):
        '''Loop over list of masks to find which one
           has same grid dimensions as self
        '''
        self.eramsk = masks[0]

        return None

    
    def fillmask(self):
        '''Fill missing values in an array with an average of nearest  
           neighbours
           From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        Order:  call after self.get_fillmask_cof()
        '''
        dist, iquery, igood, ibad = self.fillmask_cof
        weight = dist / (dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
        np.place(weight, weight > 1., 0.)
        xfill = weight * self.datain[igood[:,0][iquery], igood[:,1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:, np.newaxis]).sum(axis=1)
        self.datain[ibad[:,0], ibad[:,1]] = xfill
        return self


    def check_vars_for_downscaling(self, var_instances):
        '''Loop over list of grids to find which have
           dimensions different from self
        '''
        
        for ind, each_grid in enumerate(var_instances):
            if (each_grid.metrics != self.metrics).any():
                each_grid.to_be_downscaled = True
        return self


class ERAPrate(ERADataHourly):
    '''ERA Precipitation rate class (inherits from ERAData class)
       Responsible for one variable: 'prate'
    '''
    def __init__(self, prate_file, masks, romsgrd):
        super(ERAPrate, self).__init__(prate_file[0], prate_file[1], 'ERA',
                                        romsgrd, masks=masks)

    def convert_cmday(self):
        datain = self.datain
        self.datain[:] = datain * 86400 * 0.1
        return self


class ERARadNetlw(ERADataHourly):
    '''ERA Outgoing longwave radiation class (inherits from ERAData class)
       Responsible for two ROMS bulk variables: 'radlw' and 'radlw_in'
    '''
    def __init__(self, radlw_file, masks, romsgrd):
        super(ERARadNetlw, self).__init__(radlw_file['shflux_LW_down'][0],
                                                   radlw_file['shflux_LW_down'][1], 'ERA',
                                                   romsgrd, masks=masks)
        self.down_varname = radlw_file['shflux_LW_down'][1]
        self.net_varname = radlw_file['shflux_LW_net'][1]
        self.radlw_datain = np.ma.empty(self.datain.shape)
        self.radlw_in_datain = np.ma.empty(self.datain.shape)
        self.radlw_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.radlw_in_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.numvars = 2        
    
    def get_era_radlw_and_radlw_in(self, lw_net_datain, lw_datain):
        self.radlw_datain[:] = ne.evaluate('(-1) * lw_net_datain')
        self.radlw_in_datain[:] = lw_datain
        return self
    
    def interp2romsgrd(self, fillmask1=False, fillmask2=False):
        self.datain[:] = self.radlw_datain
        if fillmask1:
            self.fillmask()
        self._interp2romsgrd()._check_for_nans()
        self.radlw_dataout[:] = self.dataout.reshape(self.romsgrd.lon().shape)
        self.datain[:] = self.radlw_in_datain
        if fillmask2:
            self.fillmask()
        self._interp2romsgrd()._check_for_nans()
        self.radlw_in_dataout[:] = self.dataout.reshape(self.romsgrd.lon().shape)
        return self
        

class ERARadlwNet(ERADataHourly):
    '''
    '''
    def __init__(self, radlw_file, masks, romsgrd):
        super(ERARadlwNet, self).__init__(radlw_file['shflux_LW_net'][0],
                                                   radlw_file['shflux_LW_net'][1], 'ERA',
                                                   romsgrd, masks=masks)
        
class ERARadlwDown(ERADataHourly):
    '''
    '''
    def __init__(self, radlw_file, masks, romsgrd):
        super(ERARadlwDown, self).__init__(radlw_file['shflux_LW_down'][0],
                                                   radlw_file['shflux_LW_down'][1], 'ERA',
                                                   romsgrd, masks=masks)
      
class ERARadSwDown(ERADataHourly):
    '''ERA Outgoing longwave radiation class (inherits from ERADataHourly class)
       Responsible for one ROMS bulk variable: 'radsw'
    '''
    def __init__(self, radsw_file, masks, romsgrd):
        super(ERARadSwDown, self).__init__(radsw_file[0], radsw_file[1],'ERA',
                                                   romsgrd, masks=masks)


class ERARadSwNet(ERADataHourly):
    '''
    '''
    def __init__(self,radlw_file, masks, romsgrd):
        super(ERARadSwNet, self).__init__(radlw_file[0], radlw_file[1],
                                          'ERA', romsgrd, masks=masks)

    
class ERADpair(ERADataHourly):
    '''ERA qair class (inherits from ERAData class)
       Responsible for one variable: 'qair'
    '''
    def __init__(self, dpair_file, masks, romsgrd):
        super(ERADpair, self).__init__(dpair_file[0], dpair_file[1], 'ERA',
                                       romsgrd, masks=masks)
        #self.print_weights()

class ERASat(ERADataHourly):
    '''ERA surface air temperature class (inherits from ERAData class)
       Responsible for one ROMS bulk variable: 'tair'
    '''
    def __init__(self, sat_file, masks, romsgrd):
        super(ERASat, self).__init__(sat_file[0], sat_file[1], 'ERA',
                                      romsgrd, masks=masks)
        #self.print_weights()
       

class ERASap(ERADataHourly):
    '''ERA surface air pressure class (inherits from ERAData class)
       Responsible for one variable: 'qair'
    '''
    def __init__(self, sap_file, masks, romsgrd):
        super(ERASap, self).__init__(sap_file[0], sap_file[1], 'ERA',
                                      romsgrd, masks=masks)
        #self.print_weights()

class ERAWspd(ERADataHourly):
    '''ERA wind speed class (inherits from ERAData class)
       Responsible for four variables: 'uspd', 'vspd', 'sustr' and 'svstr'
       Requires: 'qair', sap' and 'rhum'
    '''
    def __init__(self, wspd_file, masks, romsgrd):
        super(ERAWspd, self).__init__(wspd_file['uspd'][0],
                                                  wspd_file['uspd'][1], 'ERA',
                                                  romsgrd, masks=masks)
        #self.print_weights()
        self.uwnd_varname = wspd_file['uspd'][1]
        self.vwnd_varname = wspd_file['vspd'][1]
        self.wspd_datain = np.ma.empty(self.lon().shape)
        self.uwnd_datain = np.ma.empty(self.lon().shape)
        self.vwnd_datain = np.ma.empty(self.lon().shape)
        self.ustrs_datain = np.ma.empty(self.lon().shape)
        self.vstrs_datain = np.ma.empty(self.lon().shape)
        self._rair_datain = np.ma.empty(self.lon().shape)
        self.wspd_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.uwnd_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.vwnd_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.ustrs_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.vstrs_dataout = np.ma.empty(self.romsgrd.lon().shape)
        
        # Physical constants
        self.G = np.asarray(9.81) # acceleration due to gravity [m/s^2] (same ROMS as scalars.h)
        self.EPS_AIR = np.asarray(0.62197) # molecular weight ratio (water/air)
        self.C2K = np.asarray(273.15) # conversion factor for [C] to [K]
        self.GAS_CONST_R = np.asarray(287.04) # gas constant for dry air [J/kg/K]
        
         # Meteorological constants
        self.Z = 10. # Default measurement height [m]
        self.KAPPA = np.asarray(0.41) # von Karman's constant (same ROMS as scalars.h)
        self.CHARNOCK_ALPHA = np.asarray(0.011) # Charnock constant (for determining roughness length
        '''                            at sea given friction velocity), used in Smith
                                       formulas for drag coefficient and also in Fairall
                                       and Edson.  use alpha=0.011 for open-ocean and
                                       alpha=0.018 for fetch-limited (coastal) regions.'''
        self.R_ROUGHNESS = np.asarray(0.11) # Limiting roughness Reynolds # for aerodynamically
        '''                        smooth flow'''
        
        # Defaults suitable for boundary-layer studies  
        #self.cp = 1004.8 # heat capacity of air [J/kg/K] (same as Roms_tools dQdSST.m)
        #self.rho_air = 1.22 # air density (when required as constant) [kg/m^2]
        #self.Ta = 10. # default air temperature [C]
        #self.Pa = 1020. # default air pressure for Kinneret [mbars]
        #self.psych_default = 'screen' # default psychmometer type (see relhumid.m)
        # Saturation specific humidity coefficient reduced by 2% over salt water
        self.QSAT_COEFF = np.asarray(0.98)
        self.TOL = np.float128(0.00001)

    def qsat(self, Ta, Pa=None):
        '''
        QSAT: computes specific humidity at saturation.
        q=QSAT(Ta) computes the specific humidity (kg/kg) at saturation at
        air temperature Ta (deg C). Dependence on air pressure, Pa, is small,
        but is included as an optional input.

          INPUT:   Ta - air temperature  [C]
                   Pa - (optional) pressure [mb]
          
           OUTPUT:  q  - saturation specific humidity  [kg/kg]

        Version 1.0 used Tetens' formula for saturation vapor pressure
        from Buck (1981), J. App. Meteor., 1527-1532.  This version
        follows the saturation specific humidity computation in the COARE
        Fortran code v2.5b.  This results in an increase of ~5% in
        latent heat flux compared to the calculation with version 1.0.
        '''
        
        #q = (622 * 6.113 * np.exp(5423 * (Ta - 273.15)
        #/(Ta * 273.15)))/self.Pa # g/kg 

        if Pa is not None:
            self.Pa = Pa # pressure in mb
        
        Tv = np.copy(Ta) - 273.15
        
        # As in Fortran code v2.5b for COARE
        ew = 6.1121 * (1.0007 + 3.46e-6 * self.Pa) * np.exp((17.502 * Tv) / \
             (240.97 + Tv)) # in mb
        q  = 0.62197 * (ew / (self.Pa - 0.378 * ew)) # mb -> kg/kg
        return q
    
    def stresstc(self, sp, u, v, Ta, rho_air):
        '''
        taux, tauy = stresstc(u, v) computes the neutral wind stress given the
        wind speed and air temperature at height z following Smith (1988),
        J. Geophys. Res., 93, 311-326

        INPUT:  u, v  - wind speed components   [m/s]
                z     - measurement height  [m]
                Ta    - air temperature [C]
                rho_air  - air density  [kg/m^3]

        OUTPUT: taux, tauy - wind stress components  [N/m^2]
        Adapted from http://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/as_consts.html
        '''
        # NOTE: windspd is calculated in __main__ before call to self.stresstc, however if
        # windspd_fill_mask == True there can be issues at the coast in taux, tauy
        # Hence, we calculate it again without mask filling
        #sp = np.hypot(u, v) # this line commented Nov2014 cos redundant from _get_wspd
        cd, u10 = self.cdntc(sp, Ta)
        taux = np.copy(rho_air)
        tauy = np.copy(rho_air)
        taux *= u10
        taux *= u
        taux *= cd
        tauy *= u10
        tauy *= v
        tauy *= cd
        return taux, tauy

    def viscair(self, Ta):
        '''
        vis=VISCAIR(Ta) computes the kinematic viscosity of dry air as a
        function of air temperature following Andreas (1989), CRREL Report
        89-11.

        INPUT: Ta - air temperature [C]
            1.326e-5*(1 + 6.542e-3*Ta + 8.301e-6*Ta.^2 - 4.84e-9*Ta.^3)
        OUTPUT: vis - air viscosity [m^2/s]
        '''
        Tv = np.asarray(Ta) - 273.15
        #va = 6.542e-3 * Ta
        #va += 8.301e-6 * Ta**2
        #va -= 4.84e-9 * Ta**3
        #va += 1
        #va *= 1.326e-5
        va = ne.evaluate('1.326e-5 * (1 + 6.542e-3 * Tv + 8.301e-6 * \
                          Tv**2 - 4.84e-9 * Tv**3)')
        return va
    
    def cdntc(self, sp, Ta):
        '''
        cd, u10 = cdntc(sp, z , Ta) computes the neutral drag coefficient and
        wind speed at 10m given the wind speed and air temperature at height z
        following Smith (1988), J. Geophys. Res., 93, 311-326.
    
        INPUT:   sp - wind speed  [m/s]
             self.Z - measurement height [m]
            self.Ta - air temperature (optional)  [C]
    
        OUTPUT:  cd - neutral drag coefficient at 10m
                  u10 - wind speed at 10m  [m/s]
        ''' 
        R_roughness = self.R_ROUGHNESS
        z = self.Z
        kappa = self.KAPPA
        Charnock_alpha = self.CHARNOCK_ALPHA
        g = self.G
        
        # Iteration endpoint
        TOL = self.TOL
        
        # Get air viscosity
        visc = self.viscair(Ta)
        #print '----------visc',visc
        
        # Remove any sp==0 to prevent division by zero
        np.place(sp, sp == 0, 0.1)
        
        # Initial guess
        ustaro = np.zeros(sp.shape)
        ustarn = np.copy(sp)
        ustarn = ne.evaluate('ustarn * 0.036')
        #print 'ustarn1', ustarn
        
        # Iterate to find z0 and ustar
        while np.any(ne.evaluate('abs(ustarn - ustaro)') > TOL):
            ustaro = np.copy(ustarn)
            z0 = ne.evaluate('Charnock_alpha * ustaro**2 / g + \
                              R_roughness * visc / ustaro')
            ustarn = ne.evaluate('sp * (kappa / log(z / z0))')
     
        #print 'ustarn2', ustarn
        sqrcd = ne.evaluate('kappa / log(10. / z0)')
        cd = ne.evaluate('sqrcd**2') # np.power(sqrcd, 2)
        #u10 = ustarn / sqrcd
        u10 = ne.evaluate('ustarn / sqrcd')
        #print cd, u10
        return cd, u10
    
    def air_dens(self, Ta, Tdc, Pa):
       
        if Pa is not None:
            self.Pa = Pa # pressure in mb
        
        tair = Ta - 273.15
        tdair = Tdc - 273.15                
        Es = 6.1121 * (1.0007 + 3.46e-6 * Pa) * np.exp((17.502 * tair) / (240.97 + tair))
        E = 6.1121 * (1.0007 + 3.46e-6 * Pa) * np.exp((17.502 * tdair) / (240.97 + tdair))         
        rhum = (E/Es)*100
        
        Q = self.qsat(Ta, Pa)
        Q *= 0.01
        Q *= rhum # specific humidity of air [kg/kg]
          
        o_six_one = 1. / self.EPS_AIR - 1. # 0.61 (moisture correction for temp.)
        o_six_one *= Q
        o_six_one += 1.
        Tv = np.copy(Ta)
        #Tv += self.C2K
        Tv *= o_six_one # air virtual temperature
        Tv *= self.GAS_CONST_R
        rhoa = np.copy(Pa)
        rhoa *= 100.
        rhoa /= Tv # air density [kg/m^3]
        return rhoa
   
    def _get_wstrs(self, dp_data, sat_data, sap_data):
        sap_data[:] = ne.evaluate('sap_data * 0.01') # convert from Pa to mb
        # Smith etal 1988
        self._rair_datain[:] = self.air_dens(sat_data, dp_data, sap_data)
        self.ustrs_datain[:], self.vstrs_datain[:] = self.stresstc(self.wspd_datain,
                                                                     self.uwnd_datain,
                                                                     self.vwnd_datain,
                                                                     sat_data,
                                                                     self._rair_datain)

    def _get_wspd(self):
        self.uwnd_datain[:] = self._get_era_data(self.uwnd_varname)
        self.vwnd_datain[:] = self._get_era_data(self.vwnd_varname)
        np.hypot(self.uwnd_datain, self.vwnd_datain, out=self.wspd_datain)

    def get_winds(self, dp, sat, sap):
        self._get_wspd()
        self._get_wstrs(dp.datain, sat.datain, sap.datain)

    def interp2romsgrd(self):
        roms_shape = self.romsgrd.lon().shape
        self.datain[:] = self.wspd_datain
        self._interp2romsgrd()._check_for_nans()
        self.wspd_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.uwnd_datain
        self._interp2romsgrd()._check_for_nans()
        self.uwnd_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.vwnd_datain
        self._interp2romsgrd()._check_for_nans()
        self.vwnd_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.ustrs_datain
        self._interp2romsgrd()._check_for_nans('Nans here could indicate')
        self.ustrs_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.vstrs_datain
        self._interp2romsgrd()._check_for_nans()
        self.vstrs_dataout[:] = self.dataout.reshape(roms_shape)
        return self

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '')
    # Print New Line on Complete
    if iteration == total: 
        print()

def Process_ERA(ERA_File,roms_grd,frc_filename,overlap,overlap_file,params):
    
    print('\n\nProcessing ERA5 single level bulk formulations')
    print('----------------------------------------------')
    print(ERA_File)
    print('\n')
    
    prate_file = (ERA_File,  'tp')
    shflux_SW_down_file = (ERA_File,  'msdwswrf')
    shflux_SW_net_file = (ERA_File,  'msnswrf')
    shflux_LW_down_file = (ERA_File,  'msdwlwrf')
    shflux_LW_net_file = (ERA_File,  'msnlwrf')
    shflux_LATENT_HEAT_file = (ERA_File,  'mslhf')
    shflux_SENSIBLE_HEAT_file = (ERA_File,  'msshf')
    sustr_file = (ERA_File,  'u10')
    svstr_file = (ERA_File,  'v10')
    sat_file = (ERA_File,  't2m')
    sap_file = (ERA_File,  'msl')
    sadp_file = (ERA_File,  'd2m')
    mask_one = (ERA_File, 'lsm') 
    era_masks = OrderedDict([('mask_one', mask_one)])
   
    cycle_length = np.float(0)
    add_dai_runoff = False
  
    # Interpolation / processing parameters_________________________________
    
    #balldist = 100000. # distance (m) for kde_ball (should be 2dx at least?)
    
    # Filling of landmask options
    # Set to True to extrapolate sea data over land
    #winds_fillmask = False # 10 m (False recommended)
    #sat_fillmask  = True # 2 m (True recommended)
    #rhum_fillmask = True # 2 m (True recommended)
    #qair_fillmask = True # 2 m (True recommended)
    
    #windspd_fillmask = True # surface (True recommended)
    fillmask_radlw, fillmask_radlw_in = True, True
    
    sigma_params = dict(theta_s=None, theta_b=None, hc=None, N=None)

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    timing_warning = 'bulk_time at tind is different from roms_day'
    
    # This dictionary of ERA files needs to supply some or all of the surface
    # forcing variables variables:
   
    era_files = OrderedDict([
            ('prate', prate_file),
            ('radnetlw', OrderedDict([
                ('shflux_LW_down', shflux_LW_down_file),
            ('shflux_LW_net', shflux_LW_net_file)])),
            ('radnetsw', OrderedDict([
                ('shflux_SW_down', shflux_SW_down_file),
            ('shflux_SW_net', shflux_SW_net_file)])),
            ('wspd', OrderedDict([
                ('sat', sat_file),
            ('sap', sap_file),
            ('sadp', sadp_file),
            ('uspd', sustr_file),
            ('vspd', svstr_file)]))])
    
    overlap_offset = 0
    
    # Instantiate a RomsGrid object
    numrec = None
    romsgrd = RomsGrid(roms_grd, sigma_params, 'ROMS')
    romsgrd.create_frc_nc(frc_filename, numrec, cycle_length, 'make_pyera2frc_one-hourly', True)
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=True).make_kdetree()   
   
    # Get all ERA mask and grid sizes
    # Final argument is kde ball distance in metres (should be +/- 2.5 * dx)
    # but by calling mask_1.check_ball() to make a figure...
   
    mask_1 = ERAMask(era_masks['mask_one'], romsgrd, 650000)
    
    # List of masks
    masks = [mask_1]

    for era_key in era_files.keys():
        
        print('\n\nProcessing variable key:', era_key.upper())        
        
        if era_key in 'prate': # Precipitation rate (used for bulk)
            
            era_prate = ERAPrate(era_files['prate'], masks, romsgrd)
            era_prate.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            era_prate.child_contained_by_parent(romsgrd)
            era_prate.make_kdetree()
            era_prate.get_delaunay_tri()  
            ERA_start = era_prate.date_start()
            ERA_end = era_prate.date_end()
     
        elif era_key in 'radnetlw': # Outgoing longwave radiation (used for bulk)
            
            era_radlw = ERARadNetlw(era_files['radnetlw'], masks, romsgrd)                     
            era_radlw_net = ERARadlwNet(era_files['radnetlw'], masks, romsgrd)
            era_radlw_down = ERARadlwDown(era_files['radnetlw'], masks, romsgrd)  
            supp_vars = [era_radlw_net, era_radlw_down]
            for supp_var in supp_vars:
                supp_var.proj2gnom(ignore_land_points=False, M=romsgrd.M)
                supp_var.get_delaunay_tri()
                era_radlw.needs_all_point_tri = True
            era_radlw.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            era_radlw.child_contained_by_parent(romsgrd)
            era_radlw.make_kdetree()
            era_radlw.get_delaunay_tri()
            ERA_start = era_radlw.date_start()
            ERA_end = era_radlw.date_end()        
            
        elif era_key in 'radnetsw': # used for bulk
            
            era_radsw_down = ERARadSwDown(era_files['radnetsw']['shflux_SW_down'], masks, romsgrd)
            era_radsw_down.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            era_radsw_down.child_contained_by_parent(romsgrd)
            era_radsw_down.make_kdetree()
            era_radsw_down.get_delaunay_tri()   
            era_radsw_net = ERARadSwNet(era_files['radnetsw']['shflux_SW_net'], masks, romsgrd)
            era_radsw_net.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            era_radsw_net.child_contained_by_parent(romsgrd)
            era_radsw_net.make_kdetree()
            era_radsw_net.get_delaunay_tri()   
            ERA_start = era_radsw_net.date_start()
            ERA_end = era_radsw_net.date_end()       
            
        elif era_key in 'wspd': # used for bulk
            era_dp = ERADpair(era_files['wspd']['sadp'], masks, romsgrd)
            era_sat = ERASat(era_files['wspd']['sat'], masks, romsgrd)
            era_sap = ERASap(era_files['wspd']['sap'], masks, romsgrd)
            era_wspd = ERAWspd(era_files['wspd'], masks, romsgrd)  
            supp_vars = [era_sat, era_sap, era_dp]
            for supp_var in supp_vars:
                supp_var.proj2gnom(ignore_land_points=False, M=romsgrd.M)
                supp_var.get_delaunay_tri()
                era_wspd.needs_all_point_tri = True  
            era_sat.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            era_sat.child_contained_by_parent(romsgrd)
            era_sat.make_kdetree()
            era_sat.get_delaunay_tri()      
            era_wspd.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            era_wspd.child_contained_by_parent(romsgrd)
            era_wspd.make_kdetree()
            era_wspd.get_delaunay_tri()       
            ERA_start = era_wspd.date_start()
            ERA_end = era_wspd.date_end()
                                  
         # Number of records at hourly frequency
        #delta = dt.datetime.timedelta(hours=1) 
        delta = 1/24
        day_zero = dt.datetime.datetime(2000,1,1)
        day_zero = dt.date2num(day_zero)
        
        #print('Time indexes:',ERA_start-day_zero, ERA_start, ERA_end)
        
        #ERA_Dates = dt.drange(ERA_start, ERA_end, delta)
        
        time_array = np.arange(ERA_start, ERA_end, delta)#dt.num2date(ERA_Dates)
        tind = 0
        ttotal = np.shape(time_array)
        
        # Loop over time
        for era_dt in time_array:
            
            #print(ERA_end-era_dt)            
            printProgressBar (tind, ttotal[0]-1)
            
            # Day count to be stored as bulk_time
            roms_day = np.float32(era_dt - day_zero)
            
            if (tind < overlap):
                # Write overlap to previous boundary file
                with netcdf.Dataset(overlap_file, 'a') as nc:      
                    if (overlap_offset == 0):
                        overlap_offset = nc.dimensions['bulk_time'].size                 
                    t_overlap = tind + overlap_offset
                    nc.variables['bulk_time'][t_overlap] = roms_day
 
            # Precipitation rate (bulk only)
            if 'prate' in era_key:
                    
                era_prate.set_date_index(era_dt)
                era_prate.get_era_data()              
                era_prate.convert_cmday()
                era_prate.interp2romsgrd()
                #era_prate.check_interp()
                    
                # Get indices and weights for Dai river climatology
                if add_dai_runoff:
                    ind_min, ind_max, weights = romsgrd.get_runoff_index_weights(era_dt)
                    # If condition so we don't read runoff data every iteration
                    if np.logical_or(ind_min != dai_ind_min, ind_max != dai_ind_max):
                        dai_ind_min, dai_ind_max = ind_min, ind_max
                        runoff1 = romsgrd.get_runoff(dai_file, dai_ind_min+1)
                        runoff2 = romsgrd.get_runoff(dai_file, dai_ind_max+1)
                    runoff = np.average([runoff1, runoff2], weights=weights, axis=0)
                    era_prate.dataout += runoff.ravel()
                    
                era_prate.dataout *= romsgrd.maskr().ravel()
                np.place(era_prate.dataout, era_prate.dataout < 0., 0)
                
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['prate'][tind] = era_prate.dataout
                    nc.variables['bulk_time'][tind] = roms_day
                    if numrec is not None:
                        nc.variables['bulk_time'].cycle_length = np.float(numrec)
                    nc.sync()
                    
                if (tind < overlap):
                    # Write overlap to previous boundary file
                    with netcdf.Dataset(overlap_file, 'a') as nc:
                        nc.variables['prate'][t_overlap] = era_prate.dataout
                        nc.sync()
              
            # Outgoing longwave radiation (bulk only)
            elif 'radnetlw' in era_key:
                
                era_radlw_net.set_date_index(era_dt)
                era_radlw_down.set_date_index(era_dt)            
                era_radlw_net.get_era_data()
                era_radlw_down.get_era_data()                  
                # Note that fillmask is called by era_radlw.interp2romsgrd
                era_radlw.get_era_radlw_and_radlw_in(era_radlw_net.datain, era_radlw_down.datain)                  
                era_radlw.interp2romsgrd(fillmask_radlw, fillmask_radlw_in)             
                era_radlw.radlw_dataout *= romsgrd.maskr()
                era_radlw.radlw_in_dataout *= romsgrd.maskr()
                    
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['radlw'][tind] = era_radlw.radlw_dataout
                    nc.variables['radlw_in'][tind] = era_radlw.radlw_in_dataout
                    assert nc.variables['bulk_time'][tind] == roms_day, timing_warning
                    nc.sync()
                    
                if (tind < overlap):
                    # Write overlap to previous boundary file
                    with netcdf.Dataset(overlap_file, 'a') as nc:
                        nc.variables['radlw'][t_overlap] = era_radlw.radlw_dataout
                        nc.variables['radlw_in'][t_overlap] = era_radlw.radlw_in_dataout
                        nc.sync()
                    
            # Net shortwave radiation (bulk only)
            elif 'radnetsw' in era_key:
         
                era_radsw_net.set_date_index(era_dt)               
                era_radsw_net.get_era_data()
                era_radsw_net.fillmask()
                era_radsw_net.interp2romsgrd()               
                era_radsw_net.dataout *= romsgrd.maskr().ravel()
                    
                np.place(era_radsw_net.dataout, era_radsw_net.dataout < 0., 0)
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['radsw'][tind] = era_radsw_net.dataout #era_radsw_down.dataout
                    assert nc.variables['bulk_time'][tind] == roms_day, timing_warning
                    nc.sync()
                    
                if (tind < overlap):
                    # Write overlap to previous boundary file
                    with netcdf.Dataset(overlap_file, 'a') as nc:
                        nc.variables['radsw'][t_overlap] = era_radsw_net.dataout
                        nc.sync()
                    
            # Wind speed (wspd, uwnd, vwnd) and stress (sustr, svstr) (bulk only)
            elif 'wspd' in era_key:
                    
                era_dp.set_date_index(era_dt)
                era_sat.set_date_index(era_dt)
                era_sap.set_date_index(era_dt)
                era_wspd.set_date_index(era_dt)                  
                era_dp.get_era_data()
                era_sat.get_era_data()
                era_sap.get_era_data()
                era_wspd.get_era_data()                    
                era_dp.get_era_data().fillmask()
                era_sat.get_era_data().fillmask()
                era_sap.get_era_data().fillmask()
                    
                for supp_var in supp_vars:
                    if supp_var.to_be_downscaled:
                        supp_var.data = horizInterp(supp_var.tri_all, supp_var.datain.ravel())(era_wspd.points_all)
                        supp_var.data = supp_var.data.reshape(era_wspd.lon().shape)
                                           
                era_wspd.get_winds(era_dp, era_sat, era_sap)
                era_wspd.interp2romsgrd()
                era_wspd.wspd_dataout *= romsgrd.maskr()                   
                era_wspd.uwnd_dataout[:], \
                era_wspd.vwnd_dataout[:] = romsgrd.rotate(era_wspd.uwnd_dataout,era_wspd.vwnd_dataout, sign=1)
                era_wspd.uwnd_dataout[:,:-1] = romsgrd.rho2u_2d(era_wspd.uwnd_dataout)
                era_wspd.uwnd_dataout[:,:-1] *= romsgrd.umask()
                era_wspd.vwnd_dataout[:-1] = romsgrd.rho2v_2d(era_wspd.vwnd_dataout)
                era_wspd.vwnd_dataout[:-1] *= romsgrd.vmask()                    
                era_wspd.ustrs_dataout[:], \
                era_wspd.vstrs_dataout[:] = romsgrd.rotate(era_wspd.ustrs_dataout,era_wspd.vstrs_dataout, sign=1)
                era_wspd.ustrs_dataout[:,:-1] = romsgrd.rho2u_2d(era_wspd.ustrs_dataout)
                era_wspd.ustrs_dataout[:,:-1] *= romsgrd.umask()
                era_wspd.vstrs_dataout[:-1] = romsgrd.rho2v_2d(era_wspd.vstrs_dataout)
                era_wspd.vstrs_dataout[:-1] *= romsgrd.vmask()
                    
                era_dp.interp2romsgrd()
                era_dp.dataout = (era_dp.dataout - 273.15) * romsgrd.maskr().ravel()                    
                era_sat.interp2romsgrd()
                era_sat.dataout = (era_sat.dataout - 273.15) * romsgrd.maskr().ravel()              
                era_sap.interp2romsgrd()
                era_sap.dataout *= romsgrd.maskr().ravel()                
                
                tdair = era_dp.dataout.reshape(romsgrd.lon().shape)
                tair = era_sat.dataout.reshape(romsgrd.lon().shape)
                pres = era_sap.dataout.reshape(romsgrd.lon().shape) * 0.01                                           
                
                Es = 6.1121 * (1.0007 + 3.46e-6 * pres) * np.exp((17.502 * tair) / (240.97 + tair))
                E = 6.1121 * (1.0007 + 3.46e-6 * pres) * np.exp((17.502 * tdair) / (240.97 + tdair))         
                rhum = (E/Es)
                
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['rhum'][tind] = rhum
                    nc.variables['tair'][tind] = tair
                    nc.variables['wspd'][tind] = era_wspd.wspd_dataout
                    nc.variables['uwnd'][tind] = era_wspd.uwnd_dataout[:,:-1]
                    nc.variables['vwnd'][tind] = era_wspd.vwnd_dataout[:-1]
                    nc.variables['sustr'][tind] = era_wspd.ustrs_dataout[:,:-1]
                    nc.variables['svstr'][tind] = era_wspd.vstrs_dataout[:-1]
                    assert nc.variables['bulk_time'][tind] == roms_day, timing_warning
                    nc.sync()
                
                if (tind < overlap):
                    # Write overlap to previous boundary file
                    with netcdf.Dataset(overlap_file, 'a') as nc:
                        nc.variables['rhum'][t_overlap] = rhum
                        nc.variables['tair'][t_overlap] = tair
                        nc.variables['wspd'][t_overlap] = era_wspd.wspd_dataout
                        nc.variables['uwnd'][t_overlap] = era_wspd.uwnd_dataout[:,:-1]
                        nc.variables['vwnd'][t_overlap] = era_wspd.vwnd_dataout[:-1]
                        nc.variables['sustr'][t_overlap] = era_wspd.ustrs_dataout[:,:-1]
                        nc.variables['svstr'][t_overlap] = era_wspd.vstrs_dataout[:-1]
                        nc.sync()

            tind += 1
    print('\nDone!')
                

if __name__ == '__main__':
    
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    np.seterr(divide='ignore', invalid='ignore')

    roms_grd = '/server/DISK3/SAnLX/INPUT/croco_grd.nc' 
    frc_files = '/server/DISK3/SAnLX/INPUT/BLK/croco_blk_{file:d}.nc'
    
    #roms_grd = '/server/DISK3/SAnL2/INPUT/croco_grd.nc' 
    #frc_files = '/server/DISK3/SAnL2/INPUT/BLK/croco_blk_{file:d}.nc'
    
    ERA_dir = '/server/DISKA/ROMS_DATASETS/ERA5/'
    
    day_zero = datetime(2000, 1, 1)
    day_zero = plb.date2num(day_zero)
    
    params = [6,0,10,42,1,1,0,1,day_zero,0.,20000.]
    
    files = sorted(glob.glob(ERA_dir + 'ERA5_20??_??.nc'))
    index = 0
    for i in files:
        if '2000_01' in i:
            file_start = index
        if '2010_12' in i:
            file_end = index    
        index = index + 1
    
    time_len = (file_end-file_start)+1
    
    for index in range(0,time_len):
        frc_file = frc_files.format(file = index)
        overlap_file = frc_files.format(file = (index-1))
        overlap = 0
        if index > 0:
            overlap = 8
        Process_ERA(files[index+file_start],roms_grd,frc_file,overlap*24,overlap_file,params)
    
                
                
                
                
