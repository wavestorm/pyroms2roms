# -*- coding: utf-8 -*-
# %run py_mercator2roms.py

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
Modifications by Dylan F. Bailey, 2020

+ Ported to Python 3+
+ Processes monthly mercator files
+ Processing all put into a single function to process a single month at a 
  time
+ Processes a predefined time overlap at the end of each file
  for those unusual ROMS run periods

===========================================================================

Create a ROMS boundary file based on Mercator data

===========================================================================
'''

import math
import netCDF4 as netcdf
import pylab as plt
import numpy as np
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
import glob as glob
import time
import scipy.interpolate.interpnd as interpnd
import collections
#from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime
import calendar
from time import strptime
import re
from py_roms2roms import vertInterp, horizInterp, bry_flux_corr, debug0, debug1, debug2, ROMS, RomsGrid, WestGrid, EastGrid, NorthGrid, SouthGrid, RomsData



class MercatorData (RomsData):
    '''
    MercatorData class (inherits from RomsData class)
    '''
    def __init__(self, filename, model_type, mercator_var, romsgrd, **kwargs):
        """
        Creates a new Mercator data object.
        
        Parameters
        ----------
        
        *filenames* : list of Mercator nc files.
        *model_type* : string specifying Mercator model.
        *romsgrd* : a `RomsGrid` instance.
        
        """
        super(MercatorData, self).__init__(filename, model_type)
        self.romsfile = filename
        self.vartype = mercator_var
        self.romsgrd = romsgrd
        self.var_dic = {'sossheig':'SSH',
                        'votemper':'TEMP', 'vosaline':'SALT',
                        'vozocrtx':'U', 'vomecrty':'V',
                        'ssh':'SSH',
                        'temperature':'TEMP', 'salinity':'SALT',
                        'u':'U', 'v':'V',
                        'zos':'SSH',
                        'thetao':'TEMP', 'so':'SALT',
                        'uo':'U', 'vo':'V'}
        #print self.var_dic
        self._set_variable_type()
        try:
            self._lon = self.read_nc('nav_lon', indices='[:]')
            self._lat = self.read_nc('nav_lat', indices='[:]')
        except Exception:
            try:
                self._lon = self.read_nc('longitude', indices='[:]')
                self._lat = self.read_nc('latitude', indices='[:]')
            except:
                self._lon = self.read_nc('lon', indices='[:]')
                self._lat = self.read_nc('lat', indices='[:]')
            self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        #print self._lon.shape
        #print self._lat.shape
        if 'i0' in kwargs:
            self.i0, self.i1 = kwargs['i0'], kwargs['i1']
            self.j0, self.j1 = kwargs['j0'], kwargs['j1']
        else:
            self.set_subgrid(romsgrd, k=50)
        
        try:
            self._depths = self.read_nc('deptht', indices='[:]')
        except Exception:
            self._depths = self.read_nc('depth', indices='[:]')
        try:
            self._time_count = self.read_nc('time_counter', indices='[:]')
        except Exception:
            self._time_count = self.read_nc('time', indices='[:]')
            
        self._set_data_shape()
        self._set_dates()
        self.datain = np.empty(self._data_shape, dtype=np.float64)
        if len(self._data_shape) == 2:
            self.dataout = np.ma.empty(self.romsgrd.lon().shape)
        else:
            tmp_shape = (self._depths.size, self.romsgrd.lon().shape[0],
                         self.romsgrd.lon().shape[1])
            self.datatmp = np.ma.empty(tmp_shape, dtype=np.float64).squeeze()
            self.dataout = np.ma.empty(self.romsgrd.mask3d().shape,
                                       dtype=np.float64).squeeze()
        self._set_maskr()
        self._set_angle()

        
    def lon(self):
        return self._lon[self.j0:self.j1, self.i0:self.i1]
    
    def lat(self):
        return self._lat[self.j0:self.j1, self.i0:self.i1]
    
    def maskr(self):
        return self._maskr[self.j0:self.j1, self.i0:self.i1]
    
    def maskr3d(self):
        return self._maskr3d[:, self.j0:self.j1, self.i0:self.i1]
    
    def angle(self):
        return self._angle[self.j0:self.j1, self.i0:self.i1]
    
    def depths(self):
        return self._depths
    
    def dates(self):
        return self._time_count
    
    
    def _set_data_shape(self):
        ssh_varnames = ('sossheig', 'zos')  # clumsy, needs to be added to if more names
        if self._depths.size == 1 or self.varname in ssh_varnames:
            _shp = (self.j1 - self.j0, self.i1 - self.i0)
            self._data_shape = _shp
            self.dimtype = '2D'
        else:
            _shp = (self._depths.size, self.j1 - self.j0, self.i1 - self.i0)
            self._data_shape = _shp
            self.dimtype = '3D'
        return self
    
    
    def _set_variable_type(self):
        self.varname = None
        for varname, vartype in zip(self.var_dic.keys(), self.var_dic.values()):
            if varname in self.list_of_variables() and vartype in self.vartype:
                self.varname = varname
                return self
        if self.varname is None:
            raise Exception  # no candidate variable identified
    
    
    def _get_maskr(self, k=None):
        """
        Called by _set_maskr()
        """
        if '3D' in self.dimtype:
            self.k = k
            indices = '[0, self.k]'
        else:
            indices = '[0]'
        try:
            _mask = self.read_nc(self.varname, indices=indices).mask
        except Exception: # triggered when all points are masked
            _mask = np.ones(self.read_nc(self.varname, indices=indices).shape)
        _mask = np.asarray(_mask, dtype=np.int)
        _mask *= -1
        _mask += 1
        return _mask
    
        
    def _set_maskr(self):
        """
        Set the landsea mask (*self.maskr*) with same shape
        as input Mercator nc file. If a 3D variable, then additional
        attribute, self._maskr3d is set.
        """
        if '3D' in self.dimtype:
            self._maskr = self._get_maskr(k=0)
            self._set_maskr3d()
        else:
            self._maskr = self._get_maskr()
        return self
    
    
    def _set_maskr3d(self):
        """
        Called by _set_maskr()
        """
        _3dshp = (self._depths.size, self._lon.shape[0], self._lon.shape[1])
        #print _3dshp
        self._maskr3d = np.empty(_3dshp)
        for k in range(self._depths.size):
            self._maskr3d[k] = self._get_maskr(k=k)
        return self
        
        
    def _set_angle(self):
        '''
        Compute angles of local grid positive x-axis relative to east
        '''
        latu = np.deg2rad(0.5 * (self._lat[:,1:] + self._lat[:,:-1]))
        lonu = np.deg2rad(0.5 * (self._lon[:,1:] + self._lon[:,:-1]))
        dellat = latu[:,1:] - latu[:,:-1]
        dellon = lonu[:,1:] - lonu[:,:-1]
        dellon[dellon >  np.pi] = dellon[dellon >  np.pi] - (2. * np.pi)
        dellon[dellon < -np.pi] = dellon[dellon < -np.pi] + (2. * np.pi)
        dellon = dellon * np.cos(0.5 * (latu[:,1:] + latu[:,:-1]))

        self._angle = np.zeros_like(self._lat)
        ang_s = np.arctan(dellat / (dellon + np.spacing(0.4)))
        deli = np.logical_and(dellon < 0., dellat < 0.)
        ang_s[deli] = ang_s[deli] - np.pi

        deli = np.logical_and(dellon < 0., dellat >= 0.)
        ang_s[deli] = ang_s[deli] + np.pi
        ang_s[ang_s >  np.pi] = ang_s[ang_s >  np.pi] - np.pi
        ang_s[ang_s < -np.pi] = ang_s[ang_s <- np.pi] + np.pi

        self._angle[:,1:-1] = ang_s
        self._angle[:,0] = self._angle[:,1]
        self._angle[:,-1] = self._angle[:,-2]
        return self
    
    
    def get_variable(self, date):
        #print(self._time_count,date)
        ind = (self._time_count == date).nonzero()[0][0]
        with netcdf.Dataset(self.romsfile) as nc:
            if '3D' in self.dimtype:
                self.datain[:] = np.ma.masked_array(
                    nc.variables[self.varname][
                        ind, :,
                        self.j0:self.j1,
                        self.i0:self.i1].astype(np.float64)).data
            else:
                self.datain[:] = np.ma.masked_array(
                    nc.variables[self.varname][
                        ind,
                        self.j0:self.j1, 
                        self.i0:self.i1].astype(np.float64)).data
        return self
    
    
    def _set_time_origin(self):
        try:
            self._time_counter_origin = self.read_nc_att(
                'time_counter', 'time_origin')
            ymd, hms = self._time_counter_origin.split(' ')
        except Exception:
            try:
                self._time_counter_origin = self.read_nc_att(
                    'time_counter', 'units')
                junk, junk, ymd, hms = self._time_counter_origin.split(' ')
                #print self._time_counter_origin
            except Exception:
                self._time_counter_origin = self.read_nc_att('time', 'units')
                try:
                    junk, junk, ymd, hms = self._time_counter_origin.split(' ')
                except: # Monthly clim
                    junk, junk, ymd = self._time_counter_origin.split(' ')
                    hms = '00:00:00'
                
                
        y, mo, d = ymd.split('-')
        h, mi, s = hms.split(':')
        
        try:
            time_origin = datetime(int(y), strptime(mo,'%b').tm_mon, int(d),
                                   int(h), int(mi), int(s))
        except Exception:
            time_origin = datetime(int(y), int(mo), int(d),
                                   int(h), int(mi), int(s))
        self.time_origin = plt.date2num(time_origin)
        return self
    
    
    def _set_dates(self):
        self._set_time_origin()
        
        with netcdf.Dataset(self.romsfile) as nc:
            try:
                date_start = nc.variables['time_counter'][0]
                date_end = nc.variables['time_counter'][-1]
            except Exception:
                date_start = nc.variables['time'][0]
                date_end = nc.variables['time'][-1]
       
        if 'second' in self._time_counter_origin:
            self._time_interval = 86400.
        elif 'hour' in self._time_counter_origin:
            self._time_interval = 24.
        elif 'days' in self._time_counter_origin:
            self._time_interval = 1.
        try:
            self._time_count = np.arange(date_start, date_end +
                                  self._time_interval, self._time_interval)
        except:
            print('here 2')
            print(date_start[0], date_end[-1], self._time_interval)
            self._time_count = np.arange(date_start[0], date_end[-1] + 
                                  self._time_interval, self._time_interval)
            
        self._time_count /= self._time_interval  # 86400.
        self._time_count += self.time_origin
        return self
    
    
    def get_date(self, ind):
        return self._time_count[ind]
        
        
    def get_fillmask_cofs(self):
        if '3D' in self.dimtype:
            self.fillmask_cof_tmp = []
            for k in range(self._depths.size):
                try:
                    self.get_fillmask_cof(self.maskr3d()[k])
                    self.fillmask_cof_tmp.append(self.fillmask_cof)
                except Exception:
                    self.fillmask_cof_tmp.append(None)
            self.fillmask_cof = self.fillmask_cof_tmp
        else:
            self.get_fillmask_cof(self.maskr())
        return self
    
    
    def _fillmask(self, dist, iquery, igood, ibad, k=None):
        '''
        '''
        weight = dist / (dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
        np.place(weight, weight > 1., 0.)
        if k is not None:
            xfill = weight * self.datain[k][igood[:,0][iquery], igood[:,1][iquery]]
            xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)
            self.datain[k][ibad[:,0], ibad[:,1]] = xfill
        else:
            xfill = weight * self.datain[igood[:,0][iquery], igood[:,1][iquery]]
            xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)
            self.datain[ibad[:,0], ibad[:,1]] = xfill
        return self
    
    def fillmask(self):
        '''Fill missing values in an array with an average of nearest  
           neighbours
           From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        Order:  call after self.get_fillmask_cof()
        '''
        if '3D' in self.dimtype:
            for k in range(len(self.fillmask_cof)):
                if self.fillmask_cof[k] is not None:
                    dist, iquery, igood, ibad = self.fillmask_cof[k]
                    #print dist, iquery, igood, ibad
                    try:
                        self._fillmask(dist, iquery, igood, ibad, k=k)
                    except:
                        self.maskr3d()[k] = 0
            self._check_and_fix_deep_levels()
        else:
            dist, iquery, igood, ibad = self.fillmask_cof
            self._fillmask(dist, iquery, igood, ibad)
        return self
    
    
    def _check_and_fix_deep_levels(self):
        for k in range(self._depths.size):
            if np.sum(self.maskr3d()[k]) == 0:
                self.datain[k] = self.datain[k-1]
        return self
    
    
    def set_2d_depths(self):
        '''
        
        '''
        self._2d_depths = np.tile(-self.depths()[::-1],
                                  (self.romsgrd.h().size, 1)).T
        return self
    
    
    def set_3d_depths(self):
        '''
        
        '''
        self._3d_depths = np.tile(-self.depths()[::-1],
                                  (self.romsgrd.h().shape[1],
                                   self.romsgrd.h().shape[0], 1)).T
        return self
    
    
    def set_map_coordinate_weights(self, j=None):
        '''
        Order : set_2d_depths or set_3d_depths
        '''
        if j is not None:
            mercator_depths = self._3d_depths[:,j]
            roms_depths = self.romsgrd.scoord2z_r()[:,j]
        else:
            mercator_depths = self._2d_depths
            roms_depths = self.romsgrd.scoord2z_r()
        self.mapcoord_weights = self.romsgrd.get_map_coordinate_weights(
                                             roms_depths, mercator_depths)
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
    
    
    def vert_interp(self):
        '''
        Vertical interpolation using ndimage.map_coordinates()
          See vertInterp class in py_roms2roms.py
        Requires self.mapcoord_weights set by set_map_coordinate_weights()
        '''
        self._vert_interp = vertInterp(self.mapcoord_weights)
        return self
    
    
    def _interp2romsgrd(self, k=None):
        '''
        '''
        if k is not None:
            interp = horizInterp(self.tri, self.datain[k].flat[self.ball])
            interp = interp(self.romsgrd.points)
            try:
                self.datatmp[k] = interp
            except Exception:
                self.datatmp[k] = interp.reshape(self.romsgrd.maskr().shape)
        else:
            interp = horizInterp(self.tri, self.datain.flat[self.ball])
            self.dataout = interp(self.romsgrd.points)
        return self
    
    
    def interp2romsgrd(self, j=None):
        '''
        '''
        if '3D' in self.dimtype:
            for k in np.arange(self._depths.size):
                self._interp2romsgrd(k)
            self.dataout[:] = self._vert_interp.vert_interp(self.datatmp[::-1])
        else:
            self._interp2romsgrd()
        return self
        
    
    def debug_figure(self):
        if self.var_dic[self.varname] in 'SALT':
            cmin, cmax = 34, 38
        elif self.var_dic[self.varname] in 'TEMP':
            cmin, cmax = 5, 25
        elif self.var_dic[self.varname] in ('U', 'V'):
            cmin, cmax = -.4, .4
        else:
            raise Exception
        
        plt.figure()
        ax = plt.subplot(211)
        ax.set_title('Mercator vertical grid')
        pcm = ax.pcolormesh(self.romsgrd.lat(), -self.depths(), self.datatmp)
        pcm.set_clim(cmin, cmax)
        ax.plot(np.squeeze(self.romsgrd.lat()), -self.romsgrd.h(), c='gray', lw=3)
        plt.colorbar(pcm)
        
        ax = plt.subplot(212)
        ax.set_title('ROMS vertical grid')
        pcm = ax.pcolormesh(np.tile(self.romsgrd.lat(), (self.romsgrd.N, 1)),
                                   self.romsgrd.scoord2z_r(),
                                   self.dataout)
        pcm.set_clim(cmin, cmax)
        ax.plot(np.squeeze(self.romsgrd.lat()), -self.romsgrd.h(), c='gray', lw=2)
        plt.colorbar(pcm)
        
        plt.figure()
        ax = plt.subplot(211)
        ax.scatter(self._vert_interp.hweights, self._vert_interp.vweights, s=10, edgecolors='none')
        
        ax = plt.subplot(212)
        lats1 = np.tile(self.romsgrd.lat(), (self.romsgrd.N, 1))
        lats2 = np.tile(self.romsgrd.lat(), (self._depths.size, 1))
        ax.scatter(lats1, self.romsgrd.scoord2z_r(), s=10, c='r', edgecolors='none', label='ROMS')
        ax.scatter(lats2, self._2d_depths, s=10, c='g', edgecolors='none', label='Mercator')
                                     
        plt.show()
        return
        
        
    def check_angle(self):
        '''
        Check angle computation
        '''
        plt.pcolormesh(self.lon(), self.lat(), self.angle())
        plt.axis('image')
        plt.colorbar()
        plt.show()
        return


def prepare_romsgrd(romsgrd):
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=False)
    romsgrd.make_kdetree()
    return romsgrd


def prepare_mercator(mercator, balldist):
    mercator.proj2gnom(ignore_land_points=False, M=mercator.romsgrd.M)
    mercator.child_contained_by_parent(mercator.romsgrd)
    mercator.make_kdetree().get_fillmask_cofs()
    ballpoints = mercator.kdetree.query_ball_tree(mercator.romsgrd.kdetree, r=balldist)
    mercator.ball = np.array(np.array(ballpoints).nonzero()[0])
    mercator.tri = sp.Delaunay(mercator.points[mercator.ball])
    if '3D' in mercator.dimtype:
        mercator.set_2d_depths().set_map_coordinate_weights()
        mercator.vert_interp()  
    proceed = True
    return mercator, proceed
    
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

def Process_Mercator(mercator_file,roms_grd,bry_file,overlap,overlap_file,params):

    print('\n\nProcessing Mercator Lateral Boundaries')
    print('----------------------------------------')
    print(mercator_file)
    
    sigma_params = dict(theta_s=params[0], theta_b=params[1], hc=params[2], N=params[3])
    obc_dict = dict(south=params[4], east=params[5], north=params[6], west=params[7]) # 1=open, 0=closed
    day_zero = params[8] # offset in days since 2000
    bry_cycle = params[9]  # days, 0 means no cycle
    balldist = params[10] # meters
    
    mercator_ssh_file = \
    mercator_temp_file = \
    mercator_salt_file = \
    mercator_u_file = \
    mercator_v_file = mercator_file
    
    plt.close('all')
    fillval = 9999.  
    
    overlap_offset = 0

    if 'IBI_daily' in mercator_file or 'ibi' in mercator_file:
        k2c = -273.15
    
    # Set up a RomsGrid object
    romsgrd = RomsGrid(roms_grd, sigma_params, 'ROMS')
    romsgrd.set_bry_dx()
    romsgrd.set_bry_maskr()
    romsgrd.set_bry_areas()
    
    # Get surface areas of open boundaries
    chd_bry_surface_areas = []
    for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
        
        if 'west' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_west.sum(axis=0) * romsgrd.maskr_west)
        
        elif 'east' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_east.sum(axis=0) * romsgrd.maskr_east)
        
        elif 'south' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_south.sum(axis=0) * romsgrd.maskr_south)
        
        elif 'north' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_north.sum(axis=0) * romsgrd.maskr_north)
    
    # Get total surface of open boundaries
    chd_bry_total_surface_area = np.array([area.sum() for area in chd_bry_surface_areas]).sum()
    
    # Set up a RomsData object for creation of the boundary file
    romsbry = RomsData(bry_file, 'ROMS')
    romsbry.create_bry_nc(romsgrd, obc_dict, bry_cycle, fillval, 'py_mercator2roms')
    
    
    mercator_vars = dict(SSH = mercator_ssh_file,
                         TEMP = mercator_temp_file,
                         SALT = mercator_salt_file,
                         U = mercator_u_file)
    
    for mercator_var, mercator_file in zip(mercator_vars.keys(), mercator_vars.values()):
        
        print('\n\nProcessing variable *%s*' % mercator_var)
        proceed = False
        
        with netcdf.Dataset(mercator_file) as nc:
            try:
                mercator_date_start = nc.variables['time_counter'][0]
                mercator_date_end = nc.variables['time_counter'][-1]
                mercator_time_units = nc.variables['time_counter'].units
                mercator_time_origin = nc.variables['time_counter'].time_origin
                mercator_time_origin = plt.date2num(plt.datetime.datetime.strptime(
                                            mercator_time_origin, '%Y-%b-%d %H:%M:%S'))
            except Exception:
                mercator_date_start = nc.variables['time'][0]
                mercator_date_end = nc.variables['time'][-1]
                mercator_time_units = nc.variables['time'].units
                mercator_time_origin = mercator_time_units.partition(' ')[-1].partition(' ')[-1]
                mercator_time_origin = plt.date2num(plt.datetime.datetime.strptime(
                                            mercator_time_origin, '%Y-%m-%d %H:%M:%S'))
                
        if 'seconds' in mercator_time_units:
            mercator_dates = np.arange(mercator_date_start, mercator_date_end + 86400, 86400)
            mercator_dates /= 86400.
        elif 'hours' in mercator_time_units:
            mercator_dates = np.arange(mercator_date_start, mercator_date_end + 24, 24)
            mercator_dates /= 24.
        else:
            raise Exception('deal_with_when_a_problem')
        
        mercator_dates += mercator_time_origin
        #mercator_dates = mercator_dates[np.logical_and(mercator_dates >= dtstr,
        #                                               mercator_dates <= dtend)]
        #print(mercator_dates)
        #exit()
        
        for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
            
            print('\n--- processing %sern boundary' % boundary)
            
            if 'west' in boundary and is_open:
                romsgrd_at_bry = WestGrid(roms_grd, sigma_params, 'ROMS')
                proceed = True
            
            elif 'north' in boundary and is_open:
                romsgrd_at_bry = NorthGrid(roms_grd, sigma_params, 'ROMS')
                proceed = True
            
            elif 'east' in boundary and is_open:
                romsgrd_at_bry = EastGrid(roms_grd, sigma_params, 'ROMS')
                proceed = True
            
            elif 'south' in boundary and is_open:
                romsgrd_at_bry = SouthGrid(roms_grd, sigma_params, 'ROMS')
                proceed = True
            
            else:
                proceed = False #raise Exception
            
            if proceed:
                
                romsgrd_at_bry = prepare_romsgrd(romsgrd_at_bry)
                mercator = MercatorData(mercator_file, 'Mercator', mercator_var, romsgrd_at_bry)
                mercator, proceed = prepare_mercator(mercator, balldist)
                
                if 'U' in mercator_var:
                    mercator_v = MercatorData(mercator_v_file, 'Mercator', 'V', romsgrd_at_bry,
                                         i0=mercator.i0, i1=mercator.i1, j0=mercator.j0, j1=mercator.j1)
                    mercator_v, junk = prepare_mercator(mercator_v, balldist)
                
                tind = 0     # index for writing records to bry file
                ttotal = np.shape(mercator_dates)
                
                for dt in mercator_dates:
                            
                    printProgressBar (tind, ttotal[0]-1)
                    
                    dtnum = math.floor(dt - day_zero)
                               
                    # Read in variables
                    mercator.get_variable(dt).fillmask()
                        
                    # Calculate barotropic velocities
                    if mercator.vartype in 'U':
                            
                        mercator_v.get_variable(dt).fillmask()
                        
                        # Rotate to zero angle
                        for k in np.arange(mercator.depths().size):
                            u, v = mercator.datain[k], mercator_v.datain[k]
                            mercator.datain[k], mercator_v.datain[k] = mercator.rotate(u, v, sign=-1)
                        
                        mercator.interp2romsgrd()
                        mercator_v.interp2romsgrd()
                        
                        # Rotate u, v to child angle
                        for k in np.arange(romsgrd.N).astype(np.int):
                            u, v = mercator.dataout[k], mercator_v.dataout[k]
                            mercator.dataout[k], mercator_v.dataout[k] = romsgrd.rotate(u, v, sign=1,
                                                                     ob=boundary)
                        
                        mercator.set_barotropic()
                        mercator_v.set_barotropic()
                        
                    else:
                        
                        mercator.interp2romsgrd()
                        
                        
                    # Write to boundary file
                    with netcdf.Dataset(romsbry.romsfile, 'a') as nc:
                            
                        if mercator.vartype in 'U':
                            
                            if boundary in ('north', 'south'):
                                u = romsgrd.half_interp(mercator.dataout[:,:-1],
                                                        mercator.dataout[:,1:])
                                ubar = romsgrd.half_interp(mercator.barotropic[:-1],
                                                           mercator.barotropic[1:])
                                u *= romsgrd_at_bry.umask()
                                ubar *= romsgrd_at_bry.umask()
                                v = mercator_v.dataout
                                vbar = mercator_v.barotropic
                                v *= romsgrd_at_bry.vmask()
                                vbar *= romsgrd_at_bry.vmask()
                                nc.variables['u_%s' % boundary][tind] = u
                                nc.variables['ubar_%s' % boundary][tind] = ubar
                                nc.variables['v_%s' % boundary][tind] = v
                                nc.variables['vbar_%s' % boundary][tind] = vbar
                            
                            elif boundary in ('east', 'west'):
                                u = mercator.dataout
                                ubar = mercator.barotropic
                                u *= romsgrd_at_bry.umask()
                                ubar *= romsgrd_at_bry.umask()
                                v = romsgrd.half_interp(mercator_v.dataout[:,:-1],
                                                        mercator_v.dataout[:,1:])
                                vbar = romsgrd.half_interp(mercator_v.barotropic[:-1],
                                                           mercator_v.barotropic[1:])
                                v *= romsgrd_at_bry.vmask()
                                vbar *= romsgrd_at_bry.vmask()
                                nc.variables['v_%s' % boundary][tind] = v
                                nc.variables['vbar_%s' % boundary][tind] = vbar
                                nc.variables['u_%s' % boundary][tind] = u
                                nc.variables['ubar_%s' % boundary][tind] = ubar
                            
                            else:
                                raise Exception('Unknown boundary: %s' % boundary)
                        
                        elif mercator.vartype in 'SSH':
                            
                            nc.variables['zeta_%s' % boundary][tind] = mercator.dataout
                            nc.variables['bry_time'][tind] = np.float(dtnum)
            
                        elif mercator.vartype in 'TEMP':
                            
                            if (mercator.dataout > 100.).any():
                                mercator.dataout += k2c # Kelvin to Celcius
                            nc.variables['temp_%s' % boundary][tind] = mercator.dataout
                        
                        else:
                            
                            varname = mercator.vartype.lower() + '_%s' % boundary
                            nc.variables[varname][tind] = mercator.dataout
                    
                    if (tind < overlap):
                        # Write overlap to previous boundary file
                        
                        with netcdf.Dataset(overlap_file, 'a') as nc:
                        
                            if (overlap_offset == 0):
                                overlap_offset = nc.dimensions['bry_time'].size
                                
                            t_overlap = tind + overlap_offset
                            
                            if mercator.vartype in 'U':
                                
                                if boundary in ('north', 'south'):
                                    u = romsgrd.half_interp(mercator.dataout[:,:-1],
                                                            mercator.dataout[:,1:])
                                    ubar = romsgrd.half_interp(mercator.barotropic[:-1],
                                                               mercator.barotropic[1:])
                                    u *= romsgrd_at_bry.umask()
                                    ubar *= romsgrd_at_bry.umask()
                                    v = mercator_v.dataout
                                    vbar = mercator_v.barotropic
                                    v *= romsgrd_at_bry.vmask()
                                    vbar *= romsgrd_at_bry.vmask()
                                    nc.variables['u_%s' % boundary][t_overlap] = u
                                    nc.variables['ubar_%s' % boundary][t_overlap] = ubar
                                    nc.variables['v_%s' % boundary][t_overlap] = v
                                    nc.variables['vbar_%s' % boundary][t_overlap] = vbar
                                
                                elif boundary in ('east', 'west'):
                                    u = mercator.dataout
                                    ubar = mercator.barotropic
                                    u *= romsgrd_at_bry.umask()
                                    ubar *= romsgrd_at_bry.umask()
                                    v = romsgrd.half_interp(mercator_v.dataout[:,:-1],
                                                            mercator_v.dataout[:,1:])
                                    vbar = romsgrd.half_interp(mercator_v.barotropic[:-1],
                                                               mercator_v.barotropic[1:])
                                    v *= romsgrd_at_bry.vmask()
                                    vbar *= romsgrd_at_bry.vmask()
                                    nc.variables['v_%s' % boundary][t_overlap] = v
                                    nc.variables['vbar_%s' % boundary][t_overlap] = vbar
                                    nc.variables['u_%s' % boundary][t_overlap] = u
                                    nc.variables['ubar_%s' % boundary][t_overlap] = ubar
                                
                                else:
                                    raise Exception('Unknown boundary: %s' % boundary)
                            
                            elif mercator.vartype in 'SSH':
                                
                                nc.variables['zeta_%s' % boundary][t_overlap] = mercator.dataout
                                nc.variables['bry_time'][t_overlap] = np.float(dtnum)
                
                            elif mercator.vartype in 'TEMP':
                                
                                if (mercator.dataout > 100.).any():
                                    mercator.dataout += k2c # Kelvin to Celcius
                                nc.variables['temp_%s' % boundary][t_overlap] = mercator.dataout
                            
                            else:
                                
                                varname = mercator.vartype.lower() + '_%s' % boundary
                                nc.variables[varname][t_overlap] = mercator.dataout
                    
                    
                        
                    tind += 1

    
    
    # Correct volume fluxes and write to boundary file
    print('\nProcessing volume flux correction')
    #with netcdf.Dataset(romsbry.romsfile, 'a') as nc:
    with netcdf.Dataset(bry_file, 'a') as nc:                
        bry_times = nc.variables['bry_time'][:]
        boundarylist = []
        for bry_ind in range(bry_times.size):
            uvbarlist = []
            for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
                if 'west' in boundary and is_open:
                    uvbarlist.append(nc.variables['ubar_west'][bry_ind])
                elif 'east' in boundary and is_open:
                    uvbarlist.append(nc.variables['ubar_east'][bry_ind])
                elif 'north' in boundary and is_open:
                    uvbarlist.append(nc.variables['vbar_north'][bry_ind])
                elif 'south' in boundary and is_open:
                    uvbarlist.append(nc.variables['vbar_south'][bry_ind])
                if bry_ind == 0 and is_open:
                    boundarylist.append(boundary)
            fc = bry_flux_corr(boundarylist,
                               chd_bry_surface_areas,
                               chd_bry_total_surface_area,
                               uvbarlist)
            print('------ barotropic velocity correction:', fc, 'm/s')
            for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
                if 'west' in boundary and is_open:
                    nc.variables['u_west'][bry_ind] -= fc
                    nc.variables['ubar_west'][bry_ind] -= fc
                elif 'east' in boundary and is_open:
                    nc.variables['u_east'][bry_ind] += fc
                    nc.variables['ubar_east'][bry_ind] += fc
                elif 'north' in boundary and is_open:
                    nc.variables['v_north'][bry_ind] += fc
                    nc.variables['vbar_north'][bry_ind] += fc
                elif 'south' in boundary and is_open:
                    nc.variables['v_south'][bry_ind] -= fc
                    nc.variables['vbar_south'][bry_ind] -= fc
                    
    if (overlap>0):
        print('\nProcessing volume flux correction for overlap')
        with netcdf.Dataset(overlap_file, 'a') as nc:                
            bry_times = nc.variables['bry_time'][:]
            boundarylist = []
            for bry_ind in range(bry_times.size-overlap,bry_times.size):
                uvbarlist = []
                for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
                    if 'west' in boundary and is_open:
                        uvbarlist.append(nc.variables['ubar_west'][bry_ind])
                    elif 'east' in boundary and is_open:
                        uvbarlist.append(nc.variables['ubar_east'][bry_ind])
                    elif 'north' in boundary and is_open:
                        uvbarlist.append(nc.variables['vbar_north'][bry_ind])
                    elif 'south' in boundary and is_open:
                        uvbarlist.append(nc.variables['vbar_south'][bry_ind])
                    if bry_ind == (bry_times.size-overlap) and is_open:
                        boundarylist.append(boundary)
                fc = bry_flux_corr(boundarylist,
                                   chd_bry_surface_areas,
                                   chd_bry_total_surface_area,
                                   uvbarlist)
                print('------ barotropic velocity correction:', fc, 'm/s')
                for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
                    if 'west' in boundary and is_open:
                        nc.variables['u_west'][bry_ind] -= fc
                        nc.variables['ubar_west'][bry_ind] -= fc
                    elif 'east' in boundary and is_open:
                        nc.variables['u_east'][bry_ind] += fc
                        nc.variables['ubar_east'][bry_ind] += fc
                    elif 'north' in boundary and is_open:
                        nc.variables['v_north'][bry_ind] += fc
                        nc.variables['vbar_north'][bry_ind] += fc
                    elif 'south' in boundary and is_open:
                        nc.variables['v_south'][bry_ind] -= fc
                        nc.variables['vbar_south'][bry_ind] -= fc
    
    
    print('all done')
   
  
    
if __name__ == '__main__':
  
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    np.seterr(divide='ignore', invalid='ignore')
    
    roms_grd = '/server/DISKA/ROMS/SAnLX/INPUT/croco_grd.nc' 
    mercator_dir = '/server/DISKA/ROMS_DATASETS/REANALYSIS_PHY_001_030/'
    bry_files = '/server/DISKA/ROMS/SAnLX/INPUT/BRY/croco_bry_{file:d}.nc'
    
    day_zero = datetime(2000, 1, 1)
    day_zero = plt.date2num(day_zero)
    
    params = [6,0,10,42,1,1,0,1,day_zero,0.,20000.]
    
    files = sorted(glob.glob(mercator_dir + 'REANALYSIS_PHY0010??_20??_??.nc'))
    index = 0
    for i in files:
        if '2000_01' in i:
            file_start = index
        if '2010_12' in i:
            file_end = index    
        index = index + 1
    
    time_len = (file_end-file_start)+1
    
    for index in range(0,time_len):
        bry_file = bry_files.format(file = index)
        overlap_file = bry_files.format(file = (index-1))
        overlap = 0
        if index > 0:
            overlap = 10
        Process_Mercator(files[index+file_start],roms_grd,bry_file,overlap,overlap_file,params)
    
    