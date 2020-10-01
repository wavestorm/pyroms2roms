# -*- coding: utf-8 -*-
# %run py_mercator2ini.py

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
===========================================================================

Create a ROMS initial file based on Mercator data

===========================================================================
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
import numexpr as ne
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
import glob as glob
import time
import scipy.interpolate.interpnd as interpnd
import collections
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime
import calendar
from time import strptime
#import copy

from py_roms2roms import vertInterp, horizInterp, bry_flux_corr, debug0, debug1, debug2
from make_mercator import RomsGrid, MercatorData, prepare_romsgrd
from pyroms2ini import RomsData


class MercatorDataIni (MercatorData):
    '''
    MercatorDataIni class (inherits from MercatorData class)
    '''
    def __init__(self, filenames, model_type, mercator_var, romsgrd, **kwargs):
        """
        Creates a new Mercator ini data object.
        
        Parameters
        ----------
        
        *filenames* : list of Mercator nc files.
        *model_type* : string specifying Mercator model.
        *romsgrd* : a `RomsGrid` instance.
        
        """
        super(MercatorDataIni, self).__init__(filenames, model_type, mercator_var, romsgrd, **kwargs)
        
        
    def interp2romsgrd(self):
        '''
        Modification of MercatorData class definition *interp2romsgrd*
        '''
        if '3D' in self.dimtype:
            for k in np.arange(self._depths.size):
                self._interp2romsgrd(k)
        else:
            self._interp2romsgrd()
        return self


    def vert_interp(self, j):
        '''
        Modification of MercatorData class definition *vert_interp*
        '''
        vinterp = vertInterp(self.mapcoord_weights)
        self.dataout[:, j] = vinterp.vert_interp(self.datatmp[::-1, j])
        return self
      




def prepare_mercator(mercator, balldist):
    mercator.proj2gnom(ignore_land_points=False, M=mercator.romsgrd.M)
    mercator.child_contained_by_parent(mercator.romsgrd)
    mercator.make_kdetree().get_fillmask_cofs()
    ballpoints = mercator.kdetree.query_ball_tree(romsgrd.kdetree, r=balldist)
    #print 'dd', print(ballpoints)
    try:
        mercator.ball = np.array(ballpoints)
    except:
        print(ballpoints)
    mercator.ball = mercator.ball.nonzero()
    #np.array(np.array(ballpoints).nonzero()[0])
    #print(ballpoints[0:10])
    mercator.ball = mercator.ball[0]
    mercator.tri = sp.Delaunay(mercator.points[mercator.ball])
    if '3D' in mercator.dimtype:
        mercator.set_3d_depths()
    return mercator
    
    
    
if __name__ == '__main__':
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    np.seterr(divide='ignore', invalid='ignore')
    
    roms_grd = '/server/DISK3/SAnLX/INPUT/croco_grd.nc'
    roms_ini = '/server/DISK3/SAnLX/INPUT/croco_ini.nc'    
    mercator_dir = '/server/DISKA/ROMS_DATASETS/REANALYSIS_PHY_001_030/'
#    mercator_dir = '/server/DISKA/ROMS_DATASETS/'
        
    chd_sigma_params = dict(theta_s=6, theta_b=0, hc=10, N=42)
    
    ini_date   = '20110101'
#    ini_date   = '20110116'     
    ini_time = 4018.  # days
    tstart = 4018.
    tend = 4018.

    date_str = (ini_date[:4], ini_date[4:6], ini_date[6:])
    mercator_ssh_file = 'REANALYSIS_PHY001030_%s_%s.nc' % (ini_date[:4], ini_date[4:6])
#    mercator_ssh_file = 'REANALYSIS_%s_%s.nc' % (ini_date[:4], ini_date[4:6])
    mercator_temp_file = mercator_dir + mercator_ssh_file
    mercator_salt_file = mercator_dir + mercator_ssh_file
    mercator_u_file = mercator_dir + mercator_ssh_file
    mercator_v_file = mercator_dir + mercator_ssh_file
    mercator_ssh_file = mercator_dir + mercator_ssh_file
    #print 'aaaaaaa', mercator_ssh_file 
    
    balldist = 20000. # meters
    
    # Scale final variables. Can help with blow ups at initialization.
    vel_scale = 1
    zeta_scale = 1

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    fillval = 9999
  
    print(mercator_ssh_file)
  
    mercator_ssh_file = glob.glob(mercator_ssh_file)
    mercator_temp_file = glob.glob(mercator_temp_file)
    mercator_salt_file = glob.glob(mercator_salt_file)
    mercator_u_file = glob.glob(mercator_u_file)
    mercator_v_file = glob.glob(mercator_v_file)
    
       
    #ini_filename = ini_filename.replace('.nc', '_%s.nc' %ini_date)

    day_zero = datetime(int(ini_date[:4]), int(ini_date[4:6]), int(ini_date[6:]))
    day_zero = plt.date2num(day_zero) + 0.5

    
    # Set up a RomsGrid object
    romsgrd = RomsGrid(roms_grd, chd_sigma_params, model_type='ROMS')
    
    # Set up a RomsData object for the initial file
    romsini = RomsData(roms_ini, model_type='ROMS')
    
    # Create the initial file
    romsini.create_ini_nc(romsgrd, fillval)

    
    mercator_vars = OrderedDict([('SSH', mercator_ssh_file),
                                 ('TEMP', mercator_temp_file),
                                 ('SALT', mercator_salt_file),
                                 ('U', mercator_u_file),
                                 ('V', mercator_v_file)])
    
    '''
    mercator_vars = OrderedDict([('U', mercator_u_file),
                                 ('V', mercator_v_file)])
    
    mercator_vars = OrderedDict([('zos', mercator_ssh_file),
                                 ('thetao', mercator_temp_file),
                                 ('so', mercator_salt_file),
                                 ('u0', mercator_u_file),
                                 ('vo', mercator_v_file)])
    '''
    
    # Set partial domain chunk sizes
    ndomx = 2# 5
    ndomy = 3#4
    
    Mp, Lp = romsgrd.h().shape
    szx = np.floor(Lp / ndomx).astype(int)
    szy = np.floor(Mp / ndomy).astype(int)
    
    icmin = np.arange(0, ndomx) * szx
    icmax = np.arange(1, ndomx + 1) * szx
    jcmin = np.arange(0, ndomy) * szy
    jcmax = np.arange(1, ndomy + 1) * szy
    
    icmin[0] = 0
    icmax[-1] = Lp + 1
    jcmin[0] = 0
    jcmax[-1] = Mp + 1
    
    
    for domx in np.arange(ndomx):
        
        for domy in np.arange(ndomy):
    
            romsgrd.i0 = icmin[domx]
            I1 = romsgrd.i1 = icmax[domx]
            romsgrd.j0 = jcmin[domy]
            J1 = romsgrd.j1 = jcmax[domy]
            
            i0, i1, j0, j1 = romsgrd.i0, romsgrd.i1, romsgrd.j0, romsgrd.j1
            
            Mp, Lp = romsgrd.h().shape
            
            
            
            for mercator_var, mercator_file in zip(mercator_vars.keys(), mercator_vars.values()):
                
                try:
                    del mercator
                except:
                    pass
                
                print('\nProcessing variable *%s*' %mercator_var)
                
                if mercator_var in 'U':
                    romsgrd.i1 += 1
                    i1 += 1
                    romsgrd.j1 += 1
                    j1 += 1
                
                elif mercator_var in 'V':
                    pass
                
                else:
                    romsgrd.i1 = i1 = I1
                    romsgrd.j1 = j1 = J1
                
                romsgrd = prepare_romsgrd(romsgrd)
                mercator = MercatorDataIni(mercator_file[0], 'Mercator', mercator_var, romsgrd)
                mercator = prepare_mercator(mercator, balldist)
        
        
                if 0: # debug
                    plt.pcolormesh(mercator.lon(), mercator.lat(),mercator.maskr())
                    plt.scatter(mercator.lon().flat[mercator.ball], mercator.lat().flat[mercator.ball],
                                s=10,c='g',edgecolors='none')
                    plt.plot(romsgrd.boundary()[0], romsgrd.boundary()[1], 'w')
                    plt.axis('image')
                    plt.show()
                    

        
                # Read in variables and interpolate to romsgrd
                mercator.get_variable(day_zero).fillmask()
                mercator.interp2romsgrd()
        
        
                if '3D' in mercator.dimtype:
                    for j in np.arange(romsgrd.maskr().shape[0]):
                        if j / 100. == np.fix(j / 100.):
                            print('------ j = %s of %s' %(j + 1, romsgrd.maskr().shape[0]))
                
                        mercator.set_map_coordinate_weights(j=j)
                        mercator.vert_interp(j=j)
                
                
                # Calculate barotropic velocities
                if mercator.vartype in ('U', 'V'):
                    mercator.set_barotropic()
                
                # Write to initial file
                print('Saving *%s* to %s' %(mercator.vartype, romsini.romsfile))
                with netcdf.Dataset(romsini.romsfile, 'a') as nc:
                    
                    if mercator.vartype in 'U':
                
                        u = mercator.dataout.copy()
                        ubar = mercator.barotropic.copy()
            
                    elif mercator.vartype in 'V':
                        #aaaaaaaa
                        ubar, vbar = romsgrd.rotate(ubar, mercator.barotropic, sign=1)
                        ubar = romsgrd.rho2u_2d(ubar)
                        ubar *= romsgrd.umask()[j0:j1, i0:i1 - 1]
                        nc.variables['ubar'][:, j0:j1, i0:i1 - 1] = ubar[np.newaxis] *vel_scale
                        del ubar
                        vbar = romsgrd.rho2v_2d(vbar)
                        vbar *= romsgrd.vmask()[j0:j1 - 1, i0:i1]
                        nc.variables['vbar'][:, j0:j1 - 1, i0:i1] = vbar[np.newaxis] * vel_scale
                        del vbar
                
                        for k in np.arange(romsgrd.N.size).astype(np.int):
                            utmp, vtmp = u[k], mercator.dataout[k]
                            u[k], mercator.dataout[k] = romsgrd.rotate(utmp, vtmp, sign=1)
                
                        u = romsgrd.rho2u_3d(u)
                        u *= romsgrd.umask3d()[:, j0:j1, i0:i1 - 1]
                        nc.variables['u'][:, :, j0:j1, i0:i1 - 1] = u[np.newaxis] * vel_scale
                        del u
                        v = romsgrd.rho2v_3d(mercator.dataout)
                        v *= romsgrd.vmask3d()[:, j0:j1 - 1, i0:i1]
                        nc.variables['v'][:, :, j0:j1 - 1, i0:i1] = v[np.newaxis] * vel_scale
                        del v
                
            
                    elif mercator.vartype in 'SSH':
                
                
                        mercator.dataout = mercator.dataout.reshape(Mp, Lp)[np.newaxis]
                        mercator.dataout *= romsgrd.maskr()
                        
                        nc.variables['zeta'][:, j0:j1, i0:i1] = mercator.dataout * zeta_scale
    
                        nc.variables['scrum_time'][:] = ini_time * 86400
                        nc.variables['tstart'][:] = tstart
                        nc.variables['tend'][:] = tend
                        nc.mercator_start_date = ini_date
            
                    elif mercator.vartype in 'TEMP' and mercator.dataout.max() > 100.:
                
                        mercator.dataout -= mercator.Kelvin
                        mercator.dataout *= romsgrd.mask3d()
                        nc.variables['temp'][:, :, j0:j1, i0:i1] = mercator.dataout[np.newaxis]
            
                    else:    
                
                        varname = mercator.vartype.lower()
                        mercator.dataout *= romsgrd.mask3d()
                        nc.variables[varname][:, :, j0:j1, i0:i1] = mercator.dataout[np.newaxis]
    
    print('all done')
            
            
      














