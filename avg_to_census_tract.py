#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
avg_to_census_tract.py
    python script to average 1km data from Shobha by contiguous US census tract
    the majority of this code is adapted from code by Dr. Gaige Kerr:
    https://github.com/gaigekerr/edf/blob/main/harmonize_afacs.py
    
    This code takes about 8 hours to run. I run it on GWUs HPC system pegasus. 
    All the code editing occured on my local machine. 
Created on Thu Mar 10 10:28:28 2022
version descriptions
v1 - intial getting code to work
v2.0 - split into annual average code and census-tract averaging code
vX indicates annual average code version; .X indicates this code version
v2.1 - added observation percent by census tract to the output
v2.2 - added alert percent by census tract to the output
v4 - updated to latest gridded data input (final version pre-submission of GeoXO paper1)
v4b - fix wyoming missing from the output csvs
v5 - use Gaige's haversine function, average alerts w/ idw, and use latest gridded ABI data
@author: kodell
"""
#%% user inputs
import sys
# version label
out_v = 'v5'
# remote paths for running on pegausus
gridded_data_fp = '/GWSPH/home/kodell/GeoXO/datafiles/out/alerts_chronic_HIA_0.01_'
census_tract_shp_fp = '/GWSPH/home/kodell/GeoXO/datafiles/in/census_tract_shapefiles_19/'
out_fp = '/GWSPH/home/kodell/GeoXO/datafiles/out/census_tract_avg_alerts/'
#load variables from the submission script - which file to read in
data_version = sys.argv[1]
version = sys.argv[2]
case = sys.argv[3]

# local paths for running locally
'''
gridded_data_fp = '/Users/kodell/Library/CloudStorage/Box-Box/Shobha_data/processed_HIA/alerts_chronic_HIA_0.01_'
census_tract_shp_fp = '/Users/kodell/Library/CloudStorage/GoogleDrive-kodell@email.gwu.edu/My Drive/Ongoing Projects/GeoXO/population_data/census_tract_shapefiles_19/'
out_fp = '/Users/kodell/Desktop/'
# to run locally, will also need to specify which files to load and version names
# for pegasus, this is done in the submission script
data_version = 'abi_og' # abi unmodified
version = 'v10' # version of abi processed data to use
case = 'base' # base or reduced alert-day concentrations, this work always uses base
'''
out_fn = data_version + '_' + version + '_'+ case + '.' + out_v + '.csv'

# specify plotting, but only if running locally
plot = False # True: make and save figures for selected area; False: don't make figures
# these plots are for zoomed in versions of both datasets mostly for a quick code check

#%% import modeules
import numpy as np
import shapefile
from shapely.geometry import shape, Point
import pandas as pd
import os
from netCDF4 import Dataset
import math
# only for plotting
if (plot):
    from ODell_udf import plt_map 
    import geopandas as gpd
    import pyproj
    import matplotlib.pyplot as plt

#%% define functions
# haversine from Gaige
def harvesine(lon1, lat1, lon2, lat2):
    """Distance calculation, degree to km (Haversine method)

    Parameters
    ----------
    lon1 : float
        Longitude of point A
    lat1 : float
        Latitude of point A
    lon2 : float
        Longitude of point B
    lat2 : float
        Latitude of point A

    Returns
    -------
    d : float
        Distance between points A and B, units of degrees
    """
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

# geo_idx from Dr. Gaige Kerr, https://github.com/gaigekerr/edf/blob/main/harmonize_afacs.py
def geo_idx(dd, dd_array):
    """Function searches for nearest decimal degree in an array of decimal 
    degrees and returns the index. np.argmin returns the indices of minimum 
    value along an axis. So subtract dd from all values in dd_array, take 
    absolute value and find index of minimum. 
    
    Parameters
    ----------
    dd : int
        Latitude or longitude whose index in dd_array is being sought
    dd_array : numpy.ndarray 
        1D array of latitude or longitude 
    
    Returns
    -------
    geo_idx : int
        Index of latitude or longitude in dd_array that is closest in value to 
        dd
    ------
    written by Dr. Gaige Kerr
    """
    geo_idx = (np.abs(dd_array - dd)).argmin()
    return geo_idx

# idwr from Dr. Gaige Kerr
def idwr(x, y, z, xi, yi):
    """Inverse distance weighting for interpolating gridded fields to census 
    tracts that are too small to intersect with the grid. 
    Parameters
    ----------
    x : list
        Longitude
    y : list
        Latitudes
    z : list
        NO2 column densities
    xi : list
        Unknown longitude
    yi : TYPE
        Unknown latitude
    Returns
    -------
    lstxyzi : list
        List comprised of unknown longitude, latitude, and interpolated 
        value for small census tract 
    ------
    """
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.nansum(sumsup)
        # The original configuration of this function had the following 
        # line of code
        # sumsup = np.nansum(np.array(sumsup) * np.array(z))
        # However, there were issues with arrays with missing data and NaNs, 
        # so it was changed to the following: 
        sumsup = np.nansum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        xyzi = [xi[p], yi[p], u]
        lstxyzi.append(xyzi)
    return lstxyzi
    
#%% load annual average file
load_fn = gridded_data_fp  + data_version + '_' + version + '_0.30.nc'
fid = Dataset(load_fn)
glats = fid['lat'][:]
glons = fid['lon'][:]
ann_count = fid['PM_obs'][:]
ann_alert = fid['PM_alerts'][:]
if case == 'base':
    annavg_pm25 = fid['annavg_PM'][:]
elif case == 'reduced':
    annavg_pm25 = fid['annavg_PM_bm'][:]
else:
    sys.exit('version not recognized')
fid.close()

#%% loop through and create census tract averages - most of this is taken
# directly from Gaige's code linked above
pm25_all = []
count_all = [] # added by me
alert_all = [] # also added by me
geoid_all = []
statefips_all = np.arange(1,57)
state_fips_files = os.listdir(census_tract_shp_fp)
for statefips in statefips_all:
#for statefips in [4]:
    statefips = str(statefips).zfill(2)
    fname = 'tl_2019_%s_tract'%statefips
    # look for the file because the state fips do not increase monotomically 
    if fname not in state_fips_files:
        continue
    print('loading',statefips)
    r = shapefile.Reader(census_tract_shp_fp+fname+'/tl_2019_%s_tract.shp'%statefips)
    # Get shapes, records
    tracts = r.shapes()
    records = r.records()
    
    # collect by state then stack all of these for full US
    tract_pm25 = []
    tract_count = []
    tract_alert = []
    tract_geoid = []
    for ti in range(len(tracts)):
        tract = shape(tracts[ti])
        record = records[ti]
        tract_geoid.append(record[3])
        area = (record['ALAND']+record['AWATER'])/(1000*1000) # convert to km2
        # convert search area to radians, and assume length as square tract
        searchrad = (np.sqrt(area)/110.) * 15.
        # still too small for really small tracts
        if searchrad < 0.05:
            searchrad = 0.05
        # large tracts will have a very large search area that is costly, so force to be smaller
        if searchrad > 2.:
            searchrad = 2.
        # Centroid of tract 
        lat_tract = tract.centroid.y
        lng_tract = tract.centroid.x
        # Subset latitude, longitude, and attributable fraction maps  
        upper = geo_idx(lat_tract+searchrad, glats)
        lower = geo_idx(lat_tract-searchrad, glats)
        left = geo_idx(lng_tract-searchrad, glons)
        right = geo_idx(lng_tract+searchrad, glons)
        lat_subset = glats[lower:upper]
        lng_subset = glons[left:right]
        pm25_subset = annavg_pm25[lower:upper, left:right]
        count_subset = ann_count[lower:upper, left:right]
        alert_subset = ann_alert[lower:upper, left:right]
        #print(pm25_subset.shape)
        # # # # Fetch coordinates within tracts (if they exist) 
        pm25_inside = []
        count_inside = []
        alert_inside = []
        interpflag = []
        for i, ilat in enumerate(lat_subset):
            for j, jlng in enumerate(lng_subset): 
                point = Point(jlng, ilat)
                if tract.contains(point) is True:
                    pm25_inside.append(pm25_subset[i,j])
                    count_inside.append(count_subset[i,j])
                    alert_inside.append(alert_subset[i,j])
                    interpflag.append(0.)
        if len(pm25_inside) == 0:
            if (statefips!='02') and (statefips!='15'): 
                idx_latnear = geo_idx(lat_tract, lat_subset)
                idx_lngnear = geo_idx(lng_tract, lng_subset)
                # Indices for 8 nearby points
                lng_idx = [idx_lngnear-1, idx_lngnear, idx_lngnear+1, 
                    idx_lngnear-1, idx_lngnear+1, idx_lngnear-1, idx_lngnear, 
                    idx_lngnear+1]
                lat_idx = [idx_latnear+1, idx_latnear+1, idx_latnear+1, 
                    idx_latnear, idx_latnear, idx_latnear-1, idx_latnear-1, 
                    idx_latnear-1]
                # Known coordinates and attributable fractions
                x = lng_subset[lng_idx]
                y = lat_subset[lat_idx]
                z1 = pm25_subset[lat_idx, lng_idx]
                z2 = alert_subset[lat_idx, lng_idx]
                pm25_inside.append(idwr(x,y,z1,[lng_tract], [lat_tract])[0][-1])
                count_inside=count_subset[lat_idx, lng_idx].flatten()
                #alert_inside=alert_subset[lat_idx,lng_idx].flatten() # old method of calculating alerts
                alert_inside.append(idwr(x,y,z2,[lng_tract], [lat_tract])[0][-1])
                #hi=bye
                interpflag.append(1.)
        pm25_inside = np.nanmean(pm25_inside)
        avail_grid_count = len(count_inside)*366 # if every grid cell had an observation every day of the year
        tract_count.append(100.0*np.nansum(count_inside)/avail_grid_count) 
        #tract_alert.append(100.0*np.nansum(alert_inside)/avail_grid_count)
        #tract_alert.append(np.nanmean(alert_inside)) # doesn't make a significant difference to do it this way
        alert_inside = np.nanmean(alert_inside) # a third option to inverse distance weight alerts as well
        tract_alert.append(alert_inside)
        tract_pm25.append(pm25_inside)
    pm25_all = np.hstack([pm25_all,tract_pm25])
    count_all = np.hstack([count_all,tract_count])
    alert_all = np.hstack([alert_all,tract_alert])
    geoid_all = np.hstack([geoid_all,tract_geoid])

#%% save output full
pm25_census_tract = pd.DataFrame(data={'geoid':geoid_all,
                                       'pm25':pm25_all,
                                       'count':count_all,
                                       'alerts':alert_all})
pm25_census_tract.to_csv(out_fp+out_fn)
print('data saved, making figures if indicated')

#%% check output with figure
if plot:
    # gridded data
    lat_plot = 34.0
    lng_plot = -112.1
    searchrad = 1.0
    upper = geo_idx(lat_plot+searchrad, glats)
    lower = geo_idx(lat_plot-searchrad, glats)
    left = geo_idx(lng_plot-searchrad, glons)
    right = geo_idx(lng_plot+searchrad, glons)
    lat_subset = glats[lower:upper]
    lng_subset = glons[left:right]
    pm25_subset = annavg_pm25[lower:upper, left:right]
    alert_subset = ann_alert[lower:upper, left:right]

    lng_plot, lat_plot = np.meshgrid(lng_subset,lat_subset)
    cnty_fn = census_tract_shp_fp+'tl_2019_04_tract/tl_2019_04_tract.shp'
    
    plt_map(lng_plot, lat_plot, alert_subset, 15,'inferno', 'Annual Alerts', 'Gridded Alerts',
            shpfile = cnty_fn,clim = [0,20])
    
    map_data = gpd.read_file(cnty_fn)
    map_data.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    map_data['pm25'] = tract_pm25
    map_data['geoID'] = tract_geoid
    map_data['alerts'] = tract_alert
    
    # census tract averages using geopandas
    # put data in geopandas array
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(10,6))
    map_data.plot(column='alerts', cmap='inferno', ax=ax, 
                         edgecolor='0.4', legend = True,vmin=0,vmax=20)
    ax.set_title('Alert Count')
    ax.axis('off')
    ax.set_xlim(lng_subset[0], lng_subset[-1])
    ax.set_ylim(lat_subset[0], lat_subset[-1])
    #plt.savefig(out_fp + 'bycause_rate_all_years.png')





