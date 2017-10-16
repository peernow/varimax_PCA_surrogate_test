import os
from urllib import urlretrieve
from datetime import datetime, date
import geo_data_loader as gdl
import cPickle
import string
import numpy
import scipy.io as sio

######################
TYPE = 'slp' # sat or slp
DNLOAD = False # if already downloaded, set to False
save_dir = '/home/jakobrunge/Daten/varimax_components/'
time_scale = 'daily'

load_from_disk = False
######################


# download data from NCEP/NCAR reanalysis
#os.chdir('/home/nikolaj/Work/climate')
y_start = 1948
y_end = 2015
if DNLOAD:
    print("[%s] Downloading daily data from NOAA ftp server for years %d - %d..." % (str(datetime.now()), y_start, y_end))
#    url_base = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/'
    url_base = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/'
    for y in range(y_start, y_end + 1):
        if TYPE == 'sat':
            file = 'air.sig995.' + str(y) + '.nc'
            save = save_dir + 'SAT%s/' %time_scale
        elif TYPE == 'slp':
            file = 'pres.sfc.' + str(y) + '.nc'
            save = save_dir + 'SLP%s/' %time_scale
        urlretrieve(url_base + file, save + file)
        if ((y + 1 - y_start) % 10 == 0):
            print("... %i / %i completed..." % (y + 1 - y_start, y_end + 1 - y_start))
    print("[%s] Download complete." % (str(datetime.now())))


# load daily data
if load_from_disk:
    if time_scale == 'daily':
        print("[%s] Loading daily %s data..." % (str(datetime.now()), TYPE.upper()))
        if TYPE == 'slp':
            fname = save_dir + 'SLPdaily/pres.sfc.%d.nc'
            varname = 'pres'
        elif TYPE == 'sat':
            fname = save_dir + 'SATdaily/air.sig995.%d.nc'
            varname = 'air'
        from_date = date(y_start, 1, 1)
        to_date = date(y_end, 1, 1)
        gf = gdl.load_daily_data_general(fname, varname, from_date, to_date, None, [-87.5, 87.5], None)  #, var_norm = True)
        print("[%s] Daily %s data loaded. Shape of the data: %s" % (str(datetime.now()), TYPE.upper(), str(gf.d.shape)))
    else:
        if TYPE == 'slp':
            fname = save_dir + 'SLPmonthly/pres.sfc.mon.mean.nc'
            varname = 'pres'
        elif TYPE == 'sat':
            fname = save_dir + 'SATmonthly/air.sig995.mon.mean.nc'
            varname = 'air'
        from_date = date(y_start, 1, 1)
        to_date = date(y_end, 1, 1)
        gf = gdl.load_monthly_data_general(fname, varname, 
                from_date=from_date, to_date=to_date, months=None, slice_lon=None, 
                slice_lat=[-89, 89], level=None, var_norm = True)


# load components
print("[%s] Loading components..." % (str(datetime.now())))
comps_dir= 'new-components-sat-slp-2013/'
#comps_suffix = '_all_var_b0_cosweights_varimax_'
comps_suffix = '_detrended_cosweights_daily_detrended'
comp_fname = TYPE + comps_suffix + '.bin'
with open(save_dir + 'daily_time_series/' + comp_fname, 'r') as f:
            data = cPickle.load(f)
comps = data['Ur']
lats, lons = data['lats'], data['lons']
print("[%s] Components loading done. Shape of the data: %s" % (str(datetime.now()), str(comps.shape)))
#print("Daily data spatial shape is %s and components lats and lons are %d x %d" % (gf.d.shape[1:], len(lats), len(lons)))


# matrix product - yielding the daily time series for components
print "[%s] Computing %s time series for components..." %(str(datetime.now()), time_scale)
if string.find(comp_fname, 'detrended') != -1: # if comp is derived from detrended data, detrend daily data as well
    print 'Detrending daily data...'
    gf.detrend()
    TYPE += '_detrended'
data = gf.data()
if string.find(comp_fname, 'cosweights') != -1: # if comp is derived from cosweighted data, cosweight daily as well
    print 'Cosweighting %s data...' %time_scale
    data *= gf.qea_latitude_weights()
    TYPE += '_cosweights'
data = numpy.reshape(data, (data.shape[0], numpy.prod(data.shape[1:]))) # flattening field of daily data
comp_ts = numpy.dot(data, comps)
print("[%s] Done. Shape of the %s time series for components is %s. Saving file..." % (str(datetime.now()), time_scale, str(comp_ts.shape)))
save_file = '/home/jakob/Daten/varimax_components/' + TYPE + '_%s' %time_scale
if truncate:
    save_file = save_file + '_truncated'
with open(save_file + '.bin', 'w') as f:
    cPickle.dump({ 'ts' : comp_ts, 'lats' : lats, 'lons' : lons, 'Ur': comps }, f)
sio.savemat(save_file + '.mat', { 'ts' : comp_ts, 'lats' : lats, 'lons' : lons, 'Ur': comps })
print("[%s] Saving done." % (str(datetime.now())))
