import os
from urllib import urlretrieve
from datetime import datetime, date
import geo_data_loader as gdl
import cPickle
import string
import numpy
import scipy.io as sio
import matplotlib
from matplotlib import pyplot
#matplotlib.rcParams.update(cfg.params)
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D

import my_cmap
# from grid import Grid

######################
TYPE = 'slp' # sat or slp
DNLOAD = False # if already downloaded, set to False
save_dir = '/media/peer/fast_data/results_rpca/xizka/data_out/'
load_dir = '/media/peer/fast_data/results_rpca/xizka/data_in/'
time_scale = 'monthly'
save_data_new_tseries=True
truncate = False
threshold_lev = .95   #.98
plot_comps = True
plot_comps_3d = False
save_data = False
save_format = 'pdf'
figure_folder = '/media/peer/fast_data/results_rpca/xizka/figures/'
correlate_time_series = False
#selected_comps = [0,1,2,4,5,8,9,18,22,26,28,33,48,52,56,59]
#selected_comps = [0,1,2,4,5,7,9,18,26,28,42,46,48,51,56]
#selected_comps = [0,1,2,4,5,7,9,18,26,46,48,52]
# selected_comps = [0,1,2,5,7,15,18,22,26,33,42,53,59]
# selected_comps = [0,1, 2,3,4,5,6,7,8,9]
selected_comps = numpy.arange(0,107,1)
final_version = False
selected_3d = [0,1,2,18,26]
# selected_comps = range(50, 60)
selected_suffix = 'talk'   #50-59_core99'   #50-59'   #'selected'  #'0-10'
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

    assert 1==2

# load daily data
if save_data:
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
# comp_fname = TYPE + comps_suffix + '.bin'
comp_fname = 'slp_all_var_b0_cosweights_varimax_detrended.bin'
# with open(save_dir + 'daily_time_series/' + comp_fname, 'r') as f:
with open(load_dir  + comp_fname, 'r') as f:
            data = cPickle.load(f)
comps = data['Ur']
lats, lons = data['lats'], data['lons']
# print lats.shape
# print lons.shape
# exit()

# print lons
# exit()
print("[%s] Components loading done. Shape of the data: %s" % (str(datetime.now()), str(comps.shape)))
#print("Daily data spatial shape is %s and components lats and lons are %d x %d" % (gf.d.shape[1:], len(lats), len(lons)))


if correlate_time_series:
                nao_file = numpy.loadtxt('/home/jakob/Daten/Indices/nao_pc-based.data',
                    skiprows=54, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                nao = nao_file[:,1:].reshape(-1)
                nao = af.anomalize(nao, time_cycle=12)
                nao_time = numpy.linspace(nao_file[0,0], nao_file[-1,0]+11./12, len(nao))

                soi_file = numpy.loadtxt('/home/jakob/Daten/Indices/soi_1866-2012.dat',
                    skiprows=82, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                soi = soi_file[:,1:].reshape(-1)
                soi = af.anomalize(soi, time_cycle=12)
                soi_time = numpy.linspace(soi_file[0,0], soi_file[-1,0]+11./12, len(soi))

                npi_file = numpy.loadtxt('/home/jakob/Daten/Indices/npi_1899-2010.dat',
                    skiprows=54, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                npi = npi_file[:,1:].reshape(-1)
                npi = af.anomalize(npi, time_cycle=12)
                npi_time = numpy.linspace(npi_file[0,0], npi_file[-1,0]+11./12, len(npi))

                ao_file = numpy.loadtxt('/home/jakob/Daten/Indices/ao.long.data',
                    skiprows=5, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                ao = ao_file[:,1:].reshape(-1)
                ao = af.anomalize(ao, time_cycle=12)
                ao_time = numpy.linspace(ao_file[0,0], ao_file[-1,0]+11./12, len(ao))

                pna_file = numpy.loadtxt('/home/jakob/Daten/Indices/pna.dat',
                    skiprows=1, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                pna = pna_file[:,1:].reshape(-1)
                pna = af.anomalize(pna, time_cycle=12)
                pna_time = numpy.linspace(pna_file[0,0], pna_file[-1,0]+11./12, len(pna))

                np_file = numpy.loadtxt('/home/jakob/Daten/Indices/np.long.data',
                    skiprows=6, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                np = np_file[:,1:].reshape(-1)
                np = af.anomalize(np, time_cycle=12)
                np_time = numpy.linspace(np_file[0,0], np_file[-1,0]+11./12, len(np))

                aao_file = numpy.loadtxt('/home/jakob/Daten/Indices/aao.dat',
                    skiprows=1, usecols=(0, 1,2,3,4,5,6,7,8,9,10,11,12))
                aao = aao_file[:,1:].reshape(-1)
                aao = af.anomalize(aao, time_cycle=12)
                aao_time = numpy.linspace(aao_file[0,0], aao_file[-1,0]+11./12, len(aao))

    #            print cfg.datatime, nao_time, soi_time, npi_time, ao_time

                ct = af.common_time_range([cfg.datatime, nao_time, soi_time, npi_time, ao_time, pna_time, np_time ])
                ct2 = af.common_time_range([cfg.datatime, aao_time])
                print ct

# truncate components
if truncate:
    lon, lat = numpy.meshgrid(lons, lats)
    grid_size, N_comps = comps.shape
    quantiles = numpy.zeros((N_comps, len(lats), len(lons)))
    comps = comps.reshape(len(lats), len(lons), N_comps)
    for i in range(N_comps):
        quantiles[i] = numpy.argsort(numpy.argsort(numpy.abs(comps[:,:,i]*numpy.cos(lat*numpy.pi/180.)).reshape(len(lats)*len(lons)))).reshape(len(lats), len(lons))/float(len(lats)*len(lons))
        print (quantiles[i]>threshold_lev).mean()
        print ((quantiles[i]>threshold_lev)*numpy.cos(lat*numpy.pi/180.)).mean()
else:
    lon, lat = numpy.meshgrid(lons, lats)
    grid_size, N_comps = comps.shape
    quantiles = numpy.zeros((N_comps, len(lats), len(lons)))
    comps = comps.reshape(len(lats), len(lons), N_comps)
        
if plot_comps:
    maximum = numpy.abs(comps).max()     
    vmin = -maximum; vmax = maximum
    comps = comps.reshape(len(lats), len(lons), N_comps)
    print numpy.cos(lat*numpy.pi/180.)
    # bm = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            # llcrnrlon=0,urcrnrlon=360,lat_ts=0,resolution='c')
    bm = Basemap(projection='robin',lon_0=180,resolution='c')
    x, y = bm(lon, lat)


    # bm.drawcoastlines(linewidth=0.4, color = 'grey')
    # bm.fillcontinents(zorder = 0)  #color='coral',lake_color='aqua')
    # bm.drawmapboundary(color='k', linewidth=0.)
    # bm.drawparallels(numpy.arange(-90.,91.,2.),labels=[0,0,0,0],linewidth=0.1, fontsize=6)
    # bm.drawmeridians(numpy.arange(0.,361.,2.),labels=[0,0,0,0],linewidth=0.1, fontsize=6)

    # pyplot.savefig(os.path.expanduser(figure_folder) + TYPE + comps_suffix + '_grid.pdf')

    # assert 1==2

    node_pos = {'y' : numpy.zeros(N_comps),  'x' : numpy.zeros(N_comps)}
    for i in range(N_comps):
        node_pos['x'][i] = lons[comps.reshape(len(lats)*len(lons), N_comps)[:,i].argmax()%192]
        node_pos['y'][i] = lats[comps.reshape(len(lats)*len(lons), N_comps)[:,i].argmax()/192]
    print node_pos
#    pp = PdfPages(os.path.expanduser(figure_folder) + TYPE + comps_suffix + '_components_truncated.pdf')


    NUM_COLORS = len(selected_comps) 
    cm = pyplot.get_cmap('hsv')

    cgen = [cm(1.*col/NUM_COLORS) for col in range(NUM_COLORS)]
#    cgen = ['blue', 'red','blue', 'red','blue', 'red','blue', 'red','blue', 'red','blue', 'red','blue', 'red']

    cmap_list = ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                 'seismic']

#    for i in range(N_comps):

    ## Plot nodes for all components

    for id, i in enumerate(selected_comps):
        print i
        fig = pyplot.figure(figsize=(3.5, 2.2), frameon=False)  #2.4))
        ax = fig.add_subplot(111, frameon=False)
        print 'plotting ', i   #, d['reg_expvar'].shape
        for idj, j in enumerate(range(N_comps)):
            px, py = bm(node_pos['x'][j], node_pos['y'][j])
            ax.scatter(px, py, s=200, 
                       facecolors = 'white', edgecolors='grey', alpha=1.,
                    clip_on=False, linewidth=.1) 

            ax.text(px, py, str(j), fontsize=10, color = 'grey', horizontalalignment='center', verticalalignment='center') #, color='grey')
        print selected_comps.shape,comps.shape,'comps'
        ax.set_title('Component %d ' 
                   %(i) , fontsize = 8)
        if truncate:
            # bm.contour(x, y, quantiles[i], [threshold_lev], linestyles = 'solid', linewidths = 3., colors = [cgen[id]])
            weights = comps[:,:,i]
            weights = numpy.ma.masked_array(comps[:,:,i], mask=quantiles[i]<threshold_lev)
#            weights[quantiles[i]<threshold_lev] = numpy.nan

            cmap_here = cmap_list[id]
            bm.contourf(x, y, weights, cmap = pyplot.get_cmap(cmap_here), 
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), alpha=.7)
        else:
            weights = comps[:,:,i]
            # cmap_here = cmap_list[id]
            cmap_here = 'bwr'
            bm.contourf(x, y, weights, cmap = pyplot.get_cmap(cmap_here), 
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), alpha=.7)
        px, py = bm(node_pos['x'][i], node_pos['y'][i])
        print [node_pos['x'][i]], [node_pos['y'][i]], px, py
        print i,'p1'
        ax.text(px, py, str(i), fontsize=10, horizontalalignment='center', verticalalignment='center') #, color='grey')
        bm.drawcoastlines(linewidth=0.4, color = 'grey')
        bm.fillcontinents(zorder = 0)  #color='coral',lake_color='aqua')
        bm.drawparallels(numpy.arange(-90.,91.,30.),labels=[0,1,0,0],linewidth=0.1, fontsize=6)
        bm.drawmeridians(numpy.arange(0.,361.,60.),labels=[1,0,0,1],linewidth=0.1, fontsize=6)
        if final_version:
            ax.text(0., 1., 'a)', fontsize=12, horizontalalignment='left', verticalalignment='top', transform=fig.transFigure)     
        fig.subplots_adjust(left = 0.01, right = .95, bottom = 0.01, top = .99)
        print i,'p2'
        print 'saving ', os.path.expanduser(figure_folder) + str(i) + TYPE + comps_suffix + '_components_truncated_allimportant.pdf'
        pyplot.savefig(os.path.expanduser(figure_folder) + str(i) + TYPE + comps_suffix + '_components_%s.%s' %(selected_suffix, save_format))
        print i


if plot_comps_3d:
    from matplotlib.collections import PolyCollection

    maximum = numpy.abs(comps).max()     
    vmin = -maximum; vmax = maximum
    comps = comps.reshape(len(lats), len(lons), N_comps)
    print numpy.cos(lat*numpy.pi/180.)

    node_pos = {'y' : numpy.zeros(N_comps),  'x' : numpy.zeros(N_comps)}
    for i in range(N_comps):
        node_pos['x'][i] = lons[comps.reshape(len(lats)*len(lons), N_comps)[:,i].argmax()%144]
        node_pos['y'][i] = lats[comps.reshape(len(lats)*len(lons), N_comps)[:,i].argmax()/144]
    print node_pos
#    pp = PdfPages(os.path.expanduser(figure_folder) + TYPE + comps_suffix + '_components_truncated.pdf')


    NUM_COLORS = len(selected_3d) 
    cm = pyplot.get_cmap('hsv')
    cgen = [cm(1.*col/NUM_COLORS) for col in range(NUM_COLORS)]
#    cgen = ['blue', 'red','blue', 'red','blue', 'red','blue', 'red','blue', 'red','blue', 'red','blue', 'red']

    fig = pyplot.figure(figsize=(5, 3), frameon=False)  #2.4))
    ax = Axes3D(fig, frame_on=False)

    # bm = Basemap(projection='robin',lon_0=180,resolution='c')
    bm = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=60,\
            llcrnrlon=0,urcrnrlon=360,lat_ts=0,resolution='c')

    x, y = bm(lon, lat)

    # ax.plot(px, py, zs=[0])
    # ax = fig.add_subplot(111, frameon=False)
    # Plot nodes for all components
    for id, i in enumerate(range(N_comps)):
        px, py = bm(node_pos['x'][i], node_pos['y'][i])
        ax.plot(px, py, zs=[0], color='grey', alpha=1.,
                    clip_on=False, linewidth=10.) 

        # ax.text(px, py, str(i), fontsize=10, color = 'grey', horizontalalignment='center', verticalalignment='center') #, color='grey')

    for id, i in enumerate(selected_3d):
        print 'plotting ', i   #, d['reg_expvar'].shape

        ## convert to quantiles
        if truncate:
            # bm.contour(X=x, Y=y, Z=quantiles[i], zs=0, levels=[threshold_lev], linestyles = 'solid', linewidths = 3., colors = [cgen[id]])
            weights = comps[:,:,i]
            weights = numpy.ma.masked_array(comps[:,:,i], mask=quantiles[i]<threshold_lev)
#            weights[quantiles[i]<threshold_lev] = numpy.nan
            bm.contourf(x, y, weights, zdir='z', offset=0, cmap = my_cmap.bluewhitered(numpy.array([vmin,vmax]),72), norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), alpha=.7)
#            bm.contourf(x, y, weights, norm=False , alpha=.7)

        px, py = bm(node_pos['x'][i], node_pos['y'][i])
#        bm.scatter(px, py , marker = '*', s = 20, color = 'black')
        print [node_pos['x'][i]], [node_pos['y'][i]], px, py
        # ax.text(px, py, str(i), fontsize=10, horizontalalignment='center', verticalalignment='center') #, color='grey')

    ax.add_collection3d(bm.drawcoastlines(linewidth=.3))  #linewidth=0.4, color = 'grey'))
    # bm.drawmapboundary(color='k', linewidth=0.)
    # ax.add_collection3d(bm.fillcontinents(zorder = 0))  #color='coral',lake_color='aqua')
    # bm.drawparallels(numpy.arange(-90.,91.,30.),labels=[0,1,0,0],linewidth=0.1, fontsize=6)
    # bm.drawmeridians(numpy.arange(0.,361.,60.),labels=[1,0,0,1],linewidth=0.1, fontsize=6)
    # ax.grid(True)
    # ax.set_zlim3d(0,1.)
    ax.set_axis_off()
    ax.azim = 270
    ax.elev = 30
    ax.dist = 7     
#    pp.close()
    pyplot.savefig(os.path.expanduser(figure_folder) + TYPE + comps_suffix + '_components_%s.%s' %('selected_3d', 'svg'))


if save_data_new_tseries:
    from_date = date(y_start, 1, 1)
    to_date = date(y_end, 1, 1)
    fname='/media/peer/fast_data/results_rpca/xizka/data_in/xizka_all_150years_sel.nc'
    varname='temp'
    gf = gdl.load_monthly_data_general(fname, varname, 
                from_date=from_date, to_date=to_date, months=None, slice_lon=None, 
                slice_lat=[-89, 89], level=None, var_norm = True)
    comps = comps.reshape(grid_size, N_comps)
    if truncate:
#            grid_size, N_comps = comps.shape
#            comps = comps.reshape(len(lats), len(lons), N_comps)
        weights = comps[:,:,i]
        comps[quantiles<threshold_lev] = 0.; comps/= weights.sum()
        comps = comps.reshape(grid_size, N_comps)

    # matrix product - yielding the daily time series for components
    print "[%s] Computing %s time series for components..." %(str(datetime.now()), time_scale)
    if string.find(comp_fname, 'detrended') != -1: # if comp is derived from detrended data, detrend daily data as well
        print 'Detrending daily data...'
        gf.detrend()
        TYPE += '_detrended'
    ### the indexing here might need to change for my own run data!!!!!!!!!!<
    data = gf.data()
    if string.find(comp_fname, 'cosweights') != -1: # if comp is derived from cosweighted data, cosweight daily as well
        print 'Cosweighting %s data...' %time_scale
        data *= gf.qea_latitude_weights()
        TYPE += '_cosweights'
    data = numpy.reshape(data, (data.shape[0], numpy.prod(data.shape[1:]))) # flattening field of daily data
    print data.shape,comps.shape,'compqre shapes'
    comp_ts = numpy.dot(data, comps)
    print("[%s] Done. Shape of the %s time series for components is %s. Saving file..." % (str(datetime.now()), time_scale, str(comp_ts.shape)))
    # save_file = '/home/jakob/Daten/varimax_components/' + TYPE + '_%s' %time_scale
    save_file = save_dir + TYPE + 'peer_%s' %time_scale
    if truncate:
        save_file = save_file + '_truncated'
    with open(save_file + '.bin', 'w') as f:
        cPickle.dump({ 'ts' : comp_ts, 'lats' : lats, 'lons' : lons, 'Ur': comps }, f)
    sio.savemat(save_file + '.mat', { 'ts' : comp_ts, 'lats' : lats, 'lons' : lons, 'Ur': comps })
    print("[%s] Saving done." % (str(datetime.now())))
    print 'lalalalal'
