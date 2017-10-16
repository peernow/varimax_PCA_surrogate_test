from pylab import *

def bluewhitered(a,N=256,vmin=None,vmax=None):
    bottom =    [0,   0,  0.5]
    botmiddle = [0,  0.5,  1]
    middle =    [1,   1,   1]
    topmiddle = [1,   0,   0]
    top =       [0.5, 0,   0]

    if vmin and vmax:
        lims=[vmin,vmax]
    else:
        lims=[a.min(),a.max()]

    if lims[0]<0 and lims[1]>0:
        ratio=abs(lims[0])/(abs(lims[0])+lims[1])

        cdict={}
        cdict['red']=[]
        cdict['green']=[]
        cdict['blue']=[]

        # negative part
        red=[(0.0, 0.0, 0.0),
             (ratio/2, 0.0, 0.0),
             (ratio, 1.0, 1.0)]
        green=[(0.0, 0.0, 0.0),
             (ratio/2, 0.5, 0.5),
             (ratio, 1.0, 1.0)]
        blue=[(0.0, 0.5, 0.5),
             (ratio/2, 1, 1),
             (ratio, 1.0, 1.0)]

        cdict['red'].extend(red)
        cdict['green'].extend(green)
        cdict['blue'].extend(blue)

        nratio=1-(1-ratio)/2.0
        # positive part
        red=[(ratio, 1.0, 1.0),
             (nratio, 1.0, 1.0),
             (1, 0.5, 0.5)]
        green=[(ratio, 1.0, 1.0),
             (nratio, 0., 0.),
             (1, 0.0, 0.0)]
        blue=[(ratio, 1., 1.),
             (nratio, 0, 0),
             (1, 0, 0)]

        cdict['red'].extend(red)
        cdict['green'].extend(green)
        cdict['blue'].extend(blue)




    elif lims[0]>=0:  # all positive
        cdict={}
        cdict['red']=[]
        cdict['green']=[]
        cdict['blue']=[]

        ratio=0.0
        nratio=0.5

        # positive part
        red=[(ratio, 1.0, 1.0),
             (nratio, 1.0, 1.0),
             (1, 0.5, 0.5)]
        green=[(ratio, 1.0, 1.0),
             (nratio, 0., 0.),
             (1, 0.0, 0.0)]
        blue=[(ratio, 1., 1.),
             (nratio, 0, 0),
             (1, 0, 0)]

        cdict['red'].extend(red)
        cdict['green'].extend(green)
        cdict['blue'].extend(blue)

    else: # all negative
        cdict={}
        cdict['red']=[]
        cdict['green']=[]
        cdict['blue']=[]

        ratio=1.0

        # negative part
        red=[(0.0, 0.0, 0.0),
             (ratio/2, 0.0, 0.0),
             (ratio, 1.0, 1.0)]
        green=[(0.0, 0.0, 0.0),
             (ratio/2, 0.5, 0.5),
             (ratio, 1.0, 1.0)]
        blue=[(0.0, 0.5, 0.5),
             (ratio/2, 1, 1),
             (ratio, 1.0, 1.0)]

        cdict['red'].extend(red)
        cdict['green'].extend(green)
        cdict['blue'].extend(blue)

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,N)


    return my_cmap



#a=randn(20,20)+10
#my_cmap=bluewhitered(a,256)



#clf()
#pcolor(a,cmap=my_cmap)
#colorbar()

#show()


