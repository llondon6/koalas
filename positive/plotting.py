#
from positive import *

# Return the min and max limits of an 1D array
def lim(x):
    # Import useful bit
    from numpy import array,ndarray
    if not isinstance(x,ndarray):
        x = array(x)
    # Columate input.
    z = x.reshape((x.size,))
    # Return min and max as list
    return array([min(z),max(z)]) + (0 if len(z)==1 else array([-1e-20,1e-20]))


# Function to produce array of color vectors
def rgb( N,                     #
         offset     = None,     #
         speed      = None,     #
         plot       = False,    #
         shift      = None,     #
         jet        = False,    #
         reverse    = False,    #
         weights    = None,     #
         grayscale  = None,     #
         verbose    = None ):   #

    '''
    Function to produce array of color vectors.
    '''

    #
    from numpy import array,pi,sin,arange,linspace,amax

    # If bad first intput, let the people know.
    if not isinstance( N, int ):
        msg = 'First input must be '+cyan('int')+'.'
        raise ValueError(msg)

    #
    if offset is None:
        offset = pi/4.0

    #
    if speed is None:
        speed = 2.0

    #
    if shift is None:
        shift = 0

    #
    if jet:
        offset = -pi/2.1
        shift = pi/2.0

    #
    if weights is None:
        t_range = linspace(1,0,N)
    else:
        if len(weights)==N:
            t_range = array(weights)
            t_range /= 1 if 0==amax(t_range) else amax(t_range)
        else:
            error('weights must be of length N','rgb')

    #
    if reverse:
        t_range = linspace(1,0,N)
    else:
        t_range = linspace(0,1,N)



    #
    r = array([ 1, 0, 0 ])
    g = array([ 0, 1, 0 ])
    b = array([ 0, 0, 1 ])

    #
    clr = []
    w = pi/2.0
    for t in t_range:

        #
        if not grayscale:
            R = r*sin( w*t                + shift )
            G = g*sin( w*t*speed + offset + shift )
            B = b*sin( w*t + pi/2         + shift )
        else:
            R = r*t
            G = g*t
            B = b*t
        #
        clr.append( abs(R+G+B) )

    #
    if plot:

        #
        from matplotlib import pyplot as p

        #
        fig = p.figure()
        fig.set_facecolor("white")

        #
        for k in range(N):
            p.plot( array([0,1]), (k+1.0)*array([1,1])/N, linewidth=20, color = clr[k] )

        #
        p.axis('equal')
        p.axis('off')

        #
        p.ylim([-1.0/N,1.0+1.0/N])
        p.show()

    #
    return array(clr)


# Plot 2d surface and related scatter points
def splot(domain,scalar_range,domain2=None,scalar_range2=None,kind=None,ms=35,cbfs=12):
    '''Plot 2d surface and related scatter points '''

    # Import usefult things
    from matplotlib.pyplot import figure,plot,scatter,xlabel,ylabel,savefig,imshow,colorbar,gca
    from numpy import linspace,meshgrid,array,angle,unwrap
    from matplotlib import cm

    #
    plot_scatter = (domain2 is not None) and (scalar_range2 is not None)

    #
    fig = figure( figsize=2*array([4,2.8]) )
    clrmap = cm.coolwarm

    #
    # Z = abs(SR) if kind=='amp' else angle(SR)
    # Z = abs(scalar_range) if kind=='amp' else scalar_range
    # Z = sunwrap(angle(scalar_range)) if kind=='phase' else scalar_range
    if kind=='amp':
        Z = abs(scalar_range)
    elif kind=='phase':
        Z = sunwrap(angle(scalar_range))
    else:
        Z = scalar_range
    #
    norm = cm.colors.Normalize(vmax=1.1*Z.max(), vmin=Z.min())

    # Plot scatter of second dataset
    if plot_scatter:
        # Set marker size
        mkr_size = ms
        # Scatter the outline of domain points
        scatter( domain2[:,0], domain2[:,1], mkr_size+5, color='k', alpha=0.6, marker='o', facecolors='none' )
        # Scatter the location of domain points and color by value
        Z_ = abs(scalar_range2) if kind=='amp' else sunwrap(angle(scalar_range2))
        scatter( domain2[:,0],domain2[:,1], mkr_size, c=Z_,
                 marker='o',
                 cmap=clrmap, norm=norm, edgecolors='none' )

    #
    extent = (domain[:,0].min(),domain[:,0].max(),domain[:,1].min(),domain[:,1].max())
    im = imshow(Z, extent=extent, aspect='auto',
                    cmap=clrmap, origin='lower', norm=norm )

    #
    cb = colorbar()
    cb_range = linspace(Z.min(),Z.max(),5)
    cb.set_ticks( cb_range )
    cb.set_ticklabels( [ '%1.3f'%k for k in cb_range ] )
    cb.ax.tick_params(labelsize=cbfs)

    #
    return gca()
