#
from positive import *

# # Return the min and max limits of an 1D array
# def lim(x):
#     # Import useful bit
#     from numpy import array,ndarray
#     if not isinstance(x,ndarray):
#         x = array(x)
#     # Columate input.
#     z = x.reshape((x.size,))
#     # Return min and max as list
#     return array([min(z),max(z)]) + (0 if len(z)==1 else array([-1e-20,1e-20]))


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
    from numpy import array,pi,sin,arange,linspace,amax,mean,sqrt

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
        # Ensure that all color vectors have a mean that is the golden ratio (less than one)
        V = abs(R+G+B)
        if not grayscale:
            V /= mean(V)*0.5*(1+sqrt(5))
            # But make sure that all values are bounded by one
            V = array([ min(v,1) for v in V ])
        # Add color vector to output
        clr.append( V )

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
def splot( domain,
           scalar_range,
           domain2=None,
           scalar_range2=None,
           kind=None,
           ms=60,
           cbfs=16,
           color_scatter=True,
           verbose=True):
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
        #
        if color_scatter:
            mkr = 'o'
        else:
            mkr = 's'
        # Set marker size
        mkr_size = ms
        # Scatter the outline of domain points
        scatter( domain2[:,0], domain2[:,1], mkr_size + 5, color='k', alpha=0.6 if color_scatter else 0.333, marker=mkr, facecolors='none' if color_scatter else 'none' )
        # Scatter the location of domain points and color by value
        if color_scatter:
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


#
def sYlm_mollweide_plot(l,m,ax=None,title=None,N=100,form=None,s=-2,colorbar_shrink=0.68):
    '''
    Plot spin weighted spherical harmonic.
    '''

    #
    from matplotlib.pyplot import subplots,gca,gcf,figure,colorbar,draw
    from numpy import array,pi,linspace,meshgrid

    # Coordinate arrays for the graphical representation
    x = linspace(-pi, pi, N)
    y = linspace(-pi/2, pi/2, N/2)
    X, Y = meshgrid(x, y)

    # Spherical coordinate arrays derived from x, y
    theta = pi/2 - y
    phi = x.copy()

    #
    if form in (None,'r','re','real'):
        SYLM_fun = lambda S,L,M,TH,PH: sYlm(S,L,M,TH,PH).real.T
        title = r'$\Re(_{%i}Y_{%i%i})$'%(s,l,m)
    elif form in ('i','im','imag'):
        SYLM_fun = lambda S,L,M,TH,PH: sYlm(S,L,M,TH,PH).imag.T
        title = r'$\Im(_{%i}Y_{%i%i})$'%(s,l,m)
    elif form in ('a','ab','abs'):
        SYLM_fun = lambda S,L,M,TH,PH: abs(sYlm(S,L,M,TH,PH)).T
        title = r'$|_{%i}Y_{%i%i}|$'%(s,l,m)
    elif form in ('+','plus'):
        SYLM_fun = lambda S,L,M,TH,PH: ( sYlm(S,L,M,TH,PH) + sYlm(S,L,-M,TH,PH) ).real.T
        title = r'$ _{%i}Y^{+}_{%i%i} = \Re \; \left[  \sum_{m\in \{%i,%i\}} \, _{%i} Y_{%i m} \; \right] $'%(s,l,m,m,-m,s,l,)
    elif form in ('x','cross'):
        SYLM_fun = lambda S,L,M,TH,PH: ( sYlm(S,L,M,TH,PH) + sYlm(S,L,-M,TH,PH) ).imag.T
        title = r'$ _{%i}Y^{\times}_{%i%i} = \Im \; \left[  \sum_{m\in \{%i,%i\}} \, _{%i} Y_{%i m} \; \right] $'%(s,l,m,m,-m,s,l,)

    #
    Z = SYLM_fun( -2,l,m,theta,phi )

    xlabels = ['$210^\circ$', '$240^\circ$','$270^\circ$','$300^\circ$','$330^\circ$',
               '$0^\circ$', '$30^\circ$', '$60^\circ$', '$90^\circ$','$120^\circ$', '$150^\circ$']

    ylabels = ['$165^\circ$', '$150^\circ$', '$135^\circ$', '$120^\circ$',
               '$105^\circ$', '$90^\circ$', '$75^\circ$', '$60^\circ$',
               '$45^\circ$','$30^\circ$','$15^\circ$']

    #
    if ax is None:
        fig, ax = subplots(subplot_kw=dict(projection='mollweide'), figsize= 1*array([10,8]) )

    #
    im = ax.pcolormesh(X,Y,Z)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.set_yticklabels(ylabels, fontsize=14)
    # ax.set_title( title, fontsize=20)
    ax.set_xlabel(r'$\phi$', fontsize=20)
    ax.set_ylabel(r'$\theta$', fontsize=20)
    ax.grid()
    colorbar(im, ax=ax, orientation='horizontal',shrink=colorbar_shrink,label=title)
    gcf().canvas.draw_idle()


# Plot a 3d meshed sphere
def plot_3d_mesh_sphere(ax=None,nth=30,nph=30,r=1,color='k',lw=1,alpha=0.1):
    #
    from numpy import sin,cos,linspace,ones_like,array,pi
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.pyplot import figure,plot,figaspect,text
    #
    if ax is None:
        fig = figure( figsize=4*figaspect(1) )
        ax = fig.add_subplot(111,projection='3d')
        axis('square')
        ax.set_xlim([-r,r])
        ax.set_ylim([-r,r])
        ax.set_zlim([-r,r])
        axis('off')
    #
    th_ = linspace(0,pi,nth)
    ph_ = linspace(0,2*pi,nph)
    #
    for th in th_:
        x = r*sin(th)*cos(ph_)
        y = r*sin(th)*sin(ph_)
        z = r*cos(th)*ones_like(ph_)
        plot(x,y,z,color=color,alpha=alpha,lw=lw)
    #
    for ph in ph_[:-1]:
        x = r*sin(th_)*cos(ph)
        y = r*sin(th_)*sin(ph)
        z = r*cos(th_)
        plot(x,y,z,color=color,alpha=alpha,lw=lw)
    #
    for ph in [ 0, pi, pi/2, 3*pi/2 ]:
        x = r*sin(th_)*cos(ph)
        y = r*sin(th_)*sin(ph)
        z = r*cos(th_)
        plot(x,y,z,color='k',alpha=0.35,lw=lw,ls='--')
    #
    for th in [ pi/2 ]:
        x = r*sin(th)*cos(ph_)
        y = r*sin(th)*sin(ph_)
        z = r*cos(th)*ones_like(ph_)
        plot(x,y,z,color='k',alpha=0.35,lw=lw,ls='--')

    # Label axes
    text_r = r*1.1
    ax.text( text_r,0,0, '$x$', zdir='x', size=24, color='k', alpha = 0.27 )
    ax.text( 0,text_r,0, '$y$', zdir='x', size=24, color='k', alpha = 0.27 )
    ax.text( 0,0,text_r, '$z$', zdir='x', size=24, color='k', alpha = 0.27 )

    # Plot the origin
    plot([0],[0],[0],'k+',alpha=0.35)
