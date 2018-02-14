#
from positive import *

# Smooth 1D data
class smooth:
    '''
    Smooth 1D data. Initially based on https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    '''

    # Class constructor
    def __init__(this,y,width=None,method=None):
        # Import useful things
        from numpy import ones,convolve,mod,hstack,arange,cumsum,mod,array
        # Handle method input; set default
        method = 'average' if method is None else method.lower()
        # Handle n input; default is None which causes method to be auto
        method = 'auto' if width is None else method
        # Store relevant inputs to this object
        this.scalar_range = array(y)
        this.width = width
        this.method = method
        # Handle different methods
        if method in ('average'):
            # Use Rolling Average (non convulative)
            y_smooth = this.__rolling_average__(width)
        elif method in ('auto','optimal'):
            # Automatically determine best smoothing length to use with average
            y_smooth = this.__auto_smooth__()
        else:
            error('unknown smoothing method requested: %s'%red(method))
        #
        this.answer = y_smooth

    # Smooth using moving average of available pionts
    def __rolling_average__(this,width):
        # Import useful things
        from numpy import ones,mod,array
        ''' Use a rolling average '''
        # NOTE: I tried using convolution, but it didnt handle general boundary conditions well; so I wrote my own algo
        if width > 0:
            width = int(width+mod(width,2))/2
            z = array(this.scalar_range)
            for k in range(len(z)):
                #
                a = max(0,k-width)
                b = min(len(this.scalar_range),k+width)
                s = min( k-a, b-k )
                a,b = k-s,k+s
                z[k] = sum( this.scalar_range[a:b] ) / (b-a) if b>a else this.scalar_range[k]
        else:
            z = this.scalar_range
        #
        ans = z
        return ans

    # Automatically determine best smoothing length to use with average
    def __auto_smooth__(this):
        '''Automatically determine best smoothing length to use with average'''
        # Import useful things
        from numpy import ones,convolve,mod,hstack,arange,cumsum,mod,array
        from numpy import poly1d,polyfit,std,argmin

        #
        err,smo = [],[]
        width_range = range(3,min(50,int(len(this.scalar_range)/2)))
        #
        for j,k in enumerate(width_range):
            smo.append( smooth(this.scalar_range,k,method='average').answer )
            err.append( std( this.scalar_range - smo[-1] )/std(this.scalar_range) )
        #
        modeled_err = poly1d( polyfit(width_range,err,2) )(width_range)
        k = argmin( modeled_err )
        best_width = width_range[k] if k>0 else 3
        #
        y_smooth = smooth(this.scalar_range,best_width,method='average').answer
        #
        this.raw_error = err
        this.modeled_error = modeled_err
        this.trial_answers = smo
        this.width_range = width_range
        this.width = best_width
        #
        ans = y_smooth
        return ans

    # Plotting function
    def plot(this):
        # Import useful things
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16
        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,legend,subplots,gca,sca,xlim,title,subplot
        from numpy import array,arange,argmin
        #
        if this.method in ('auto'):
            #
            fsz = 1.2*array([12,4])
            fig1 = figure( figsize=fsz )
            subplot(1,2,1)
            plot( this.scalar_range,'ok',alpha=0.5)
            xlim( lim(arange(len(this.scalar_range))) )
            clr = rgb( len(this.width_range), jet=True, reverse=True )
            for j,k in enumerate(this.width_range):
                plot( this.trial_answers[j], color = clr[j], alpha=0.2 )
            #
            plot( this.answer, '-k' )
            xlabel('$x$')
            ylabel('$y(x)$')
            title('Method = "%s"'%this.method)
            #
            subplot(1,2,2)
            plot( this.width_range, this.raw_error, 'k', alpha=0.5 )
            plot( this.width_range, this.modeled_error, 'g' )
            k = argmin( this.modeled_error )
            best_n = this.width_range[k] if k>0 else 0
            plot( this.width_range[k], this.modeled_error[k], 'og', mec='none' )
            xlim( lim(this.width_range) )
            xlabel('$x$')
            ylabel('error for $y(x)$')
            title('Smoothed with $width = %d$'%this.width)
        else:
            fsz = 1.2*array([6,4])
            fig = figure( figsize=fsz )
            #
            x = arange(len(this.scalar_range))
            y = this.scalar_range
            plot(x, y,'ok',alpha=0.3,label='Input Data')
            plot(x, this.answer, 'r', label='Smoothed Data' )
            xlim( lim(x) )
            xlabel('$x$')
            ylabel('$y(x)$')
            legend(frameon=False)
            title('Smoothed with $width = %d$'%this.width)


# Given an array, return a processed array such that, from 0 to k, the value of the array taken on the maximum value on [0,k]. The result is monotomic. NOTE that this function is useful for feature selection.
def upbow(a):
    '''
    Given an array, return a processed array such that, from 0 to k, the value of the array taken on the maximum value on [0,k]. The result is monotomic. NOTE that this function is useful for feature selection.
    ~llondon
    '''
    from numpy import ndarray,array
    if not isinstance(a,ndarray):
        error('input must be ndarray, instead it\'s %s'%(type(a).__class__.__name__))
    b = a.copy()
    u = a[0]
    for k,v in enumerate(a):
        b[k] = max(u,a[k])
        u = b[k]
    return b


# [Depreciated???] custome function for setting desirable ylimits
def pylim( x, y, axis='both', domain=None, symmetric=False, pad_y=0.1 ):
    '''Try to automatically determine nice xlim and ylim settings for the current axis'''
    #
    from matplotlib.pyplot import xlim, ylim
    from numpy import ones

    #
    if domain is None:
        mask = ones( x.shape, dtype=bool )
    else:
        mask = (x>=min(domain))*(x<=max(domain))

    #
    if axis == 'x' or axis == 'both':
        xlim( lim(x) )

    #
    if axis == 'y' or axis == 'both':
        limy = lim(y[mask]); dy = pad_y * ( limy[1]-limy[0] )
        if symmetric:
            ylim( [ -limy[-1]-dy , limy[-1]+dy ] )
        else:
            ylim( [ limy[0]-dy , limy[-1]+dy ] )



# Calculate teh positive definite represenation of the input's complex phase
def anglep(x):
    '''Calculate teh positive definite represenation of the input's complex phase '''
    from numpy import angle,amin,pi,exp,amax
    #
    initial_shape = x.shape
    x_ = x.reshape( (x.size,) )
    #
    x_phase = angle(x_)
    C = 2*pi # max( abs(amin(x_phase)), abs(amax(x_phase))  )
    x_phase -= C
    for k,y in enumerate(x_phase):
        while y < 0:
            y += 2*pi
        x_phase[k] = y
    return x_phase.reshape(initial_shape)+C


# Sort an array, unwrap it, and then reimpose its original order
def sunwrap( a ):
    ''' Sort an array, unwrap it, and then reimpose its original order '''

    # Import useful things
    from numpy import unwrap,array,pi,amin,amax,isnan,nan,isinf,isfinite,mean

    # Flatten array by size
    true_shape = a.shape
    b = a.reshape( (a.size,) )

    # Handle non finites
    nanmap = isnan(b) | isinf(b)
    b[nanmap] = -200*pi*abs(amax(b[isfinite(b)]))

    # Sort
    chart = sorted(  range(len(b))  ,key=lambda c: b[c])

    # Apply the sort
    c = b[ chart ]

    # Unwrap the sorted
    d = unwrap(c)
    d -= 2*pi*( 1 + int(abs(amax(d))) )
    while amax(d)<0:
        d += 2*pi

    # Re-order
    rechart = sorted(  range(len(d))  ,key=lambda r: chart[r])

    # Restore non-finites
    e = d[ rechart ]
    e[nanmap] = nan

    #
    f = e - mean(e)
    pm = mean( f[f>=0] )
    mm = mean( f[f<0] )
    while pm-mm > pi:
        f[ f<0 ] += 2*pi
        mm = mean( f[f<0] )
    f += mean(e)


    # Restore true shape and return
    return f.reshape( true_shape )
    # from numpy import unwrap
    # return unwrap(a)


#
def sunwrap_dev(X_,Y_,Z_):
    '''Given x,y,z unwrap z using x and y as coordinates'''

    #
    from numpy import unwrap,array,pi,amin,amax,isnan,nan
    from numpy import sqrt,isinf,isfinite,inf
    from numpy.linalg import norm

    #
    true_shape = X_.shape
    X = X_.reshape( (X_.size,) )
    Y = Y_.reshape( (Y_.size,) )
    Z = Z_.reshape( (Z_.size,) )

    #
    threshold = pi

    #
    skip_dex = []
    for k,z in enumerate(Z):
        #
        if isfinite(z) and ( k not in skip_dex ):
            #
            x,y = X[k],Y[k]
            #
            min_dr,z_min,j_min = inf,None,None
            for j,zp in enumerate(Z):
                if j>k:
                    dr = norm( [ X[j]-x, Y[j]-y ] )
                    if dr < min_dr:
                        min_dr = dr
                        j_min = j
                        z_min = zp
            #
            if z_min is not None:
                skip_dex.append( j_min )
                dz = z - z_min
                if dz < threshold:
                    Z[k] += 2*pi
                elif dz> threshold:
                    Z[k] -= 2*pi

    #
    ans = Z.reshape( true_shape )

    #
    return ans


# Useful identity function of two inputs --- this is here becuase pickle cannot store lambdas in python < 3
def IXY(x,y): return y

# Rudimentary single point outlier detection based on cross validation of statistical moments
# NOTE that this method is to be used sparingly. It was developed to help extrapolate NR data ti infinity
def single_outsider( A ):
    '''Rudimentary outlier detection based on cross validation of statistical moments'''

    # Import useful things
    from numpy import std,array,argmin,ones,mean

    #
    true_shape = A.shape

    #
    a = array( abs( A.reshape( (A.size,) ) ) )
    a = a - mean(a)

    #
    std_list = []
    for k in range( len(a) ):

        #
        b = [ v for v in a if v!=a[k]  ]
        std_list.append( std(b) )

    #
    std_arr = array(std_list)

    #
    s = argmin( std_arr )

    # The OUTSIDER is the data point that, when taken away, minimizes the standard deviation of the population.
    # In other words, the outsider is the point that adds the most diversity.

    mask = ones( a.shape, dtype=bool )
    mask[s] = False
    mask = mask.reshape( true_shape )

    # Return the outsider's location and a mask to help locate it within related data
    return s,mask


# Return the min and max limits of an 1D array
def lim(x,dilate=0):
    '''
    Return the min and max limits of an 1D array.

    INPUT
    ---
    x,              ndarray
    dilate=0,       fraction of max-min by which to expand or contract output

    RETURN
    ---
    array with [min(x),max(x)]

    '''

    # Import useful bit
    from numpy import array,amin,amax,ndarray,diff

    # ensure is array
    if not isinstance(x,ndarray): x = array(x)

    # Columate input.
    z = x.reshape((x.size,))

    #
    ans = array([min(z),max(z)]) + (0 if len(z)>1 else array([-1e-20,1e-20]))

    #
    if dilate != 0: ans += diff(ans)*dilate*array([-1,1])

    # Return min and max as list
    return ans

# Determine whether numpy array is uniformly spaced
def isunispaced(x,tol=1e-5):

    # import usefull fun
    from numpy import diff,amax

    # If t is not a numpy array, then let the people know.
    if not type(x).__name__=='ndarray':
        msg = '(!!) The first input must be a numpy array of 1 dimension.'

    # Return whether the input is uniformly spaced
    return amax(diff(x,2))<tol

# Calculate rfequency domain (~1/t Hz) given time series array
def getfreq( t, shift=False ):

    #
    from numpy.fft import fftfreq
    from numpy import diff,allclose,mean

    # If t is not a numpy array, then let the people know.
    if not type(t).__name__=='ndarray':
        msg = '(!!) The first input must be a numpy array of 1 dimension.'

    # If nonuniform time steps are found, then let the people know.
    if not isunispaced(t):
        msg = '(!!) The time input (t) must be uniformly spaced.'
        raise ValueError(msg)

    #
    if shift:
        f = fftshift( fftfreq( len(t), mean(diff(t)) ) )
    else:
        f = fftfreq( len(t), mean(diff(t)) )

    #
    return f

# Low level function for fixed frequency integration (FFI)
def ffintegrate(t,y,w0,n=1):

    # This function is based upon 1006.1632v1 Eq 27

    #
    from numpy import array,allclose,ones,pi
    from numpy.fft import fft,ifft,fftfreq,fftshift
    from numpy import where

    # If x is not a numpy array, then let the people know.
    if not type(y).__name__=='ndarray':
        msg = '(!!) The second input must be a numpy array of 1 dimension.'
        error(msg)

    # If nonuniform time steps are found, then let the people know.
    if not isunispaced(t):
        msg = '(!!) The time input (t) must be uniformly spaced.'
        raise ValueError(msg)

    # Define the lowest level main function which applies integration only once.
    def ffint(t_,y_,w0=None):

        # Note that the FFI method is applied in a DOUBLE SIDED way, under the assumpion tat w0 is posistive
        if w0<0: w0 = abs(w0);

        # Calculate the fft of the inuput data, x
        f = getfreq(t_) # NOTE that no fftshift is applied

        # Replace zero frequency values with very small number
        if (f==0).any :
            f[f==0] = 1e-9

        #
        w = f*2*pi

        # Find masks for positive an negative fixed frequency regions
        mask1 = where( (w>0) * (w<w0)  ) # Positive and less than w0
        mask2 = where( (w<0) * (w>-w0) ) # Negative and greater than -w0

        # Preparare fills for each region of value + and - w0
        fill1 =  w0 * ones( w[mask1].shape )
        fill2 = -w0 * ones( w[mask2].shape )

        # Apply fills to the frequency regions
        w[ mask1 ] = fill1; w[ mask2 ] = fill2

        # Take the FFT
        Y_ = fft(y_)

        # Calculate the frequency domain integrated vector
        Y_int = Y_ / (w*1j)

        # Inverse transorm, and make sure that the inverse is of the same nuerical type as what was input
        tol = 1e-8
        y_isreal = allclose(y_.imag,0,atol=tol)
        y_isimag = allclose(y_.real,0,atol=tol)
        if y_isreal:
            y_int = ifft( Y_int ).real
        elif y_isimag:
            y_int = ifft( Y_int ).imag
        else:
            y_int = ifft( Y_int )

        # Share knowledge with the people.
        return y_int


    #
    x = y
    for k in range(n):
        #
        x = ffint(t,x,w0)

    #
    return x


# Derivative function that preserves array length: [(d/dt)^n y(t)] is returned
def intrp_diff( t,        # domain values
                y,        # range values
                n = 1 ):  # degree of derivative

    #
    from numpy import diff,append
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    if 1 == n :
        #
        dt = t[1]-t[0]
        dy = diff(y)/dt
        dy_left  = append( dy, spline( t[:-1], dy )(t[-1]) )
        dy_right = append( spline( t[:-1], dy )(t[0]-dt), dy )
        dy_center = 0.5 * ( dy_left + dy_right )
        return dy_center
    elif n > 1:
        #
        dy = intrp_diff( t, y )
        return intrp_diff( t, dy, n-1 )
    elif n == 0 :
        #
        return y


# Find peaks adaptation from Matlab. Yet another example recursion's power!
def findpeaks( y, min_distance = None ):

    # Algorithm copied from Matlab's findLocalMaxima within findpeaks.m
    # lionel.london@ligo.org

    #
    from numpy import array,ones,append,arange,inf,isfinite,diff,sign,ndarray,hstack,where,abs
    import warnings

    #
    thisfun = inspect.stack()[0][3]

    if min_distance is None:

        #
        if not isinstance(y,ndarray):
            msg = red('Input must be numpy array')
            error(msg,thisfun)

        # bookend Y by NaN and make index vector
        yTemp = hstack( [ inf, y, inf ] )
        iTemp = arange( len(yTemp) )

        # keep only the first of any adjacent pairs of equal values (including NaN).
        yFinite = isfinite(yTemp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iNeq = where(  ( abs(yTemp[1:]-yTemp[:-1])>1e-12 )  *  ( yFinite[:-1]+yFinite[1:] )  )
        iTemp = iTemp[ iNeq ]

        # take the sign of the first sample derivative
        s = sign( diff(  yTemp[iTemp]  ) )

        # find local maxima
        iMax = where(diff(s)<0)

        # find all transitions from rising to falling or to NaN
        iAny = 1 + array( where( s[:-1]!=s[1:] ) )

        # index into the original index vector without the NaN bookend.
        iInflect = iTemp[iAny]-1
        iPk = iTemp[iMax]

        # NOTE that all inflection points are found, but note used here. The function may be updated in the future to make use of inflection points.

        # Package outputs
        locs    = iPk
        pks     = y[locs]

    else:

        #
        pks,locs = findpeaks(y)
        done = min( diff(locs) ) >= min_distance
        pks_ = pks
        c = 0
        while not done:

            #
            pks_,locs_ = findpeaks(pks_)
            print 'length is %i' % len(locs_)

            #
            if len( locs_ ) > 1 :
                #
                locs = locs[ locs_ ]
                pks = pks[ locs_ ]
                #
                done = min( diff(locs_) ) >= min_distance
            else:
                #
                done = True

            #
            c+=1
            print c

    #
    return pks,locs

# Find the roots of a descrete array.
def findroots( y ):

    from numpy import array,arange,allclose

    n = len(y)

    w =[]

    for k in range(n):
        #
        l = min(k+1,n-1)
        #
        if y[k]*y[l]<0 and abs(y[k]*y[l])>1e-12:
            #
            w.append(k)

        elif allclose(0,y[k],atol=1e-12) :
            #
            w.append(k)

    #
    root_mask = array( w )

    # #
    # _,root_mask = findpeaks( root_mask, min_distance=10 )

    #
    return root_mask

# Clone of MATLAB's find function: find all of the elements in a numpy array that satisfy a condition.
def find( bool_vec ):

    #
    from numpy import where

    #
    return where(bool_vec)[0]

# Low level function that takes in numpy 1d array, and index locations of start and end of wind, and then outputs the taper (a hanning taper). This function does not apply the taper to the data.
def maketaper(arr,state,window_type='hann',ramp=True):
    '''
    Low level function that takes in numpy 1d array, and index locations of start and end of wind, and then outputs the taper (a hanning taper). This function does not apply the taper to the data.

    For all window types allowed, see:
    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.get_window.html
    '''

    # Import useful things
    from numpy import ones,zeros
    from numpy import hanning as hann
    from scipy.signal import get_window

    # Parse taper state
    a = state[0]
    b = state[-1]

    #
    use_nr_window = window_type in ('nr')

    # Only proceed if a valid taper is given
    proceed = True
    true_width = abs(b-a)

    #
    if ramp:

        if window_type in ('nr'):
            #
            twice_ramp = nrwindow(2*true_width)
        elif window_type in ('exp'):
            #
            twice_ramp = expsin_window(2*true_width)
        else:
            #
            twice_ramp = get_window( window_type, 2*true_width )

        if b>a:
            true_ramp = twice_ramp[ :true_width ]
        elif b<=a:
            true_ramp = twice_ramp[ true_width: ]
        else:
            proceed = False
            print a,b
            alert('Whatght!@!')
    else:
        print 'ramp is false'
        if window_type in ('nr'):
            true_ramp = nrwindow(true_width)
        elif window_type in ('exp'):
            true_ramp = expsin_window(true_width)
        else:
            true_ramp = get_window( window_type,true_width )

    # Proceed (or not) with tapering
    taper = ones( len(arr) ) if ramp else zeros( len(arr) )
    if proceed:
        # Make the taper
        if b>a:
            taper[ :min(state) ] = 0*taper[ :min(state) ]
            taper[ min(state) : max(state) ] = true_ramp
        else:
            taper[ max(state): ] = 0*taper[ max(state): ]
            taper[ min(state) : max(state) ] = true_ramp

    #
    if len(taper) != len(arr):
        error('the taper length is inconsistent with input array')

    #
    return taper


# James Healy 6/27/2012
# modifications by spxll'16
# conversion to python by spxll'16
def diff5( time, ff ):

    #
    from numpy import var,diff

    # check that time and func are the same size
    if length(time) != length(ff) :
        error('time and function arrays are not the same size.')

    # check that dt is fixed:
    if var(diff(time))<1e-8 :
        dt = time[1] - time[0]
        tindmax = len(time)
    else:
        error('Time step is not uniform.')

    # first order at the boundaries:
    deriv[1]         = ( -3.0*ff[4] + 16.0*ff[3] -36.0*ff[2] + 48.0*ff[1] - 25.0*ff[0] )/(12.0*dt)
    deriv[2]         = ( ff[5] - 6*ff[4] +18*ff[3] - 10*ff[2] - 3*ff[1] )/(12.0*dt)
    deriv[-2] = (  3.0*ff[-1] + 10.0*ff[-2] - 18*ff[-3] + 6*ff[-4] -   ff[-5])/(12.0*dt)
    deriv[-1]   = ( 25.0*ff[-1] - 48*ff[-2] + 36.0*ff[-3] -16*ff[-4] + 3*ff[-5])/(12.0*dt)

    # second order at interior:
    deriv[3:-2] = ( -ff[5:] + 8*ff[4:-1] - 8*ff[2:-3] + ff[1:-4] ) / (12.0*dt)

    #
    return deriv


# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))


# High level function for spin weighted spherical harmonics
def sYlm(s,l,m,theta,phi,return_mesh=False):

    # Import useful things
    from numpy import array,vstack,ndarray,exp,double

    # Enforce that theta and phi are arrays
    phi   = array( phi   if isinstance(phi  ,(list,tuple)) else [double(phi  )]  ) if not isinstance(phi  ,ndarray) else phi
    theta = array( theta if isinstance(theta,(list,tuple)) else [double(theta)]  ) if not isinstance(theta,ndarray) else theta

    #
    theta_is_matrix = len(theta.shape)>1
    phi_is_matrix = len(phi.shape)>1
    if theta_is_matrix or phi_is_matrix :
        error('theta and phi inputs must not have dimension greater than 1')

    # Define function to encapsulate azimuthal dependence
    Am = lambda M,PHI: exp( 1j*M*PHI )

    # IF more than one phi value is given
    if len(phi)>1 :
        D = sDlm(s,l,m,theta)
        Y = vstack(  [ D * Am(m,ph) for ph in phi ]  )
    else: # ELSE if a single value is given
        Y = sDlm(s,l,m,theta) * Am(m,phi)

    #
    if not return_mesh:
        return Y
    else:
        from numpy import meshgrid
        THETA,PHI = meshgrid(theta,phi)
        return Y,THETA,PHI

# Use formula from wikipedia to calculate the harmonic
# See http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
# for more information.
def sDlm(s,l,m,theta):

    #
    from numpy import pi,ones,sin,tan,exp,array,double,sqrt,zeros,ones_like
    from scipy.misc import factorial,comb

    #
    if isinstance(theta,(float,int,double)): theta = [theta]
    theta = array(theta)

    #
    theta = array([ double(k) for k in theta ])

    # Ensure regular output (i.e. no nans)
    theta[theta==0.0] = 1e-9

    # Name anonymous functions for cleaner syntax
    f = lambda k: double(factorial(k))
    c = lambda x: double(comb(x[0],x[1]))
    cot = lambda x: 1.0/double(tan(x))

    # Pre-allocatetion array for calculation (see usage below)
    X = ones_like( theta )


    # Calcualte the "pre-sum" part of sYlm
    a = (-1.0)**(m)
    a = a * sqrt( f(l+m)*f(l-m)*(2.0*l+1) )
    a = a / sqrt( 4.0*pi*f(l+s)*f(l-s) )
    a = a * sin( theta/2.0 )**(2.0*l)
    A = a * X

    # Calcualte the "sum" part of sYlm
    B = zeros(theta.shape)
    for k in range(len(theta)):
        B[k] = 0
        for r in range(l-s+1):
            if (r+s-m <= l+s) and (r+s-m>=0) :
                a = c([l-s,r])*c([l+s,r+s-m])
                a = a * (-1)**(l-r-s)
                a = a * cot( theta[k]/2.0 )**(2*r+s-m)
                B[k] = B[k] + a

    # Calculate final output array
    D = A*B

    #
    if (sum(abs(D.imag)) <= 1e-7).all():
        D = D.real

    #
    return D



# Time shift array data, h, using a frequency diomain method
def tshift( t,      # time sries of data
            h,      # data that will be shifted
            t0,     # time by which to shift the data
            verbose=False,  # Toggle to let the people know
            method=None ):   # amount to shift data


    # Import usefuls
    from scipy.fftpack import fft, fftfreq, fftshift, ifft
    from numpy import diff,mean,exp,pi

    # Determine if the data is all real
    is_real = sum( h.imag ) == 0

    #
    if verbose: alert( 'The data are real valued.' )

    #
    if method is None:
        method = 'fft'
        if verbose: alert('Using the default time shifting method.')

    #
    if verbose: alert('The method is "%s"'%yellow(method))


    # Apply the time shift
    if method.lower() in ('fft'):

        # take fft of input
        H = fft(h)

        # get frequency domain of H in hertz (non-monotonic,
        # i.e. not the same as the "getfrequencyhz" function)
        dt = mean(diff(t))
        f = fftfreq( len(t), dt )

        # shift, and calculate ifft
        H_ = H * exp( -2*pi*1j*t0*f )

        #
        if is_real:
            h_ = ifft( H_ ).real
        else:
            h_ = ifft( H_ ) # ** here, errors in ifft process are ignored **

    elif method.lower() in ('td','index','ind'):

        # Use index shifting
        if verbose:
            alert('Note that this method assumes the data are equally spaced in time.')

        #
        di = int( t0/mean(diff(t)) )
        h_ = ishift(h, di)


    # Return the answer
    return h_

# Time shift array data, h, using a index shifting method
def ishift( h, di ):

    #
    from numpy import mod,arange

    #
    di = int( mod(di,len(h)) )

    #
    space = arange( len(h) )
    new_space = space - di
    ans = h[ new_space ]

    return ans


# Find the interpolated global max location of a data series
def intrp_max( y, domain=None, verbose=False, return_argmax=False, plot = False, pad = 3 ):

    #
    from scipy.interpolate import UnivariateSpline as spline
    from scipy.optimize import minimize
    from numpy import allclose,linspace,argmax,arange,hstack,diff,argmax,argmin,mod,array,mean,std
    #
    PLOT = plot
    if PLOT: from matplotlib.pyplot import plot,show,xlim,ylim,xlabel,ylabel,title,figure

    #
    t = arange(len(y)) if domain is None else domain

    # Determine if y is flat
    c = (y - mean(y))/std(y)
    # the centered version of y, c, is determined to be flat if the largest difference is small
    y_is_flat = allclose( y, y[::-1], rtol=1e-3 )

    '''
    If the input vector is flat, simply take its numerical max.
    Otherwise, use the intrp_max algorithm.
    '''

    # IF THE INPUT IS NOT FLAT
    if not y_is_flat:

        #
        if PLOT:
            #
            from positive import rgb
            ts = linspace( min(t), max(t), 2e2 )
            ys = spline(t,y,s=0,k=4)(ts)
            #
            clr= rgb(3)
            #
            fig1 = figure()
            plot( t,y, 'ok' )
            plot( ts,ys, color=clr[0], linestyle='--' )
            #
            dy = diff( lim(y) )*0.1
            ylim( array([-1,1])*dy + lim(y) )
            xlim( lim(t) )
            #
            xlabel('domain')
            ylabel('range')

        #
        k_max = argmax( y )
        t_max = t[k_max]
        y_max = y[k_max]

        #
        if PLOT:
            plot( t_max, y_max, 'o', mfc='none', mec='k', ms=16 )

        # Determine points to right and left of numerical max

        # This many points to right and left of numerical max will be taken
        pad = pad

        #
        a = k_max - pad
        b = k_max + pad

        #
        left = arange( a, k_max )
        right = arange( k_max, b+1 )
        #
        raw_space = hstack( [left,right] )
        #
        space = mod( raw_space, len(y)-1 )
        #
        raw_kspace = range( len(space) )

        #
        if PLOT:
            plot( t[ space[0] ], y[ space[0] ], '>', mfc='none', mec='g', ms = 19 )
            plot( t[ space[-1] ], y[ space[-1] ], '<', mfc='none', mec='g', ms = 19 )

        #
        raw_suby = array( [ y[k] for k in space ] ) # y[space]

        # -------------------------------------------- #
        # Enforce adjacent symmetry about numerical max
        # -------------------------------------------- #
        left_k  =  1 + argmin( abs(raw_suby[0] - raw_suby[1:]) )
        right_k =  argmin( abs(raw_suby[-1] - raw_suby[:-1]) )
        center_k = argmax(raw_suby)
        # print left_k, right_k, center_k

        #
        if PLOT:
            fig2 = figure()
            plot( raw_kspace, raw_suby, 'ok' )

        # IF the clostest point is on the other side of the peak AND there is an assymetry detected
        # THEN make more symmetric by removing points from left or right
        mask = range( len(raw_suby) )
        if (right_k < center_k): # and (left_k != len(raw_suby)-1) :
            mask = range( right_k, len(raw_suby) )
        elif (left_k > center_k): # and (right_k != 0) :
            mask = range( 0, left_k+1 )

        # Apply the mask
        kspace = array([ raw_kspace[v] for v in mask ])
        suby = array([ raw_suby[v] for v in mask ])

        # -------------------------------------------- #
        # Interpolate local space to estimate max
        # -------------------------------------------- #
        try:
            intrp_suby = spline( kspace, suby, k=4, s=0 )
        except:
            from matplotlib import pyplot as pp
            pp.figure()
            pp.plot( kspace, suby, '-o' )
            pp.title( diff(lim(c)) )
            pp.show()
            raise
        # Location of the max is determined analytically, given the local spline model
        kspace_maxes = intrp_suby.derivative().roots()
        try:
            kspace_max = kspace_maxes[ argmax( intrp_suby(kspace_maxes) ) ]
        except:
            warning('somthing\'s wrong folks ....')
            print kspace_maxes
            from matplotlib import pyplot as pp
            pp.figure()
            from numpy import isnan
            print sum(isnan(y))
            pp.plot( kspace, suby, '-o' )
            pp.title( diff(lim(c)) )
            pp.show()
            raise

        #
        if PLOT:
            #
            plot( kspace_max, intrp_suby(kspace_max), '*', ms=20, mec=clr[-1], mfc=clr[-1] )
            kspace_sm = linspace(min(kspace),max(kspace))
            plot( kspace_sm, intrp_suby(kspace_sm), color=clr[0], linestyle='--' )
            plot( kspace, suby, 'ow', ms=4 )
            #
            dy = diff( lim(suby) )*0.2
            ylim( array([-1,1])*dy + lim(raw_suby) )
            xlim( lim(raw_kspace) )
            xlabel('mapped index domain')
            ylabel('wrapped range')

        max_val = intrp_suby(kspace_max)
        index_arg_max = spline( raw_kspace, raw_space, k=1, s=0 )(kspace_max)
        arg_max = spline( range(len(t)), t )( index_arg_max )

        #
        if verbose:
            print '\n>> Results of intrp_max:\n%s' % ( '--'*20 )
            print '    intrp_max \t = \t %f' % max_val
            print 'intrp_arg_max \t = \t %f\n' % arg_max

        #
        if PLOT:
            figure( fig1.number )
            plot( arg_max, max_val, '*', ms=20, mec=clr[-1], mfc=clr[-1]  )

    else: # IF THE INPUT IS FLAT

        #
        if verbose: warning('Input is determined to be flat. A simple numerical mex will be used.')
        arg_max_dex = argmax( y )
        arg_max = t[ arg_max_dex ]
        max_val = y[ arg_max_dex ]

    #
    if return_argmax:
        ans = (max_val,float(arg_max))
    else:
        ans = max_val

    #
    return ans



# Find the interpolated global max location of a data series
# NOTE that this version does not localize around numerical max of input; this is a bad thing
def intrp_argmax( y,
                  domain=None,
                  plot=False,
                  verbose=False ):

    #
    max_val,arg_max = intrp_max( y,domain=domain,verbose=verbose,return_argmax=True,plot=plot )

    #
    ans = arg_max
    return ans


# Find the interpolated global max location of a data series
# NOTE that this version does not localize around numerical max of input; this is a bad thing
def intrp_max_depreciated( y,
               domain=None,
               verbose=False, return_argmax=False ):

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from scipy.optimize import minimize
    from numpy import linspace,argmax

    #
    x = range(len(y)) if domain is None else domain

    #
    yspline = spline( x, y )

    # Find the approximate max location in index
    k = argmax( y )

    # NOTE that we use minimize with bounds as it was found to have better behavior than fmin with no bounding
    x0 = x[k]
    f = lambda X: -yspline(X)
    dx = 0.1*x0
    q = minimize(f,x0,bounds=[(max(x0-dx,min(x)),min(x0+dx,max(x)))])
    xmax = q.x[0]

    #
    if yspline(xmax)<max(y):
        # warning('yspline(xmax)<max(y): spline optimization failed; now taking numerical max of input series')
        maxval = max(y)
    else:
        maxval = yspline(xmax)

    #
    if return_argmax:
        ans = (maxval,xmax)
    else:
        ans = maxval

    # #
    # from matplotlib.pyplot import plot,xlim,ylim,title,show,gca
    # plot(x,y,'bo',mfc='none')
    # x_ = linspace(min(x),max(x),2e2)
    # plot( x_,yspline(x_),'k',alpha=0.5 )
    # plot( xmax, yspline(xmax), 'or', mfc='none' )
    # show()

    #
    return ans


#
def expsin_window( N ):
    #
    from numpy import hstack,array,linspace,exp,log,pi,sin
    #
    t =  log(1e16) * (1+ sin( linspace( pi/2, -pi/2, int(N)/2 ) ))*0.5
    A = exp( -t )
    A -= min(A)
    A /= max(A)
    #
    ans = hstack( [A, A[range(len(A)-1,0,-1)] ] ) if 2*len(A)==N else hstack( [A, A[range(len(A)-1,1,-1)] ] )
    #
    return ans

#
def spline_diff(t,y,k=3):
    '''
    Wrapper for InterpolatedUnivariateSpline derivative function
    '''

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    return spline(t,y,k=k).derivative()(t)

#
def spline_antidiff(t,y,k=3):
    '''
    Wrapper for InterpolatedUnivariateSpline antiderivative function
    '''

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    return spline(t,y,k=k).antiderivative()(t)

# Sinc Intepolation
# from -- https://gist.github.com/endolith/1297227
def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    if len(x) != len(s):
        raise Exception, 'x and s must be the same length'

    # Find the period
    T = s[1] - s[0]

    sincM = tile(u, (len(s), 1)) - tile(s[:, newaxis], (1, len(u)))
    y = dot(x, sinc(sincM/T))
    return y

#
def nrwindow( N ):
    '''
    The point here is to define a taper to be used for the low frequency part of waveforms from NR data samples.
    '''
    #
    from scipy.interpolate import CubicSpline as spline
    from numpy import hstack,array,linspace,pi,sin
    #
    numerical_data = array([ [0.000235599, 0.164826], [0.000471197, 0.140627],\
                             [0.000706796, 0.139527], [0.000942394, 0.154408],\
                             [0.00117799, 0.144668], [0.00141359, 0.0820655],\
                             [0.00164919, 0.107215], [0.00188479, 0.326988],\
                             [0.00212039, 0.612349], [0.00235599, 0.928147],\
                             [0.00259158, 1.25567], [0.00282718, 1.61068],\
                             [0.00306278, 2.05771], [0.00329838, 2.69093],\
                             [0.00353398, 3.58197], [0.00376958, 4.74465],\
                             [0.00400517, 6.14815], [0.00424077, 7.76167],\
                             [0.00447637, 9.66762], [0.00471197, 12.1948],\
                             [0.00494757, 16.2907], [0.00518317, 23.0923],\
                             [0.00541877, 33.2385], [0.00565436, 49.4065],\
                             [0.00588996, 73.3563], [0.00612556, 101.84],\
                             [0.00636116, 121.165], ])
    #
    a = numerical_data[:,1]/max(numerical_data[:,1])
    n = len(a)
    f = linspace(0,1,n)
    #
    A = spline(f,a)( linspace(0,1,int(N)/2) )
    #
    ans = hstack( [A, A[range(len(A)-1,0,-1)] ] ) if 2*len(A)==N else hstack( [A, A[range(len(A)-1,1,-1)] ] )
    #
    return ans

'''
Given data set xx yy constrcut an interpolating polynomial that passes through all points (xx,yy). The output is a function object.
http://stackoverflow.com/questions/14823891/newton-s-interpolating-polynomial-python
'''
def newtonpoly(xx,yy):

    import numpy as np
    #import matplotlib.pyplot as plt

    def coef(x, y):
        '''x : array of data points
           y : array of f(x)  '''
        x.astype(float)
        y.astype(float)
        n = len(x)
        a = []
        for i in range(n):
            a.append(y[i])

        for j in range(1, n):

            for i in range(n-1, j-1, -1):
                a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])

        return np.array(a) # return an array of coefficient

    def Eval(a, x, r):

        ''' a : array returned by function coef()
            x : array of data points
            r : the node to interpolate at  '''

        x.astype(float)
        n = len( a ) - 1
        temp = a[n]
        for i in range( n - 1, -1, -1 ):
            temp = temp * ( r - x[i] ) + a[i]
        return temp # return the y_value interpolation

    #
    A = coef(xx,yy)
    return lambda r: Eval( A, xx, r )

#-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#

"""
An OrderedSet is a custom MutableSet that remembers its order, so that every
entry has an index that can be looked up.

Based on a recipe originally posted to ActiveState Recipes by Raymond Hettiger,
and released under the MIT license.

Rob Speer's changes are as follows:

    - changed the content from a doubly-linked list to a regular Python list.
      Seriously, who wants O(1) deletes but O(N) lookups by index?
    - add() returns the index of the added item
    - index() just returns the index of an item
    - added a __getstate__ and __setstate__ so it can be pickled
    - added __getitem__
"""
import collections

SLICE_ALL = slice(None)
__version__ = '1.3'


def is_iterable(obj):
    """
    Are we being asked to look up a list of things, instead of a single thing?
    We check for the `__iter__` attribute so that this can cover types that
    don't have to be known by this module, such as NumPy arrays.

    Strings, however, should be considered as atomic values to look up, not
    iterables.

    We don't need to check for the Python 2 `unicode` type, because it doesn't
    have an `__iter__` attribute anyway.
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


class OrderedSet(collections.MutableSet):
    """
    An OrderedSet is a custom MutableSet that remembers its order, so that
    every entry has an index that can be looked up.
    """
    def __init__(self, iterable=None):
        self.items = []
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        """
        Get the item at a given index.

        If `index` is a slice, you will get back that slice of items. If it's
        the slice [:], exactly the same object is returned. (If you want an
        independent copy of an OrderedSet, use `OrderedSet.copy()`.)

        If `index` is an iterable, you'll get the OrderedSet of items
        corresponding to those indices. This is similar to NumPy's
        "fancy indexing".
        """
        if index == SLICE_ALL:
            return self
        elif hasattr(index, '__index__') or isinstance(index, slice):
            result = self.items[index]
            if isinstance(result, list):
                return OrderedSet(result)
            else:
                return result
        elif is_iterable(index):
            return OrderedSet([self.items[i] for i in index])
        else:
            raise TypeError("Don't know how to index an OrderedSet by %r" %
                    index)

    def copy(self):
        return OrderedSet(self)

    def __getstate__(self):
        if len(self) == 0:
            # The state can't be an empty list.
            # We need to return a truthy value, or else __setstate__ won't be run.
            #
            # This could have been done more gracefully by always putting the state
            # in a tuple, but this way is backwards- and forwards- compatible with
            # previous versions of OrderedSet.
            return (None,)
        else:
            return list(self)

    def __setstate__(self, state):
        if state == (None,):
            self.__init__([])
        else:
            self.__init__(state)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        """
        Add `key` as an item to this OrderedSet, then return its index.

        If `key` is already in the OrderedSet, return the index it already
        had.
        """
        if key not in self.map:
            self.map[key] = len(self.items)
            self.items.append(key)
        return self.map[key]
    append = add

    def index(self, key):
        """
        Get the index of a given entry, raising an IndexError if it's not
        present.

        `key` can be an iterable of entries that is not a string, in which case
        this returns a list of indices.
        """
        if is_iterable(key):
            return [self.index(subkey) for subkey in key]
        return self.map[key]

    def discard(self, key):
        raise NotImplementedError(
            "Cannot remove items from an existing OrderedSet"
        )

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and self.items == other.items
        try:
            other_as_set = set(other)
        except TypeError:
            # If `other` can't be converted into a set, it's not equal.
            return False
        else:
            return set(self) == other_as_set



"""
An OrderedSet is a custom MutableSet that remembers its order, so that every
entry has an index that can be looked up.

Based on a recipe originally posted to ActiveState Recipes by Raymond Hettiger,
and released under the MIT license.

Rob Speer's changes are as follows:

    - changed the content from a doubly-linked list to a regular Python list.
      Seriously, who wants O(1) deletes but O(N) lookups by index?
    - add() returns the index of the added item
    - index() just returns the index of an item
    - added a __getstate__ and __setstate__ so it can be pickled
    - added __getitem__
"""
from collections import MutableSet


def is_iterable(obj):
    """
    Are we being asked to look up a list of things, instead of a single thing?
    We check for the `__iter__` attribute so that this can cover types that
    don't have to be known by this module, such as NumPy arrays.

    Strings, however, should be considered as atomic values to look up, not
    iterables.

    We don't need to check for the Python 2 `unicode` type, because it doesn't
    have an `__iter__` attribute anyway.
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str)

# Class for ordered sets
class OrderedSet(MutableSet):
    __version__ = '1.3'
    """
    An OrderedSet is a custom MutableSet that remembers its order, so that
    every entry has an index that can be looked up.
    """
    def __init__(self, iterable=None):
        self.items = []
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        """
        Get the item at a given index.

        If `index` is a slice, you will get back that slice of items. If it's
        the slice [:], exactly the same object is returned. (If you want an
        independent copy of an OrderedSet, use `OrderedSet.copy()`.)

        If `index` is an iterable, you'll get the OrderedSet of items
        corresponding to those indices. This is similar to NumPy's
        "fancy indexing".
        """
        if index == slice(None):
            return self
        elif hasattr(index, '__index__') or isinstance(index, slice):
            result = self.items[index]
            if isinstance(result, list):
                return OrderedSet(result)
            else:
                return result
        elif is_iterable(index):
            return OrderedSet([self.items[i] for i in index])
        else:
            raise TypeError("Don't know how to index an OrderedSet by %r" %
                    index)

    def copy(self):
        return OrderedSet(self)

    def __getstate__(self):
        if len(self) == 0:
            # The state can't be an empty list.
            # We need to return a truthy value, or else __setstate__ won't be run.
            #
            # This could have been done more gracefully by always putting the state
            # in a tuple, but this way is backwards- and forwards- compatible with
            # previous versions of OrderedSet.
            return (None,)
        else:
            return list(self)

    def __setstate__(self, state):
        if state == (None,):
            self.__init__([])
        else:
            self.__init__(state)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        """
        Add `key` as an item to this OrderedSet, then return its index.

        If `key` is already in the OrderedSet, return the index it already
        had.
        """
        if key not in self.map:
            self.map[key] = len(self.items)
            self.items.append(key)
        return self.map[key]
    append = add

    def index(self, key):
        """
        Get the index of a given entry, raising an IndexError if it's not
        present.

        `key` can be an iterable of entries that is not a string, in which case
        this returns a list of indices.
        """
        if is_iterable(key):
            return [self.index(subkey) for subkey in key]
        return self.map[key]

    def discard(self, key):
        raise NotImplementedError(
            "Cannot remove items from an existing OrderedSet"
        )

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and self.items == other.items
        try:
            other_as_set = set(other)
        except TypeError:
            # If `other` can't be converted into a set, it's not equal.
            return False
        else:
            return set(self) == other_as_set






'''
Maths functions for implementation of matrix method for corotating frame.
Reference: https://arxiv.org/pdf/1110.2965.pdf
'''
