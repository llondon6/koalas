#
from positive import *

# Reference factorial from scipy
try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial


'''
Useful Method for estimating QNM locations in leaver solution space: Estimate the
local minima of a 2D array
'''
def localmins(arr,edge_ignore=False):

    import numpy as np
    import scipy.ndimage.filters as filters
    import scipy.ndimage.morphology as morphology

    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    # detected_minima = local_min - eroded_background
    __localmin__ =  list( np.where(detected_minima) )

    # Option: Ignore mins on domain boundaries
    if edge_ignore:
        isonedge0 =  lambda x: (x==0) or (x==(len(arr[:,0])-1))
        isonedge1 =  lambda x: (x==0) or (x==(len(arr[0,:])-1))
        mask = np.ones( __localmin__[0].shape, dtype=bool )
        for k in range(len(__localmin__[0])):
            mask[k] = not ( isonedge0( __localmin__[0][k] ) or isonedge1( __localmin__[1][k] ) )
        __localmin__[0] = __localmin__[0][mask]
        __localmin__[1] = __localmin__[1][mask]

    #
    return __localmin__


# Lentz's continued fration solver
def lentz( aa, bb, tol=None, tiny=1e-30, mpm=False ):
    '''
    Lentz's method for accurate continued fraction calculation of a function
    f:

    f = b(0) + a(1)/( b(1) + a(2)/( b(2) + a(3)/( b(3) + ... ) ) )

    (Equivalent notation)

    f = b(0) + [a(1)/b(1)+][a(2)/b(2)+][a(3)/b(3)+]...[a(n)/b(n)+]...

    References:

    http://www.mpi-hd.mpg.de/astrophysik/HEA/internal/Numerical_Recipes/f5-2.pdf
    http://epubs.siam.org/doi/pdf/10.1137/1.9780898717822.ch6

    ~ llondon6'12
    [CONVERTED TO PYTHON FROM MATLAB by llondon2'14]
    '''

    #
    if tol == None:
        tol = 1e-10

    #
    f = bb(0)
    if 0==f: f = tiny

    if mpm:
        from mpmath import mpc
        C,D = mpc(f),mpc(0)
    else:
        from numpy import complex256
        C,D = complex256(f),complex256(0)

    done,state = False,False
    j,jmax = 0,2e3
    while not done:

        #
        j = 1+j

        #
        D = bb(j) + aa(j)*D
        if 0==D: D = tiny
        #
        C = bb(j) + aa(j)/C
        if 0==C: C = tiny
        #
        D = 1.0/D
        DELTA = C*D
        #
        f = f*DELTA
        #
        err = abs( DELTA - 1.0 )
        done = err<tol
        if j>=jmax:
            warning('Maximum number of iterations reached before error criteria passed. (err=%d)\n'%err)
            state = True
            done = state

    return (f,state)


# Smooth 1D data
class smooth:
    '''
    Smooth 1D data. Initially based on https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    '''

    # Class constructor
    def __init__(this,y,width=None,method=None,auto_method=None,polynomial_order=2):
        # Import useful things
        from numpy import ones,convolve,mod,hstack,arange,cumsum,mod,array
        # Handle method input; set default
        method = 'savgol' if method is None else method.lower()
        # # Handle n input; default is None which causes method to be auto
        # method = 'auto' if width is None else method
        # Store relevant inputs to this object
        this.scalar_range = array(y)
        this.width = width
        this.method = method

        # Handle different methods
        if method in ('average','avg','mean'):
            # Use Rolling Average (non convulative)
            y_smooth = this.__rolling_average__(width)
        elif method in ('savgol'):
            # Automatically determine best smoothing length to use with average
            y_smooth = this.__savgol__(width=width,polynomial_order=polynomial_order)
        elif method in ('auto','optimal'):
            # Automatically determine best smoothing length to use with average
            y_smooth = this.__auto_smooth__(method=auto_method)
        else:
            error('unknown smoothing method requested: %s'%red(method))
        #
        this.answer = y_smooth

    # Smooth using savgol filter from scipy
    def __savgol__(this,width=None,polynomial_order=2):

        # Import usefuls
        from scipy.signal import savgol_filter as savgol
        from numpy import mod

        # Handle inputs
        if width is None: width = max( ceil( len(this.scalar_rang)/10 ), polynomial_order+1 )
        if not isinstance(width,int):
            error('width muist be int')
        if width<(polynomial_order+1):
            width += 2
        if not mod(width,2):
            width += 1

        #
        # print '>> ',width,polynomial_order
        ans = savgol( this.scalar_range, width, polynomial_order )
        return ans

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
    def __auto_smooth__(this,method=None):
        '''Automatically determine best smoothing length to use with average'''
        # Import useful things
        from numpy import ones,convolve,mod,hstack,arange,cumsum,mod,array,mean
        from numpy import poly1d,polyfit,std,argmin

        #
        if method is None: method='savgol'

        #
        err,smo = [],[]
        width_range = array(list(range(5,min(50,int(len(this.scalar_range)/2)))))
        # print lim(width_range)

        if method=='savgol':
            mask = mod(width_range,2).astype(bool)
            width_range = width_range[ mask ]

        #
        for j,k in enumerate(width_range):
            smo.append( smooth(this.scalar_range,int(k),method=method).answer )
            dif = this.scalar_range - smo[-1]
            # err.append( -mean( dif ) if method=='savgol' else std(dif)/std(this.scalar_range) )
            err.append( -mean( dif ) )
        #
        modeled_err = poly1d( polyfit(width_range,err,2) )(width_range)
        k = argmin( modeled_err )
        best_width = int( width_range[k] if k>0 else 3 )
        # print 'best width = ',width_range[k]
        #
        y_smooth = smooth(this.scalar_range,best_width,method=method).answer
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
    chart = sorted(  list(range(len(b)))  ,key=lambda c: b[c])

    # Apply the sort
    c = b[ chart ]

    # Unwrap the sorted
    d = unwrap(c)
    d -= 2*pi*( 1 + int(abs(amax(d))) )
    while amax(d)<0:
        d += 2*pi

    # Re-order
    rechart = sorted(  list(range(len(d)))  ,key=lambda r: chart[r])

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

    '''
    Given fing the indeces and values of the input vector's local maxima.

    INTPUT
    --
    y                       numpy 1D array of reals
    min_distance = None     minimum allowed distance between consecutive peaks

    OUTPUT
    --
    pks                     peak values
    locs                    indeces of peaks


    Algorithm copied from Matlab's findLocalMaxima within findpeaks.m
    lionel.london@ligo.org
    '''

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
            print('length is %i' % len(locs_))

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
            print(c)

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

    # Vlaidate inputs
    for k in state:
        if k+1 > len(arr):
            error('state incompatible with array dimensions: the array shape is %s, but the state is %s'%(yellow(str(arr.shape)),yellow(str(state))) )

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
            print(a,b)
            alert('Whatght!@!')
    else:
        print('ramp is false')
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
            # print state, state[1]-state[0], taper.shape, true_ramp.shape, taper[ min(state) : max(state) ].shape
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
    from numpy import array,vstack,ndarray,exp,double,zeros_like

    # Enforce that theta and phi are arrays
    phi   = array( phi   if isinstance(phi  ,(list,tuple)) else [double(phi  )]  ) if not isinstance(phi  ,ndarray) else phi
    theta = array( theta if isinstance(theta,(list,tuple)) else [double(theta)]  ) if not isinstance(theta,ndarray) else theta

    #
    (s,l,m) = [ int(x) for x in (s,l,m) ]

    #
    isvalid = (abs(l)>=abs(s)) and (abs(m)<=l)

    #
    if isvalid:

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

    else:

        Y = zeros_like(theta)

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
    from numpy import pi,ones,exp,array,double,zeros_like,ones_like,sum
    from scipy.misc import factorial,comb
    from scipy import sqrt,tan,sin

    #
    if isinstance(theta,(float,int,double)): theta = [theta]
    theta = array(theta)

    #
    # theta = array([ double(k) for k in theta ])

    # Ensure regular output (i.e. no nans)
    theta[theta==0.0] = 1e-9

    # Name anonymous functions for cleaner syntax
    f = lambda k: double(factorial(k))
    c = lambda x: double(comb(x[0],x[1]))
    # cot = lambda x: 1.0/double(tan(x))
    cot = lambda x: 1.0/tan(x)

    # Pre-allocatetion array for calculation (see usage below)
    X = ones_like( theta )


    # Calcualte the "pre-sum" part of sYlm
    a = (-1.0)**(m)
    a = a * sqrt( f(l+m)*f(l-m)*(2.0*l+1) )
    a = a / sqrt( 4.0*pi*f(l+s)*f(l-s) )
    a = a * sin( theta/2.0 )**(2.0*l)
    A = a * X

    # Calcualte the "sum" part of sYlm
    B = zeros_like(theta)
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

    elif method.lower() in ('td','index','ind','roll'):

        # Use index shifting
        if verbose:
            alert('Note that this method assumes the data are equally spaced in time.')

        #
        from numpy import roll
        di = int( t0/mean(diff(t)) )
        h_ = roll(h, di)

    else:

        error('unhandled method for time shifting')


    # Return the answer
    return h_

# Time shift array data, h, using a index shifting method
def ishift( h, di ):
    #
    from numpy import roll
    return roll(h,di)


# Find the interpolated global max location of a data series
def intrp_max( y, domain=None, verbose=False, return_argmax=False, plot = False, pad = 3, ref_index=None ):

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
    y_is_flat = allclose( y, y[::-1], rtol=1e-3 ) and (std(diff(y)/diff(lim(y))))<1e-3

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
        if ref_index: k_max = ref_index
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
        raw_kspace = list(range( len(space)))

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
        mask = list(range( len(raw_suby)))
        if (right_k < center_k): # and (left_k != len(raw_suby)-1) :
            mask = list(range( right_k, len(raw_suby)))
        elif (left_k > center_k): # and (right_k != 0) :
            mask = list(range( 0, left_k+1))

        # Apply the mask
        kspace = array([ raw_kspace[v] for v in mask ])
        suby = array([ raw_suby[v] for v in mask ])

        # -------------------------------------------- #
        # Interpolate local space to estimate max
        # -------------------------------------------- #
        try:
            intrp_suby = spline( kspace, suby, k=4, s=0 )
        except:
            warning('Interpolative max failed. Using index.')
            #
            arg_max = argmax(y)
            max_val = y[arg_max]
            if return_argmax:
                ans = (max_val,float(arg_max))
            else:
                ans = max_val
            return ans

        # Location of the max is determined analytically, given the local spline model
        kspace_maxes = intrp_suby.derivative().roots()
        try:
            kspace_max = kspace_maxes[ argmax( intrp_suby(kspace_maxes) ) ]
        except:
            warning('somthing\'s wrong folks ....')
            print(kspace_maxes)
            from matplotlib import pyplot as pp
            pp.figure()
            from numpy import isnan
            print(sum(isnan(y)))
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
        arg_max = spline( list(range(len(t))), t )( index_arg_max )

        #
        if verbose:
            print('\n>> Results of intrp_max:\n%s' % ( '--'*20 ))
            print('    intrp_max \t = \t %f' % max_val)
            print('intrp_arg_max \t = \t %f\n' % arg_max)

        #
        if PLOT:
            figure( fig1.number )
            plot( arg_max, max_val, '*', ms=20, mec=clr[-1], mfc=clr[-1]  )

    else: # IF THE INPUT IS FLAT

        #
        if verbose: warning('Input is determined to be flat. A simple numerical mex will be used.')
        arg_max_dex = argmax( y )
        if ref_index: arg_max_dex = ref_index
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
                  ref_index = None,
                  verbose=False ):

    #
    max_val,arg_max = intrp_max( y,domain=domain,verbose=verbose,return_argmax=True,plot=plot,ref_index=ref_index )

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
    x = list(range(len(y))) if domain is None else domain

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
    ans = hstack( [A, A[list(range(len(A)-1,0,-1))] ] ) if 2*len(A)==N else hstack( [A, A[list(range(len(A)-1,1,-1))] ] )
    #
    return ans

#
def spline_diff(t,y,k=3,n=1):
    '''
    Wrapper for InterpolatedUnivariateSpline derivative function
    '''

    #
    from numpy import sum
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Calculate the desired number of derivatives
    ans = spline(t,y.real,k=k).derivative(n=n)(t) \
          + ( 1j*spline(t,y.imag,k=k).derivative(n=n)(t) if (sum(abs(y.imag))!=0) else 0 )

    return ans

#
def spline_antidiff(t,y,k=3,n=1):
    '''
    Wrapper for InterpolatedUnivariateSpline antiderivative function
    '''

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Calculate the desired number of integrals
    ans = spline(t,y.real,k=k).antiderivative(n=n)(t) + ( 1j*spline(t,y.imag,k=k).antiderivative(n=n)(t) if isinstance(y[0],complex) else 0 )

    # Return the answer
    return ans

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
        raise Exception('x and s must be the same length')

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
    ans = hstack( [A, A[list(range(len(A)-1,0,-1))] ] ) if 2*len(A)==N else hstack( [A, A[list(range(len(A)-1,1,-1))] ] )
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

# """
# An OrderedSet is a custom MutableSet that remembers its order, so that every
# entry has an index that can be looked up.
#
# Based on a recipe originally posted to ActiveState Recipes by Raymond Hettiger,
# and released under the MIT license.
#
# Rob Speer's changes are as follows:
#
#     - changed the content from a doubly-linked list to a regular Python list.
#       Seriously, who wants O(1) deletes but O(N) lookups by index?
#     - add() returns the index of the added item
#     - index() just returns the index of an item
#     - added a __getstate__ and __setstate__ so it can be pickled
#     - added __getitem__
# """
# import collections
#
# SLICE_ALL = slice(None)
# __version__ = '1.3'
#
#
# def is_iterable(obj):
#     """
#     Are we being asked to look up a list of things, instead of a single thing?
#     We check for the `__iter__` attribute so that this can cover types that
#     don't have to be known by this module, such as NumPy arrays.
#
#     Strings, however, should be considered as atomic values to look up, not
#     iterables.
#
#     We don't need to check for the Python 2 `unicode` type, because it doesn't
#     have an `__iter__` attribute anyway.
#     """
#     return hasattr(obj, '__iter__') and not isinstance(obj, str)
#
#
# class OrderedSet(collections.MutableSet):
#     """
#     An OrderedSet is a custom MutableSet that remembers its order, so that
#     every entry has an index that can be looked up.
#     """
#     def __init__(self, iterable=None):
#         self.items = []
#         self.map = {}
#         if iterable is not None:
#             self |= iterable
#
#     def __len__(self):
#         return len(self.items)
#
#     def __getitem__(self, index):
#         """
#         Get the item at a given index.
#
#         If `index` is a slice, you will get back that slice of items. If it's
#         the slice [:], exactly the same object is returned. (If you want an
#         independent copy of an OrderedSet, use `OrderedSet.copy()`.)
#
#         If `index` is an iterable, you'll get the OrderedSet of items
#         corresponding to those indices. This is similar to NumPy's
#         "fancy indexing".
#         """
#         if index == SLICE_ALL:
#             return self
#         elif hasattr(index, '__index__') or isinstance(index, slice):
#             result = self.items[index]
#             if isinstance(result, list):
#                 return OrderedSet(result)
#             else:
#                 return result
#         elif is_iterable(index):
#             return OrderedSet([self.items[i] for i in index])
#         else:
#             raise TypeError("Don't know how to index an OrderedSet by %r" %
#                     index)
#
#     def copy(self):
#         return OrderedSet(self)
#
#     def __getstate__(self):
#         if len(self) == 0:
#             # The state can't be an empty list.
#             # We need to return a truthy value, or else __setstate__ won't be run.
#             #
#             # This could have been done more gracefully by always putting the state
#             # in a tuple, but this way is backwards- and forwards- compatible with
#             # previous versions of OrderedSet.
#             return (None,)
#         else:
#             return list(self)
#
#     def __setstate__(self, state):
#         if state == (None,):
#             self.__init__([])
#         else:
#             self.__init__(state)
#
#     def __contains__(self, key):
#         return key in self.map
#
#     def add(self, key):
#         """
#         Add `key` as an item to this OrderedSet, then return its index.
#
#         If `key` is already in the OrderedSet, return the index it already
#         had.
#         """
#         if key not in self.map:
#             self.map[key] = len(self.items)
#             self.items.append(key)
#         return self.map[key]
#     append = add
#
#     def index(self, key):
#         """
#         Get the index of a given entry, raising an IndexError if it's not
#         present.
#
#         `key` can be an iterable of entries that is not a string, in which case
#         this returns a list of indices.
#         """
#         if is_iterable(key):
#             return [self.index(subkey) for subkey in key]
#         return self.map[key]
#
#     def discard(self, key):
#         raise NotImplementedError(
#             "Cannot remove items from an existing OrderedSet"
#         )
#
#     def __iter__(self):
#         return iter(self.items)
#
#     def __reversed__(self):
#         return reversed(self.items)
#
#     def __repr__(self):
#         if not self:
#             return '%s()' % (self.__class__.__name__,)
#         return '%s(%r)' % (self.__class__.__name__, list(self))
#
#     def __eq__(self, other):
#         if isinstance(other, OrderedSet):
#             return len(self) == len(other) and self.items == other.items
#         try:
#             other_as_set = set(other)
#         except TypeError:
#             # If `other` can't be converted into a set, it's not equal.
#             return False
#         else:
#             return set(self) == other_as_set
#
#

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





#-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#

# Return data with common sample rates and lengths
def format_align( domain_A,range_A,         # Domain and range of first 1d dataset
                  domain_B,range_B,         # Domain and range of second 1d dataset
                  center_domains=False,     # Toggle for setting domains to 0 at start
                  verbose=False):

    '''
    Determine time spacing of each array, and choose the larger spacing as the common one
    '''

    # Imoprt usefuls
    from numpy import array,pad,argmax,mod,arange,angle,exp,roll,std,diff,unwrap,allclose
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Validate domains
    if not isunispaced(domain_A):
        error('First domain must be unispaced.')
    if not isunispaced(domain_B):
        error('Second domain must be unispaced.')

    # Let the people know
    alert('Verbose mode ON.',verbose=verbose)

    # Do nothing if the data are already in the same format
    if len(domain_A)==len(domain_B):
        if allclose(domain_A,domain_B):
            alert('Inputs already in the same format. You may wish to apply domain transformations (e.g. time shifts) outside of this function.',verbose=verbose)
            return domain_A,range_A,range_B


    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Determine bounaries of common domain
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    if center_domains:
        # Center domains at start
        alert('Setting domains to start at zero.',verbose=verbose)
        domain_min = 0
        domain_max = max( (domain_A-domain_A[0])[-1], (domain_B-domain_B[0])[-1] )
    else:
        # Be agnostic about whether shifts in domain may apply
        domain_min = min( min(domain_A), min(domain_B) )
        domain_max = max( max(domain_A), max(domain_B) )

    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Generate a common domain
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    alert('Choosing the smallest domain spacing for calculation of common domain.',verbose=verbose)
    d_A = domain_A[1]-domain_A[0]
    d_B = domain_B[1]-domain_B[0]
    d = min( [d_A,d_B] )
    domain = arange( domain_min, domain_max+d, d )
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Interpolate to common domain
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    def __interpolate_domain__(dom,ran):
        dom_ = dom - dom[0]
        _amp = abs(ran)
        _phi = unwrap(angle(ran))
        _ran = spline(dom,_amp)(domain) * exp(1j*spline(dom,_phi)(domain))
        mask = (domain<min(dom)) | (domain>max(dom))
        _ran[mask] = 0
        # Return answer
        return _ran
    #
    alert('Interpolating data to common domain.',verbose=verbose)
    range_A = __interpolate_domain__(domain_A,range_A)
    range_B = __interpolate_domain__(domain_B,range_B)

    #
    alert('Done.',verbose=verbose)
    return domain,range_A,range_B


# Given two datasets, use numpy's xcorr to align the domains and ranges.
def corr_align( domain_A,range_A,
                domain_B,range_B,
                plot=False,
                center_domains=True,
                domain_align=True ):
    '''
    Given two datasets, use numpy's xcorr to align the domains and ranges.

    INPUTS
    ---
    domain_A,   Domain values for first dataset
    range_A,    Range values for first dataset
    domain_B,   Domain values for second dataset
    range_B,    Range values for second dataset
    plot=False  Optional plotting

    OUTPUTS
    ---
    domain_A,   Aligned Domain values for first dataset
    range_A,    Aligned Range values for first dataset
    domain_B,   Aligned Domain values for second dataset
    range_B,    = range_A
    foo         Dictionary containing information about the aignment

    '''
    # Imoprt usefuls
    from numpy import correlate, allclose
    from numpy import array,pad,argmax,mod,arange,angle,exp,roll,std,diff
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Validate domains
    if not isunispaced(domain_A):
        error('First domain must be unispaced.')
    if not isunispaced(domain_B):
        error('Second domain must be unispaced.')

    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Pad inputs to the same length (again)
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    domain,range_A,range_B = format_align(domain_A,range_A,domain_B,range_B,center_domains=True,verbose=False)

    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Use cross-correlation to determine optimal time and phase shift
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    x = correlate(range_A,range_B,mode='full')
    k = argmax( abs(x) )
    x0 = x[k]
    k0 = mod( k+1, len(domain) ) # NOTE that the +1 here ensures
                                 # k0=dom0=phi0=0 when trying to align data with itself
    dom0 = domain[k0]
    phi0 = angle(x0)

    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Apply the alignment parameters to input B
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    _range_B = range_B * exp( 1j*phi0 )
    if domain_align: _range_B = roll( _range_B, k0 )

    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    # Plot
    # ~-~-~-~-~-~-~-~--~-~-~--~-~-~-~ #
    if plot:
        #
        from matplotlib.pyplot import plot,xlim,figure,figaspect,ylim
        ref_d = domain[argmax( abs(range_A) )]
        #
        fig = figure( figsize=1*figaspect(1.0/7) )
        plot( domain, abs(range_A) )
        plot( domain, abs(_range_B) )
        #
        plot( domain, range_A.imag, lw=1, color='r', alpha=0.8 )
        plot( domain,_range_B.imag, 'k', alpha=0.9 )
        #
        dd = 0.25*diff(lim(domain))
        xlim(lim(domain))
        #
        dt = domain[1]-domain[0]
        figure(figsize=1*figaspect(1.0/7))
        plot( arange(len(x))*dt,abs(x) )
        xlim(lim(arange(len(x))*dt))

    #
    foo = {}
    foo['phase_shift'] = phi0
    foo['domain_shift'] = dom0
    foo['index_shift'] = k0
    foo['frmse'] = abs( std( range_A-_range_B )/std(range_A) )

    # Return in same order as input with additional info
    return domain,range_A,domain,_range_B,foo



# A fucntion that calculates a smoothness measure on the input 1D data.
def smoothness(y,r=20,stepsize=1,domain=None,unsigned=False):

    '''
    This fucntion calculates a smoothness measure on the input 1D data.
    The concept is similar to that decribed here: http://www.goldensoftware.com/variogramTutorial.pdf

    USAGE
    ---
    u,x = smoothness(t,y,r=4)

    INPUTS
    ---
    t,          Domain points of data set
    y,          Range of data set
    r=4,        Radius of which to consider variations (derivates)
    stepsize=1, The average will be considered every stepsize points. Increasing
                this from its default of 1 can be useful when processing large
                data sets; however, when stepsize is not 1, the length of the
                output will differ from that of the inputs.

    OUTPUTS
    ---
    u,       Sub-domain which is one-to-one with smoothness measure
    x,       Smoothness measure -- the data, y, is smooth when |x| is approx. 1
             NOTE that x=-1 is smooth and decreasing while x=1 is smooth and increasing

    '''

    # Import usefuls
    from numpy import arange,var,std,polyfit,poly1d,mean,diff,zeros_like,array
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    #
    if domain is None: domain = list(range(0,len(y)))

    x,u = [],[]
    for k in arange( 0, len(y), stepsize ):
        a = max(0,k-r)
        b = min(len(y),k+r)-1
        D = ( y[b]-y[a] ) / (b-a)
        if unsigned: D = abs(D)
        d = abs( mean(diff(y[a:b])) )
        x.append( ( D / d ) if d!=0 else 0 )
        u.append( (domain[a]+domain[b])/2 )

    # Preserve length
    x = array(x)
    if stepsize > 1:
        x = spline( u, x, k=1 )(domain)

    # Return the domain subseries and the smoothness measure
    return x


# Given a 1D vec of values, clump together adjacent identical values
def clump( data ):
    '''
    Given a 1D vec of values, clump together adjacent identical values.

    INTPUTS
    ---
    data,       1D iterable

    OUTPUTS
    ---
    clumps,     list of lists; each sublist is of like adjacent values
    maps        a list of index mask corresponding to the clumps (i.e. the sublists mentioned above)

    EXAMPLE
    ---
    clump([0,0,0,1,0,0,1,1,1,1,0,0,1,0,1])[0]

    ... ([[0, 0, 0], [1], [0, 0], [1, 1, 1, 1], [0, 0],   [1],  [0],  [1]],
         [[0, 1, 2], [3], [4, 5], [6, 7, 8, 9], [10, 11], [12], [13], [14]])

    --> the largest clump is at indeces [6, 7, 8, 9]

    spxll ~2018
    '''

    # Import usefuls
    from numpy import array,diff,arange

    # Find constant regions and their boundaries
    d = array( [0]+list(diff(data)), dtype=bool )
    e = find(d)

    # For all boundaries
    clump = []
    for j,k in enumerate(e):

        # The space between bounaries are to be clumped together

        if j==0:
            a = 0
        else:
            a = e[j-1]

        b = e[j]
        clump.append( data[a:b] )

    # Add the trailing clump manually
    clump.append(data[e[-1]:])

    # Create a pullback map
    M = []
    k = 0
    for c in clump:
        M.append( list(arange(len(c))+k) )
        k += len(c)

    # Return the ans
    ans = (clump,M)
    return ans


# Given a 1d data vector, determine a mask for the largest smooth region
def smoothest_part( data,
                    smoothness_radius=100,
                    smoothness_stepsize=10,
                    smooth_length=80,
                    smoothness_tolerance=1,
                    unsigned=False,
                    verbose=False ):

    '''
    Given a 1d data vector, determine a mask for the largest smooth region.

    smoothest_part( data,                       # 1D data of interest -- real
                    smoothness_radius=100,
                    smoothness_stepsize=20,
                    smooth_length=80
                    smoothness_tolerance=2,
                    verbose=False


    ~ spxll 2018
    '''

    # Import usefuls
    from numpy import isreal,argmax

    # Validate input(s)
    if not isreal(data).all():
        warning('Input array not real. The real part will be taken.')
        data = data.real

    # Calculate the smoothness of the input dataset
    x = smooth( smoothness( smooth(data,smooth_length).answer ,r=smoothness_radius,stepsize=smoothness_stepsize,unsigned=unsigned), smooth_length ).answer
    # x = smooth( smoothness( data ,r=smoothness_radius,stepsize=smoothness_stepsize), smooth_length ).answer

    # Create a boolean represenation of smoothness
    k = abs(x-1) < smoothness_tolerance

    # Clump the boolean represenation and then determine the largest clump
    if k.all():
        #
        warning('the data appears to be smooth everywhere; please consider using this function\'s optional inputs to set your smoothing criteria')
        mask = list(range(len(data)))
    elif k.any():
        clumps,clump_masks = clump(k)
        mask = clump_masks[ argmax( [ len(_) for _ in clump_masks ] ) ]
    else:
        warning('the data appears to not be smooth anywhere; please consider using this function\'s optional inputs to set your smoothing criteria')
        mask = list(range(len(data)))

    # Return answer
    ans = mask
    return ans

# Rotate a 3 vector using Euler angles
def rotate3(vector,alpha,beta,gamma,invert=False):
    '''
    Rotate a 3 vector using Euler angles under conventions defined at:
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Rotation_matrix

    Science reference: https://arxiv.org/pdf/1110.2965.pdf (Appendix)

    Specifically, the Z1,Y2,Z3 ordering is used: https://wikimedia.org/api/rest_v1/media/math/render/svg/547e522037de6467d948ecf3f7409975fe849d07

    *  alpha represents a rotation around the z axis

    *  beta represents a rotation around the x' axis

    *  gamma represents a rotation around the z'' axis

    NOTE that in order to perform the inverse rotation, it is *not* enough to input different rotation angles. One must use the invert=True keyword. This takes the same angle inputs as the forward rotation, but correctly applies the transposed rotation matricies in the reversed order.

    spxll'18
    '''

    # Import usefuls
    from numpy import cos,sin,array,dot,ndarray,vstack

    # Hangle angles as arrays
    angles_are_arrays = isinstance(alpha,ndarray) and isinstance(beta,ndarray) and isinstance(gamma,ndarray)
    if angles_are_arrays:
        # Check for consistent array shapes
        if not ( alpha.shape == beta.shape == gamma.shape ):
            # Let the people know and halt
            error( 'input angles as arrays must have identical array shapes' )

    # Validate input(s)
    if isinstance(vector,(list,tuple,ndarray)):
        vector = array(vector)
    else:
        error('first input must be iterable compatible 3D vector; please check')


    # Rotation around z''
    Ra = array( [
                    [cos(alpha),-sin(alpha),0],
                    [sin(alpha),cos(alpha),0],
                    [0,0,1]
        ] )

    # Rotation around y
    Rb = array( [
                    [cos(beta),0,sin(beta)],
                    [0,1,0],
                    [-sin(beta),0,cos(beta)]
        ] )

    # Rotation around z
    Rg = array( [
                    [cos(gamma),-sin(gamma),0],
                    [sin(gamma),cos(gamma),0],
                    [0,0,1]
        ] )

    # Perform the rotation
    # ans = (  Ra * ( Rb * ( Rg * vector ) )  )
    # NOTE that this is the same convention of equation A9 of Boyle et al : https://arxiv.org/pdf/1110.2965.pdf
    R = dot(  Ra, dot(Rb,Rg)  )
    if invert: R = R.T
    ans = dot( R, vector )

    # If angles are arrays, then format the input such that rows in ans correspond to rows in alpha, beta and gamma
    if angles_are_arrays:
        ans = vstack( ans ).T

    #
    return ans

# Look for point reflections in vector and correct
def reflect_unwrap( vec ):
    '''Look for point reflections in vector and correct'''

    #
    from numpy import array,sign,zeros_like

    #
    ans = array(vec)

    #
    for k,v in enumerate(vec):

        #
        if (k>0) and ( (k+1) < len(vec) ):

            #
            l = vec[k-1]
            c = vec[k]
            r = vec[k+1]

            #
            apply_reflection = (sign(l)==sign(r)) and (sign(l)==-sign(c))
            if apply_reflection:

                #
                ans[k] *= -1

    #
    return ans

# Look for reflections in vector and correct
def reflect_unwrap2( VEC, tol=0.1, domain = None ):
    '''Look for point reflections in vector and correct'''

    #
    from numpy import array,sign,zeros_like,arange,hstack

    #
    def ru2(vec):

        #
        ans = array(vec)

        #
        for k,v in enumerate(vec):

            #
            if (k>0):

                #
                l = ans[k-1]
                c = ans[k]

                #
                apply_reflection = abs(l+c) < tol*abs(l)
                if apply_reflection:

                    #
                    ans[k:] *= -1
        #
        return ans

    #
    if domain is None: domain = arange(len(VEC))
    mask_ = domain  > 0
    _mask = domain <= 0
    ans = hstack( [ ru2(VEC[_mask][::-1])[::-1], ru2(VEC[mask_]) ] )

    #
    return ans


def acos(X,sig=1,branch=0):
    '''
    Inverse cosine function compatible with arguments greater than 1 and complex numbers.
    ~ londonl@mit.edu '19

    USAGE:
        ans = acos(X,sig=1)

    NOTE:
        acos(X,sig=1) = -acos(X,sig=-1)
    '''
    #
    from positive import isiterable
    from numpy import sqrt,log,array,complex,angle,pi

    #
    def _acos_(x):
        x2 = x*x
        if x2<1:
            det = 1j * sqrt( 1-x2 )
        else:
            det = sqrt( x2-1 )
        #
        a = x + sig*det
        #
        if (a.real<0) or isinstance(a,complex):
            ans = -1j * (  log( abs(a) ) + 1j*(angle(a) + 2*pi*branch)  )
        else:
            ans = -1j * log( a  )
        if x<1:
            return ans.real
        else:
            return 1j*ans.imag

    #
    if isiterable(X):
        return array( [ _acos_(x) for x in X ] )
    else:
        return _acos_(X)



def sYlmAdj(s,l,lref,m,theta,phi,verbose=False,norm=True):

    '''

    AdjY = sYlmAdj(s,l,lref,m,theta,phi)

    Given indeces for a spin weighted spherical at (s,lref,m) calculate a relative adjoint (s,l,m). Note that we cannot currently handle add l-lref cases.

    '''

    # Import usefuls
    from positive import sYlm
    from scipy import sqrt,cos
    from numpy import zeros_like,conj

    # make indeces floats for safe arithmetic
    (s,l,lref,m) = [ float(x) for x in (s,l,lref,m) ]

    # Define u coordinate
    u = cos(theta)

    # Raising ell: Define dictionary for handled cases
    P = {
    		(1) : lambda ll: (2*sqrt((3 + 2*ll)*1.0/(1 + 2*ll))*(m*s + ll*(1 + ll)*u)*sYlm(s, ll, m, theta, phi))*1.0/sqrt((1 + 2*ll + ll**2 - m**2)*(1 + 2*ll + ll**2 - s**2)) - sYlm(s, 1 + ll, m, theta, phi),

    		(2) : lambda ll: (sqrt((5 + 2*ll)*1.0/(1 + 2*ll))*(2 + 6*m**2*s**2 + 4*m*s*u - 2*u**2 + 4*ll**5*u**2 + ll*(9 + 4*m**2*s**2 + 20*m*s*u - 5*u**2) + 2*ll**2*(7 + 12*m*s*u + 2*u**2) + 2*ll**4*(1 + 8*u**2) + ll**3*(9 + 8*m*s*u + 19*u**2))*sYlm(s, ll, m, theta, phi))*1.0/((1 + ll)*sqrt((1 + 2*ll + ll**2 - m**2)*(1 + 2*ll + ll**2 - s**2))*sqrt((4 + 4*ll + ll**2 - m**2)*(4 + 4*ll + ll**2 - s**2))) - (2*sqrt((5 + 2*ll)*1.0/(3 + 2*ll))*((3 + 2*ll)*m*s + (2 + 7*ll + 7*ll**2 + 2*ll**3)*u)*sYlm(s, 1 + ll, m, theta, phi))*1.0/((1 + ll)*sqrt((4 + 4*ll + ll**2 - m**2)*(4 + 4*ll + ll**2 - s**2))) + sYlm(s, 2 + ll, m, theta, phi),

    		(3) : lambda ll: (2*sqrt((5 + 2*ll)*1.0/(1 + 2*ll))*sqrt((7 + 2*ll)*1.0/(5 + 2*ll))*(2*(6 + 7*ll + 2*ll**2)*m**3*s**3 + 2*(13 + 49*ll + 63*ll**2 + 33*ll**3 + 6*ll**4)*m**2*s**2*u + 2*ll*(1 + ll)**4*(6 + 5*ll + ll**2)*u*(3 + 2*(-1 + ll)*u**2) + (1 + ll)**2*m*s*(25 - 13*u**2 + 12*ll**4*u**2 + 4*ll*(13 + 5*u**2) + 4*ll**2*(8 + 21*u**2) + ll**3*(6 + 60*u**2)))*sYlm(s, ll, m, theta, phi))*1.0/((1 + ll)**2*sqrt((1 + 2*ll + ll**2 - m**2)*(1 + 2*ll + ll**2 - s**2))*sqrt((4 + 4*ll + ll**2 - m**2)*(4 + 4*ll + ll**2 - s**2))*sqrt((9 + 6*ll + ll**2 - m**2)*(9 + 6*ll + ll**2 - s**2))) - (2*sqrt((7 + 2*ll)*1.0/(3 + 2*ll))*(36 + 33*m**2*s**2 + 76*m*s*u + 6*ll**7*u**2 + 2*ll*(78 + 29*m**2*s**2 + 145*m*s*u + 36*u**2) + 6*ll**5*(5 + 2*m*s*u + 40*u**2) + 12*ll**4*(10 + 8*m*s*u + 41*u**2) + ll**6*(3 + 60*u**2) + ll**3*(246 + 6*m**2*s**2 + 292*m*s*u + 546*u**2) + ll**2*(33*m**2*s**2 + 422*m*s*u + 39*(7 + 8*u**2)))*sYlm(s, 1 + ll, m, theta, phi))*1.0/((1 + ll)**2*(2 + ll)*sqrt((4 + 4*ll + ll**2 - m**2)*(4 + 4*ll + ll**2 - s**2))*sqrt((9 + 6*ll + ll**2 - m**2)*(9 + 6*ll + ll**2 - s**2))) + (2*sqrt((7 + 2*ll)*1.0/(3 + 2*ll))*((11 + 12*ll + 3*ll**2)*m*s + 3*(1 + ll)**2*(6 + 5*ll + ll**2)*u)*sYlm(s, 2 + ll, m, theta, phi))*1.0/((1 + ll)*(2 + ll)*sqrt((5 + 2*ll)*1.0/(3 + 2*ll))*sqrt((9 + 6*ll + ll**2 - m**2)*(9 + 6*ll + ll**2 - s**2))) - sYlm(s, 3 + ll, m, theta, phi),

    		(4) : lambda ll: (sqrt((3 + 2*ll)*1.0/(1 + 2*ll))*sqrt((7 + 2*ll)*1.0/(3 + 2*ll))*sqrt((9 + 2*ll)*1.0/(7 + 2*ll))*(120*m**4*s**4 + 548*m**3*s**3*u + 16*ll**11*u**4 + 2*m**2*s**2*(401 - 29*u**2) + 216*(-1 + u**2)**2 - 72*m*s*u*(-9 + 5*u**2) + 16*ll**10*u**2*(3 + 13*u**2) + 4*ll**9*(3 + 174*u**2 + 16*m*s*u**3 + 262*u**4) + ll**6*(3981 + 26214*u**2 - 3211*u**4 + 48*m**2*s**2*(1 + 21*u**2) + 176*m*s*u*(37 + 50*u**2)) + ll**7*(1137 + 13650*u**2 + 96*m**2*s**2*u**2 + 1913*u**4 + 16*m*s*u*(77 + 229*u**2)) + 2*ll**4*(5847 + 272*m**3*s**3*u + 8772*u**2 - 4111*u**4 + 2*m*s*u*(7748 + 1021*u**2) + 2*m**2*s**2*(543 + 2225*u**2)) + ll**2*(5529 + 96*m**4*s**4 + 2704*m**3*s**3*u - 4362*u**2 + 1353*u**4 + 4*m*s*u*(4554 - 1523*u**2) + 2*m**2*s**2*(2749 + 2907*u**2)) + ll**3*(10233 + 16*m**4*s**4 + 1760*m**3*s**3*u + 1650*u**2 - 2279*u**4 + 4*m*s*u*(7787 - 1146*u**2) + 2*m**2*s**2*(2351 + 5059*u**2)) + 2*ll**5*(4272 + 32*m**3*s**3*u + 14844*u**2 - 4498*u**4 + 4*m**2*s**2*(64 + 525*u**2) + m*s*u*(9261 + 5321*u**2)) + 4*ll**8*(45 + 1047*u**2 + 614*u**4 + 24*m*s*(u + 8*u**3)) + 2*ll*(94*m**4*s**4 + 986*m**3*s**3*u - 3*m*s*u*(-927 + 431*u**2) + m**2*s**2*(1653 + 625*u**2) + 9*(93 - 138*u**2 + 61*u**4)))*sYlm(s, ll, m, theta, phi))*1.0/((1 + ll)**3*sqrt((1 + 2*ll + ll**2 - m**2)*(1 + 2*ll + ll**2 - s**2))*sqrt((4 + 4*ll + ll**2 - m**2)*(4 + 4*ll + ll**2 - s**2))*sqrt((9 + 6*ll + ll**2 - m**2)*(9 + 6*ll + ll**2 - s**2))*sqrt((16 + 8*ll + ll**2 - m**2)*(16 + 8*ll + ll**2 - s**2))) - (4*sqrt((7 + 2*ll)*1.0/(3 + 2*ll))*sqrt((9 + 2*ll)*1.0/(7 + 2*ll))*((195 + 538*ll + 584*ll**2 + 310*ll**3 + 80*ll**4 + 8*ll**5)*m**3*s**3 + 2*(1 + ll)**2*(482 + 1203*ll + 1179*ll**2 + 565*ll**3 + 132*ll**4 + 12*ll**5)*m**2*s**2*u + (2 + 3*ll + ll**2)**3*(36 + 117*ll + 107*ll**2 + 36*ll**3 + 4*ll**4)*u*(3 + (-1 + 2*ll)*u**2) + (2 + 3*ll + ll**2)**2*m*s*(247 + 108*u**2 + 24*ll**5*u**2 + 12*ll**4*(1 + 19*u**2) + 3*ll**2*(123 + 385*u**2) + 2*ll**3*(56 + 389*u**2) + ll*(507 + 701*u**2)))*sYlm(s, 1 + ll, m, theta, phi))*1.0/((1 + ll)**3*(2 + ll)**2*sqrt((4 + 4*ll + ll**2 - m**2)*(4 + 4*ll + ll**2 - s**2))*sqrt((9 + 6*ll + ll**2 - m**2)*(9 + 6*ll + ll**2 - s**2))*sqrt((16 + 8*ll + ll**2 - m**2)*(16 + 8*ll + ll**2 - s**2))) + (2*sqrt((9 + 2*ll)*1.0/(5 + 2*ll))*(550*m**2*s**2 + 2484*m*s*u + 12*ll**9*u**2 + 1296*(1 + u**2) + 6*ll**8*(1 + 36*u**2) + 60*ll**6*(13 + 6*m*s*u + 120*u**2) + 3*ll**7*(35 + 8*m*s*u + 555*u**2) + 2*ll**5*(1605 + 6*m**2*s**2 + 1120*m*s*u + 9603*u**2) + 2*ll**4*(3999 + 69*m**2*s**2 + 3745*m*s*u + 16344*u**2) + ll**3*(12345 + 614*m**2*s**2 + 14530*m*s*u + 35385*u**2) + 2*ll**2*(660*m**2*s**2 + 8177*m*s*u + 180*(32 + 65*u**2)) + 2*ll*(685*m**2*s**2 + 4947*m*s*u + 54*(55 + 79*u**2)))*sYlm(s, 2 + ll, m, theta, phi))*1.0/((1 + ll)**2*(2 + ll)**2*(3 + ll)*sqrt((9 + 6*ll + ll**2 - m**2)*(9 + 6*ll + ll**2 - s**2))*sqrt((16 + 8*ll + ll**2 - m**2)*(16 + 8*ll + ll**2 - s**2))) - (4*sqrt((9 + 2*ll)*1.0/(7 + 2*ll))*((25 + 35*ll + 15*ll**2 + 2*ll**3)*m*s + (72 + 198*ll + 205*ll**2 + 100*ll**3 + 23*ll**4 + 2*ll**5)*u)*sYlm(s, 3 + ll, m, theta, phi))*1.0/((1 + ll)*(2 + ll)*(3 + ll)*sqrt((16 + 8*ll + ll**2 - m**2)*(16 + 8*ll + ll**2 - s**2))) + sYlm(s, 4 + ll, m, theta, phi)
        }

    # Lowering ell: Define dictionary for handled cases
    Q = {
    		(1) : lambda ll: -sYlm(s, -1 + ll, m, theta, phi) + (2*sqrt((-1 + 2*ll)*1.0/(1 + 2*ll))*(m*s + ll*(1 + ll)*u)*sYlm(s, ll, m, theta, phi))*1.0/sqrt((ll**2 - m**2)*(ll**2 - s**2)),

    		(2) : lambda ll: sYlm(s, -2 + ll, m, theta, phi) + (2*sqrt((3 - 2*ll)*1.0/(1 - 2*ll))*(m*(s - 2*ll*s) + ll*(1 + ll - 2*ll**2)*u)*sYlm(s, -1 + ll, m, theta, phi))*1.0/(ll*sqrt((1 - 2*ll + ll**2 - m**2)*(1 - 2*ll + ll**2 - s**2))) + (sqrt((-3 + 2*ll)*1.0/(1 + 2*ll))*(-2*m**2*s**2 + 4*ll*m*s*(m*s - u) + 4*ll**5*u**2 + ll**3*(1 + 8*m*s*u - 5*u**2) + ll**2*(1 - 3*u**2) + ll**4*(-2 + 4*u**2))*sYlm(s, ll, m, theta, phi))*1.0/(ll*sqrt((ll**2 - m**2)*(ll**2 - s**2))*sqrt((1 - 2*ll + ll**2 - m**2)*(1 - 2*ll + ll**2 - s**2))),

    		(3) : lambda ll: (-((-1 + ll)*sYlm(s, -3 + ll, m, theta, phi)) + (2*sqrt((5 - 2*ll)*1.0/(1 - 2*ll))*((2 - 6*ll + 3*ll**2)*m*s + 3*ll**2*(2 - 3*ll + ll**2)*u)*sYlm(s, -2 + ll, m, theta, phi))*1.0/(sqrt((3 - 2*ll)*1.0/(1 - 2*ll))*ll*sqrt((4 - 4*ll + ll**2 - m**2)*(4 - 4*ll + ll**2 - s**2))) + (2*sqrt((5 - 2*ll)*1.0/(3 - 2*ll))*(-3 + 2*ll)*(2*m**2*s**2 - 6*ll**7*u**2 + 2*ll*m*s*(-5*m*s + u) + ll**2*m*s*(15*m*s + 2*u) + 3*ll**4*(5 + 12*m*s*u - 6*u**2) - 2*ll**3*(3 + 3*m**2*s**2 + 14*m*s*u - 6*u**2) - 6*ll**5*(2 + 2*m*s*u + u**2) + 3*ll**6*(1 + 6*u**2))*sYlm(s, -1 + ll, m, theta, phi))*1.0/(sqrt((3 - 2*ll)*1.0/(1 - 2*ll))*ll**2*(-1 + 2*ll)*sqrt((4 - 4*ll + ll**2 - m**2)*(4 - 4*ll + ll**2 - s**2))*sqrt((1 - 2*ll + ll**2 - m**2)*(1 - 2*ll + ll**2 - s**2))))*1.0/(-1 + ll) + (2*sqrt((5 - 2*ll)*1.0/(3 - 2*ll))*(-3 + 2*ll)*sqrt((-1 + 2*ll)*1.0/(1 + 2*ll))*(2*m**3*s**3 - 6*ll**7*u + 4*ll**8*u**3 + 2*ll*m**2*s**2*(-3*m*s + 2*u) - 2*ll**3*m*s*(3 + 9*m*s*u - 8*u**2) + ll**2*m*s*(-1 + 4*m**2*s**2 + 3*u**2) - 4*ll**6*u*(-3 - 3*m*s*u + 5*u**2) + 2*ll**4*(-6*u + 6*m**2*s**2*u + 8*u**3 + m*s*(7 - 12*u**2)) - 6*ll**5*(-u + m*(s + 2*s*u**2)))*sYlm(s, ll, m, theta, phi))*1.0/(sqrt((3 - 2*ll)*1.0/(1 - 2*ll))*ll**2*(-1 + 2*ll)*sqrt((ll**2 - m**2)*(ll**2 - s**2))*sqrt((4 - 4*ll + ll**2 - m**2)*(4 - 4*ll + ll**2 - s**2))*sqrt((1 - 2*ll + ll**2 - m**2)*(1 - 2*ll + ll**2 - s**2))),

    		(4) : lambda ll: sYlm(s, -4 + ll, m, theta, phi) - (4*sqrt((7 - 2*ll)*1.0/(5 - 2*ll))*((-3 + 11*ll - 9*ll**2 + 2*ll**3)*m*s + ll*(6 - 23*ll + 28*ll**2 - 13*ll**3 + 2*ll**4)*u)*sYlm(s, -3 + ll, m, theta, phi))*1.0/((-2 + ll)*(-1 + ll)*ll*sqrt((9 - 6*ll + ll**2 - m**2)*(9 - 6*ll + ll**2 - s**2))) + (2*sqrt((7 - 2*ll)*1.0/(5 - 2*ll))*(-5 + 2*ll)*(-12*m**2*s**2 + 12*ll**9*u**2 + 8*ll*m*s*(10*m*s + 3*u) + 2*ll**3*(96 + 91*m**2*s**2 + 305*m*s*u - 60*u**2) - 2*ll**2*(18 + 93*m**2*s**2 + 100*m*s*u - 18*u**2) - ll**4*(393 + 78*m**2*s**2 + 850*m*s*u - 9*u**2) - 6*ll**8*(1 + 18*u**2) + 3*ll**7*(19 + 8*m*s*u + 123*u**2) - 3*ll**6*(71 + 64*m*s*u + 195*u**2) + ll**5*(399 + 12*m**2*s**2 + 584*m*s*u + 387*u**2))*sYlm(s, -2 + ll, m, theta, phi))*1.0/(sqrt((5 - 2*ll)*1.0/(3 - 2*ll))*(-2 + ll)*(-1 + ll)**2*ll**2*(-3 + 2*ll)*sqrt((9 - 6*ll + ll**2 - m**2)*(9 - 6*ll + ll**2 - s**2))*sqrt((4 - 4*ll + ll**2 - m**2)*(4 - 4*ll + ll**2 - s**2))) - (4*sqrt((3 - 2*ll)*1.0/(1 - 2*ll))*sqrt((7 - 2*ll)*1.0/(3 - 2*ll))*(-3*m**3*s**3 + 20*ll*m**3*s**3 + 8*ll**11*u**3 - ll**2*m*s*(9 + 54*m**2*s**2 + 26*m*s*u - 12*u**2) - 4*ll**10*u*(3 + 13*u**2) + 2*ll**9*u*(48 + 12*m*s*u + 47*u**2) - ll**4*(40*m**3*s**3 - 69*u + 312*m**2*s**2*u + 57*u**3 + m*s*(228 - 197*u**2)) + ll**3*(70*m**3*s**3 + 144*m**2*s**2*u + m*s*(75 - 91*u**2) + 18*u*(-1 + u**2)) + ll**5*(8*m**3*s**3 + 314*m**2*s**2*u + m*s*(331 - 63*u**2) - 2*u*(15 + 8*u**2)) + ll**7*(24*m**2*s**2*u + 8*u*(48 - 31*u**2) + 2*m*s*(44 + 173*u**2)) - ll**6*(144*m**2*s**2*u + 4*u*(51 - 56*u**2) + m*s*(245 + 269*u**2)) - ll**8*(285*u - 29*u**3 + 12*m*(s + 13*s*u**2)))*sYlm(s, -1 + ll, m, theta, phi))*1.0/((-1 + ll)**2*ll**3*sqrt((9 - 6*ll + ll**2 - m**2)*(9 - 6*ll + ll**2 - s**2))*sqrt((4 - 4*ll + ll**2 - m**2)*(4 - 4*ll + ll**2 - s**2))*sqrt((1 - 2*ll + ll**2 - m**2)*(1 - 2*ll + ll**2 - s**2))) + (sqrt((-7 + 2*ll)*1.0/(1 + 2*ll))*(-12*m**4*s**4 + 4*ll*m**3*s**3*(11*m*s - 3*u) + 16*ll**11*u**4 - 12*ll**2*m**2*s**2*(2 + 4*m**2*s**2 + 4*m*s*u - 3*u**2) - 16*ll**10*u**2*(3 + 2*u**2) + 4*ll**9*(3 + 54*u**2 + 16*m*s*u**3 - 38*u**4) + ll**7*(129 - 558*u**2 + 96*m**2*s**2*u**2 + 313*u**4 + 16*m*s*u*(29 - 11*u**2)) - 2*ll**6*(27 - 228*u**2 + 211*u**4 - 72*m*s*u*(-4 + 5*u**2) + 24*m**2*s**2*(1 + 7*u**2)) + 2*ll**3*m*s*(8*m**3*s**3 + 112*m**2*s**2*u + 3*m*s*(29 - 47*u**2) + 6*u*(-8 + 9*u**2)) + ll**5*(-33 + 64*m**3*s**3*u + 126*u**2 - 69*u**4 + 56*m**2*s**2*(4 + 3*u**2) - 2*m*s*u*(27 + 79*u**2)) - 4*ll**8*(18 + 21*u**2 - 64*u**4 + 24*m*s*(u + 2*u**3)) - 2*ll**4*(112*m**3*s**3*u - 2*m**2*s**2*(-83 + 85*u**2) + m*s*u*(-169 + 165*u**2) - 9*(1 - 6*u**2 + 5*u**4)))*sYlm(s, ll, m, theta, phi))*1.0/(ll**3*sqrt((ll**2 - m**2)*(ll**2 - s**2))*sqrt((9 - 6*ll + ll**2 - m**2)*(9 - 6*ll + ll**2 - s**2))*sqrt((4 - 4*ll + ll**2 - m**2)*(4 - 4*ll + ll**2 - s**2))*sqrt((1 - 2*ll + ll**2 - m**2)*(1 - 2*ll + ll**2 - s**2)))
        }

    # Define shorthand
    k = lref
    r = l-k

    #
    if (abs(r) in P) and (r>0):
    	#
    	aY = P[r](k)
    elif (abs(r) in Q) and (r<0):
    	#
    	if (l>=abs(s)) and (l>=abs(m)):
    		aY = Q[abs(r)](k)
    	else:
    		aY = zeros_like(theta)
    elif r==0:
        if verbose: alert('returning conjugate of input')
    	aY = sYlm(s,l,m,theta,phi)
    else:
    	#
    	warning('l-lref=%i currently not handled'%r)
    	aY = zeros_like(theta)

    #
    ans = conj( aY )
    if norm:
        c = sqrt(prod(ans,ans,theta))
        ans = ans/c

    #
    return ans
