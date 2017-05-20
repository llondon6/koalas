
# -------------------------------------------------------- #
'''                 Define some useful BASICS            '''
# These are useful for terminal printing & system commanding
# -------------------------------------------------------- #

# Make "mkdir" function for directories
def mkdir(dir_,rm=False,verbose=False):
    # Import useful things
    import os
    import shutil
    # Delete the directory if desired and if it already exists
    if os.path.exists(dir_) and (rm is True):
        if verbose:
            alert('Directory at "%s" already exists %s.'%(magenta(dir_),red('and will be removed')),'mkdir')
        shutil.rmtree(dir_,ignore_errors=True)
    # Check for directory existence; make if needed.
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        if verbose:
            alert('Directory at "%s" does not yet exist %s.'%(magenta(dir_),green('and will be created')),'mkdir')
    # Return status
    return os.path.exists(dir_)

# Alert wrapper
def alert(msg,fname=None,say=False):
    import os
    if fname is None:
        fname = 'note'
    print '('+cyan(fname)+')>> '+msg
    if say: os.system( 'say "%s"' % msg )

# Wrapper for OS say
def say(msg,fname=None):
    import os
    if fname is None:
        fname = 'a function'
    if msg:
        os.system( 'say "%s says: %s"' % (fname,msg) )

# Warning wrapper
def warning(msg,fname=None):
    if fname is None:
        fname = 'warning'
    print '('+yellow(fname)+')>> '+msg

# Error wrapper
def error(msg,fname=None):
    if fname is None:
        fname = 'error'
    raise ValueError( '('+red(fname)+')!! '+msg )

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

# Useful function for getting parent directory
def parent(path):
    '''
    Simple wrapper for getting absolute parent directory
    '''
    import os
    return os.path.abspath(os.path.join(path, os.pardir))+'/'

# Class for basic print manipulation
class print_format:
   magenta = '\033[95m'
   cyan = '\033[96m'
   darkcyan = '\033[36m'
   blue = '\033[94m'
   green = '\033[92m'
   yellow = '\033[93m'
   red = '\033[91m'
   bold = '\033[1m'
   grey = gray = '\033[1;30m'
   ul = '\033[4m'
   end = '\033[0m'
   underline = '\033[4m'

# Function that uses the print_format class to make tag text for bold printing
def bold(string):
    return print_format.bold + string + print_format.end
def red(string):
    return print_format.red + string + print_format.end
def green(string):
    return print_format.green + string + print_format.end
def magenta(string):
    return print_format.magenta + string + print_format.end
def blue(string):
    return print_format.blue + string + print_format.end
def grey(string):
    return print_format.grey + string + print_format.end
def yellow(string):
    return print_format.yellow + string + print_format.end
def cyan(string):
    return print_format.cyan + string + print_format.end
def darkcyan(string):
    return print_format.darkcyan + string + print_format.end
def textul(string):
    return print_format.underline + string + print_format.end
def underline(string):
    return print_format.underline + string + print_format.end

# Function to produce array of color vectors
def rgb( N,                     #
         offset     = None,     #
         speed      = None,     #
         plot       = False,    #
         shift      = None,     #
         jet        = False,     #
         reverse    = False,     #
         verbose    = None ):   #

    #
    from numpy import array,pi,sin,arange,linspace

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
        R = r*sin( w*t                + shift )
        G = g*sin( w*t*speed + offset + shift )
        B = b*sin( w*t + pi/2         + shift )

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

#
def apolyfit(x,y,order=None,tol=1e-3):
    #
    from numpy import polyfit,poly1d,std,inf

    #
    givenorder = False if order is None else True

    #
    done = False; k = 0; ordermax = len(x)-1; oldr = inf
    while not done:

        order = k if givenorder is False else order
        fit = poly1d(polyfit(x,y,order))
        r = std( fit(x)-y ) / ( std(y) if std(y)>1e-15 else 1.0 )
        k += 1

        dr = oldr-r # ideally dr > 0

        if order==ordermax:
            done = True
        if dr <= tol:
            done = True
        if dr < 0:
            done = True

        if givenorder:
            done = True

    #
    return fit


# custome function for setting desirable ylimits
def pylim( x, y, axis='both', domain=None, symmetric=False, pad_y=0.1 ):

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

# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))

#
# Use formula from wikipedia to calculate the harmonic
# See http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
# for more information.
# NOTE: See also Proc. R. Soc. Lond. A-1977-Breuer-71-86
def sYlm(s,l,m,theta,phi):

    #
    from numpy import pi,ones,sin,tan,exp,array,double,sqrt,zeros
    from scipy.misc import factorial,comb

    #
    if isinstance(theta,(float,int,double)): theta = [theta]
    if isinstance(phi,(float,int,double)): phi = [phi]
    theta = array(theta)
    phi = array(phi)

    #
    theta = array([ double(k) for k in theta ])
    phi = array([ double(k) for k in phi ])

    # Ensure regular output (i.e. no nans)
    theta[theta==0.0] = 1e-9

    # Name anonymous functions for cleaner syntax
    f = lambda k: double(factorial(k))
    c = lambda x: double(comb(x[0],x[1]))
    cot = lambda x: 1.0/double(tan(x))

    # Pre-allocatetion array for calculation (see usage below)
    if min(theta.shape)!=1 and min(phi.shape)!=1:
        X = ones( len(theta) )
        if theta.shape != phi.shape:
            error('Input dim error: theta and phi inputs must be same size.')
    else:
        X = ones( theta.shape )


    # Calcualte the "pre-sum" part of sYlm
    a = (-1.0)**m
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
    Y = A*B*exp( 1j*m*phi )

    #
    if sum(abs(Y.imag)) < 1e-20:
        Y = Y.real

    #
    return Y


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Given a 1D array, determine the set of N lines that are optimally representative  #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
def romline(  domain,           # Domain of Map
              range_,           # Range of Map
              N,                # Number of Lines to keep for final linear interpolator
              positive=False,   # Toggle to use positive greedy algorithm ( where rom points are added rather than removed )
              verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_
    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1

    #
    if not positive:
        #
        done = False
        space = range( len(d) )
        raw_space = range( len(d) )
        err = lambda x: mean( abs(x) ) # std(x) #
        raw_mask = []
        while not done:
            #
            min_sigma = inf
            for k in range(len(space)):
                # Remove a trial domain point
                trial_space = list(space)
                trial_space.pop(k)
                # Determine the residual error incured by removing this trial point after linear interpolation
                # Apply linear interpolation ON the new domain TO the original domain
                trial_domain = d[ trial_space ]
                trial_range = r[ trial_space ]
                # Calculate the ROM's representation error using ONLY the points that differ from the raw domain, as all other points are perfectly represented by construction. NOTE that doing this significantly speeds up the algorithm.
                trial_mask = list( raw_mask ).append( k )
                sigma = err( linterp( trial_domain, trial_range )( d[trial_mask] ) - r[trial_mask] ) / ( err(r[trial_mask]) if err(r[trial_mask])!=0 else 1e-8  )
                #
                if sigma < min_sigma:
                    min_k = k
                    min_sigma = sigma
                    min_space = array( trial_space )

            #
            raw_mask.append( min_k )
            #
            space = list(min_space)

            #
            done = len(space) == N

        #
        rom = linterp( d[min_space], R[min_space] )
        knots = min_space

    else:
        from numpy import inf,argmin,argmax
        seed_list = [ 0, argmax(R), argmin(R), len(R)-1 ]
        min_sigma = inf
        for k in seed_list:
            trial_knots,trial_rom,trial_sigma = positive_romline( d, R, N, seed = k )
            print trial_sigma
            if trial_sigma < min_sigma:
                knots,rom,min_sigma = trial_knots,trial_rom,trial_sigma

    #
    print min_sigma

    return knots,rom


#
def positive_romline(   domain,           # Domain of Map
                        range_,           # Range of Map
                        N,                # Number of Lines to keep for final linear interpolator
                        seed = None,      # First point in domain (index) to use
                        verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_
    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1

    #
    if seed is None:
        seed = argmax(r)
    else:
        if not isinstance(seed,int):
            msg = 'seed input must be int'
            error( msg, 'positive_romline' )

    #
    done = False
    space = [ seed ]
    domain_space = range(len(d))
    err = lambda x: mean( abs(x) ) # std(x) #
    while not done:
        #
        min_sigma = inf
        for k in [ a for a in domain_space if not (a in space) ]:
            # Add a trial point
            trial_space = list(space)
            trial_space.append(k)
            trial_space.sort()
            # Apply linear interpolation ON the new domain TO the original domain
            trial_domain = d[ trial_space ]
            trial_range = r[ trial_space ]
            #
            sigma = err( linterp( trial_domain, trial_range )( d ) - r ) / ( err(r) if err(r)!=0 else 1e-8  )
            #
            if sigma < min_sigma:
                min_k = k
                min_sigma = sigma
                min_space = array( trial_space )

        #
        space = list(min_space)
        #
        done = len(space) == N

    #
    rom = linterp( d[min_space], R[min_space] )
    knots = min_space

    return knots,rom,min_sigma
