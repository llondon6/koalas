
# -------------------------------------------------------- #
# Import Libs
import os,shutil
import glob
import urllib2
import tarfile,sys
import time
import subprocess
import re
import inspect
import pickle
import numpy
import string
import random
import h5py
import copy
# -------------------------------------------------------- #

# Return name of calling function
def thisfun():
    import inspect
    return inspect.stack()[2][3]

# Make "mkdir" function for directories
def mkdir(dir_,rm=False,verbose=False):
    # Import useful things
    import os
    import shutil
    # Expand user if needed
    dir_ = os.path.expanduser(dir_)
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
def alert(msg,fname=None,say=False,output_string=False):
    import os
    if fname is None:
        fname = thisfun()
    if say: os.system( 'say "%s"' % msg )
    _msg = '('+cyan(fname)+')>> '+msg
    if not output_string:
        print _msg
    else:
        return _msg

# Wrapper for OS say
def say(msg,fname=None):
    import os
    if fname is None:
        fname = thisfun()
    if msg:
        os.system( 'say "%s says: %s"' % (fname,msg) )

# Warning wrapper
def warning(msg,fname=None,output_string=False):
    if fname is None:
        fname = thisfun()
    _msg = '('+yellow(fname)+')>> '+msg
    if not output_string:
        print _msg
    else:
        return _msg

# Error wrapper
def error(msg,fname=None):
    if fname is None:
        fname = thisfun()
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
         jet        = False,    #
         reverse    = False,    #
         weights    = None,     #
         verbose    = None ):   #

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

# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))

#
from nrutils import sYlm

# Generate all positive integer *pairs* that sum to n
def twosum(n):
    '''Generate all positive integer *pairs* that sum to n'''
    ans = []
    for j in range(n):
        if (j+1) <= (n-j-1):
            # NOTE that it important here that the smaller number is first
            # Also NOTE that the orientation of the above inequality means that no zeros will be output
            ans.append( [ j+1,n-j-1 ] )
    return ans

# NOTE that the algorithm below *may* present a novel solution to the subset-sum problem
# Recursively generate all sets of length k whose sum is n
def rnsum(order,degree,lst=None):
    '''Recursively generate all sets of length k whose sum is n'''
    # Use shorthand
    k = order
    n = degree
    #
    if 1==k:
        # -------------------------------- #
        # Handle the trivial order 1 case
        # -------------------------------- #
        A = [ [n] ]
    else:
        # -------------------------------- #
        # Handle the general order case
        # -------------------------------- #
        # The answer will have the form [a1,a2...ak], where the sum over all ai is n
        # We will proceed by treating the list above as [a1,b], in which case the twosum solution applies
        A = twosum(n)
        # We only wish to proceed if lists of length greater than two are requested
        proceed = k>len(A[0]) if len(A) else False
        #
        while proceed:
            B = []
            for a in A:
                # Generate sumlists, again from teh twosum perspective
                U = twosum( a[-1] )
                # Now create 3 element list whose sum is n
                V = []
                for u in U:
                    V.append( sorted( a[:-1]+u ) )
                B+=V
            # Remove duplicates
            A = [ list(y) for y in set([tuple(x) for x in sorted(B)]) ]
            proceed = k>len(A[0]) if len(A) else False
    #
    ans = A
    return ans

# Given the domain space dimension, and maximum degree of interest, generate a list of strings to be used with gmvpfit. The strings correspond to unique n-dimensional multinomial terms. NOTE that this function's output is formatted to be constistent with mvpolyfit().
def mvsyms( dimension, max_degree, verbose=False ):
    # Import useful things
    from itertools import permutations,combinations
    from numpy import arange
    # Create reference strings for the encoding dimnesions (used once below)
    dims = ''.join( [ str(k) for k in range(dimension) ] )
    # Create temporary holder for all symbols
    _ans = []
    # For all orders
    for order in arange(1,dimension+1):
        basis = [ ''.join(s) for s in combinations(dims,order) ]
        if verbose: print 'Order=%i\n%s'%(order,10*'####')
        if verbose: print 'basis = %s'%basis
        for B in basis:
            # For degrees between 1 and max degree
            # NOTE that while degree can be greater than order, the converse is not true: terms of order N have a degree of at least N
            for degree in arange(order,max_degree+1):
                # Create all symbols of the desired order that have the desired degree
                if verbose: print '\nDegree=%i\n%s'%(degree,10*'----')
                # Power basis
                pwrbasis = rnsum(order,degree)
                if verbose: print pwrbasis
                # For all power lists
                for pwr in pwrbasis:
                    # Create symbols for all permutations of the current power list
                    if verbose: print pwr
                    for P in permutations( pwr ):
                        this_symbol = ''.join([ p*B[k] for k,p in enumerate(P) ])
                        if verbose: print this_symbol
                        _ans += [this_symbol]

    # Remove duplicate symbols, ans sort according to order
    # And then add symbold for constant term to the start
    ans = ['K'] + sorted(sorted(list(set(_ans))),key = lambda x: len(x))

    # Return output
    return ans

# Flatten N-D domain and related range
def ndflatten( domain_list,      # List of meshgrid arrays
               range_array=None ):  # Array of related range for some external map

    # Import useful things
    from numpy import reshape,vstack

    #
    if not (range_array is None):
        range_shape = range_array.shape

    #
    flat_domain_list = []
    for d in domain_list:
        flat_domain_list.append( reshape(d,(d.size,)) )
        this_shape = d.shape
        # Check for shape consistency
        if not (range_array is None):
            if this_shape != range_shape:
                warning('all input objects must be of the same shape: range shape is %s, but domain input found with shape %s'%(list(range_shape),list(this_shape)))

    #
    flat_domain = vstack( flat_domain_list ).T
    if not (range_array is None):
        flat_range = reshape( range_array, (range_array.size,) )

    #
    if not (range_array is None):
        return flat_domain,flat_range
    else:
        return flat_domain

# Multivariate polynomial fitting algorithm
class mvpolyfit:
    '''
    # Low Level Multivariate Polynomial Fitting
    ---

    fit_object = mvpolyfilt( ... )

    ## Inputs

    domain          The N-D domain over which a scalar will be modeled: list of vector coordinates. Numpy ndarray of shape number_of_samples x number_of_domain_dimnesions.

    scalar_range    The scalar range to model on the domain: 1d iterable of range values corresponding to domain entries

    basis_symbols   1D iterable of string symbols to be interpreted as multinomial functions of the domain variables. Example: if domain variables are (x,y), then '001' is x** * y and 'K' is a constant term.

    labels          Example:

                    labels = {
                                'python':[ 'function_name_for_python',('var1','var2',...) ],
                                'latex' :[ 'function_var_name',('var1','var2',...) ]
                             }

    ## Outputs

    The output object is a memeber of a class defined within this function. Please see

        print fit_object.__dict__.keys()

    Of particular use:

        fit_object.plot()
        fit_object.__str__

    '''
    # Initialize the fit
    def __init__( this,                # The Current Object
                  domain,              # The N-D domain over which a scalar will be modeled: list of vector coordinates
                  scalar_range,        # The scalar range to model on the domain: 1d iterable of range values corresponding to domain entries
                  basis_symbols,       # These symbols encode which combinations of dimensions will be used for regression
                  labels = None,       # Domain and range labels
                  range_map = None,    # Operation to apply to range before fitting, and inverse. EXAMPLE: range_map = { 'forward': lambda domain,range: ..., 'backward': lambda domain,forward_range: ... }
                  plot = False,
                  data_label = None,
                  verbose = False ):   # Let the people know


        # Import useful things
        from kerr import alert,error,warning
        from numpy import array,mean,unwrap,angle,std,isfinite
        from scipy.stats import norm

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        ''' Validate Inputs '''
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        this.__validate_inputs__(domain,scalar_range,basis_symbols,labels,range_map,data_label)

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        ''' Perform the fit '''
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        this.__fit__()

        # Compute the fit residuals
        # NOTE that we will use a similarity transformation based on the range_map
        def U(rr):  return this.range_map[ 'forward'](this.domain,rr )
        def V(rr_): return this.range_map[ 'backward'](this.domain,rr_)
        fit_range = this.eval( this.domain )
        residuals = fit_range - this.range
        fractional_residuals = (fit_range - this.range)/this.range
        frmse = abs( std(U(fit_range)-U(this.range)) / std(U(this.range)) )

        # Model the residuals as normally distributed random points
        ampres = ( abs(fit_range)-abs(this.range) ) / abs(this.range)
        phares = ( sunwrap(angle(fit_range)) - sunwrap(angle(this.range)) ) / sunwrap(angle(this.range))
        ar_mu,ar_std = norm.fit( ampres )
        pr_mu,pr_std = norm.fit( phares )
        # Also do this for the real values, in case the data is only real
        rr_mu,rr_std = norm.fit( residuals.real / this.range.real )

        # Store bulk information about fit
        this.residual = residuals
        this.frmse = frmse
        this.prompt_string = str(this)
        this.python_string = this.__str_python__()
        this.latex_string = this.__str_latex__()
        this.bin = {}   # A holder for misc information

        # Store characteristics of amplitude and phase residuals
        this.bin['frac_amp_res_mu_std'] = (ar_mu,ar_std)
        this.bin['frac_pha_res_mu_std'] = (pr_mu,pr_std)
        this.bin['frac_real_res_mu_std'] = (rr_mu,rr_std)
        this.bin['frac_amp_res'] = ampres
        this.bin['frac_pha_res'] = phares
        this.bin['frac_real_res'] = fractional_residuals

    # Perform the fit
    def __fit__(this):

        # Import usefuls
        from numpy import array,dot
        from numpy.linalg import pinv,lstsq

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        ''' Given the basis symbols, construct a generalized Vandermonde matrix '''
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        P = array( [this.symeval(sym) for sym in this.basis_symbols] ).T

        # Compute the pseudo inverse of P
        Q = pinv( P )
        # NOTE that pinv correctly handles complex matricies

        # Extract the forward domain map to apply before least-squares
        U = this.range_map['forward']

        # Estimate the coefficients of the basis symbols
        a = dot( Q, U(this.domain,this.range) )

        # # Largely the same as above, but in certain circumstances, a more complicated approach is taken -- line kept for future consideration
        # a = lstsq( P,U(this.domain,this.range) )[0]
        # print a

        # Store the fit coefficients
        this.coeffs = a

    # Create a functional representation of the fit
    def eval( this, vec ):
        '''A functional representation of the fit'''
        #
        ans_ = 0
        for k,b in enumerate(this.coeffs):
            ans_ += b*this.symeval( this.basis_symbols[k], dom=vec )
        #
        V = this.range_map['backward']
        ans = V(vec,ans_)
        #
        return ans

    # Each symbol encodes a simple multinomial function of the basis vectors.
    # Example: sym = "001111" encodes x^2 * y^4
    # NOTE that a constant term will be added independently of the given symbols, unless as toggle is implemented?
    def symeval( this, sym, dom = None ):

        if dom is None:
            dom = this.domain

        if sym.upper() == 'K':
            # Handle symbolic representation of constant term
            from numpy import ones
            ans = ones( dom[:,0].shape, dtype=dom.dtype ) if len(dom.shape)>1 else ones( dom.shape, dtype=dom.dtype )
        elif sym.isdigit():
            # Handle non-constant symbols
            map_ = [ int(k) for k in sym ]
            ans = 1.0 # NOTE that the final answer will be of the shape of the domain vectors
            for k in map_:
                # IF the domain has dimension greater than 1
                if len(dom.shape)>1:
                    # Allow referencing of each dimnesion
                    ans *= dom[:,k]
                else:
                    # ELSE, simply use teh domain
                    ans *= dom
                    # NOTE that this IF-ELSE structure results from 1D domains not being able to be refernced in a matrix like manner
        else:
            raise TypeError('"%s" is an invalid symbol. Multivariate symbols must be "K" for constant term, or string of integers corresponding to domain dimensions, such as "0001" which, if the domain is [x,y,...], corresponds to x*x*x*y.'%sym)

        #
        return ans

    # Create model string for prompt printing
    def __str__(this,labels=None):

        # Handle the labels input
        return this.__str_python__(labels=labels)

    # Create model string for python module
    def __str_python__(this,labels=None,precision=8):

        python_labels = ( this.labels['python'] if 'python' in this.labels else None ) if labels is None else labels

        # Extract desired labels and handle defaults
        funlabel = 'f' if python_labels is None else python_labels[0]
        varlabel = None if python_labels is None else python_labels[1]

        prefix = '' if python_labels is None else python_labels[2]
        postfix = '' if python_labels is None else ( python_labels[3] if len(python_labels)==4 else '' )

        if varlabel is None:
            varlabel = [ 'x%s'%str(k) for k in range(this.domain_dimension) ]
        elif len(varlabel) != this.domain_dimension:
            error( 'Number of variable labels, %i, is not equal to the number of domain dimensions found, %i. One posiility is that youre fitting with a 1D domain, and have attempted to use a domain label that is a tuple containing a single string which python may interpret as a string -- try defining the label as a list by using square brackets.'%( len(varlabel), this.domain_dimension ) , 'mvpolyfit' )

        # Replace minus signs in function name with M
        funlabel = funlabel.replace('-','M')

        # Create a simple string representation of the fit
        model_str = '%s = lambda %s:%s%s*(x%s)' % ( funlabel, ','.join(varlabel), (' %s('%prefix) if prefix else ' '  , complex2str(this.coeffs[0],precision=precision) if isinstance(this.coeffs[0],complex) else '%1.4e'%this.coeffs[0], '*x'.join( list(this.basis_symbols[0]) ) )
        for k,b in enumerate(this.coeffs[1:]):
            model_str += ' + %s*(x%s)' % ( complex2str(b,precision=precision) if isinstance(b,complex) else '%1.4e'%b , '*x'.join( list(this.basis_symbols[k+1]) ) )

        # Correct for a lingering multiply sign
        model_str = model_str.replace('(*','(')

        # Correct for the constant term not being an explicit function of a domain variable
        model_str = model_str.replace('*(xK)','')

        # if there is a prefix, then close the automatic ()
        model_str += ' )' if prefix else ''

        #
        model_str += postfix

        # Replace variable labels with input
        if not ( varlabel is None ):
            for k in range(this.domain_dimension):
                model_str = model_str.replace( 'x%i'%k, varlabel[k] )

        return model_str

    # Create model string for latex output
    def __str_latex__(this,labels=None,precision=8):

        # Import useful things
        from numpy import mod

        #
        term_split = 3
        split_state = False

        # Handle the labels input
        latex_labels = ( this.labels['latex'] if 'latex' in this.labels else None ) if labels is None else labels

        # Extract desired labels and handle defaults
        funlabel = r'f(\vec{x})' if latex_labels is None else latex_labels[0]
        varlabel = [ 'x%i'%k for k in range(this.domain_dimension) ] if latex_labels is None else latex_labels[1]
        prefix = '' if latex_labels is None else latex_labels[2]
        if varlabel is None:
            varlabel = [ r'x_%i'%k for k in range(this.domain_dimension) ]
        elif len(varlabel) != this.domain_dimension:
            error( 'Number of variable labels, %i, is not equal to the number of domain dimensions found, %i.'%( len(varlabel), M ) , 'mvpolyfit' )

        # Create a simple string representation of the fit
        latex_str = r'%s  \; &= \; %s %s\,x%s%s' % ( funlabel,
                                                   (prefix+r' \, ( \,') if prefix else '',
                                                   complex2str(this.coeffs[0],
                                                   latex=True,precision=precision) if isinstance(this.coeffs[0],complex) else '%1.4e'%this.coeffs[0], r'\,x'.join( list(this.basis_symbols[0]) ), '' if len(this.coeffs)>1 else (r' \; )' if prefix else '') )
        for k,b in enumerate(this.coeffs[1:]):
            latex_str += r' \; + \; (%s)\,x%s%s' % ( complex2str(b,latex=True,precision=precision) if isinstance(b,complex) else '%1.4e'%b ,
                                                     r'\,x'.join( list(this.basis_symbols[k+1]) ),
                                                     (r' \; )' if prefix else '') if (k+1)==len(this.coeffs[1:]) else '' )
            #
            if ( not mod(k+2,term_split) ) and ( (k+1) < len(this.coeffs[1:]) ):
                latex_str += '\n  \\\\ \\nonumber\n & \quad '
                if not split_state:
                    split_state = not split_state

        # Correct for a lingering multiply sign
        latex_str = latex_str.replace('(\,','(')

        # Correct for the constant term not being an explicit function of a domain variable
        latex_str = latex_str.replace('\,xK','')

        # Replace variable labels with input
        for k in range(this.domain_dimension):
            latex_str = latex_str.replace( 'x%i'%k, varlabel[k] )

        # Replace repeated variable labels with power notation
        for pattern in varlabel:
            latex_str = rep2pwr( latex_str, pattern, r'\,'  )

        return latex_str

    # Write python formula ans save
    def save_as_python( this,
                        variable_labels=None,
                        function_label=None,
                        writedir=None,
                        verbose=False):
        ''' Given an optional write directory, save the fit formula as a python module.'''
        # Open file for writing
        return None

    # Plot 1D domain, 1D range
    def __plot2D__(this,
                   ax=None,
                   _map=None,
                   fit_xmin=None,   # Lower bound to evaluate fit domain
                   fit_xmax=None,   # Upper bound to evaluate fit domain
                   verbose=None):

        # Import useful things
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16

        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,legend,subplots,gca,sca,xlim
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import diff,linspace, meshgrid, amin, amax, ones, array

        # Handle optional axes input
        if ax is None:
            fig = figure()
            ax = fig.subplot(111,projection='3d')

        # Handle optinal map input: transform the range values for plotting use
        _map = (lambda x: x) if _map is None else _map

        #
        ax.plot( this.domain[:,0], _map(this.range), 'ok',label='Data', mfc='none', ms=8, alpha=1 )

        # Take this.domain over which to plot fit either from data or from inputs
        dx = ( max(this.domain[:,0])-min(this.domain[:,0]) ) * 0.1
        fit_xmin = min(this.domain[:,0])-dx if fit_xmin is None else fit_xmin
        fit_xmax = max(this.domain[:,0])+dx if fit_xmax is None else fit_xmax

        # NOTE that this could be replaced b a general system where a list of bounds is input

        #
        fitx = linspace( fit_xmin, fit_xmax, 2e2 )
        # fitx = linspace( min(domain[:,0])-dx, max(domain[:,0])+dx, 2e2 )
        ax.plot( fitx, _map(this.eval(fitx)), '-r', alpha=1,label='Fit', linewidth=1 )

        #
        xlim(lim(fitx))

        #
        xlabel( '$x_0$' )
        ylabel( '$f(x_0,x_1)$' )

    # Plot 2D domain, 1D Range
    def __plot3D__(this,ax=None,_map=None):

        # Import useful things
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16

        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,legend,subplots,gca,sca
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import diff,linspace, meshgrid, amin, amax, ones, array

        import matplotlib.tri as mtri
        from scipy.spatial import Delaunay

        # Handle optional axes input
        if ax is None:
            fig = figure()
            ax = fig.subplot(111,projection='3d')

        # Handle optinal map input: transform the range values for plotting use
        _map = (lambda x: x) if _map is None else _map

        #
        ax.view_init(60,30)

        # # Plot the fit evaluated on the domain
        # ax.scatter(domain[:,0],domain[:,1],_map(this.eval( domain )),marker='x',s=20,color='b',zorder=30)

        # # Try triangulating the input surface: Does this incerase visibility?
        # tri = Delaunay(domain[:,:2])
        # ax.plot_trisurf(domain[:,0], domain[:,1], _map(this.eval( domain )), triangles=tri.simplices, color='none', edgecolor='k' )

        # Setup grid points for model
        padf = 0.05
        dx = ( max(this.domain[:,0])-min(this.domain[:,0]) ) * padf
        dy = ( max(this.domain[:,1])-min(this.domain[:,1]) ) * padf
        fitx = linspace( min(this.domain[:,0])-dx, max(this.domain[:,0])+dx, 20 )
        fity = linspace( min(this.domain[:,1])-dy, max(this.domain[:,1])+dy, 20 )
        xx,yy = meshgrid( fitx, fity )
        fitdomain,_ = ndflatten( [xx,yy], yy )
        # Plot model on grid
        zz = this.eval( fitdomain ).reshape( xx.shape )
        ax.plot_wireframe(xx, yy, _map(zz), color='r', rstride=1, cstride=1,label='Model',zorder=1,alpha=0.8)

        # Plot the raw data points
        ax.scatter(this.domain[:,0],this.domain[:,1],_map(this.range),marker='o',color='k',label='Data',zorder=1, facecolors='k')

        xlabel( '$x_0$' )
        ylabel( '$x_1$' )
        ax.set_zlabel( '$f(x_0,x_1)$' )
        dz = (-amin(_map(this.range))+amax(_map(this.range)))*0.05
        ax.set_zlim( amin(_map(this.range))-dz, amax(_map(this.range))+dz )
        # title('$%s$'%this)
        legend(frameon=False)

    # Plot N-D domain, 1D Range
    def __plotND__(this,ax=None,_map=None):

        # Import useful things
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16

        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,legend,subplots,gca,sca
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import diff,linspace, meshgrid, amin, amax, ones, array

        # Handle optional axes input
        if ax is None:
            fig = figure()
            ax = fig.subplot(111,projection='3d')

        # Handle optinal map input: transform the range values for plotting use
        _map = (lambda x: x) if _map is None else _map

        #
        ax.view_init(60,30)

        # Plot the fit evaluated on the domain
        ax.scatter(this.domain[:,0],this.domain[:,1],_map(this.eval( this.domain )),marker='x',s=20,color='r')

        # # Setup grid points for model
        # padf = 0
        # dx = ( max(this.domain[:,0])-min(this.domain[:,0]) ) * padf
        # dy = ( max(this.domain[:,1])-min(this.domain[:,1]) ) * padf
        # fitx = linspace( min(this.domain[:,0])-dx, max(this.domain[:,0])+dx, 20 )
        # fity = linspace( min(this.domain[:,1])-dy, max(this.domain[:,1])+dy, 20 )
        # xx,yy = meshgrid( fitx, fity )
        # rawfitdomain = [xx,yy]
        # for k in range( 2,len(domain[0,:]-2) ):
        #     rawfitdomain += [ mean(this.domain[:,k])*ones( xx.shape, dtype=this.domain[:,k].dtype ) ]
        # fitdomain,_ = ndflatten( rawfitdomain, yy )
        # # Plot model on grid
        # zz = this.eval( fitdomain ).reshape( xx.shape )
        # ax.plot_wireframe(xx, yy, _map(zz), color='r', rstride=1, cstride=1,label='Model Slice',zorder=1,alpha=0.8)

        # Plot the raw data points
        ax.scatter(this.domain[:,0],this.domain[:,1],_map(this.range),marker='o',color='k',label='Data',zorder=1, facecolors='none')

        xlabel( '$x_0$' )
        ylabel( '$x_1$' )
        ax.set_zlabel( '$f(x_0,x_1)$' )
        dz = (-amin(_map(this.range))+amax(_map(this.range)))*0.05
        ax.set_zlim( amin(_map(this.range))-dz, amax(_map(this.range))+dz )
        # title('$%s$'%this)
        legend(frameon=False)

    # Plot residual histograms
    def __plotHist__(this,ax=None,_map=None,kind=None):

        # Import useful things
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16

        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,legend,subplots,gca,sca,xlim
        from numpy import diff,linspace,meshgrid,amin,amax,ones,array,angle,ones,sqrt,pi,mean
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import norm

        # Handle optinal map input: transform the range values for plotting use
        _map = (lambda x: x) if _map is None else _map

        # Handle optional axes input
        if ax is None:
            fig = figure()
            ax = fig.subplot(111,projection='3d')

        # Extract desired residuals
        res = this.bin['frac_real_res'] if kind in (None,'real') else ( this.bin['frac_amp_res'] if kind.lower() == 'amp' else this.bin['frac_pha_res'] )
        res = res.real

        # Extract desired normal fit
        mu,std = this.bin['frac_real_res_mu_std'] if kind in (None,'real') else ( this.bin['frac_amp_res_mu_std'] if kind.lower() == 'amp' else this.bin['frac_pha_res_mu_std'] )

        # Plot histogram
        n, bins, patches = ax.hist( res, max([len(res)/5,3]), facecolor=0.92*ones((3,)), alpha=1.0 )

        # Plot estimate normal distribution
        xmin,xmax=xlim()
        x = linspace( mu-5*std, mu+5*std, 2e2 )
        from matplotlib import mlab
        pdf =  norm.pdf( x, mu, std ) * sum(n) * (bins[1]-bins[0])
        # pdf =  norm.pdf( x, mu, std ) * len(res) * (bins[1]-bins[0])
        plot( x, pdf, 'r', label='Normal Approx.' )

        # Decorate plot
        title(r'$frmse = %1.4e$, $\langle res \rangle =%1.4e$'%(this.frmse,mean(res)))
        xlabel('Fractional Residaul Error')
        ylabel('Count in Bin')
        # legend( frameon=False )

    # Plot flattened ND data on index
    def __plot1D__(this,ax=None,_map=None):

        # Import useful things
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16

        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,\
                                      legend,subplots,gca,sca,xlim,text
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import diff,linspace, meshgrid, amin, amax, ones, array, arange

        # Handle optional axes input
        if ax is None:
            fig = figure()
            ax = fig.subplot(111)

        # Handle optinal map input: transform the range values for plotting use
        _map = (lambda x: x) if _map is None else _map

        #
        index = arange( len(this.range) )+1
        plot( index, _map(this.range), 'ok', mfc=0.6*array([1,1,1]), mec='k', alpha=0.95, ms=6 )
        plot( index, _map(this.eval(this.domain)), 'o', ms = 12, mfc='none', mec='r', alpha=0.95  )
        plot( index, _map(this.eval(this.domain)), 'x', ms = 9, mfc='none', mec='r', alpha=0.95  )
        ax.set_xlim( [-1,len(_map(this.range))+1] )
        dy = 0.05 * ( amax(_map(this.range)) - amin(_map(this.range)) )
        ax.set_ylim( array([amin(_map(this.range)),amax(_map(this.range))]) + dy*array([-1,1]) )


        for k in index:
            j = k-1
            text( k, _map(this.range[j]), str(j), ha='center', va='center', size=12, alpha=0.6 )
            if this.data_label is not None: print '[%i]>> %s' % (j,this.data_label[j])

        #
        xlabel('Domain Index')
        ylabel(r'$f( \vec{x} )$')

    # High level plotting function
    def plot(this,show=False,fit_xmin=None):

        from matplotlib.pyplot import plot,figure,title,xlabel,ylabel,legend,subplots,gca,sca,subplot,tight_layout
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import diff,linspace,meshgrid,amin,amax,array,arange,angle,unwrap,pi

        # Determine if range is complex; this effects plotting flow control
        range_is_complex = this.range.dtype == complex

        # Setup the dimnesions of the plot
        spdim = '23' if range_is_complex else '13'

        # Initiate the figure
        fig = figure( figsize=2.5*( array([7,4]) if range_is_complex else array([7,2]) ) )
        subplot( spdim+'1' )
        fig.patch.set_facecolor('white')
        if range_is_complex:
            tight_layout(w_pad=5,h_pad=5,pad=4)
        else:
            tight_layout(w_pad=9,h_pad=0,pad=4)

        # Calculate phase of somplex array such that there are only positive values; used for complex valued ranges
        def get_phase(s):
            x = anglep(s)
            x = sunwrap(x)
            # while amin(x)<0: x += 2*pi
            return x

        # --------------------------------------------- #
        # Plot "Normal Space" represenation
        # --------------------------------------------- #
        if this.domain.shape[-1] == 2:
            if range_is_complex:
                ax = subplot(spdim+'1',projection='3d')
                this.__plot3D__( ax, _map = lambda s: abs(s) )
                ax = subplot(spdim+'4',projection='3d')
                this.__plot3D__( ax, _map = lambda s: get_phase(s) )
            else:
                ax = subplot(spdim+'1',projection='3d')
                this.__plot3D__( ax )
        elif this.domain.shape[-1] == 1:
            if range_is_complex:
                ax = subplot(spdim+'1')
                this.__plot2D__( ax, _map = lambda s: abs(s), fit_xmin=fit_xmin )
                ax = subplot(spdim+'4')
                this.__plot2D__( ax, _map = lambda s: get_phase(s), fit_xmin=fit_xmin )
            else:
                ax = subplot(spdim+'1')
                this.__plot2D__( ax )
        else:
            if range_is_complex:
                ax = subplot(spdim+'1',projection='3d')
                this.__plotND__( ax, _map = lambda s: abs(s))
                ax = subplot(spdim+'4',projection='3d')
                this.__plotND__( ax, _map = lambda s: get_phase(s) )
            else:
                ax = subplot(spdim+'1',projection='3d')
                this.__plotND__( ax )

            print '2/3d plotting is not enabled as the map is %id'% (1+this.domain.shape[-1])

        # --------------------------------------------- #
        # Plot Histogram of Fractional Residuals
        # --------------------------------------------- #
        if range_is_complex:
            ax = subplot(spdim+'3')
            this.__plotHist__(ax,kind='amp')
            # this.__plotHist__(ax,_map = lambda s: abs(s))
            ax.yaxis.set_label_position('right')
            ax = subplot(spdim+'6')
            this.__plotHist__(ax,kind='phase')
            # this.__plotHist__(ax,_map = lambda s: get_phase(s))
            ax.yaxis.set_label_position('right')
        else:
            ax = subplot(spdim+'3')
            this.__plotHist__(ax,kind='real')
            ax.yaxis.set_label_position('right')

        # --------------------------------------------- #
        # Plot 1D data points
        # --------------------------------------------- #
        if range_is_complex:
            ax = subplot(spdim+'2')
            this.__plot1D__(ax,_map = lambda s: abs(s))
            # ax.yaxis.set_label_position('right')
            ax = subplot(spdim+'5')
            this.__plot1D__(ax,_map = lambda s: get_phase(s))
            # ax.yaxis.set_label_position('right')
        else:
            ax = subplot(spdim+'2')
            this.__plot1D__(ax)
            # ax.yaxis.set_label_position('right')

        #
        if show:
            from matplotlib.pyplot import show,draw
            draw();show()

        #
        return fig

    # Validate inputs and store important low-level fields
    def __validate_inputs__(this,domain,scalar_range,basis_symbols,labels,range_map,data_label):

        # Import usefuls
        from numpy import ndarray,isfinite,complex256,float128,double

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        ''' Validate the domain: '''
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        # * This input should be an NxM ndarray, where N is thew number of dimensions in the domain, and M is the number of samples along the given dimensions.
        # * NOTE the M>=N will be inforced so that the regression problem is well posed.

        # Type check
        if not isinstance(domain,ndarray):
            msg = 'domain input must be numpy array'
            error(msg,'mvpolyfit')

        # Check for 1D domain; reshape for consistent indexing below
        if len(domain.shape)==1: domain = domain.reshape(len(domain),1)

        # Validate the range map; set to identitiy op if none given
        range_map = {'forward':IXY, 'backward':IXY} if range_map is None else range_map
        #
        if not isinstance( range_map, dict ):
            error('range_map input must be dictionary with keys "forward" and "backward" which are functions of the input range to be applied to the range (forward) before fitting, and (backward) after fitting.','mvpolyfit')
        else:
            required_keys = ['forward','backward']
            for k in required_keys:
                if not ( k in range_map ):
                    error('required key %s not found in range_map input'%k)

        # Check for nonfinite values in range
        mask = isfinite( range_map['forward'](domain,scalar_range) )
        if sum(mask) != len(scalar_range):
            scalar_range = scalar_range[mask]
            domain = domain[mask,:]
            msg = 'Non-finite values detected in scalar_range or its forward mapped representation. The offending values will be masked away before fitting.'
            warning(msg,'mvpolyfit')

        # Determine the number of domain dimensions:
        if len(domain.shape)==1: domain = domain.reshape(len(domain),1)
        this.domain_dimension = domain.shape[-1]

        # Dimension check for well-posedness
        N,M = domain.shape
        if N<M:
            msg = 'number of samples (%i) found to be less than number of dimensions in domain (%i); this means that the problem as posed is underconstrained; please add more points to your sample space' % (N,M)
            error(msg,'mnpolyfit')

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        ''' Validate the range '''
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        # The range must be an iterable of length M
        if M != len(scalar_range):
            msg = 'the sample range must be an iterable of length M; length of %i found insted'%len(scalar_range)
        # Members of the range must all be floats
        for r in scalar_range:
            if not isinstance(r,(float,int,complex,complex256,double,float128)):
                msg = 'all member of the input range must be floats; %s found instead'%r
                error(msg,'mvpolyfit')

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        ''' Validate the basis symbols '''
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        # The basis symbols should be stored in an iterable, and each should be a string
        try:
            for k in basis_symbols:
                _ = None
        except:
            msg = 'input basis_symbols must be iterable and contain strings '
            error( msg, 'mvpolyfit' )
        # They must all be strings
        for k in basis_symbols:
            if not isinstance(k,str):
                msg = 'all basis symbols must be str'
                error(msg,'mvpolyfit')

        # TODO: Check content of strings to ensure valid format -- can it be inderstood by mvpolyfit?

        # Cleanly format basis symbols, and remove possible duplicates, or linearly dependent symbols
        for s in basis_symbols:
            s = sorted(s)
        # Get unique values
        basis_symbols = list( set(basis_symbols) )
        basis_symbols = sorted( basis_symbols, key=lambda k: len(k) )
        if 'K' in basis_symbols:
            basis_symbols = ['K'] + [ s for s in basis_symbols if s!='K' ]

        # Store low level inputs (More items are stored later)
        this.domain = domain
        this.range = scalar_range
        this.basis_symbols = basis_symbols
        this.labels = {} if labels is None else labels
        this.range_map = range_map

        #
        this.data_label = None

# High level Positive greedy algorithm ("Positive" becuase points are added greedily)
# NOTE that this function answers: Which model basis elements need to be added in order to optimize efffectualness?
class pgreedy:
    # Constructor
    def __init__( this,
                  bulk,                      # Symbols to be greedily selected to minimize estimator
                  action,                    # real_valued_estimator,answer_object = action(subset_of_bulk)
                  plot = False,
                  show = False,
                  plotfun = None,            # plotfun(answer_object) produces an informative plot
                  fitatol = 1e-3,            # Absolute tolerance to be used for change in estimator
                  initial_boundary = None,   # Initializing boundary (optional initial guess for model content)
                  verbose = False ):

        # Import stuff
        from numpy import arange,inf,array,ndarray
        from copy import deepcopy as copy

        # Let the people know
        if verbose: print '\n############################################\n# Applying a Positive Greedy Algorithm\n############################################\n'

        # The positive greedy process will move information from the bulk to the boundary based upon interactions with the input data
        boundary= [] if initial_boundary is None else initial_boundary
        # Ensure that starting bulk and boundary are disjoint
        bulk = list( set(bulk).difference(set(initial_boundary)) ) if not ( initial_boundary is None ) else bulk

        # Positive greedy process
        est_list = []
        this.estimator_list = est_list
        last_min_est = inf
        done,ans = False,None
        itercount = 0
        while not done:

            # Try to add a single term, and make note of the one that results in the smallest (i.e. best) fractional root-mean-square-error (frmse)
            min_est,min_term = last_min_est,None
            for k,term in enumerate(bulk):
                trial_boundary = boundary + [term]
                est,foo = action(trial_boundary)
                if est < min_est:
                    min_index = k
                    min_est = est
                    min_ans = copy(foo)
                    min_term = term

            # Measure the stopping condition
            d_est = min_est - last_min_est
            the_estimator_hasnt_changed_a_lot = abs( d_est ) < fitatol
            done = the_estimator_hasnt_changed_a_lot
            if done: state_msg = '>> Exiting because |min_est-last_min_est| = |%f - %f| = |%f| < %f.\n>> NOTE that the result of the previous iteration will be kept.' % (min_est,last_min_est,d_est,fitatol)
            done = done or ( len(bulk) == 0 )
            if (len(bulk) == 0): state_msg = '>> Exiting because there are no more symbols to draw from.\n>> NOTE that the result of the previous iteration will be kept.'

            # Tidy up before looping back
            if not done:
                # Keep track of the frmse
                est_list.append( min_est )
                # Move the current best symbol from the bulk to the boundary so that it is used on the next iteration
                boundary.append( bulk[min_index] )
                del bulk[ min_index ]
                # Store current estimator value
                last_min_est = min_est
                # Store the current optimal answer object
                ans = min_ans

            # If a plotting function is given, plot per iteration
            if plotfun and plot: plotfun(ans)

            #
            if verbose:
                itercount += 1
                print '\n%sIteration #%i (Positive Greedy)\n%s' % ( 'Final ' if done else '', itercount, 12*'---' )
                print '>> The current estimator value is %1.4e' % min_est
                print '>> %s was added to the boundary' % ( min_term if isinstance(min_term,(list,str,ndarray)) else (list(min_term) if not (min_term is None) else 'Nothing' ) )
                print '>> This caused the estimator value to change by %f' % d_est
                print '>> The current boundary is %s' % boundary
                if done: print state_msg

        #
        this.boundary = boundary
        this.answer =  ans
        this.estimator_list = est_list

        # Plot stuff
        if plot: this.plot(show=show)
    # Plotting function
    def plot(this,show=False):
        from matplotlib.pyplot import figure,plot,axes,gca,xlabel,ylabel,title
        from numpy import array
        fig = figure( figsize=2*array([4,3]) )
        gca().set_yscale('log')
        plot( range(1,len(this.estimator_list)+1), this.estimator_list, '-o' )
        xlabel('Iteration #')
        ylabel('Estimator Value')
        title('Convergence: Positive Greedy')
        if show:
            from matplotlib.pyplot import show
            show()

# High level Negative greedy algorithm ("Negative" becuase points are removed greedily)
# NOTE that this function answers: Which model basis elements can be removed without significantly impacting effectualness?
class ngreedy:
    # Constructor
    def __init__( this,
                  boundary,          # Symbol list to be greedily shrank to minimize the estimator
                  action,            # real_valued_estimator,answer_object = action(subset_of_boundary)
                  plot = False,
                  show = False,
                  plotfun = None,    # plotfun(answer_object) produces an informative plot
                  ref_est_list = None,
                  permanent = None,
                  verbose = False ):

        # Import stuff
        from numpy import arange,inf,array,ndarray
        from copy import deepcopy as copy

        # Let the people know
        if verbose: print '\n############################################\n# Applying a Negative Greedy Algorithm\n############################################\n'

        # The negative greedy process will move information from the boundary ("to the bulk") based upon interactions with the input data
        # bulk = [] # NOTE that the information about the "bulk" is not used, and so it is not stored

        # NOTE that the value below could be input for efficiency
        initial_estimator_value,ans = action(boundary)

        #
        permanent = [] if permanent is None else permanent

        # Positive greedy process
        est_list = [initial_estimator_value]
        last_min_est = initial_estimator_value
        fitatol = 0.0125 * initial_estimator_value if ref_est_list is None else 0.99*(abs( ref_est_list[-2] - ref_est_list[-1] ) if len(ref_est_list)>1 else 0.005*(max(ref_est_list)-min(min(ref_est_list),initial_estimator_value) ) )
        # NOTE the dependence of "done" on the length of boundary: if the model only contains one dictionary element, then do not proceed.
        done = False or (len(boundary)<=1)
        itercount = 0
        while not done:

            #
            if verbose:
                itercount += 1
                print '\n%sIteration #%i (Negative Greedy)\n%s' % ( '(Final) ' if done else '', itercount, 12*'---' )

            # Try to add a single term, and make note of the one that results in the smallest (i.e. best) fractional root-mean-square-error (frmse)
            min_est,min_term = inf,None
            for k,term in enumerate(boundary):
                if not term in permanent:
                    trial_boundary = copy(boundary)
                    trial_boundary.remove( term )
                    est,foo = action(trial_boundary)
                    if est < min_est:
                        min_index = k
                        min_est = est
                        min_term = term
                        min_ans = copy(foo)

            # Measure the stopping condition
            # ---
            # * The estimator for the stopping condition
            d_est = min_est - initial_estimator_value
            # * Stop if the estimator is greater than the tolerance AND it's greater than zero; thus allow arbitrary improvements to the fit while limiting how bad it can get
            done = (abs(d_est) > fitatol) and (d_est>0)
            if done: state_msg = '>> Exiting because |min_est-initial_estimator_value| = |%f-%f| = |%f| > %f.\n>> NOTE that the result of the previous iteration will be kept.' % (min_est,initial_estimator_value,d_est,fitatol)
            # * Stop if there are no more symbols to remove after the next iteration
            done = done or ( len(trial_boundary) == 1 )
            if (len(trial_boundary) == 1): state_msg = '>> Exiting because there are no more symbols to draw from.\n>> NOTE that the result of the previous iteration will be kept, unless this is the first iteration.'

            # Tidy up before looping back
            if not done:
                # Keep track of the frmse
                est_list.append( min_est )
                # Refine the current boundary by removing the optimal member
                # NOTE that we don't use list.remove here becuase we do not wish to delete the item from memory
                boundary = [ item for k,item in enumerate(boundary) if k!=min_index  ]
                # Store current estimator value
                last_min_est = min_est
                # Store the current optimal answer object
                ans = min_ans
            else:
                # NOTE that this step is needed for actions that affect the internal state of the ans object (namely its pointer like behavior)
                final_estimator_value,ans = action(boundary)

            # If a plotting function is given, plot per iteration
            if plotfun and plot: plotfun(ans)

            #
            if verbose:
                print '>> min_estimator = %1.4e' % min_est
                if not done:
                    print '>> "%s" was removed from the boundary.' % ( min_term if isinstance(min_term,(list,str,ndarray)) else list(min_term) )
                    print '>> As a result, the estimator value changed by %f. The tolerance for this change is %f' % (d_est,fitatol)
                print '>> The current boundary = %s' % boundary
                if done: print state_msg

        #
        this.boundary = boundary
        this.answer =  ans
        this.estimator_list = est_list
        this.reference_estimator_list = ref_est_list

        # Plot stuff
        if plot: this.plot(show=show)

    # Plotting function
    def plot(this,show=False):
        from matplotlib.pyplot import figure,plot,axes,gca,xlabel,ylabel,title,legend
        from numpy import array
        fig = figure( figsize=2*array([4,3]) )
        gca().set_yscale('log')
        offset = 1 if this.reference_estimator_list is None else len(this.reference_estimator_list)
        if this.reference_estimator_list:
            x = range(1,len(this.reference_estimator_list)+1)
            plot( x, this.reference_estimator_list, '-ob', label = 'Reference Values' )
        plot( range(offset,len(this.estimator_list)+offset), this.estimator_list, '-vr', label='Negative Greedy Steps' )
        xlabel('Iteration #')
        ylabel('Estimator Value')
        title('Convergence: Negative Greedy')
        legend( frameon=False )
        if show:
            from matplotlib.pyplot import show
            show()


# Adaptive (Greedy) Multivariate polynomial fitting
# NOTE that this version optimizes over degree
def gmvpfit( domain,              # The N-D domain over which a scalar will be modeled: list of vector coordinates
             scalar_range,        # The scalar range to model on the domain: 1d iterable of range values corresponding to domain entries
             maxdeg = 8,          # polynomial degrees of at most this order in domain variables will be considered: e.g. x*y and y*y are both degree 2. NOTE that the effective degree will be optimized over: increased incrementally until the model becomes less optimal.
             mindeg = 1,          # minimum degree to consider for tempering
             plot = False,        # Toggle plotting
             show = False,        # Toggle for showing plots as they are created
             fitatol  = 1e-3,     # Tolerance in fractional chance in estimator
             permanent_symbols = None,  # If given, these symbols (compatible with mvpfit) will always be used in the final fit
             initial_boundary = None,   # Seed boundary for positive greedy process
             apply_negative = True,     # Toggle for the application of a negtative greedy algorithm following the positive one. This is set true by default to handle overmodelling.
             temper = True,         # Toggle for applying degree tempering, where the positive greedy process is forward optimized over max polynomial degree
             range_map = None,      # Operation to apply to range before fitting, and inverse. EXAMPLE: range_map = { 'forward': lambda x: external_variable*x, 'backward': lambda y: y/external_variable }
             verbose = False,       # Let the people know
             **kwargs ):
    '''Adaptive (Greedy) Multivariate polynomial fitting: general domain dimension, but range must be 1D (possibly complex).'''
    # Import stuff
    from itertools import combinations as combo
    from numpy import arange,isfinite,inf,amin
    from copy import deepcopy as copy

    # NOTE that this algorithm does not try to optimize over maximum degree. This is necessary in some cases as an overmodeling bias for maxdegree too large can overwhelm the fitatol threshold before a decent fit is found.

    # Determine the number of domain dimensions:
    if len(domain.shape)==1: domain = domain.reshape(len(domain),1)
    domain_dimension = domain.shape[-1]

    # Let the people know about the apply_negative input
    if verbose and (not apply_negative):
        msg = 'Found %s. The negative greedy step will not be applied. Please consider turing the option on (its true by default) to investigate whether the result of the positive greedy algorithm is over-modeled.' % cyan("apply_negative = True")
        alert(msg)

    # Validate the range map; set to identitiy op if none given
    range_map = {'forward':IXY, 'backward':IXY} if range_map is None else range_map
    #
    if not isinstance( range_map, dict ):
        error('range_map input must be dictionary with keys "forward" and "backward" which are functions of the input range to be applied to the range (forward) before fitting, and (backward) after fitting.','mvpolyfit')
    else:
        required_keys = ['forward','backward']
        for k in required_keys:
            if not ( k in range_map ):
                error('required key %s not found in range_map input'%k)

    # Check for nonfinite values in range
    mask = isfinite( range_map['forward'](domain,scalar_range) )
    if sum(mask) != len(scalar_range):
        scalar_range = scalar_range[mask]
        domain = domain[mask,:]
        msg = 'Non-finite values detected in scalar_range. They will be masked away before fitting.'
        warning(msg,'gmvpfit')

    # Prepare inputs for generalized positive greedy algorithm
    def action( trial_boundary ):
        foo = mvpolyfit(domain,scalar_range,trial_boundary,range_map=range_map,**kwargs)
        estimator = foo.frmse
        return estimator,foo
    def mvplotfun( foo ): foo.plot(show=show)

    # Create a lexicon of symbols to consider for model learning
    # NOTE the manual adding of a constant term symbol, "K"
    maxbulk = mvsyms( domain_dimension, maxdeg )

    # Define the space of all possible degrees bounded above by maxdeg
    degree_space = range(mindeg,maxdeg+1) if temper else [maxdeg]

    # Increment through degrees
    # NOTE that this for-loop is not directly castable as a greedy algorithm because the dictionary set is of size 1 (i.e. one degree at a time), and corresponds to a heuristically structured family
    last_min_est = inf
    for deg in degree_space:

        # Determine the bulk for this degree value
        bulk = [ s for s in maxbulk if len(s)<=deg ] if deg>0 else mvsyms(domain_dimension,deg)

        # Let the people know
        if verbose:
            msg = 'Now working deg = %i' % deg
            alert(msg)

        # Apply a positive greedy process to estimate the optimal model's symbol content (i.e. "boundary", the greedy process moves symbols from the bulk to the boundary)
        A_ = pgreedy( bulk, action, fitatol=fitatol, initial_boundary=initial_boundary, verbose = verbose if (not temper) else False  )

        # Meter the stopping condition
        d_est = amin(A_.estimator_list)-last_min_est
        the_estimator_is_worse = amin(A_.estimator_list) > last_min_est
        temper_fitatol = fitatol
        the_estimator_hasnt_changed_a_lot = abs(d_est)<temper_fitatol and d_est!=0
        we_are_at_maxdeg = deg == maxdeg
        done =    the_estimator_is_worse \
               or the_estimator_hasnt_changed_a_lot \
               or we_are_at_maxdeg

        # Let the people know
        if verbose:

            if the_estimator_is_worse:
                exit_msg = 'the estimator was made worse by using this max degree'
            elif the_estimator_hasnt_changed_a_lot:
                exit_msg = 'the estimator has changed by |%f| < %f' % (d_est,temper_fitatol)
            elif we_are_at_maxdeg:
                exit_msg = 'we are at the maximum degree of %i' % deg

            print '&& The estimator has changed by %f' % d_est
            print '&& '+ ('Degree tempering will continue.' if not done else 'Degree tempering has completed becuase %s. The results of the last iteration wil be kept.'%exit_msg)

        #
        if (not done) or not ('A' in locals()):
            A = A_
            boundary,est_list = A.boundary, A.estimator_list
            last_min_est = est_list[-1]
            if verbose:
                print '&& The current boundary is %s' % boundary
                print '&& The current estimator value is %f\n' % est_list[-1]
        else:
            if plot:
                A.answer.plot()
                A.plot()
            if verbose:
                print '&& The Final boundary is %s' % boundary
                print '&& The Final estimator value is %f\n' % est_list[-1]
            break



    if verbose: print '\n%s\n# Degree Tempered Positive Greedy Solution:\n%s\n'%(10*'====',10*'====')+str(A.answer)

    # Apply a negative greedy process to futher refine the symbol content
    B = ngreedy( boundary, action, plot = plot, show=show, plotfun = mvplotfun, verbose = verbose, ref_est_list = est_list, permanent = permanent_symbols ) if apply_negative==True else None

    if apply_negative and verbose: print '\n%s\n# Negative Greedy Solution:\n%s\n'%(10*'====',10*'====')+str(B.answer)

    #
    ans = B.answer if apply_negative else A.answer

    # Store the greedy results in an ad-hoc way
    ans.bin['pgreedy_result'] = A
    ans.bin['ngreedy_result'] = B if apply_negative else None

    #
    if verbose: print '\nFit Information:\n%s\n'%(10*'----')+str(ans)

    #
    return ans

# Convert complex number to string in exponential form
def complex2str( x, precision=None, latex=False ):
    '''Convert complex number to string in exponential form '''
    # Import useful things
    from numpy import ndarray,angle,abs,pi
    # Handle optional precision input
    precision = 8 if precision is None else precision
    precision = -precision if precision<0 else precision
    # Create function to convert single number to string
    def c2s(y):

        # Check type
        if not isinstance(y,complex):
            msg = 'input must be complex number or numpy array of complex datatype'

        #
        handle_as_real = abs(y.imag) < (10**(-precision))

        if handle_as_real:
            #
            fmt = '%s1.%if'%(r'%',precision)
            ans_ = '%s' % ( fmt ) % y.real
        else:

            # Compute amplitude and phase
            amp,phase = abs(y),angle(y)
            # Write phase as positive number
            phase = phase+2*pi if phase<0 else phase
            # Create string
            fmt = '%s1.%if'%(r'%',precision)
            ans_ = '%s*%s%s%s' % (fmt, 'e^{' if latex else 'exp(' ,fmt, 'i}' if latex else 'j)') % (amp,phase)
            if latex: ans_ = ans_.replace('*',r'\,')

        return ans_

    # Create the final string representation
    if isinstance(x,(list,ndarray,tuple)):
        s = []
        for c in x:
            s += [c2s(c)]
        ans = ('\,+\,' if latex else ' + ').join(s)
    else:
        ans = c2s(x)
    # Return the answer
    return ans

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

# Given a string with repeated symbols, convert the repetitions to power notation
def rep2pwr( string,        # String to be processed
             pattern,       # Pattern to look for
             delimiter,     # Delimeter
             latex=True,
             pwrfun=None ): # Operation to apply to repetitios of the pattern, eg pwrfun = lambda pattern,N: '%s^%i'%(pattern,N)
    '''
    Given a string with repeated symbols, convert the repetitions to power notation.

    Example:

    >> a = '*x*x*x'
    >> enpower(a,'*x')

        x^{3}

    '''

    # Handle the power-function input
    pwrfun = (lambda pattern,N: '{%s}^{%i}'%(pattern,N)) if pwrfun is None else pwrfun

    # Find the maximum number of repetitions by noting it is bound above by the total number of parrtern instances
    maxdeg = len(string.split(pattern)) # Technically there should be a -1 here, but let's leave it out to simplify things later on, and avoid buggy behavior

    # Copy the input
    ans = str(string) # .replace(' ','')

    # Look for repetitions
    for deg in range( maxdeg, 1, -1 ):
        # Create a repeated pattern
        reppat = delimiter.join( [ pattern for k in range(deg) ] )
        # Look for the pattern, and replace it with the power representation
        ans = ans.replace( reppat, pwrfun(pattern,deg) )

    # Return the answer
    return ans


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


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Given a 1D array, determine the set of N lines that are optimally representative  #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Hey, here's a function that approximates any 1d curve as a series of lines
def romline(  domain,           # Domain of Map
              range_,           # Range of Map
              N,                # Number of Lines to keep for final linear interpolator
              positive=True,   # Toggle to use positive greedy algorithm ( where rom points are added rather than removed )
              verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_
    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/( R1 if abs(R1)!=0 else 1 )

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
            # print trial_sigma
            if trial_sigma < min_sigma:
                knots,rom,min_sigma = trial_knots,trial_rom,trial_sigma

    #
    # print min_sigma

    return knots,rom


# Hey, here's a function related to romline
def positive_romline(   domain,           # Domain of Map
                        range_,           # Range of Map
                        N,                # Number of Lines to keep for final linear interpolator
                        seed = None,      # First point in domain (index) to use
                        verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    #
    weights = (r-amin(r)) / amax( r-amin(r) )
    weights = ones( d.size )

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
    min_space = list(space)
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
            sigma = err( weights * (linterp( trial_domain, trial_range )( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
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


''' Code from nrutils '''



# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# NOTE that uncommenting the line below may cause errors in OSX install relating to fonts
# rc('text', usetex=True)

def linenum():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# # Class for basic print manipulation
# class print_format:
#    magenta = '\033[1;35m'
#    cyan = '\033[0;36m'
#    darkcyan = '\033[0;36m'
#    blue = '\033[0;34m'
#    # green = '\033[0;32m'
#    green = '\033[92m'
#    yellow = '\033[0;33m'
#    red = '\033[31m'
#    bold = '\033[1m'
#    grey = gray = '\033[1;30m'
#    ul = '\033[4m'
#    end = '\x1b[0m'
#    hlb = '\033[5;30;42m'
#    underline = '\033[4m'
#
# # Function that uses the print_format class to make tag text for bold printing
# def bold(string):
#     return print_format.bold + string + print_format.end
# def red(string):
#     return print_format.red + string + print_format.end
# def green(string):
#     return print_format.green + string + print_format.end
# def magenta(string):
#     return print_format.magenta + string + print_format.end
# def blue(string):
#     return print_format.blue + string + print_format.end
# def grey(string):
#     return print_format.grey + string + print_format.end
# def yellow(string):
#     return print_format.yellow + string + print_format.end
# def orange(string):
#     return print_format.orange + string + print_format.end
# def cyan(string):
#     return print_format.cyan + string + print_format.end
# def darkcyan(string):
#     return print_format.darkcyan + string + print_format.end
# def hlblack(string):
#     return print_format.hlb + string + print_format.end
# def textul(string):
#     return print_format.underline + string + print_format.end
#
# # Return name of calling function
# def thisfun():
#     import inspect
#     return inspect.stack()[2][3]

# #
# def parent(path):
#     '''
#     Simple wrapper for getting absolute parent directory
#     '''
#     return os.path.abspath(os.path.join(path, os.pardir))+'/'


# # Make "mkdir" function for directories
# def mkdir(dir_,rm=False,verbose=False):
#     # Import useful things
#     import os
#     import shutil
#     # Expand user if needed
#     dir_ = os.path.expanduser(dir_)
#     # Delete the directory if desired and if it already exists
#     if os.path.exists(dir_) and (rm is True):
#         if verbose:
#             alert('Directory at "%s" already exists %s.'%(magenta(dir_),red('and will be removed')),'mkdir')
#         shutil.rmtree(dir_,ignore_errors=True)
#     # Check for directory existence; make if needed.
#     if not os.path.exists(dir_):
#         os.makedirs(dir_)
#         if verbose:
#             alert('Directory at "%s" does not yet exist %s.'%(magenta(dir_),green('and will be created')),'mkdir')
#     # Return status
#     return os.path.exists(dir_)
#

# Function that returns true if for string contains l assignment that is less than l_max
def l_test(string,l_max):
    '''
    Function that returns true if for string contains l assignment that is <= l_max:
    score = ltest('Ylm_l3_m4_stuff.asc',3)
          = True
    score = ltest('Ylm_l3_m4_stuff.asc',5)
          = True
    score = ltest('Ylm_l6_m4_stuff.asc',2)
          = False
    '''
    # break string into bits by l
    score = False
    for bit in string.split('l'):
        if bit[0].isdigit():
            score = score or int( bit[0] )<= l_max

    # return output
    return score

#
def h5tofiles( h5_path, save_dir, file_filter= lambda s: True, cleanup = False, prefix = '' ):
    '''
    Function that takes in h5 file location, and and writes acceptable contents to files using groups as directories.
    ~ lll2'14
    '''

    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    #
    def group_to_files( group, work_dir ):
        '''
        Recurssive fucntion to make folder trees from h5 groups and files.
        ~ lll2'14
        '''

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        if type(group) is h5py._hl.group.Group or \
           type(group) is h5py._hl.files.File:
            # make a directory with the group name
            this_dir = work_dir + group.name.split('.')[0]
            if this_dir[-1] is not '/': this_dir = this_dir + '/'
            mkdir( this_dir )
            #
            for key in group.keys():
                #
                if type(group[key]) is h5py._hl.group.Group or \
                   type(group[key]) is h5py._hl.files.File:
                    #
                    group_to_files( group[key], this_dir )
                elif type(group[key]) is h5py._hl.dataset.Dataset:
                    #
                    data_file_name = prefix + key.split('.')[0]+'.asc'
                    if file_filter( data_file_name ):
                        #
                        data_file_path = this_dir + data_file_name
                        #
                        data = numpy.zeros( group[key].shape )
                        group[key].read_direct(data)
                        #
                        print( '[%s]>> ' % thisfun + bold('Writing') + ': "%s"'% data_file_path)
                        numpy.savetxt( data_file_path, data, delimiter="  ", fmt="%20.8e")
                else:
                    #
                    raise NameError('Unhandled object type: %s' % type(group[key]))
        else:
            #
            raise NameError('Input must be of the class "h5py._hl.group.Group".')

    #
    if os.path.isfile( h5_path ):

        # Open the file
        h5_file = h5py.File(h5_path,'r')

        # Begin pasing each key, and use group to recursively make folder trees
        for key in h5_file.keys():

            # reset output directory
            this_dir = save_dir

            # extract reference object with h5 file
            ref = h5_file[ key ]

            # If the key is a group
            if type(ref) is h5py._hl.group.Group:

                #
                group_to_files( ref, this_dir )


            else: # Else, if it's a writable object

                print('[%s]>> type(%s) = %s' % (thisfun,key,type(ref)) )

        # If the cleanup option is true, delete the original h5 file
        if cleanup:
            #
            print('[%s]>> Removing the original h5 file at: "%s"' % (thisfun,h5_path) )
            os.remove(h5_path)

    else:

        # Raise Error
        raise NameError('No file at "%s".' % h5_path)

#
def replace_line(file_path, pattern, substitute, **kwargs):
    '''
    Function started from: https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python.

    This function replaces an ENTIRE line, rather than a string in-line.

    ~ ll2'14
    '''

    #
    from tempfile import mkstemp
    from shutil import move
    # Get the string for this function name
    thisfun = inspect.stack()[0][3]

    # Look for verbose key
    keys = ('verbose','verb')
    VERB = parsin( keys, kwargs )
    if VERB:
        print('[%s]>> VERBOSE mode on.' % thisfun)

    #
    if substitute[-1] is not '\n':
        substitute = substitute + '\n'

    # If the file exists
    if os.path.isfile(file_path):
        #
        if VERB:
            print( '[%s]>> Found "%s"' % (thisfun,file_path) )
        # Create temp file
        fh, abs_path = mkstemp()
        if VERB: print( '[%s]>> Temporary file created at "%s"' % (thisfun,abs_path) )
        new_file = open(abs_path,'w')
        old_file = open(file_path)
        for line in old_file:
            pattern_found = line.find(pattern) != -1
            if pattern_found:
                if VERB:
                    print( '[%s]>> Found pattern "%s" in line:\n\t"%s"' % (thisfun,pattern,line) )
                new_file.write(substitute)
                if VERB:
                    print( '[%s]>> Line replaced with:\n\t"%s"' % (thisfun,substitute) )
            else:
                new_file.write(line)
        # Close temp file
        new_file.close()
        os.close(fh)
        old_file.close()
        # Remove original file
        os.remove(file_path)
        # Move new file
        move(abs_path, file_path)
        # NOTE that the temporary file is automatically removed
        if VERB: print( '[%s]>> Replacing original file with the temporary file.' % (thisfun) )
    else:
        #
        if VERB:
            print( '[%s]>> File not found at "%s"' % (thisfun,file_path) )
        if VERB:
            print( '[%s]>> Creating new file at "%s"' % (thisfun,file_path) )
        #
        file = open( file_path, 'w' )
        if substitute[-1]!='\n':
            substitute = substitute + '\n'
        #
        if VERB:
            print( '[%s]>> Writing "%s"' % (thisfun,substitute) )
        #
        file.write(substitute)
        file.close()
    #
    if VERB:
        print('[%s] All done!',thisfun)

# Function that returns randome strings of desired length and component of the desired set of characters
def rand_str(size=2**4, characters=string.ascii_uppercase + string.digits):
    '''
    Function that returns randome strings of desired length and component of the desired set of characters. Started from: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    -- ll2'14
    '''
    # Ensure that each character has the same probability of being selected by making the set unique
    characters = ''.join(set(characters))
    # return the random string
    return ''.join(random.choice(characters) for _ in range(size))

#
def parsin( keys, dict, default=False, verbose=False, fname='*', **kwarg ):
    '''
    Function for interpretive keyword parsing:
    1. Given the dictionary arguments of a fuction,
    scan for a member of the set "keys".
    2. If a set member is found, output it's dictionary reference.
    The net result is that multiple keywords can be mapped to a
    single internal keyword for use in the host function. Just as traditional
    keywords are initialized once, this function should be used within other
    functions to initalize a keyword only once.
    -- ll2'14
    '''

    if type(keys)==str:
        keys = [keys]

    # print('>> Given key list of length %g' % len(keys))
    value = default
    for key in keys:
        #if verbose:
        #    print('>> Looking for "%s" input...' % key)
        if key in dict:

            if verbose:
                print('[%s]>> Found "%s" or variant thereof.' % (fname,key) )

            value = dict[key]
            break
    #
    return value


# Bash emulator
def bash( cmd ):
    # Pass the command to the operating system
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    raw_output = process.communicate()[0]
    #
    return raw_output

# Rough grep equivalent using the subprocess module
def grep( flag, file_location, options=None, comment=None ):
    #
    if options is None: options = ''
    if comment is None: comment = []
    if not isinstance(comment,list): comment = [comment]
    # Create string for the system command
    cmd = "grep " + '"' + flag + '" ' + file_location + options
    # Pass the command to the operating system
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    raw_output = process.communicate()[0]
    # Split the raw output into a list whose elements are the file's lines
    output = raw_output.splitlines()
    # Mask the lines that are comments
    if comment:
        for commet in comment:
            if not isinstance(commet,str):
                raise TypeError('Hi there!! Comment input must be string or list of stings. :D ')
            # Masking in Python:
            mask = [line[0]!=commet for line in output]
            output = [output[k] for k in xrange(len(output)) if mask[k]]

    # Return the list of lines
    return output

# Simple function to determine whether or not a string is intended to be a
# number: can it be cast as float?
def isnumeric( s ):
    try:
        float(s)
        ans = True
    except:
        ans = False
    return ans

# Rudimentary function for printing text in the center of the terminal window
def center_space(str):
    x = os.popen('stty size', 'r').read()
    if x:
        rows, columns = x.split()
        a = ( float(columns) - float(len(str)+1.0) ) /2.0
    else:
        a = 0
    return ' '*int(a)
def center_print(str):
    pad = center_space(str)
    print pad + str

# Print a short about statement to the prompt
def print_hl(symbol="<>"):
    '''
    Simple function for printing horizontal line across terminal.
    ~ ll2'14
    '''
    x = os.popen('stty size', 'r').read()
    if x:
        rows, columns = x.split()
        if columns:
            print symbol*int(float(columns)/float(len(symbol)))

# Function for untaring datafiles
def untar(tar_file,savedir='',verbose=False,cleanup=False):
    # Save to location of tar file if no other directory given
    if not savedir:
        savedir = os.path.dirname(tar_file)
    # Open tar file and extract
    tar = tarfile.open(tar_file)
    internal_files = tar.getnames()
    tar.extractall(savedir)
    tar.close()
    if verbose:
        print ">> untar: Found %i files in tarball." % len(internal_files)
    if cleanup:
        os.remove(tar_file)

# Function for file downloading from urls
def download( url, save_path='', save_name='', size_floor=[], verbose=False, overwrite=True ):

    # set default file name for saving
    if not save_name:
        save_name = url.split('/')[-1]

    # Create full path of file that will be downloaded using URL
    path,file_type = os.path.splitext(url)
    file_location = save_path + save_name
    u = urllib2.urlopen(url)

    # Determine whether the download is desired
    DOWNLOAD = os.path.isfile(file_location) and overwrite
    DOWNLOAD = DOWNLOAD or not os.path.isfile(file_location)

    # Set the default output
    done = False

    #
    if DOWNLOAD:
        f = open(file_location, 'wb')
        file_size_dl = 0
        block_sz = 10**4 # bites
        # Time the download by getting the current system time
        t0 = time.time()
        # Perform the download
        k=0
        while True:
            t1 = time.time();
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            mb_downloaded = file_size_dl/(10.0**6.0);
            dt = time.time() - t1
            if k==0:
                status = r"   Download Progress:%1.2f MB downloaded" % mb_downloaded
            else:
                status = r"   Download Progress:%1.2f MB downloaded at %1.2f Mb/sec     " % (mb_downloaded,(len(buffer)/(10.0**6.0))/dt)
            status = status + chr(8)*(len(status)+1)
            k += 1
            if verbose: print status,
        # Close file
        f.close()
        # Get the final time
        tf = time.time()
        # Show completion notice
        if verbose: print "   Download of %1.4f MB completed in %1.4f sec" % ((file_size_dl/(10.0**6.0)),tf-t0)
        if verbose: print "   Average download rate: %1.4f Mb/sec" % ((file_size_dl/(10.0**6.0))/(tf-t0))
        if verbose: print('   Saving:"%s"' % file_location )
        # If the size of this file is below the floor, delete it.
        if size_floor:
            if file_size_dl<size_floor:
                os.remove(file_location)
                if verbose: print( '   *File is smaller than %i bytes and has been deleted.' % size_floor )
                done = True
    else:
        #
        print('   *File exists and overwrite is not turned on, so this file will be skipped.')

    return (done,file_location)




# Class for dynamic data objects such as sim-catalog-entries (scentry's)
class smart_object:
    '''
    This class has the ability to learn files and string by making file elemnts
    its attributes and automatically setting the attribute values.
    ~ll2'14
    '''

    def __init__(this,attrfile=None,id=None,overwrite=False,**kwargs):
        #
        this.valid = False
        this.source_file_path = []
        this.source_dir  = []

        #
        this.overwrite = overwrite
        if attrfile is not None:

            if isinstance( attrfile, list ):

                # Learn list of files
                for f in attrfile:
                    this.learn_file( f, **kwargs )

            elif isinstance( attrfile, str ):

                # Learn file
                this.learn_file( attrfile, **kwargs )
            else:

                msg = 'first input (attrfile key) must of list containing strings, or single string of file location'
                raise ValueError(msg)

    #
    def show( this ):

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        #
        for attr in this.__dict__.keys():
            value = this.__dict__[attr]
            print( '[%s]>> %s = %s' % (thisfun,attr,str(value)) )

    # Function for parsing entire files into class attributes and values
    def learn_file( this, file_location, eqls="=", **kwargs ):
        # Use grep to read each line in the file that contains an equals sign
        line_list = grep(eqls,file_location,**kwargs)
        for line in line_list:
            this.learn_string( line,eqls, **kwargs )
        # Learn file location
        this.source_file_path.append(file_location)
        # Learn location of parent folder
        this.source_dir.append( parent(file_location) )

    # Function for parsing single lines strings into class attributes and values
    def learn_string(this,string,eqls='=',comment=None,**kwargs):

        #
        from numpy import array,ndarray,append

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        # Look for verbose key
        keys = ('verbose','verb')
        VERB = parsin( keys, kwargs )
        if VERB:
            print '[%s]>> VERBOSE mode on.' % thisfun
            print 'Lines with %s will not be considered.' % comment

        # Get rid of partial line comments. NOTE that full line comments have been removed in grep
        done = False
        if comment is not None:
            if not isinstance(comment,list): comment = [comment]
            for c in comment:
                if not isinstance(c,str):
                    raise TypeError('Hi there!! Comment input must be string or list of stings. I found %s :D '%[c])
                for k in range( string.count(c) ):
                    h = string.find(c)
                    # Keep only text that comes before the comment marker
                    string = string[:h]

        # The string must be of the format "A eqls B", in which case the result is
        # that the field A is added to this object with the value B
        part = string.split(eqls)

        # Remove harmful and unneeded characters from the attribute side
        attr = part[0].replace('-','_')
        attr = attr.replace(' ','')
        attr = attr.replace('#','')

        # Detect space separated lists on the value side
        # NOTE that this will mean that 1,2,3,4 5 is treated as 1,2,3,4,5
        part[1] = (','.join( [ p for p in part[1].split(' ') if p ] )).replace(',,',',')

        if VERB: print( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1]))
        # if True: print( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1]))

        # Correctly formatted lines will be parsed into exactly two parts
        if [2 == len(part)]:
            #
            value = []
            if part[1].split(','):
                is_number = True
                for val in part[1].split(','):
                    #
                    if  not isnumeric(val):   # IF
                        is_number = False
                        if VERB: print( '>> Learning character: %s' % val )
                        value.append( val )
                    else:                       # Else
                        if VERB: print( '>> Learning number: %s' % val)
                        if val:
                            # NOTE that the line below contains eval rather than float becuase we want our data collation process to preserve type
                            value.append( eval(val) )
                #
                if is_number:
                    value = array(value)
            else:
                value.append("none")
            #
            if 1==len(value):
                value = value[0]

            if this.overwrite is False:
                # If the attr does not already exist, then add it
                if not ( attr in this.__dict__.keys() ):
                    setattr( this, attr, value )
                else:
                    # If it's already a list, then append
                    if isinstance( getattr(this,attr), (list,ndarray) ):
                        setattr(  this, attr, list(getattr(this,attr))  )
                        setattr(  this, attr, getattr(this,attr)+[value]  )
                    else:
                        # If it's not already a list, then make it one
                        old_value = getattr(this,attr)
                        setattr( this, attr, [old_value,value] )

            else:
                setattr( this, attr, value )

        else:
            raise ValueError('Impoperly formatted input string.')


# Function for loading various file types into numerical array
def smart_load( file_location,        # absolute path location of file
                  verbose = None ):     # if true, let the people know

    #
    from os.path import isfile
    from numpy import array

    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    #
    status = isfile(file_location)
    if status:

        # Handle various file types
        if file_location.split('.')[-1] is 'gz':
            # Load from gz file
            import gzip
            with gzip.open(file_location, 'rb') as f:
                raw = f.read()
        else:
            # Load from ascii file
            try:
                raw = numpy.loadtxt( file_location, comments='#')
            except:
                alert('Could not load: %s'%red(file_location),thisfun)
                alert(red('None')+' will be output',thisfun)
                raw = None
                status = False

    else:

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        #
        alert('Could not find file: "%s". We will proceed, but %s will be returned.'%(yellow(file_location),red('None')),thisfun)
        raw = None

    #
    return raw,status

# # Function to produce array of color vectors
# def rgb( N,                     #
#          offset     = None,     #
#          speed      = None,     #
#          plot       = False,    #
#          shift      = None,     #
#          jet        = False,     #
#          reverse    = False,     #
#          verbose    = None ):   #
#
#     #
#     from numpy import array,pi,sin,arange,linspace
#
#     # If bad first intput, let the people know.
#     if not isinstance( N, int ):
#         msg = 'First input must be '+cyan('int')+'.'
#         raise ValueError(msg)
#
#     #
#     if offset is None:
#         offset = pi/4.0
#
#     #
#     if speed is None:
#         speed = 2.0
#
#     #
#     if shift is None:
#         shift = 0
#
#     #
#     if jet:
#         offset = -pi/2.1
#         shift = pi/2.0
#
#     #
#     if reverse:
#         t_range = linspace(1,0,N)
#     else:
#         t_range = linspace(0,1,N)
#
#     #
#     r = array([ 1, 0, 0 ])
#     g = array([ 0, 1, 0 ])
#     b = array([ 0, 0, 1 ])
#
#     #
#     clr = []
#     w = pi/2.0
#     for t in t_range:
#
#         #
#         R = r*sin( w*t                + shift )
#         G = g*sin( w*t*speed + offset + shift )
#         B = b*sin( w*t + pi/2         + shift )
#
#         #
#         clr.append( abs(R+G+B) )
#
#     #
#     if 1 == N :
#         clr = clr[0]
#
#     #
#     if plot:
#
#         #
#         from matplotlib import pyplot as p
#
#         #
#         fig = p.figure()
#         fig.set_facecolor("white")
#
#         #
#         for k in range(N):
#             p.plot( array([0,1]), (k+1.0)*array([1,1])/N, linewidth=20, color = clr[k] )
#
#         #
#         p.axis('equal')
#         p.axis('off')
#
#         #
#         p.ylim([-1.0/N,1.0+1.0/N])
#         p.show()
#
#     #
#     return array(clr)

# # custome function for setting desirable ylimits
# def pylim( x, y, axis='both', domain=None, symmetric=False, pad_y=0.1 ):
#
#     #
#     from matplotlib.pyplot import xlim, ylim
#     from numpy import ones
#
#     #
#     if domain is None:
#         mask = ones( x.shape, dtype=bool )
#     else:
#         mask = (x>=min(domain))*(x<=max(domain))
#
#     #
#     if axis == 'x' or axis == 'both':
#         xlim( lim(x) )
#
#     #
#     if axis == 'y' or axis == 'both':
#         limy = lim(y[mask]); dy = pad_y * ( limy[1]-limy[0] )
#         if symmetric:
#             ylim( [ -limy[-1]-dy , limy[-1]+dy ] )
#         else:
#             ylim( [ limy[0]-dy , limy[-1]+dy ] )

# Return the min and max limits of an 1D array
def lim(x):

    # Import useful bit
    from numpy import array,amin,amax

    # Columate input.
    z = x.reshape((x.size,))

    # Return min and max as list
    return array([amin(z),amax(z)])

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

# #
# def alert(msg,fname=None):
#
#     if fname is None:
#         fname = thisfun()
#
#     print '('+cyan(fname)+')>> '+msg

# #
# def warning(msg,fname=None):
#
#     if fname is None:
#         fname = thisfun()
#
#     print '('+yellow(fname+'!')+')>> '+msg

# #
# def error(msg,fname=None):
#
#     if fname is None:
#         fname = thisfun()
#
#     raise ValueError( '('+red(fname+'!!')+')>> '+msg )



# Usual find methods can be slow AND non-verbose about what's happening. This is one possible solution that at least lets the user know what's happening in an online fashion.
def rfind( path , pattern = None, verbose = False, ignore = None ):

    #
    import fnmatch
    import os
    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    # # Use find with regex to get matches
    # from subprocess import Popen, PIPE
    # (stdout, stderr) = Popen(['find',path,'-regex','.*/[^/]*%s*'%(pattern)], stdout=PIPE).communicate()
    #
    # if 'None' is stderr:
    #     raise ValueError( 'Unable to find files matching '+red(pattern)+' in '+red(path)+'. The system says '+red(stderr) )
    #
    # #
    # matches = stdout.split('\n')


    # All items containing these string will be ignored
    if ignore is None:
        ignore = ['.git','.svn']

    # Searching for pattern files. Let the people know.
    msg = 'Seaching for %s in %s:' % (cyan(pattern),cyan(path))
    if verbose: alert(msg,thisfun)

    matches = []
    for root, dirnames, filenames in os.walk( path ):
        for filename in filenames:

            proceed = len(filename)>=len(pattern)
            for k in ignore: proceed = proceed and not (k in filename)

            if proceed:

                if pattern in filename:
                    parts = os.path.join(root, filename).split(pattern)
                    if len(parts)==2:
                        if verbose: print magenta('  ->  '+parts[0])+cyan(pattern)+magenta(parts[1])
                    else:
                        if verbose: print magenta('  ->  '+os.path.join(root, filename) )
                    matches.append(os.path.join(root, filename))

    return matches


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


# # Standard factorial function
# def factorial(n):
#     x = 1.0
#     for k in range(n):
#         x *= (k+1)
#     return x

# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))

#
# Use formula from wikipedia to calculate the harmonic
# See http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
# for more information.
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
    Y = A*B*exp( 1j*m*phi )

    #
    if sum(abs(Y.imag)) == 1e-7:
        Y = Y.real

    #
    return Y

# Interpolate waveform array to a given spacing in its first column
def intrp_wfarr(wfarr,delta=None,domain=None):

    #
    from numpy import linspace,array,diff,zeros,arange
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Validate inputs
    if (delta is None) and (domain is None):
        msg = red('First "delta" or "domain" must be given. See traceback above.')
        error(msg,'intrp_wfarr')
    if (delta is not None) and (domain is not None):
        msg = red('Either "delta" or "domain" must be given, not both. See traceback above.')
        error(msg,'intrp_wfarr')

    # Only interpolate if current delta is not input delta
    proceed = True
    if delta is not None:
        d = wfarr[0,0]-wfarr[1,0]
        if abs(delta-d)/delta < 1e-6:
            proceed = False

    # If there is need to interpolate, then interpolate.
    if proceed:

        # Encapsulate the input domain for ease of reference
        input_domain = wfarr[:,0]

        # Generate or parse the new domain
        if domain is None:
            N = diff(lim(input_domain))[0] / delta
            intrp_domain = delta * arange( 0, N  ) + wfarr[0,0]
        else:
            intrp_domain = domain

        # Pre-allocate the new wfarr
        _wfarr = zeros( (len(intrp_domain),wfarr.shape[1]) )

        # Store the new domain
        _wfarr[:,0] = intrp_domain

        # Interpolate the remaining columns
        for k in range(1,wfarr.shape[1]):
            _wfarr[:,k] = spline( input_domain, wfarr[:,k] )( intrp_domain )

    else:

        # Otherwise, return the input array
        _wfarr = wfarr

    #
    return _wfarr


# Fucntion to pad wfarr with zeros. NOTE that this should only be applied to a time domain waveform that already begins and ends with zeros.
def pad_wfarr(wfarr,new_length,where=None):

    #
    from numpy import hstack,zeros,arange

    # Only pad if size of the array is to increase
    length = len(wfarr[:,0])
    proceed = length < new_length

    #
    if isinstance(where,str):
        where = where.lower()

    #
    if where is None:
        where = 'sides'
    elif not isinstance(where,str):
        error('where must be string: left,right,sides','pad_wfarr')
    elif where not in ['left','right','sides']:
        error('where must be in {left,right,sides}','pad_wfarr')


    # Enforce integer new length
    if new_length != int(new_length):
        msg = 'Input pad length is not integer; I will apply int() before proceeding.'
        alert(msg,'pad_wfarr')
        new_length = int( new_length )

    #
    if proceed:


        # Pre-allocate the new array
        _wfarr = zeros(( new_length, wfarr.shape[1] ))

        # Create the new time series
        dt = wfarr[1,0] - wfarr[0,0]
        _wfarr[:,0] = dt * arange( 0, new_length ) + wfarr[0,0]

        if where is 'sides':

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length)/2 )
            right_pad = zeros( new_length-length-len(left_pad) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k],right_pad] )

        elif where == 'right':

            # Create the pads for the other columns
            right_pad = zeros( new_length-length )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [wfarr[:,k],right_pad] )

        elif where == 'left':

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k]] )

    else:

        # Otherwise, do nothing.
        _wfarr = wfarr

        # Warn the user that nothing has happened.
        msg = 'The desired new length is <= the current array length (i.e. number of time domain points). Nothing will be padded.'
        warning( msg,fname='pad_wfarr'+cyan('@%i'%linenum()) )

    # Return padded array
    return _wfarr

# Shift a waveform arra by some "shift" amount in time
def tshift_wfarr( _wfarr, shift ):
    '''Shift a waveform arra by some "shift" amount in time'''
    # Import useful things
    from numpy import array
    # Unpack waveform array
    t,p,c = _wfarr[:,0],_wfarr[:,1],_wfarr[:,2]
    _y = p + 1j*c
    # Shift the waveform array data using tshift
    y = tshift( t,_y,shift )
    # Repack the input array
    wfarr = array(_wfarr)
    wfarr[:,0] = t
    wfarr[:,1] = y.real
    wfarr[:,2] = y.imag
    # Return answer
    ans = wfarr
    return ans


# Time shift array data, h, using a frequency diomain method
def tshift( t,      # time sries of data
            h,      # data that will be shifted
            t0 ):   # amount to shift data


    #
    from scipy.fftpack import fft, fftfreq, fftshift, ifft
    from numpy import diff,mean,exp,pi

    #
    is_real = sum( h.imag ) == 0

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

    #
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

#
def pnw0(m1,m2,D=10.0):
    # https://arxiv.org/pdf/1310.1528v4.pdf
    # Equation 228
    # 2nd Reference: arxiv:0710.0614v1
    # NOTE: this outputs orbital frequency
    from numpy import sqrt,zeros,pi,array,sum
    #
    G = 1.0
    c = 1.0
    r = float(D)
    M = float( m1+m2 )
    v = m1*m2/( M**2 )
    gamma = G*M/(r*c*c)     # Eqn. 225
    #
    trm = zeros((4,))
    #
    trm[0] = 1.0
    trm[1] = v - 3.0
    trm[2] = 6 + v*41.0/4.0 + v*v
    trm[3] = -10.0 + v*( -75707.0/840.0 + pi*pi*41.0/64.0 ) + 19.0*0.5*v*v + v*v*v
    #
    w0 = sqrt( (G*M/(r*r*r)) * sum( array([ term*(gamma**k) for k,term in enumerate(trm) ]) ) )

    #
    return w0


# Find the interpolated global max location of a data series
def intrp_argmax( y,
                  domain=None,
                  verbose=False ):

    #
    max_val,argmax = intrp_max( y,domain=domain,verbose=verbose,return_argmax=True )

    #
    ans = argmax
    return ans


# Find the interpolated global max location of a data series
def intrp_max( y,
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
    q = minimize(f,x0,bounds=[(x0-10,x0+10)])
    xmax = q.x[0]

    #
    if return_argmax:
        ans = (yspline(xmax),xmax)
    else:
        ans = yspline(xmax)

    #
    return ans

# Shift phase of waveform array
def shift_wfarr_phase(wfarr,dphi):

    #
    from numpy import array,ndarray,sin,cos

    #
    if not isinstance(wfarr,ndarray):
        error( 'input must be numpy array type' )

    #
    t,r,c = wfarr[:,0],wfarr[:,1],wfarr[:,2]

    #
    r_ = r*cos(dphi) - c*sin(dphi)
    c_ = r*sin(dphi) + c*cos(dphi)

    #
    wfarr[:,0],wfarr[:,1],wfarr[:,2] = t , r_, c_

    #
    return wfarr

# Find the average phase difference and align two wfarr's
def align_wfarr_average_phase(this,that,mask=None,verbose=False):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    #
    from numpy import angle,unwrap,mean

    #
    if mask is None:
        u = this[:,1]+1j*this[:,2]
        v = that[:,1]+1j*that[:,2]
    else:
        u = this[mask,1]+1j*this[mask,2]
        v = that[mask,1]+1j*that[mask,2]

    #
    _a = unwrap( angle(u) )
    _b = unwrap( angle(v) )


    #
    a,b = mean( _a ), mean( _b )
    dphi = -a + b

    #
    if verbose:
        alert('The phase shift applied is %s radians.'%magenta('%1.4e'%(dphi)))

    #
    this_ = shift_wfarr_phase(this,dphi)

    #
    return this_


#
def get_wfarr_relative_phase(this,that):

    #
    from numpy import angle,unwrap,mean

    #
    u = this[:,1]+1j*this[:,2]
    v = that[:,1]+1j*that[:,2]

    #
    _a = unwrap( angle(u) )[0]
    _b = unwrap( angle(v) )[0]

    #
    dphi = -_a + _b

    #
    return dphi

# Find the average phase difference and align two wfarr's
def align_wfarr_initial_phase(this,that):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    dphi = get_wfarr_relative_phase(this,that)

    #
    this_ = shift_wfarr_phase(this,dphi)

    #
    return this_


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Here are some phenomenological fits used in PhenomD                               #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
# s is defined around Equation 3.6.
''' Copied from LALSimulation Version '''
def FinalSpin0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    s2 = s*s
    s3 = s2*s
    s4 = s3*s
    return 3.4641016151377544*eta - 4.399247300629289*eta2 +\
    9.397292189321194*eta3 - 13.180949901606242*eta4 +\
    (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +\
    (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +\
    (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +\
    (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4

#Wrapper function for FinalSpin0815_s.
''' Copied from LALSimulation Version '''
def FinalSpin0815(eta,chi1,chi2):
    from numpy import sqrt
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(abs(1.0 - 4.0*float(eta)))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # s defined around Equation 3.6 arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2)
    return FinalSpin0815_s(eta, s)

# Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
# Input parameter s defined around Equation 3.7 and 3.8.
def EradRational0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*\
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 \
    - 1.4756929437702908*eta + 7.304676214885011*eta2)*s)


# Wrapper function for EradRational0815_s.
def EradRational0815(eta, chi1, chi2):
    from numpy import sqrt,round
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(1.0 - 4.0*eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)
    return EradRational0815_s(eta,s)


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Given a 1D array, determine the set of N lines that are optimally representative  #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Hey, here's a function that approximates any 1d curve as a series of lines
def romline(  domain,           # Domain of Map
              range_,           # Range of Map
              N,                # Number of Lines to keep for final linear interpolator
              positive=True,   # Toggle to use positive greedy algorithm ( where rom points are added rather than removed )
              verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_
    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/( R1 if abs(R1)!=0 else 1 )

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
            # print trial_sigma
            if trial_sigma < min_sigma:
                knots,rom,min_sigma = trial_knots,trial_rom,trial_sigma

    #
    # print min_sigma
    knots = array([ int(k) for k in knots ])

    return knots,rom


# Hey, here's a function related to romline
def positive_romline(   domain,           # Domain of Map
                        range_,           # Range of Map
                        N,                # Number of Lines to keep for final linear interpolator
                        seed = None,      # First point in domain (index) to use
                        verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    #
    weights = (r-amin(r)) / amax( r-amin(r) )
    weights = ones( d.size )

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
    min_space = list(space)
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
            sigma = err( weights * (linterp( trial_domain, trial_range )( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
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



#
def romspline( domain, range_,tol=1e-2,N=None,verbose=False ):
    #
    knots,rom,min_sigma = positive_romspline( domain, range_,tol=1e-2,N=None,verbose=False )
    return None



# Hey, here's a function related to romspline
def positive_romspline(   domain,           # Domain of Map
                          range_,           # Range of Map
                          tol=1e-2,         # Tolerance of normalized data
                          N = None,         # Optional number of points
                          verbose = False ):

    # Use an interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from scipy.interpolate import interp1d
    from matplotlib.pyplot import plot,show,figure,xlabel,ylabel,legend,yscale
    from kerr import pgreedy

    # Domain and range shorthand
    d = domain
    R = range_

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    #
    # weights = (r-amin(r)) / amax( r-amin(r) )
    weights = ones( d.size )

    #
    done = False
    space,_ = romline( domain,range_,N=3 )
    domain_space = range(len(d))
    err = lambda x: std(x) # mean( abs(x) ) # std(x) #
    min_space = list(space)
    e = []

    #
    foo = pgreedy()

    while not done:
        #
        min_sigma = inf
        for k in [ a for a in domain_space if not (a in space) ]:
            # Add a trial point
            trial_space = list(space)
            trial_space.append(k)
            # trial_space.sort()
            # Apply linear interpolation ON the new domain TO the original domain
            trial_domain = d[ sorted(trial_space) ]
            trial_range = r[ sorted(trial_space) ]
            #
            sigma = err( weights * (spline( trial_domain, trial_range )( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
            # print spline( trial_domain, trial_range, k=3 )( d )
            # print interp1d( trial_domain, trial_range, kind='cubic' )( d )
            # raise
            #
            if sigma < min_sigma:
                min_k = k
                min_sigma = sigma
                min_space = array( trial_space )

        #
        e.append(min_sigma)

        #
        space = list(min_space)
        #
        low_enough_error = min_sigma<tol
        enough_points = len(space) == N
        done = low_enough_error or enough_points

    #
    rom = spline( d[sorted(min_space)], R[sorted(min_space)] )
    knots = min_space

    #
    figure()
    plot( e )
    from numpy import arange,log,amax,diff,argmax
    lknots,lrom = romline( arange(len(e)), log(e), 9 )
    dlknots = abs(diff(lrom(lknots))/diff(lknots))
    dlknots /= amax(dlknots)
    print find(dlknots>0.1)[-1] + 1
    lknots = lknots[ find(dlknots>0.1)[-1] + 1 ]
    plot( lknots, e[lknots], 'or', mfc='r', mec='r', alpha=0.8 )
    yscale('log')
    show()

    knots = knots[:(lknots+1)]
    rom = spline( d[sorted(knots)], R[sorted(knots)] )
    # min_sigma = err( weights * (rom( d ) - R) ) / ( err(R) if err(R)!=0 else 1e-8  )

    return knots,rom,min_sigma



# Hey, here's a function related to romspline
def positive_romspline(   domain,           # Domain of Map
                          range_,           # Range of Map
                          tol=1e-2,         # Tolerance of normalized data
                          N = None,         # Optional number of points
                          verbose = False ):

    # Use an interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from scipy.interpolate import interp1d
    from matplotlib.pyplot import plot,show,figure,xlabel,ylabel,legend,yscale

    # Domain and range shorthand
    d = domain
    R = range_

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    #
    # weights = (r-amin(r)) / amax( r-amin(r) )
    weights = ones( d.size )

    #
    done = False
    space,_ = romline( domain,range_,N=3 )
    domain_space = range(len(d))
    err = lambda x: std(x) # mean( abs(x) ) # std(x) #
    min_space = list(space)
    e = []
    while not done:
        #
        min_sigma = inf
        for k in [ a for a in domain_space if not (a in space) ]:
            # Add a trial point
            trial_space = list(space)
            trial_space.append(k)
            # trial_space.sort()
            # Apply linear interpolation ON the new domain TO the original domain
            trial_domain = d[ sorted(trial_space) ]
            trial_range = r[ sorted(trial_space) ]
            #
            sigma = err( weights * (spline( trial_domain, trial_range )( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
            # print spline( trial_domain, trial_range, k=3 )( d )
            # print interp1d( trial_domain, trial_range, kind='cubic' )( d )
            # raise
            #
            if sigma < min_sigma:
                min_k = k
                min_sigma = sigma
                min_space = array( trial_space )

        #
        e.append(min_sigma)

        #
        space = list(min_space)
        #
        low_enough_error = min_sigma<tol
        enough_points = len(space) == N
        done = low_enough_error or enough_points

    #
    rom = spline( d[sorted(min_space)], R[sorted(min_space)] )
    knots = min_space

    #
    figure()
    plot( e )
    from numpy import arange,log,amax,diff,argmax
    lknots,lrom = romline( arange(len(e)), log(e), 9 )
    dlknots = abs(diff(lrom(lknots))/diff(lknots))
    dlknots /= amax(dlknots)
    print find(dlknots>0.1)[-1] + 1
    lknots = lknots[ find(dlknots>0.1)[-1] + 1 ]
    plot( lknots, e[lknots], 'or', mfc='r', mec='r', alpha=0.8 )
    yscale('log')
    show()

    knots = knots[:(lknots+1)]
    rom = spline( d[sorted(knots)], R[sorted(knots)] )
    # min_sigma = err( weights * (rom( d ) - R) ) / ( err(R) if err(R)!=0 else 1e-8  )

    return knots,rom,min_sigma



# Fix nans, nonmonotinicities and jumps in time series waveform array
def straighten_wfarr( wfarr, verbose=False ):
    '''
    Some waveform arrays (e.g. from the BAM code) may have non-monotonic time series
    (gaps, duplicates, and crazy backwards referencing). This method seeks to identify
    these instances and reformat the related array. Non finite values will also be
    removed.
    '''

    # Import useful things
    from numpy import arange,sum,array,diff,isfinite,hstack
    thisfun = 'straighten_wfarr'

    # Remove rows that contain non-finite data
    finite_mask = isfinite( sum( wfarr, 1 ) )
    if sum(finite_mask)!=len(finite_mask):
        if verbose: alert('Non-finite values found in waveform array. Corresponding rows will be removed.',thisfun)
    wfarr = wfarr[ finite_mask, : ]

    # Sort rows by the time series' values
    time = array( wfarr[:,0] )
    space = arange( wfarr.shape[0] )
    chart = sorted( space, key = lambda k: time[k] )
    if (space != chart).all():
        if verbose: alert('The waveform array was found to have nonmonotinicities in its time series. The array will now be straightened.',thisfun)
    wfarr = wfarr[ chart, : ]

    # Remove rows with duplicate time series values
    time = array( wfarr[:,0] )
    diff_mask = hstack( [ True, diff(time).astype(bool) ] )
    if sum(diff_mask)!=len(diff_mask):
        if verbose: alert('Repeated time values were found in the array. Offending rows will be removed.',thisfun)
    wfarr = wfarr[ diff_mask, : ]

    # The wfarr should now be straight
    # NOTE that the return here is optional as all operations act on the original input
    return wfarr


#
def rISCO_14067295(a):
    """
    Calculate the ISCO radius of a Kerr BH as a function of the Kerr parameter using eqns. 2.5 and 2.8 from Ori and Thorne, Phys Rev D 62, 24022 (2000)

    Parameters
    ----------
    a : Kerr parameter

    Returns
    -------
    ISCO radius
    """

    import numpy as np
    a = np.array(a)

    # Ref. Eq. (2.5) of Ori, Thorne Phys Rev D 62 124022 (2000)
    z1 = 1.+(1.-a**2.)**(1./3)*((1.+a)**(1./3) + (1.-a)**(1./3))
    z2 = np.sqrt(3.*a**2 + z1**2)
    a_sign = np.sign(a)
    return 3+z2 - np.sqrt((3.-z1)*(3.+z1+2.*z2))*a_sign

# https://arxiv.org/pdf/1406.7295.pdf
def Mf14067295( m1,m2,chi1,chi2,chif=None ):

    import numpy as np

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m
    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    #
    if chif is None:
        chif = jf14067295(m1, m2, chi1, chi2)
    r_isco = rISCO_14067295(chif)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    M0  = 0.951507
    K1  = -0.051379
    K2a = -0.004804
    K2b = -0.054522
    K2c = -0.000022
    K2d = 1.995246
    K3a = 0.007064
    K3b = -0.017599
    K3c = -0.119175
    K3d = 0.025000
    K4a = -0.068981
    K4b = -0.011383
    K4c = -0.002284
    K4d = -0.165658
    K4e = 0.019403
    K4f = 2.980990
    K4g = 0.020250
    K4h = -0.004091
    K4i = 0.078441

    # binding energy at ISCO -- Eq.(2.7) of Ori, Thorne Phys Rev D 62 124022 (2000)
    E_isco = (1. - 2./r_isco + chif/r_isco**1.5)/np.sqrt(1. - 3./r_isco + 2.*chif/r_isco**1.5)

    # final mass -- Eq. (14) of Healy et al Phys Rev D 90, 104004 (2014)
    mf = (4.*eta)**2*(M0 + K1*S + K2a*Delta*delta_m + K2b*S**2 + K2c*Delta**2 + K2d*delta_m**2 \
        + K3a*Delta*S*delta_m + K3b*S*Delta**2 + K3c*S**3 + K3d*S*delta_m**2 \
        + K4a*Delta*S**2*delta_m + K4b*Delta**3*delta_m + K4c*Delta**4 + K4d*S**4 \
        + K4e*Delta**2*S**2 + K4f*delta_m**4 + K4g*Delta*delta_m**3 + K4h*Delta**2*delta_m**2 \
        + K4i*S**2*delta_m**2) + (1+eta*(E_isco+11.))*delta_m**6.

    return mf*m

#
def jf14067295_diff(a_f,eta,delta_m,S,Delta):
    """ Internal function: the final spin is determined by minimizing this function """

    #
    import numpy as np

    # calculate ISCO radius
    r_isco = rISCO_14067295(a_f)

    # angular momentum at ISCO -- Eq.(2.8) of Ori, Thorne Phys Rev D 62 124022 (2000)
    J_isco = (3*np.sqrt(r_isco)-2*a_f)*2./np.sqrt(3*r_isco)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    L0  = 0.686710
    L1  = 0.613247
    L2a = -0.145427
    L2b = -0.115689
    L2c = -0.005254
    L2d = 0.801838
    L3a = -0.073839
    L3b = 0.004759
    L3c = -0.078377
    L3d = 1.585809
    L4a = -0.003050
    L4b = -0.002968
    L4c = 0.004364
    L4d = -0.047204
    L4e = -0.053099
    L4f = 0.953458
    L4g = -0.067998
    L4h = 0.001629
    L4i = -0.066693

    a_f_new = (4.*eta)**2.*(L0  +  L1*S +  L2a*Delta*delta_m + L2b*S**2. + L2c*Delta**2 \
        + L2d*delta_m**2. + L3a*Delta*S*delta_m + L3b*S*Delta**2. + L3c*S**3. \
        + L3d*S*delta_m**2. + L4a*Delta*S**2*delta_m + L4b*Delta**3.*delta_m \
        + L4c*Delta**4. + L4d*S**4. + L4e*Delta**2.*S**2. + L4f*delta_m**4 + L4g*Delta*delta_m**3. \
        + L4h*Delta**2.*delta_m**2. + L4i*S**2.*delta_m**2.) \
        + S*(1. + 8.*eta)*delta_m**4. + eta*J_isco*delta_m**6.

    daf = a_f-a_f_new
    return daf*daf

#
def jf14067295(m1, m2, chi1, chi2):
    """
    Calculate the spin of the final BH resulting from the merger of two black holes with non-precessing spins using fit from Healy et al Phys Rev D 90, 104004 (2014)

    Parameters
    ----------
    m1, m2 : component masses
    chi1, chi2 : dimensionless spins of two BHs

    Returns
    -------
    dimensionless final spin, chif
    """
    import numpy as np
    import scipy.optimize as so

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")

    # Vectorize the function if arrays are provided as input
    if np.size(m1) * np.size(m2) * np.size(chi1) * np.size(chi2) > 1:
        return np.vectorize(bbh_final_spin_non_precessing_Healyetal)(m1, m2, chi1, chi2)

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m

    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # compute the final spin
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    x, cov_x = so.leastsq(jf14067295_diff, 0., args=(eta, delta_m, S, Delta))
    chif = x[0]

    return chif


#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
# Find the polarization and orbital phase shifts that maximize the real part
# of  gwylm object's (2,2) and (2,1) multipoles at merger (i.e. the sum)
''' See gwylm.selfalign for higher level Implementation '''
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#

def vectorize( _gwylmo, dphi, dpsi, k_ref=0 ):
    from numpy import array
    vec = []
    select_modes = [ (2,2), (2,1) ]
    valid_count = 0
    gwylmo = _gwylmo.rotate( dphi=dphi, dpsi=dpsi, apply=False, verbose=False, fast=True )
    for y in gwylmo.ylm:
        l,m = y.l,y.m
        if (l,m) in select_modes:
            vec.append( y.plus[ k_ref ] )
            valid_count += 1
    if valid_count != 2:
        error('input gwylm object must have both the l=m=2 and (l,m)=(2,1) multipoles; only %i of these was found'%valid_count)
    return array(vec)

def alphamax(_gwylmo,dphi,plt=False,verbose=False,n=13):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,sum,argmax,array
    action = lambda x: sum( vectorize( _gwylmo, x[0], x[1] ) )
    dpsi_range = linspace(-1,1,n)*pi
    dpsis = linspace(-1,1,1e2)*pi
    a = array( [ action([dphi,dpsi]) for dpsi in dpsi_range ] )
    aspl = spline( dpsi_range, a, kind='cubic' )(dpsis)
    dpsi_opt_guess = dpsis[argmax(aspl)]
    K = minimize( lambda PSI: -action([dphi,PSI]), dpsi_opt_guess )
    dpsi_opt = K.x[-1]
    if plt:
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot, xlabel
        plot( dpsi_range, a, linewidth=4, color='k', alpha=0.1 )
        plot( dpsis, aspl, label=dpsi )
        plot( dpsis[argmax(aspl)], aspl[argmax(aspl)], 'or', mfc='none' )
        xlabel(r'$\psi$')
    if verbose: print dpsi_opt,action([dphi,dpsi_opt])
    return [ dpsi_opt, action([dphi,dpsi_opt])    ]

def betamax(_gwylmo,n=10,plt=False,opt=True,verbose=False):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,argmax,array
    dphi_list = pi*linspace(-1,1,n)
    dpsi,val = [],[]
    for dphi in dphi_list:
        [dpsi_,val_] = alphamax(_gwylmo,dphi,plt=False,n=n)
        dpsi.append( dpsi_ )
        val.append( val_ )

    dphis = linspace(min(dphi_list),max(dphi_list),1e3)
    vals = spline( dphi_list, val, kind='cubic' )( dphis )
    dpsi_s = spline( dphi_list, dpsi, kind='cubic' )( dphis )

    action = lambda x: -sum( vectorize( _gwylmo, x[0], x[1] ) )
    dphi_opt_guess = dphis[argmax(vals)]
    dpsi_opt_guess = dpsi_s[argmax(vals)]
    if opt:
        K = minimize( action, [dphi_opt_guess,dpsi_opt_guess] )
        # print K
        dphi_opt,dpsi_opt = K.x
        val_max = -K.fun
    else:
        dphi_opt = dphi_opt_guess
        dpsi_opt = dpsi_opt_guess
        val_max = vals.max()

    if plt:
        # Setup plotting backend
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot,xlabel,title
        plot( dphi_list, val, linewidth=4, alpha=0.1, color='k' )
        plot( dphi_opt, val_max, 'or', alpha=0.5 )
        plot( dphis, vals )
        xlabel(r'$\phi$')
        title(val_max)

    if verbose:
        print 'dphi_opt = ' + str(dphi_opt)
        print 'dpsi_opt = ' + str(dpsi_opt)
        print 'val_max = ' + str(val_max)

    return dphi_opt,dpsi_opt

def betamax2(_gwylmo,n=10,plt=False,opt=True,verbose=False):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,argmax,array

    action = lambda x: -sum( vectorize( _gwylmo, x[0], x[1] ) )

    dphi,dpsi,done,k = pi,pi/2,False,0
    while not done:
        dpsi_action = lambda _dpsi: action( [dphi,_dpsi] )
        dpsi = minimize( dpsi_action, dpsi, bounds=[(0,2*pi)] ).x[0]
        dphi_action = lambda _dphi: action( [_dphi,dpsi] )
        dphi = minimize( dphi_action, dphi, bounds=[(0,2*pi)] ).x[0]
        done = k>n
        print '>> ',dphi,dpsi,action([dphi,dpsi])
        k+=1

    return dphi,dpsi

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
def spline_diff(t,y):

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    return spline(t,y).derivative()(t)
