#
from maths import *
from positive import *

#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Low level methods for ration function modeling                                    #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Invert Psi = A - Psi*B, for Psi = A/(1+B)
def mvrslv0( domain, centered_scalar_range, numerator_symbols, denominator_symbols, mu=0, sigma=1.0 ):
    ''' Invert Psi = A - Psi*B, for Psi = A/(1+B) '''
    #
    from numpy import mean,std,array
    from numpy.linalg import pinv,lstsq
    #
    if 'K' in denominator_symbols: error('denominator_symbols cannot have symbol for constant term "K"')
    #
    M = len( numerator_symbols )
    alpha = array( [ symeval(sym,domain)*( 1 if k < M else -centered_scalar_range) for k,sym in enumerate( numerator_symbols+denominator_symbols ) ] ).T
    #
    coeffs = lstsq( alpha, centered_scalar_range )[0]
    #
    numerator_coeffs = coeffs[:M]
    denominator_coeffs = coeffs[M:]
    #
    def fitfun( DOMAIN ):
        # Evaluate the model
        A = sym_poly_eval( numerator_symbols,   numerator_coeffs,   DOMAIN )
        B = sym_poly_eval( denominator_symbols, denominator_coeffs, DOMAIN )
        # Shift and scale to preserve mean and sigma of input scalar_range
        ans = mu + sigma*A/(1+B)
        return ans
    #
    return numerator_coeffs, denominator_coeffs, fitfun


# Invert: Psi-1 = A - Psi*B, for Psi = (1+A)/(1+B)
def mvrslv1( domain, centered_scalar_range, numerator_symbols, denominator_symbols, mu=0, sigma=1.0 ):
    ''' Invert Psi-1 = A - Psi*B, for Psi = (1+A)/(1+B) '''
    #
    from numpy import mean,std,ones_like,array,dot
    from numpy.linalg import pinv,lstsq
    #
    if 'K' in numerator_symbols: error('numerator_symbols cannot have symbol for constant term "K"')
    if 'K' in denominator_symbols: error('denominator_symbols cannot have symbol for constant term "K"')
    #
    M = len( numerator_symbols )
    alpha = array( [ symeval(sym,domain)*( 1 if k < M else -centered_scalar_range) for k,sym in enumerate( numerator_symbols+denominator_symbols ) ] ).T
    #
    coeffs = lstsq( alpha, centered_scalar_range-1 )[0]
    #
    numerator_coeffs = coeffs[:M]
    denominator_coeffs = coeffs[M:]
    #
    def fitfun( DOMAIN ):
        # Evaluate the model
        A = sym_poly_eval( numerator_symbols,   numerator_coeffs,   DOMAIN )
        B = sym_poly_eval( denominator_symbols, denominator_coeffs, DOMAIN )
        # Shift and scale to preserve mean and sigma of input scalar_range
        ans = mu + sigma * (1+A)/(1+B)
        return ans
    #
    return numerator_coeffs, denominator_coeffs, fitfun


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


# Hey, here's a function related to romspline
def romspline(   domain,           # Domain of Map
                 range_,           # Range of Map
                 tol=1e-4,         # HALTING tolerance of normalized data
                 N = None,         # Optional number of points
                 weights=None,     # Optional weights for error estimation
                 use_smoothing = False, # Optional toggle to spline auto smoothed data
                 verbose = False ):

    # Use an interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from scipy.interpolate import interp1d
    from matplotlib.pyplot import plot,show,figure,xlabel,ylabel,legend,yscale
    from kerr import pgreedy

    # Domain and range shorthand
    d = domain
    R = range_ if not use_smoothing else smooth(range_).answer

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    # Handle weights input
    weights = ones( d.size ) if weights is None else weights
    if weights.shape != range_.shape:
        error('weights and range should one-to-one in reference --> they should at have the same shape')

    #
    domain_space = range(len(d))
    err = lambda x: std(x)**2

    # Define an action for each greedy step
    def action(trial_space):
        # Apply interpolation ON the new domain TO the original domain
        trial_domain = d[ sorted(trial_space) ]
        trial_range = r[ sorted(trial_space) ]
        # Compute error estimate
        trial_rom = spline( trial_domain, trial_range )
        estimator = err( weights * (trial_rom( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
        # Trial rom
        answer = ( trial_space, spline( trial_domain, R[ sorted(trial_space) ] ) )
        return estimator,answer

    # Set fixed initial boundary for greedy learning
    initial_boundary,_ = romline( domain,range_,N=3 )
    A = pgreedy(domain_space,action,initial_boundary=initial_boundary,fitatol=tol,verbose=verbose)

    # Apply a negative greedy process to futher refine the symbol content
    boundary = A.boundary
    est_list = A.estimator_list
    B = ngreedy( boundary, action, verbose = verbose, ref_est_list = est_list )

    # Return an answer
    return B # knots,rom,min_sigma


# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))

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
                # Generate sumlists, again from the twosum perspective
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



# Create a functional representation of a polynomial symbols consistent with symeval
def sym_poly_eval( symbols, coeffs, domain, range_map=None ):
    '''A functional representation of the fit'''
    #
    ans_ = 0
    for k,b in enumerate(coeffs):
        ans_ += b*symeval( symbols[k], domain )
    #
    if range_map is not None:
        V = range_map['backward']
        ans = V(domain,ans_)
    else:
        ans = ans_
    #
    return ans

# SYMEVAL -- Helper function for polynomial fitting
# Each symbol encodes a simple multinomial function of the basis vectors.
# Example: sym = "001111" encodes x^2 * y^4
def symeval( sym, domain, verbose=False ):
    '''
    SYMEVAL -- Helper function for polynomial fitting
    Each symbol encodes a simple multinomial function of the basis vectors.
    Example: sym = "001111" encodes x^2 * y^4
    '''
    #
    from numpy import ones_like,ones
    #
    if verbose: alert('sym = "%s"'%sym)
    #
    if sym.upper() == 'K':
        # Handle symbolic representation of constant term
        #--
        # from numpy import ones_like
        # ans = ones( domain[:,0].shape, dtype=domain.dtype ) if len(domain.shape)>1 else ones( domain.shape, dtype=domain.dtype )
        ans = ones_like( domain[:,0] ) if len(domain.shape)>1 else ones_like( domain )
    elif sym.isdigit():
        # Handle non-constant symbols
        map_ = [ int(k) for k in sym ]
        ans = 1.0 # NOTE that the final answer will be of the shape of the domain vectors
        for k in map_:
            # IF the domain has dimension greater than 1
            if len(domain.shape)>1:
                # Allow referencing of each dimnesion
                ans *= domain[:,k]
            else:
                # ELSE, simply use the domain
                ans *= domain
                # NOTE that this IF-ELSE structure results from 1D domains not being able to be refernced in a matrix like manner
    else:
        raise TypeError('"%s" is an invalid symbol. Multivariate symbols must be "K" for constant term, or string of integers corresponding to domain dimensions, such as "0001" which, if the domain is [x,y,...], corresponds to x*x*x*y.'%sym)

    #
    return ans


# Multivariate polynomial fitting algorithm
class mvpolyfit:
    '''
    # Low Level Multivariate Polynomial Fitting
    ---

    fit_object = mvpolyfit( ... )

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
        from numpy import array,dot,mean,std
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

        # Center and adjust range values
        mapped_range = U(this.domain,this.range)
        u = mean( mapped_range )
        v =  std( mapped_range )
        centered_adjusted_range = ( mapped_range - u ) / v

        # Estimate the coefficients of the basis symbols
        a = dot( Q, centered_adjusted_range )

        # Un-center and un-scale, thus return to original (mapped by U) feature space
        a *= v
        a[0] += u

        # NOTE that a[0] corresponds to a constant term; see this.__validate_inputs__ for handling of 'K' in this.basis_symbols

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

        # if sym.upper() == 'K':
        #     # Handle symbolic representation of constant term
        #     from numpy import ones
        #     ans = ones( dom[:,0].shape, dtype=dom.dtype ) if len(dom.shape)>1 else ones( dom.shape, dtype=dom.dtype )
        # elif sym.isdigit():
        #     # Handle non-constant symbols
        #     map_ = [ int(k) for k in sym ]
        #     ans = 1.0 # NOTE that the final answer will be of the shape of the domain vectors
        #     for k in map_:
        #         # IF the domain has dimension greater than 1
        #         if len(dom.shape)>1:
        #             # Allow referencing of each dimnesion
        #             ans *= dom[:,k]
        #         else:
        #             # ELSE, simply use the domain
        #             ans *= dom
        #             # NOTE that this IF-ELSE structure results from 1D domains not being able to be refernced in a matrix like manner
        # else:
        #     raise TypeError('"%s" is an invalid symbol. Multivariate symbols must be "K" for constant term, or string of integers corresponding to domain dimensions, such as "0001" which, if the domain is [x,y,...], corresponds to x*x*x*y.'%sym)
        #
        # #
        # return ans

        # Use the class external function
        return symeval( sym, dom )

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
        padf = 0.1
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
        from numpy import ndarray,isfinite,complex256,float128,double,mean,std

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
        # Ensure that the first entry is the constant term. This is important for centering.
        basis_symbols = ['K'] + [ s for s in basis_symbols if s!='K' ]

        # Store low level inputs (More items are stored later)
        # this.__raw_range__ = scalar_range
        # this.range_mean = mean(this.__raw_range__)
        # this.range_std = std(this.__raw_range__)
        # this.range = (this.__raw_range__ - this.range_mean) / this.range_std
        this.range = scalar_range
        #
        this.domain = domain
        this.basis_symbols = basis_symbols
        this.labels = {} if labels is None else labels
        this.range_map = range_map

        #
        this.data_label = None


#
class mvratifit(mvpolyfit):

    #
    def __init__(this):
        return None


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
        boundary= [] if initial_boundary is None else list(initial_boundary)
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
            itercount += 1
            if verbose:
                print '\n%sIteration #%i (Positive Greedy)\n%s' % ( 'Final ' if done else '', itercount, 12*'---' )
                print '>> The current estimator value is %1.4e' % min_est
                print '>> %s was added to the boundary' % ( min_term if isinstance(min_term,(list,str,ndarray)) else (str(min_term) if not (min_term is None) else 'Nothing' ) )
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
                    print '>> "%s" was removed from the boundary.' % ( min_term if isinstance(min_term,(list,str,ndarray)) else str(min_term) )
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
             fitatol  = 1e-2,     # Tolerance in fractional chance in estimator
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
        msg = 'Found %s. The negative greedy step will not be applied. Please consider turning the option on (its true by default) to investigate whether the result of the positive greedy algorithm is over-modeled.' % cyan("apply_negative = True")
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
        # A_ = pgreedy( bulk, action, fitatol=fitatol, initial_boundary=initial_boundary, verbose = True  )
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
