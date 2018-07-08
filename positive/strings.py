#
from positive import *

# safely join directory strings
def osjoin(a,b):
    import os
    return str( os.path.join( a, b ) )

# Class for basic print manipulation
class print_format:
   magenta = '\033[0;35m'
   cyan = '\033[0;36m'
   darkcyan = '\033[0;36m'
   blue = '\033[0;34m'
   green = '\033[92m'
   yellow = '\033[0;33m'
   red = '\033[31m'
   bold = '\033[1m'
   grey = gray = '\033[1;30m'
   ul = '\033[4m'
   end = '\x1b[0m'
   hlb = '\033[5;30;42m'
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
def hlblack(string):
    return print_format.hlb + string + print_format.end


#
def poly2latex(basis_symbols,coeffs,latex_labels=None,precision=8,term_split=1000):

    # Import useful things
    from numpy import mod

    #
    split_state = False

    # Count the number of unique domain variables
    domain_dimension = len( set(''.join(basis_symbols).replace('K','')) )

    #
    if len(basis_symbols)==len(coeffs)==0:
        basis_symbols=['']
        coeffs = [0]

    # Extract desired labels and handle defaults
    funlabel = r'f(\vec{x})' if latex_labels is None else latex_labels[0]
    varlabel = [ 'x%i'%k for k in range(domain_dimension) ] if latex_labels is None else latex_labels[1]
    prefix = '' if latex_labels is None else latex_labels[2]
    if varlabel is None:
        varlabel = [ r'x_%i'%k for k in range(domain_dimension) ]
    elif len(varlabel) != domain_dimension:
        error( 'Number of variable labels, %i, is not equal to the number of domain dimensions found, %i.'%( len(varlabel), domain_dimension )  )

    # Create a simple string representation of the fit
    latex_str = r'%s  \; &= \; %s %s\,x%s%s' % ( funlabel,
                                               (prefix+r' \, ( \,') if prefix else '',
                                               complex2str(coeffs[0],
                                               latex=True,precision=precision) if isinstance(coeffs[0],complex) else '%1.4e'%coeffs[0], r'\,x'.join( list(basis_symbols[0]) ), '' if len(coeffs)>1 else (r' \; )' if prefix else '') )
    for k,b in enumerate(coeffs[1:]):
        latex_str += r' \; + \; (%s)\,x%s%s' % ( complex2str(b,latex=True,precision=precision) if isinstance(b,complex) else '%1.4e'%b ,
                                                 r'\,x'.join( list(basis_symbols[k+1]) ),
                                                 (r' \; )' if prefix else '') if (k+1)==len(coeffs[1:]) else '' )
        #
        if ( not mod(k+2,term_split) ) and ( (k+1) < len(coeffs[1:]) ):
            latex_str += '\n  \\\\ \\nonumber\n & \quad '
            if not split_state:
                split_state = not split_state

    # Correct for a lingering multiply sign
    latex_str = latex_str.replace('(\,','(')

    # Correct for the constant term not being an explicit function of a domain variable
    latex_str = latex_str.replace('\,xK','')

    # Replace variable labels with input
    for k in range(domain_dimension):
        latex_str = latex_str.replace( 'x%i'%k, varlabel[k] )

    # Replace repeated variable labels with power notation
    for pattern in varlabel:
        latex_str = rep2pwr( latex_str, pattern, r'\,'  )

    return latex_str

# Convert poylnomial (basis symbols and coefficients) to python string
def poly2pystr(basis_symbols,coeffs,labels=None,precision=8):

    '''
    It's useful to know:

    * That "labels" is of the following form
        * labels = [range_label,domain_labels,python_prefix]
        * EXAMPLE: labels = [ 'temperature', ['day','longitude','latitude','aliens_influence_measure'], '' ]

    * The length of basis_symbols and coeffs must match.

    * basis_symbols must be consistent with positive.learning.symeval
        * EXAMPLE: basis_symbols = ['K','0','1','00']
        * this corresponds to a c0 + c1*x + c2*y + c3*x^2, and coeffs = [c0,c1,c2,c3]

    '''

    # Import usefuls
    from positive.api import error

    #
    if len(basis_symbols)==len(coeffs)==0:
        basis_symbols=['']
        coeffs = [0]

    # Count the number of unique domain variables
    domain_dimension = len( set(''.join(basis_symbols).replace('K','')) )

    # Extract desired labels and handle defaults
    funlabel = 'f' if labels is None else labels[0]
    varlabels = None if labels is None else labels[1]

    prefix = '' if labels is None else labels[2]
    postfix = '' if labels is None else ( labels[3] if len(labels)==4 else '' )

    if varlabels is None:
        varlabels = [ 'x%s'%str(k) for k in range(domain_dimension) ]
    elif len(varlabels) != domain_dimension:
        error( 'Number of variable labels, %i, is not equal to the number of domain dimensions found, %i. One posiility is that youre fitting with a 1D domain, and have attempted to use a domain label that is a tuple containing a single string which python may interpret as a string -- try defining the label as a list by using square brackets.'%( len(varlabels), domain_dimension ) , 'mvpolyfit' )

    # Replace minus signs in function name with M
    funlabel = funlabel.replace('-','M')

    # Create a simple string representation of the fit
    model_str = '%s = lambda %s:%s%s*(x%s)' % ( funlabel, ','.join(varlabels), (' %s('%prefix) if prefix else ' '  , complex2str(coeffs[0],precision=precision) if isinstance(coeffs[0],complex) else ('%%1.%ie'%precision)%coeffs[0], '*x'.join( list(basis_symbols[0]) ) )
    for k,b in enumerate(coeffs[1:]):
        model_str += ' + %s*(x%s)' % ( complex2str(b,precision=precision) if isinstance(b,complex) else ('%%1.%ie'%precision)%b , '*x'.join( list(basis_symbols[k+1]) ) )

    # Correct for a lingering multiply sign
    model_str = model_str.replace('(*','(')

    # Correct for the constant term not being an explicit function of a domain variable
    model_str = model_str.replace('*(xK)','')

    # if there is a prefix, then close the automatic ()
    model_str += ' )' if prefix else ''

    #
    model_str += postfix

    # Replace variable labels with input
    if not ( varlabels is None ):
        for k in range(domain_dimension):
            model_str = model_str.replace( 'x%i'%k, varlabels[k] )

    return model_str

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
            fmt = '%s.%if'%(r'%',precision)
            # fmt = '%s1.%if'%(r'%',precision)
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
    print(pad + str)


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
            print(symbol*int(float(columns)/float(len(symbol))))


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
