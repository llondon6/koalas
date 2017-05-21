#
from positive import *

# Class for basic print manipulation
class print_format:
   magenta = '\033[1;35m'
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
