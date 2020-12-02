#
from positive import *
from .strings import *

#
def isiterable( obj ):
    '''
    Return TRUE if object is iterable; else, FALSE.
    USAGE:
      boolean = isiterable( object )
    '''
    #
    try:
        _ = obj.__getitem__
        return True
    except:
        return False

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
                print(('[%s]>> Found "%s" or variant thereof.' % (fname,key) ))

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
            output = [output[k] for k in range(len(output)) if mask[k]]

    # Return the list of lines
    return output



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
                        if verbose: print(magenta('  ->  '+parts[0])+cyan(pattern)+magenta(parts[1]))
                    else:
                        if verbose: print(magenta('  ->  '+os.path.join(root, filename) ))
                    matches.append(os.path.join(root, filename))

    return matches

# Returns the current line number in our program.
def linenum():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Return name of calling function
def thisfun():
    import inspect
    return inspect.stack()[2][3]

# Return the current line number
def thisline():
    import inspect
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Alert wrapper
def alert(msg,fname=None,say=False,output_string=False,heading=None,header=None,pattern=None,verbose=True,fname_color=None):
    if verbose:
        import os
        msg = str(msg)
        if fname is None:
            fname = thisfun()
            if fname == "<module>": fname = 'notebook'
            if fname == "__main__": fname = 'script'
        if say: os.system( 'say "%s"' % msg )
        fname_color_fun = eval( 'cyan' if (fname_color is None) else fname_color )
        _msg = '('+fname_color_fun(fname)+')>> '+msg
        if heading or header:
            hl = ('-~' if pattern is None else pattern) * int( len(_msg.replace('033','') )/2 )
            _msg = '\n# %s #\n%s\n# %s #\n'%(hl,bold(_msg),hl)
        if verbose:
            if not output_string:
                print(_msg)
            else:
                return _msg

# Wrapper for OS say
def say(msg,fname=None):
    import os
    msg = str(msg)
    if fname is None:
        fname = thisfun()
    if msg:
        os.system( 'say "%s says: %s"' % (fname,msg) )

# Warning wrapper
def warning(msg,fname=None,output_string=False,heading=None,header=None,pattern=None,verbose=True):
    #
    import sys,os
    # Get line number
    exc_type, exc_obj, exc_tb = sys.exc_info()
    flname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1] if exc_tb is not None else 'unknown'
    #
    msg = str(msg)
    if fname is None:
        fname = thisfun()

    if exc_type is not None:
        _msg = bold('('+yellow(fname+'!,%s,@%i'%(flname,exc_tb.tb_lineno))+')>> ')+msg
    else:
        _msg = bold('('+yellow(fname+'!')+')')+'>> '+msg

    if heading or header:
        hl = ('-~' if pattern is None else pattern) * int( len(_msg.replace('033','') )/2 )
        _msg = '\n# %s #\n%s\n# %s #\n'%(hl,bold(_msg),hl)
    if not output_string:
        if verbose: print(_msg)
    else:
        return _msg

# Error wrapper
def error(msg,fname=None):
    #
    if fname is None:
        fname = thisfun()
    #
    import sys,os
    exc_type, exc_obj, exc_tb = sys.exc_info()
    #
    if exc_type is not None:
        flname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        _msg = bold('('+red(fname+'!!,%s,@%i'%(flname,exc_tb.tb_lineno))+')>> ')+msg
    else:
        _msg = bold('('+red(fname+'!!')+')')+'>> '+msg
    # Treat all as raise AssertionErrors
    raise AssertionError( _msg )


#
def parts(dict_object):
    '''
    Given dictionary with sortable keys [a1,a2,a3 ...] and values [v1,v2,v3 ...]
    i.e. dict_object = { a1:v1, a2:v2, ... }
    Output a tuple with [a1,a2,a3 ...] and [v1,v2,v3 ...]
    i.e. [a1,a2,a3 ...], [ v1,v2,v3 ...] 
    '''
    
    # Import usefuls
    from numpy import sort, array
    
    #
    if not isinstance(dict_object,dict):
        error('First input must be dictionary.')
    
    #
    keys = sort(dict_object.keys())
    
    #
    vals = array( [ dict_object[k] for k in keys ] )
    
    #
    return keys,vals