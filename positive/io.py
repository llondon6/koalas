#
from __future__ import print_function
from positive import *
from positive.api import *
from positive.io import *
from six.moves import urllib

# Function to copy files
def copyfile(src,dst,verbose=True,overwrite=False):
    from shutil import copyfile as cp
    from os.path import exists,realpath
    file_already_exists = exists(dst)
    overwrite = True if not file_already_exists else overwrite
    if (file_already_exists and overwrite) or ( not file_already_exists ):
        if verbose: alert('Copying: %s --> %s'%(magenta(src),magenta(dst)))
        if file_already_exists: alert(yellow('The file already exists and will be overwritten.'))
        cp( src,dst )
    elif file_already_exists and (not overwrite):
        warning( 'File at %s already exists. Since you have not set the overwrite keyword to be True, nothing will be copied.'%red(dst) )

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
        print(">> untar: Found %i files in tarball." % len(internal_files))
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
    u = urllib.request.urlopen(url)

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
            if verbose: print(status, end=' ')
        # Close file
        f.close()
        # Get the final time
        tf = time.time()
        # Show completion notice
        if verbose: print("   Download of %1.4f MB completed in %1.4f sec" % ((file_size_dl/(10.0**6.0)),tf-t0))
        if verbose: print("   Average download rate: %1.4f Mb/sec" % ((file_size_dl/(10.0**6.0))/(tf-t0)))
        if verbose: print(('   Saving:"%s"' % file_location ))
        # If the size of this file is below the floor, delete it.
        if size_floor:
            if file_size_dl<size_floor:
                os.remove(file_location)
                if verbose: print(( '   *File is smaller than %i bytes and has been deleted.' % size_floor ))
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

    def __init__(this,attrfile=None,id=None,overwrite=False,cleanup=False,defaults=None,unstring=False,verbose=False,**kwargs):

        #
        from os.path import isfile

        #
        this.__valid__ = False
        this.source_file_path = []
        this.source_dir  = []
        if attrfile is None: attrfile = ''
        this.__unstring__ = unstring

        #
        this.overwrite = overwrite

        # Check for list input and determine if both files exist; if so, proceed
        __isfile__ = attrfile != ''
        if isinstance(attrfile,(tuple,list)):
            for k in attrfile:
                __isfile__ = __isfile__ and isfile(k)
                # print isfile(k)

        # If the input file(s) exist, learn its contents
        if __isfile__:

            if isinstance( attrfile, list ):

                # Learn list of files
                for f in attrfile:
                    this.learn_file( f, **kwargs )

            elif isinstance( attrfile, str ):

                # Learn file
                this.learn_file( attrfile, **kwargs )
            else:

                msg = 'first input (attrfile key) must of list containing strings, or single string of file location, instead it is %s'%yellow(str(attrfile.__class__.__name__))
                raise ValueError(msg)

        elif not isfile(attrfile):

            if verbose: warning('Could not find configuration file at "%s"'%red(attrfile),'smart_object')

        # Apply defaults
        if isinstance(defaults,dict):
            for attr in defaults:
                if isinstance(attr,str):
                    if attr not in this.__dict__:
                        warning('Attribute %s not found; therefore, the defualt value will be used.'%yellow(attr),this.__class__.__name__)
                        setattr(  this, attr, defaults[attr]  )


        # Only keep attributes found in file
        if cleanup:
            trash_bin = [ 'valid', 'overwrite', 'source_dir', 'source_file_path', 'unstring' ]
            for item in trash_bin:
                if item in this.__dict__:
                    delattr(this,item)

    #
    def show( this ):

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        #
        for attr in list(this.__dict__.keys()):
            value = this.__dict__[attr]
            alert( '%s = %s ' % (yellow(attr),green(str(value))) )

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
            print('[%s]>> VERBOSE mode on.' % thisfun)
            print('Lines with %s will not be considered.' % comment)

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

        if VERB: print(( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1])))
        # if True: print( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1]))

        # Correctly formatted lines will be parsed into exactly two parts
        if [2 == len(part)]:
            #
            value = []
            if part[1].split(','):
                is_number = True
                for val in part[1].split(','):
                    #
                    if  (not isnumeric(val)) or ( isnumeric(val) and (val[0]=='0') and (not ('.' in val)) ):   # IF
                        # NOTE that the conditional here interprets numbers with leading zeros as strings for python 3 compatability
                        is_number = False
                        if 'unstring' in this.__dict__:
                            if this.__unstring__: val = val.replace("'",'').replace('"','')
                        if VERB: print(( '>> Learning character: %s' % val ))
                        value.append( val )
                    else:                       # Else
                        if VERB: print(( '>> Learning number: %s' % val))
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
                if not ( attr in list(this.__dict__.keys()) ):
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
            raw = numpy.loadtxt( file_location, comments='#')
        #    try:
        #        raw = numpy.loadtxt( file_location, comments='#')
        #    except:
        #        alert('Could not load: %s'%red(file_location),thisfun)
        #        alert(red('None')+' will be output',thisfun)
        #        raw = None
        #        status = False

    else:

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        #
        alert('Could not find file: "%s". We will proceed, but %s will be returned.'%(yellow(file_location),red('None')),thisfun)
        raw = None

    #
    return raw,status


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
    elif os.path.exists(dir_):
        if verbose: alert('Directory at "%s" already exists %s.'%(green(dir_),yellow('and will be not be altered or created.')),'mkdir')
    # Check for directory existence; make if needed.
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        if verbose:
            if verbose: alert('Directory at "%s" does not yet exist %s.'%(magenta(dir_),green('and will be created')),'mkdir')
    # Return status
    return os.path.exists(dir_)


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
            for key in list(group.keys()):
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
                        print(( '[%s]>> ' % thisfun + bold('Writing') + ': "%s"'% data_file_path))
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
        for key in list(h5_file.keys()):

            # reset output directory
            this_dir = save_dir

            # extract reference object with h5 file
            ref = h5_file[ key ]

            # If the key is a group
            if type(ref) is h5py._hl.group.Group:

                #
                group_to_files( ref, this_dir )


            else: # Else, if it's a writable object

                print(('[%s]>> type(%s) = %s' % (thisfun,key,type(ref)) ))

        # If the cleanup option is true, delete the original h5 file
        if cleanup:
            #
            print(('[%s]>> Removing the original h5 file at: "%s"' % (thisfun,h5_path) ))
            os.remove(h5_path)

    else:

        # Raise Error
        raise NameError('No file at "%s".' % h5_path)

# Useful function for getting parent directory
def parent(path):
    '''
    Simple wrapper for getting absolute parent directory
    '''
    import os
    return os.path.abspath(os.path.join(path, os.pardir))+'/'

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
        print(('[%s]>> VERBOSE mode on.' % thisfun))

    #
    if substitute[-1] is not '\n':
        substitute = substitute + '\n'

    # If the file exists
    if os.path.isfile(file_path):
        #
        if VERB:
            print(( '[%s]>> Found "%s"' % (thisfun,file_path) ))
        # Create temp file
        fh, abs_path = mkstemp()
        if VERB: print(( '[%s]>> Temporary file created at "%s"' % (thisfun,abs_path) ))
        new_file = open(abs_path,'w')
        old_file = open(file_path)
        for line in old_file:
            pattern_found = line.find(pattern) != -1
            if pattern_found:
                if VERB:
                    print(( '[%s]>> Found pattern "%s" in line:\n\t"%s"' % (thisfun,pattern,line) ))
                new_file.write(substitute)
                if VERB:
                    print(( '[%s]>> Line replaced with:\n\t"%s"' % (thisfun,substitute) ))
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
        if VERB: print(( '[%s]>> Replacing original file with the temporary file.' % (thisfun) ))
    else:
        #
        if VERB:
            print(( '[%s]>> File not found at "%s"' % (thisfun,file_path) ))
        if VERB:
            print(( '[%s]>> Creating new file at "%s"' % (thisfun,file_path) ))
        #
        file = open( file_path, 'w' )
        if substitute[-1]!='\n':
            substitute = substitute + '\n'
        #
        if VERB:
            print(( '[%s]>> Writing "%s"' % (thisfun,substitute) ))
        #
        file.write(substitute)
        file.close()
    #
    if VERB:
        print(('[%s] All done!',thisfun))


# Simple function to determine whether or not a string is intended to be a
# number: can it be cast as float?
def isnumeric( s ):
    try:
        float(s)
        ans = True
    except:
        ans = False
    return ans
