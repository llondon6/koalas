# Installing lalsuite locally for development of lalsimulation

It's easy to install lalsuite for use via `conda install -c conda-forge lalsuite `. But it can a right pain the arse to install it for development on your local machine (rather than a LIGO cluster) due to various libraries and dependencies that will naturally be missing.

The pain can be lessened, and perhaps avoided all together by noting the following:

* If you're doing development (coding in C) then it's a good bet that only a small part of lalsuite is of interest. 
* For waveform modelers that part is usually lalsimulation. This means that everything else need not be bothered with, and this means that there are a lot of dependencies that can be ignored.
* The remaining dependencies are usually simple to install (though some googling may be needed on a system by system basis).

Below I walk through the steps that worked for me during my recent install of lalsuite on my macOS Mojaje 10.14.6 macbook. 

## 1. Make a conda env and clone lalsuite

Here I call my new environment ```hack-phenomx``` but you can all yours anything. 

```bash
# Make a conda env (note that "conda create env" may be needed depending on your conda version)
conda create --name hack-phenomx python=2.7
conda activate hack-phenomx
# Clone lalsuite (you should first make your own fork of the official version)
cd ${CONDA_PREFIX}
mkdir src
cd src
git clone git@git.ligo.org:lionel.london/lalsuite.git
# At this stage you may with to immediately switch to a particular branch. If so, switch to your desired branch before compiling!
```

## 2. Install python dependencies (otherwise you'll get compile errors)

```bash
conda install numpy swig cython matplotlib h5py
```

## 3. Configure, make and make install (and cross your fingers)

Before trying this step, you may want to preimptively install fftw3f and zlib, as you could get compile errors if these libraries are not already installed. It's also possible that you may need to install other libraries. If so, google is your friend, and hopefully (like zlib) they are as simple as wget, configure, make, make install. 

```bash
#
./00boot

# NOTE that there are MANY things disabled here so that you don't waste time faffing about with unnecessary libs -- we are only interested in lalsimulation and swig here
./configure --prefix=${CONDA_PREFIX} --enable-swig-python --disable-lalstochastic --disable-lalxml --disable-lalinference --disable-laldetchar --disable-lalapps --disable-lalframe --disable-lalmetaio

# 
make -j

#
make install 
```

If this works, note the location of your lalsuite activation script (ie the script that lets your shell know where lalsuite lives). In my case its
```
/Users/book/opt/anaconda2/envs/hack-phenomx/etc/lalsuite-user-env.sh
```
So regardless of whether I'm in my conda environment, I can tell bash and/or python scripts to use this version of lalsuite by entering the bash command

```
source /Users/book/opt/anaconda2/envs/hack-phenomx/etc/lalsuite-user-env.sh
```

prior to running the bash or python code of interest.

## 4. Add your new environment to jupyter so that you can work with your development code in python via swig

```bash
# Make sure that the env is acticated
conda activate hack-phenomx
# These are the usual googlable steps for adding an env to jupyter as a kernel
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=hack-phenomx
```

After entering the above commands, you can test your configuration by:
1. Start a jupyter lab session
2. Create a new notebook -- you should see your new kernel "hack-phenomx" as an option
3. Once in your new notebook, import lalsumlation (assuming you've been able to configure make and install according to the above instructions)
```python
import lalsimulation as lals
```
4. Print the location of lalsimulation
```python
print lals.__path__
```

The output of this should be the location of your lalsimulation install in your environment. In my case its

```
['/Users/book/opt/anaconda2/envs/hack-phenomx/lib/python2.7/site-packages/lalsimulation']
```


# Installing zlib

Download, unzip, configure, make, install. Don't be shy about asking google for help.

**Reference**: https://stackoverflow.com/a/52900711

```
wget https://zlib.net/zlib-1.2.11.tar.gz
tar -xvf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure
make
make install 
```

# Installing fftw3f

1. Download: http://www.fftw.org/download.html
2. Unzip and enter fftw-3.3.9 folder
```
tar -xvf fftw-3.3.9.tar.gz
cd fftw-3.3.9
```
2. Configure and install with flag to ensure that the version lalsuite wants is compiled

```bash
./configure --enable-float --enable-sse 
make
sudo make install
```

The flags here are key to getting the correct (for lalsuite) library file compiled. 
**Reference**: https://stackoverflow.com/questions/37267441/configure-warning-fftw3f-library-not-found-the-slower-fftpack-library-will-be
