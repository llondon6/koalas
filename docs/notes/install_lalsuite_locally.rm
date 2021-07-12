# Installing lalsuite

## 1. Make a conda env and clone lalsuite

```bash
conda create --name hack-phenomx python=2.7
conda activate hack-phenomx
cd ${CONDA_PREFIX}
mkdir src
cd src
git clone git@git.ligo.org:lionel.london/lalsuite.git
```

## 2. Install python dependencies

```bash
conda install numpy swig cython matplotlib h5py
```

## 3. Configure, make and make install

```bash
# NOTE that there are many things diabled here so that you don't waste time faffing about with unnecessary libs -- we are onyl interested in lalsimulation and swig here
./configure --prefix=${CONDA_PREFIX} --enable-swig-python --disable-lalstochastic --disable-lalxml --disable-lalinference --disable-laldetchar --disable-lalapps --disable-lalframe --disable-lalmetaio

#
make -j

#
make install 
```

## 4. Add env to jupyter 

```bash
conda activate hack-phenomx
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

2. Configure and install with 

```bash
./configure --enable-float --enable-sse 
make
sudo make install
```

The flags here are key to getting the correct library file compiled. Reference: https://stackoverflow.com/questions/37267441/configure-warning-fftw3f-library-not-found-the-slower-fftpack-library-will-be
