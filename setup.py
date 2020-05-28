from setuptools import setup, find_packages
import subprocess
import os.path
import codecs
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='moseq2-pca',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version=get_version('moseq2_pca/__init__.py'),
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py==2.10.0', 'matplotlib==3.1.2', 'scipy==1.4.1', 'pathspec==0.5.3',
                      'tqdm==4.40.0', 'numpy==1.18.3', 'joblib==0.14.0',
                      'click==7.0', 'ruamel.yaml==0.16.5', 'dask==2.11.0',
                      'distributed==2.11.0', 'chest==0.2.3', 'seaborn==0.9.0', 'dask-jobqueue==0.7.0',
                      'scikit-image==0.16.2', 'bokeh==1.4.0', 'psutil==5.6.7', 'ipywidgets==7.5.1'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-pca = moseq2_pca.cli:cli']}
)
