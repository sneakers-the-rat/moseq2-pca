from setuptools import setup

setup(
    name='moseq2-pca',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.02a',
    platforms=['mac', 'unix'],
    install_requires=['h5py', 'matplotlib', 'scipy>=0.19',
                      'tqdm', 'numpy==1.13.1', 'joblib==0.11',
                      'opencv-python', 'click', 'ruamel.yaml',
                      'dask[complete]', 'chest', 'seaborn', 'dask_jobqueue',
                      'scikit-image==0.13'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-pca = moseq2_pca.cli:cli']}
)
