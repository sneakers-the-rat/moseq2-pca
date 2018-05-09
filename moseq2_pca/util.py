from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from chest import Chest
from copy import deepcopy
import ruamel.yaml as yaml
import os
import cv2
import h5py
import numpy as np
import click
import scipy.signal
import time
import warnings
import tqdm


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            param_defaults = {}
            for param in self.params:
                if type(param) is click.core.Option:
                    param_defaults[param.human_readable_name] = param.default

            if config_file is not None:
                config_data = read_yaml(config_file)
                for param, value in ctx.params.items():
                    if param in config_data:
                        if type(value) is tuple and type(config_data[param]) is int:
                            ctx.params[param] = tuple([config_data[param]])
                        elif type(value) is tuple:
                            ctx.params[param] = tuple(config_data[param])
                        else:
                            ctx.params[param] = config_data[param]

                        if param_defaults[param] != value:
                            ctx.params[param] = value

            return super(custom_command_class, self).invoke(ctx)

    return custom_command_class


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            yaml_file = yaml_string.format(os.path.splitext(file)[0])
            if file.endswith(ext) and os.path.exists(os.path.join(root, yaml_file)):
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'frames' not in f.keys():
                        continue
                h5s.append(os.path.join(root, file))
                yamls.append(os.path.join(root, yaml_file))
                dicts.append(read_yaml(os.path.join(root, yaml_file)))

    return h5s, dicts, yamls


def gauss_smooth(signal, win_length=None, sig=1.5, kernel=None):

    if kernel is None:
        kernel = gaussian_kernel1d(n=win_length, sig=sig)

    result = scipy.signal.fftconvolve(signal, kernel, mode='same')

#    win_length = len(kernel)

#     result[:win_length] = np.nan
#     result[-win_length:] = np.nan

    return result


def gaussian_kernel1d(n=None, sig=3):

    if n is None:
        n = np.ceil(sig*4)

    points = np.arange(-n, n)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 medfilter_time=None, gaussfilter_time=None, detrend_time=None,
                 tailfilter=None, tail_threshold=5):

    cleaned_frames = frames

    if tailfilter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(cleaned_frames[i], cv2.MORPH_OPEN, tailfilter) > tail_threshold
            cleaned_frames[i] = cleaned_frames[i] * mask.astype(cleaned_frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                cleaned_frames[i] = cv2.medianBlur(cleaned_frames[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            cleaned_frames[i] = cv2.GaussianBlur(cleaned_frames[i], (21, 21),
                                                 gaussfilter_space[0], gaussfilter_space[1])

    if medfilter_time is not None and np.all(np.array(medfilter_time) > 0):
        for idx, i in np.ndenumerate(cleaned_frames[0]):
            for medfilt in medfilter_time:
                cleaned_frames[:, idx[0], idx[1]] = \
                    scipy.signal.medfilt(cleaned_frames[:, idx[0], idx[1]], medfilt)

    if gaussfilter_time is not None and gaussfilter_time > 0:
        kernel = gaussian_kernel1d(sig=gaussfilter_time)
        for idx, i in np.ndenumerate(cleaned_frames[0]):
            cleaned_frames[:, idx[0], idx[1]] = \
                np.convolve(cleaned_frames[:, idx[0], idx[1]], kernel, mode='same')

    if detrend_time is not None and detrend_time > 0:
        kernel = gaussian_kernel1d(sig=detrend_time)
        for idx, i in np.ndenumerate(cleaned_frames[0]):
            cleaned_frames[:, idx[0], idx[1]] = \
                cleaned_frames[:, idx[0], idx[1]] -\
                gauss_smooth(cleaned_frames[:, idx[0], idx[1]], kernel=kernel)

    return cleaned_frames


def select_strel(string='e', size=(10, 10)):
    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel


def insert_nans(timestamps, data, fps=30):

    df_timestamps = np.diff(np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))
    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)
    nframes, nfeatures = filled_data.shape

    for idx in fill_idx[::-1]:
        ninserts = int(missing_frames[idx]-1)
        data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
        insert_timestamps = timestamps[idx-1] + np.cumsum(np.ones(ninserts,) * 1.0 / fps)
        filled_data = np.insert(filled_data, idx,
                                np.ones((ninserts, nfeatures)) * np.nan, axis=0)
        filled_timestamps = np.insert(filled_timestamps, idx, insert_timestamps)

    return filled_data, data_idx, filled_timestamps


def read_yaml(yaml_file):

    with open(yaml_file, 'r') as f:
        dat = f.read()
        try:
            return_dict = yaml.load(dat, Loader=yaml.RoundTripLoader)
        except yaml.constructor.ConstructorError:
            return_dict = yaml.load(dat, Loader=yaml.Loader)

    return return_dict


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def initialize_dask(nworkers, processes, memory, threads, wall_time, queue,
                    cluster_type='local', scheduler='dask', timeout=10):

    # only use distributed if we need it

    client = None
    workers = None
    cache = None
    cluster = None

    if cluster_type == 'local' and scheduler == 'dask':

        cache = Chest()

    elif cluster_type == 'local' and scheduler == 'distributed':

        client = Client(processes=False)

    elif cluster_type == 'slurm':

        cluster = SLURMCluster(processes=processes, threads=threads,
                               memory=memory, queue=queue, wall_time=wall_time)
        workers = cluster.start_workers(nworkers)
        client = Client(cluster)

        nworkers = 0

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            pbar = tqdm.tqdm(total=len(workers)*processes, desc="Intializing workers")

            elapsed_time = (time.time() - start_time) / 60.0

            while nworkers < len(workers)*processes and elapsed_time < timeout:
                tmp = len(client.scheduler_info()['workers'])
                if tmp - nworkers > 0:
                    pbar.update(tmp - nworkers)
                nworkers += tmp - nworkers
                time.sleep(5)
                elapsed_time = (time.time() - start_time) / 60.0

            pbar.close()

    return client, cluster, workers, cache
