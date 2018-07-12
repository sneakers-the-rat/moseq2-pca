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
import pathlib


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
            try:
                if file.endswith(ext):
                    with h5py.File(os.path.join(root, file), 'r') as f:
                        if 'frames' not in f.keys():
                            continue
                    h5s.append(os.path.join(root, file))
                    yamls.append(os.path.join(root, yaml_file))
                    dicts.append(read_yaml(os.path.join(root, yaml_file)))
            except OSError:
                print('Error loading {}'.format(os.path.join(root, file)))

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
        n = np.ceil(sig * 4)

    points = np.arange(-n, n)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 medfilter_time=None, gaussfilter_time=None, detrend_time=None,
                 tailfilter=None, tail_threshold=5):
    out = np.zeros_like(frames)
    if tailfilter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(
                frames[i], cv2.MORPH_OPEN, tailfilter) > tail_threshold
            out[i] = frames[i] * mask.astype(frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                out[i] = cv2.medianBlur(frames[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(frames[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    if medfilter_time is not None and np.all(np.array(medfilter_time) > 0):
        for idx, i in np.ndenumerate(frames[0]):
            for medfilt in medfilter_time:
                out[:, idx[0], idx[1]] = \
                    scipy.signal.medfilt(frames[:, idx[0], idx[1]], medfilt)

    if gaussfilter_time is not None and gaussfilter_time > 0:
        kernel = gaussian_kernel1d(sig=gaussfilter_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = \
                np.convolve(frames[:, idx[0], idx[1]], kernel, mode='same')

    if detrend_time is not None and detrend_time > 0:
        kernel = gaussian_kernel1d(sig=detrend_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = \
                frames[:, idx[0], idx[1]] -\
                gauss_smooth(frames[:, idx[0], idx[1]], kernel=kernel)

    return out


def select_strel(string='e', size=(10, 10)):
    if string is None or 'none' in string or np.all(np.array(size) == 0):
        strel = None
    elif string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return strel


def insert_nans(timestamps, data, fps=30):

    df_timestamps = np.diff(
        np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))
    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)

    if filled_data.ndim == 1:
        isvec = True
        filled_data = filled_data[:, None]
    else:
        isvec = False

    nframes, nfeatures = filled_data.shape

    for idx in fill_idx[::-1]:
        ninserts = int(missing_frames[idx] - 1)
        data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
        insert_timestamps = timestamps[idx - 1] + \
            np.cumsum(np.ones(ninserts,) * 1.0 / fps)
        filled_data = np.insert(filled_data, idx,
                                np.ones((ninserts, nfeatures)) * np.nan, axis=0)
        filled_timestamps = np.insert(
            filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def read_yaml(yaml_file):

    try:
        with open(yaml_file, 'r') as f:
            dat = f.read()
            try:
                return_dict = yaml.load(dat, Loader=yaml.RoundTripLoader)
            except yaml.constructor.ConstructorError:
                return_dict = yaml.load(dat, Loader=yaml.Loader)
    except IOError:
        return_dict = {}

    return return_dict


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}

    if type(h5file) is str:
        with h5py.File(h5file, 'r') as f:
            ans = recursively_load_dict_contents_from_group(f, path)
            return ans

    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans


def initialize_dask(nworkers=50, processes=4, memory='4GB', cores=2,
                    wall_time='01:00:00', queue='debug',
                    cluster_type='local', scheduler='dask', timeout=10,
                    cache_path=os.path.join(pathlib.Path.home(), 'moseq2_pca')):

    # only use distributed if we need it

    client = None
    workers = None
    cache = None
    cluster = None

    if cluster_type == 'local' and scheduler == 'dask':

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        cache = Chest(path=cache_path)

    elif cluster_type == 'local' and scheduler == 'distributed':

        client = Client(processes=False)

    elif cluster_type == 'slurm':

        cluster = SLURMCluster(processes=processes,
                               cores=cores,
                               memory=memory,
                               queue=queue,
                               wall_time=wall_time,
                               local_directory=cache_path)

        workers = cluster.start_workers(nworkers)
        client = Client(cluster)

        nworkers = 0

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            pbar = tqdm.tqdm(total=len(workers) * processes,
                             desc="Intializing workers")

            elapsed_time = (time.time() - start_time) / 60.0

            while nworkers < len(workers) * processes and elapsed_time < timeout:
                tmp = len(client.scheduler_info()['workers'])
                if tmp - nworkers > 0:
                    pbar.update(tmp - nworkers)
                nworkers += tmp - nworkers
                time.sleep(5)
                elapsed_time = (time.time() - start_time) / 60.0

            pbar.close()

    return client, cluster, workers, cache


def get_rps(frames, rps=600, normalize=True):

    if frames.ndim == 3:
        use_frames = frames.reshape(-1, np.prod(frames.shape[1:]))
    elif frames.ndim == 2:
        use_frames = frames

    rproj = use_frames.dot(np.random.randn(use_frames.shape[1], rps))

    if normalize:
        rproj = scipy.stats.zscore(scipy.stats.zscore(rproj).T)

    return rproj


def get_changepoints(scores, k=5, sigma=3, peak_height=.5, peak_neighbors=1, baseline=True, timestamps=None):

    if type(k) is not int:
        k = int(k)

    if type(peak_neighbors) is not int:
        peak_neighbors = int(peak_neighbors)

    with np.errstate(all='ignore'):

        normed_df = deepcopy(scores)
        nanidx = np.isnan(normed_df)
        normed_df[nanidx] = 0

        if sigma is not None and sigma > 0:
            for i in range(scores.shape[0]):
                normed_df[i, :] = gauss_smooth(normed_df[i, :], sigma)

        normed_df[:, k // 2:-k // 2] = (normed_df[:, k:] - normed_df[:, :-k])**2

        normed_df[nanidx] = np.nan
        normed_df[:, :int(6 * sigma)] = np.nan
        normed_df[:, -int(6 * sigma):] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            normed_df = np.nanmean(normed_df, axis=0)

        if baseline:
            normed_df -= np.nanmin(normed_df)

        if timestamps is not None:
            normed_df, _, _ = insert_nans(
                timestamps, normed_df, fps=int(1 / np.mean(np.diff(timestamps))))

        normed_df = np.squeeze(normed_df)
        cps = scipy.signal.argrelextrema(
            normed_df, np.greater, order=peak_neighbors)[0]
        cps = cps[np.argwhere(normed_df[cps] > peak_height)]

    return cps, normed_df
