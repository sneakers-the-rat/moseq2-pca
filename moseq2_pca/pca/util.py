from moseq2_pca.util import clean_frames, insert_nans, read_yaml
from dask.distributed import as_completed
from dask.diagnostics import ProgressBar
from dask.distributed import progress
import dask.array.linalg as lng
import dask
import numpy as np
import h5py
import dask.array as da
import tqdm


def train_pca_dask(h5s, h5_path, chunk_size, clean_params, use_fft, rank,
                   cluster_type, client, cluster, workers, cache):

    dsets = [h5py.File(h5, mode='r')[h5_path] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0).astype('float32')
    nfeatures = stacked_array.shape[1] * stacked_array.shape[2]

    if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
        stacked_array = stacked_array.map_overlap(
            clean_frames, depth=(20, 0, 0), boundary='reflect', dtype='float32', **clean_params)
    else:
        stacked_array = stacked_array.map_blocks(clean_frames, dtype='float32', **clean_params)

    if use_fft:
        print('Using FFT...')
        stacked_array = stacked_array.map_blocks(
            lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
            dtype='float32')

    # todo, abstract this into another function, add support for missing data
    # (should be simple, just need a mask array, then repeat calculation to convergence)

    stacked_array = stacked_array.reshape(-1, nfeatures)
    nsamples, nfeatures = stacked_array.shape
    mean = stacked_array.mean(axis=0)
    u, s, v = lng.svd_compressed(stacked_array-mean, rank, 0)
    total_var = stacked_array.var(ddof=1, axis=0).sum()

    if cluster_type == 'local':
        with ProgressBar():
            s, v, mean, total_var = dask.compute(s, v, mean, total_var,
                                                 cache=cache)
    elif cluster_type == 'slurm':
        futures = client.compute([s, v, mean, total_var])
        progress(futures)
        s, v, mean, total_var = client.gather(futures)
        cluster.stop_workers(workers)

    print('\nCalculation complete...')

    # correct the sign of the singular vectors

    tmp = np.argmax(np.abs(v), axis=1)
    correction = np.sign(v[np.arange(v.shape[0]), tmp])
    v *= correction[:, None]

    explained_variance = s**2 / (nsamples-1)
    explained_variance_ratio = explained_variance / total_var

    output_dict = {
        'components': v,
        'singular_values': s,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'mean': mean
    }

    return output_dict


def apply_pca_local(pca_components, h5s, yamls, use_fft, clean_params,
                    save_file, chunk_size, h5_metadata_path, h5_timestamp_path,
                    h5_path, fps=30):

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:
        for h5, yml in tqdm.tqdm(zip(h5s, yamls), total=len(h5s),
                                 desc='Computing scores'):

            data = read_yaml(yml)
            uuid = data['uuid']

            with h5py.File(h5, 'r') as f:

                frames = clean_frames(f[h5_path].value.astype('float32'), **clean_params)

                if use_fft:
                    frames = np.fft.fftshift(np.abs(np.fft.fft2(frames)), axes=(1, 2))

                frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

                if h5_timestamp_path is not None:
                    timestamps = f[h5_timestamp_path].value / 1000.0
                else:
                    timestamps = np.arange(frames.shape[0]) / fps

                if h5_metadata_path is not None:
                    metadata_name = 'metadata/{}'.format(uuid)
                    f.copy(h5_metadata_path, f_scores, name=metadata_name)

            scores = frames.dot(pca_components.T)
            scores, score_idx, _ = insert_nans(data=scores, timestamps=timestamps,
                                               fps=int(1 / np.mean(np.diff(timestamps))))

            f_scores.create_dataset('scores/{}'.format(uuid), data=scores,
                                    dtype='float32', compression='gzip')
            f_scores.create_dataset('scores_idx/{}'.format(uuid), data=score_idx,
                                    dtype='float32', compression='gzip')


def apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                   save_file, chunk_size, h5_metadata_path, h5_timestamp_path,
                   h5_path, client, fps=30):

    futures = []
    uuids = []

    for h5, yml in zip(h5s, yamls):
        data = read_yaml(yml)
        uuid = data['uuid']

        dset = h5py.File(h5, mode='r')[h5_path]
        frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')

        if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
            frames = frames.map_overlap(
                clean_frames, depth=(20, 0, 0), boundary='reflect', dtype='float32', **clean_params)
        else:
            frames = frames.map_blocks(clean_frames, dtype='float32', **clean_params)

        if use_fft:
            frames = frames.map_blocks(
                lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
                dtype='float32')

        frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])
        scores = frames.dot(pca_components.T)
        futures.append(scores)
        uuids.append(uuid)

    futures = client.compute(futures)
    keys = [tmp.key for tmp in futures]

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:
        for future, result in tqdm.tqdm(as_completed(futures, with_results=True), total=len(futures),
                                        desc="Computing scores"):

            file_idx = keys.index(future.key)

            with h5py.File(h5s[file_idx], mode='r') as f:
                if h5_timestamp_path is not None:
                    timestamps = f[h5_timestamp_path].value / 1000.0
                else:
                    timestamps = np.arange(frames.shape[0]) / fps

                if h5_metadata_path is not None:
                    metadata_name = 'metadata/{}'.format(uuids[file_idx])
                    f.copy(h5_metadata_path, f_scores, name=metadata_name)

            scores, score_idx, _ = insert_nans(data=result, timestamps=timestamps,
                                               fps=int(1 / np.mean(np.diff(timestamps))))

            f_scores.create_dataset('scores/{}'.format(uuids[file_idx]), data=scores,
                                    dtype='float32', compression='gzip')
            f_scores.create_dataset('scores_idx/{}'.format(uuids[file_idx]), data=score_idx,
                                    dtype='float32', compression='gzip')
