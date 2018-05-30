from moseq2_pca.util import clean_frames, insert_nans, read_yaml, get_changepoints, get_rps
from dask.distributed import as_completed
from dask.diagnostics import ProgressBar
from dask.distributed import progress
import dask.array.linalg as lng
import dask.array as da
import dask
import numpy as np
import h5py
import tqdm


def mask_data(original_data, mask, new_data):

    # need to make a copy otherwise distributed scheduler barfs
    output = original_data.copy()
    output[mask] = new_data[mask]

    return output


def train_pca_dask(dask_array, clean_params, use_fft, rank,
                   cluster_type, client, cluster, workers,
                   cache, mask=None, iters=10, recon_pcs=3,
                   min_height=10, max_height=100):

    missing_data = False
    _, r, c = dask_array.shape
    nfeatures = r * c

    if cluster_type == 'slurm':
        dask_array = client.persist(dask_array)

        if mask is not None:
            mask = client.persist(mask)

    if mask is not None:
        missing_data = True
        dask_array[mask] = 0
        mask = mask.reshape(-1, nfeatures)

    if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
        dask_array = dask_array.map_overlap(
            clean_frames, depth=(20, 0, 0), boundary='reflect', dtype='float32', **clean_params)
    else:
        dask_array = dask_array.map_blocks(clean_frames, dtype='float32', **clean_params)

    if use_fft:
        print('Using FFT...')
        dask_array = dask_array.map_blocks(
            lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
            dtype='float32')

    # todo, abstract this into another function, add support for missing data
    # (should be simple, just need a mask array, then repeat calculation to convergence)

    dask_array = dask_array.reshape(-1, nfeatures)
    nsamples, nfeatures = dask_array.shape
    mean = dask_array.mean(axis=0)

    if cluster_type == 'slurm':
        mean = client.persist(mean)

    # todo compute reconstruction error

    if not missing_data:
        u, s, v = lng.svd_compressed(dask_array-mean, rank, 0)
    else:
        for iter in range(iters):
            u, s, v = lng.svd_compressed(dask_array-mean, rank, 0)
            recon = u[:, :recon_pcs].dot(da.diag(s[:recon_pcs]).dot(v[:recon_pcs, :])) + mean
            recon[recon < min_height] = 0
            recon[recon > max_height] = 0
            dask_array = da.map_blocks(mask_data, dask_array, mask, recon, dtype=dask_array.dtype)
            mean = dask_array.mean(axis=0)

    total_var = dask_array.var(ddof=1, axis=0).sum()

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


# todo: for applying pca, run once to impute missing data, then get scores
def apply_pca_local(pca_components, h5s, yamls, use_fft, clean_params,
                    save_file, chunk_size, h5_metadata_path, h5_timestamp_path,
                    h5_path, h5_mask_path, mask_params, missing_data, fps=30):

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:
        for h5, yml in tqdm.tqdm(zip(h5s, yamls), total=len(h5s),
                                 desc='Computing scores'):

            data = read_yaml(yml)
            uuid = data['uuid']

            with h5py.File(h5, 'r') as f:

                frames = f[h5_path].value.astype('float32')

                if missing_data:
                    mask = f[h5_mask_path].value
                    mask = np.logical_and(mask < mask_params['mask_threshold'],
                                          frames > mask_params['mask_height_threshold'])
                    frames[mask] = 0
                    mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

                frames = clean_frames(frames, **clean_params)

                if use_fft:
                    frames = np.fft.fftshift(np.abs(np.fft.fft2(frames)), axes=(1, 2))

                frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

                if h5_timestamp_path is not None and h5_timestamp_path in f.keys():
                    timestamps = f[h5_timestamp_path].value / 1000.0
                else:
                    timestamps = np.arange(frames.shape[0]) / fps

                if h5_metadata_path is not None and 'metadata' in f.keys():
                    metadata_name = 'metadata/{}'.format(uuid)
                    f.copy(h5_metadata_path, f_scores, name=metadata_name)

            scores = frames.dot(pca_components.T)

            # if we have missing data, simply fill in, repeat the score calculation,
            # then move on
            if missing_data:
                recon = scores.dot(pca_components)
                frames[mask] = recon[mask]
                scores = frames.dot(pca_components.T)

            scores, score_idx, _ = insert_nans(data=scores, timestamps=timestamps,
                                               fps=int(1 / np.mean(np.diff(timestamps))))

            f_scores.create_dataset('scores/{}'.format(uuid), data=scores,
                                    dtype='float32', compression='gzip')
            f_scores.create_dataset('scores_idx/{}'.format(uuid), data=score_idx,
                                    dtype='float32', compression='gzip')


def apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                   save_file, chunk_size, h5_metadata_path, h5_timestamp_path,
                   h5_path, h5_mask_path, mask_params, missing_data, client, fps=30):

    futures = []
    uuids = []

    for h5, yml in zip(h5s, yamls):
        data = read_yaml(yml)
        uuid = data['uuid']

        dset = h5py.File(h5, mode='r')[h5_path]
        frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')

        if missing_data:
            mask_dset = h5py.File(h5, mode='r')[h5_mask_path]
            mask = da.from_array(mask_dset, chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

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

        if missing_data:
            recon = scores.dot(pca_components)
            frames = da.map_blocks(mask_data, frames, mask, recon, dtype=frames.dtype)
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
                if h5_timestamp_path is not None and h5_timestamp_path in f.keys():
                    timestamps = f[h5_timestamp_path].value / 1000.0
                else:
                    timestamps = np.arange(frames.shape[0]) / fps

                if h5_metadata_path is not None and 'metadata' in f.keys():
                    metadata_name = 'metadata/{}'.format(uuids[file_idx])
                    f.copy(h5_metadata_path, f_scores, name=metadata_name)

            scores, score_idx, _ = insert_nans(data=result, timestamps=timestamps,
                                               fps=int(1 / np.mean(np.diff(timestamps))))

            f_scores.create_dataset('scores/{}'.format(uuids[file_idx]), data=scores,
                                    dtype='float32', compression='gzip')
            f_scores.create_dataset('scores_idx/{}'.format(uuids[file_idx]), data=score_idx,
                                    dtype='float32', compression='gzip')


def get_changepoints_dask(changepoint_params, pca_components, h5s, yamls,
                          save_file, chunk_size, h5_path, h5_timestamp_path, h5_mask_path,
                          mask_params, missing_data, client, fps=30, pca_scores=None):

    futures = []
    uuids = []
    nrps = changepoint_params.pop('rps')

    for h5, yml in tqdm.tqdm(zip(h5s, yamls), desc='Setting up calculation', total=len(h5s)):
        data = read_yaml(yml)
        uuid = data['uuid']

        with h5py.File(h5, 'r') as f:

            dset = h5py.File(h5, mode='r')[h5_path]
            frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')

            if h5_timestamp_path is not None and h5_timestamp_path in f.keys():
                timestamps = f[h5_timestamp_path].value / 1000.0
            else:
                timestamps = np.arange(frames.shape[0]) / fps

        if missing_data and pca_scores is None:
            raise RuntimeError("Need to compute PC scores to impute missing data")
        elif missing_data:
            mask_dset = h5py.File(h5, mode='r')[h5_mask_path]
            mask = da.from_array(mask_dset, chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

            with h5py.File(pca_scores, 'r') as f:
                scores = f['scores/{}'.format(uuid)]
                scores_idx = f['scores_idx/{}'.format(uuid)].value
                scores = scores[~np.isnan(scores_idx), :]

            scores = da.from_array(scores, chunks=(frames.chunks[0], scores.shape[1]))

        frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

        if missing_data:
            recon = scores.dot(pca_components)
            frames = da.map_blocks(mask_data, frames, mask, recon, dtype=frames.dtype)

        rps = dask.delayed(lambda x: get_rps(x, rps=nrps, normalize=True), pure=False)(frames)
        cps = dask.delayed(lambda x: get_changepoints(x, timestamps=timestamps, **changepoint_params), pure=True)(rps)
        futures.append(cps)
        uuids.append(uuid)

    futures = client.compute(futures)
    keys = [tmp.key for tmp in futures]

    with h5py.File('{}.h5'.format(save_file), 'w') as f_cps:

        f_cps.create_dataset('metadata/fps', data=fps, dtype='float32')

        for future, result in tqdm.tqdm(as_completed(futures, with_results=True), total=len(futures),
                                        desc="Collecting results"):
            file_idx = keys.index(future.key)
            f_cps.create_dataset('cps_score/{}'.format(uuids[file_idx]), data=result[1],
                                 dtype='float32', compression='gzip')
            f_cps.create_dataset('cps/{}'.format(uuids[file_idx]), data=result[0] / fps,
                                 dtype='float32', compression='gzip')
