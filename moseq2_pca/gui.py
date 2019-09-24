from moseq2_pca.util import recursive_find_h5s, command_with_config,\
    select_strel, initialize_dask, recursively_load_dict_contents_from_group,\
    shutdown_dask, get_timestamp_path, get_metadata_path
from moseq2_pca.viz import display_components, scree_plot, changepoint_dist
from moseq2_pca.pca.util import apply_pca_dask, apply_pca_local,\
    train_pca_dask, get_changepoints_dask
import click
import os
import ruamel.yaml as yaml
import datetime
import h5py
import warnings
import dask.array as da
import tqdm
import pathlib


def train_pca_command(input_dir, cluster_type, output_dir, gaussfilter_space,
              gaussfilter_time, medfilter_space, medfilter_time, missing_data, missing_data_iters,
              mask_threshold, mask_height_threshold, min_height, max_height, tailfilter_size,
              tailfilter_shape, use_fft, recon_pcs, rank, output_file, chunk_size,
              config_file, local_processes, queue, nworkers,
              cores, processes, memory, wall_time, timeout):

    dask_cache_path = os.path.join(pathlib.Path.home(), 'moseq2_pca')
    # find directories with .dat files that either have incomplete or no extractions

    if missing_data and use_fft:
        raise NotImplementedError("FFT and missing data not implemented yet")

    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    params['start_time'] = timestamp
    params['inputs'] = h5s

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    if os.path.exists('{}.h5'.format(save_file)):
        raise IOError('{}.h5 already exists, delete before recomputing'.format(save_file))

    config_store = '{}.yaml'.format(save_file)
    with open(config_store, 'w') as f:
        yaml.dump(params, f, Dumper=yaml.RoundTripDumper)

    tailfilter = select_strel(tailfilter_shape, tailfilter_size)

    clean_params = {
        'gaussfilter_space': gaussfilter_space,
        'gaussfilter_time': gaussfilter_time,
        'tailfilter': tailfilter,
        'medfilter_time': medfilter_time,
        'medfilter_space': medfilter_space
    }

    # dask.set_options(temporary_directory='/home/jmarkow/dask-tmp')

    client, cluster, workers, cache =\
        initialize_dask(cluster_type=cluster_type,
                        nworkers=nworkers,
                        cores=cores,
                        processes=processes,
                        local_processes=local_processes,
                        memory=memory,
                        wall_time=wall_time,
                        queue=queue,
                        timeout=timeout,
                        scheduler='distributed',
                        cache_path=dask_cache_path)

    dsets = [h5py.File(h5, mode='r')['/frames'] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0)
    stacked_array[stacked_array < min_height] = 0
    stacked_array[stacked_array > max_height] = 0

    print('Processing {:d} total frames'.format(stacked_array.shape[0]))

    if missing_data:
        mask_dsets = [h5py.File(h5, mode='r')['/frames_mask'] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in mask_dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = da.logical_and(stacked_array_mask < mask_threshold,
                                            stacked_array > mask_height_threshold)
        print('Loaded mask...')
        # stacked_array_mask = dask.compute(stacked_array_mask)
    else:
        stacked_array_mask = None

    output_dict =\
        train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                       clean_params=clean_params, use_fft=use_fft,
                       rank=rank, cluster_type=cluster_type, min_height=min_height,
                       max_height=max_height, client=client,
                       iters=missing_data_iters, workers=workers, cache=cache,
                       recon_pcs=recon_pcs)


    if cluster is not None:
        try:
            shutdown_dask(cluster.scheduler)
        except:
            pass

    if True:
        plt, _ = display_components(output_dict['components'], headless=True)
        plt.savefig('{}_components.png'.format(save_file))
        plt.savefig('{}_components.pdf'.format(save_file))
        plt.close()

        plt = scree_plot(output_dict['explained_variance_ratio'], headless=True)
        plt.savefig('{}_scree.png'.format(save_file))
        plt.savefig('{}_scree.pdf'.format(save_file))
        plt.close()

    with h5py.File('{}.h5'.format(save_file)) as f:
        for k, v in output_dict.items():
            f.create_dataset(k, data=v, compression='gzip', dtype='float32')

    return True


def apply_pca_command(input_dir, cluster_type, output_dir, output_file, h5_path, h5_mask_path,
              pca_path, pca_file, chunk_size, fill_gaps, fps, detrend_window,
              config_file, queue, nworkers, cores, processes, memory, wall_time, timeout):
    # find directories with .dat files that either have incomplete or no extractions
    # TODO: additional post-processing, intelligent mapping of metadata to group names, make sure
    # moseq2-model processes these files correctly

    dask_cache_path = os.path.join(pathlib.Path.home(), 'moseq2_pca')
    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    # automatically get the correct timestamp path
    h5_timestamp_path = get_timestamp_path(h5s[0])
    h5_metadata_path = get_metadata_path(h5s[0])

    if pca_file is None:
        pca_file = os.path.join(output_dir, 'pca.h5')

    if not os.path.exists(pca_file):
        raise IOError('Could not find PCA components file {}'.format(pca_file))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    print('Loading PCs from {}'.format(pca_file))
    with h5py.File(pca_file, 'r') as f:
        pca_components = f[pca_path][...]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = '{}.yaml'.format(os.path.splitext(pca_file)[0])

    # todo detect missing data and mask parameters, then 0 out, fill in, compute scores...
    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)
            if 'use_fft' in pca_config.keys() and pca_config['use_fft']:
                print('Will use FFT...')
                use_fft = True
            else:
                use_fft = False

            tailfilter = select_strel(pca_config['tailfilter_shape'],
                                      tuple(pca_config['tailfilter_size']))

            clean_params = {
                'gaussfilter_space': pca_config['gaussfilter_space'],
                'gaussfilter_time': pca_config['gaussfilter_time'],
                'tailfilter': tailfilter,
                'medfilter_time': pca_config['medfilter_time'],
                'medfilter_space': pca_config['medfilter_space'],
            }

            mask_params = {
                'mask_height_threshold': pca_config['mask_height_threshold'],
                'mask_threshold': pca_config['mask_threshold'],
                'min_height': pca_config['min_height'],
                'max_height': pca_config['max_height']
            }

            if 'missing_data' in pca_config.keys() and pca_config['missing_data']:
                print('Detected missing data...')
                missing_data = True
            else:
                missing_data = False

    else:
        IOError('Could not find {}'.format(pca_yaml))

    if use_fft:
        print('Using FFT...')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
        if cluster_type == 'nodask':
            apply_pca_local(pca_components=pca_components, h5s=h5s, yamls=yamls,
                            use_fft=use_fft, clean_params=clean_params,
                            save_file=save_file, chunk_size=chunk_size,
                            mask_params=mask_params, fps=fps,
                            missing_data=missing_data)

        else:
            client, cluster, workers, cache =\
             initialize_dask(cluster_type=cluster_type,
                             nworkers=nworkers,
                             cores=cores,
                             processes=processes,
                             memory=memory,
                             wall_time=wall_time,
                             queue=queue,
                             scheduler='distributed',
                             timeout=timeout,
                             cache_path=dask_cache_path)
            apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                           use_fft=use_fft, clean_params=clean_params,
                           save_file=save_file, chunk_size=chunk_size,
                           fps=fps, client=client, missing_data=missing_data,
                           mask_params=mask_params)

            if cluster is not None:
                try:
                    shutdown_dask(cluster.scheduler)
                except:
                    pass
    return True

def compute_changepoints_command(input_dir, output_dir, output_file, cluster_type, pca_file_components,
                         pca_file_scores, pca_path, neighbors, threshold, klags, sigma, dims, fps, h5_path,
                         h5_mask_path, chunk_size, config_file,
                         queue, nworkers, cores, processes, memory, wall_time, timeout):

    dask_cache_path = os.path.join(pathlib.Path.home(), 'moseq2_pca')
    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    h5_timestamp_path = get_timestamp_path(h5s[0])

    if pca_file_components is None:
        pca_file_components = os.path.join(output_dir, 'pca.h5')

    if pca_file_scores is None:
        pca_file_scores = os.path.join(output_dir, 'pca_scores.h5')

    if not os.path.exists(pca_file_components):
        raise IOError('Could not find PCA components file {}'.format(pca_file_components))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    print('Loading PCs from {}'.format(pca_file_components))
    with h5py.File(pca_file_components, 'r') as f:
        pca_components = f[pca_path][...]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = '{}.yaml'.format(os.path.splitext(pca_file_components)[0])

    # todo detect missing data and mask parameters, then 0 out, fill in, compute scores...
    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

            if 'missing_data' in pca_config.keys() and pca_config['missing_data']:
                print('Detected missing data...')
                missing_data = True
                mask_params = {
                    'mask_height_threshold': pca_config['mask_height_threshold'],
                    'mask_threshold': pca_config['mask_threshold']
                }
            else:
                missing_data = False
                pca_file_scores = None
                mask_params = None

            if missing_data and not os.path.exists(pca_file_scores):
                raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    changepoint_params = {
        'k': klags,
        'sigma': sigma,
        'peak_height': threshold,
        'peak_neighbors': neighbors,
        'rps': dims
    }

    client, cluster, workers, cache =\
        initialize_dask(cluster_type=cluster_type,
                        nworkers=nworkers,
                        cores=cores,
                        processes=processes,
                        memory=memory,
                        wall_time=wall_time,
                        queue=queue,
                        scheduler='distributed',
                        timeout=timeout,
                        cache_path=dask_cache_path)

    get_changepoints_dask(pca_components=pca_components, pca_scores=pca_file_scores,
                          h5s=h5s, yamls=yamls, changepoint_params=changepoint_params,
                          save_file=save_file, chunk_size=chunk_size,
                          fps=fps, client=client, missing_data=missing_data,
                          mask_params=mask_params)

    if cluster is not None:
        try:
            shutdown_dask(cluster.scheduler)
        except:
            pass

    if True:
        import numpy as np
        with h5py.File('{}.h5'.format(save_file), 'r') as f:
            cps = recursively_load_dict_contents_from_group(f, 'cps')
        block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])
        out = changepoint_dist(block_durs, headless=True)
        if out:
            fig, _ = out
            fig.savefig('{}_dist.png'.format(save_file))
            fig.savefig('{}_dist.pdf'.format(save_file))
            fig.close('all')

    return True