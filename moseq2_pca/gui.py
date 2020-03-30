from moseq2_pca.util import recursive_find_h5s, command_with_config,\
    select_strel, initialize_dask, recursively_load_dict_contents_from_group,\
    shutdown_dask, get_timestamp_path, get_metadata_path
from moseq2_pca.viz import display_components, scree_plot, changepoint_dist
from moseq2_pca.pca.util import apply_pca_dask, apply_pca_local,\
    train_pca_dask, get_changepoints_dask
from moseq2_pca.command_helpers.data import setup_train_pca, setup_cp_command, \
                                        get_pca_yaml_data, load_pcs_for_cp
import click
import os
import ruamel.yaml as yaml
import h5py
import warnings
import dask.array as da
import logging
import pathlib

def train_pca_command(input_dir, config_file, output_dir, output_file, output_directory=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    h5s, client, cluster, workers, cache, clean_params, save_file = \
        setup_train_pca(input_dir, config_data, output_dir, output_file, output_directory)

    dsets = [h5py.File(h5, mode='r')['/frames'] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=(config_data['chunk_size'], -1, -1)) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0)
    stacked_array[stacked_array < config_data['min_height']] = 0
    stacked_array[stacked_array > config_data['max_height']] = 0

    print('Processing {:d} total frames'.format(stacked_array.shape[0]))

    if config_data['missing_data']:
        mask_dsets = [h5py.File(h5, mode='r')['/frames_mask'] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=(config_data['chunk_size'], -1, -1)) for dset in mask_dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = da.logical_and(stacked_array_mask < config_data['mask_threshold'],
                                            stacked_array > config_data['mask_height_threshold'])
        print('Loaded mask...')

    else:
        stacked_array_mask = None

    output_dict =\
        train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                       clean_params=clean_params, use_fft=config_data['use_fft'],
                       rank=config_data['rank'], cluster_type=config_data['cluster_type'], min_height=config_data['min_height'],
                       max_height=config_data['max_height'], client=client,
                       iters=config_data['missing_data_iters'], workers=workers, cache=cache,
                       recon_pcs=config_data['recon_pcs'])


    if cluster is not None:
        try:
            shutdown_dask(cluster.scheduler)
        except:
            pass

    try:
        plt, _ = display_components(output_dict['components'], headless=True)
        plt.savefig('{}_components.png'.format(save_file))
        plt.savefig('{}_components.pdf'.format(save_file))
        plt.close()
    except:
        print('could not plot components')
    try:
        plt = scree_plot(output_dict['explained_variance_ratio'], headless=True)
        plt.savefig('{}_scree.png'.format(save_file))
        plt.savefig('{}_scree.pdf'.format(save_file))
        plt.close()
    except:
        print('could not plot scree')

    with h5py.File('{}.h5'.format(save_file), 'w') as f:
        for k, v in output_dict.items():
            f.create_dataset(k, data=v, compression='gzip', dtype='float32')

    config_data['pca_file'] = f'{save_file}.h5'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    return 'PCA has been trained successfully.'


def apply_pca_command(input_dir, index_file, config_file, output_dir, output_file, output_directory=None):

    # find directories with .dat files that either have incomplete or no extractions
    # TODO: additional post-processing, intelligent mapping of metadata to group names, make sure
    # moseq2-model processes these files correctly
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    dask_cache_path = os.path.join(pathlib.Path.home(), 'moseq2_pca')
    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    if output_directory is None:
        if 'aggregate_results' in input_dir:
            outpath = '/'.join(input_dir.split('/')[:-2])
            output_dir = os.path.join(outpath, output_dir)  # outputting pca folder in inputted base directory.
        else:
            output_dir = os.path.join(input_dir, output_dir)  # outputting pca folder in inputted base directory.
    else:
        output_dir = os.path.join(output_directory, output_dir)

    # automatically get the correct timestamp path
    h5_timestamp_path = get_timestamp_path(h5s[0])
    h5_metadata_path = get_metadata_path(h5s[0])

    if config_data['pca_file'] is None:
        pca_file = os.path.join(output_dir, 'pca.h5')

    if not os.path.exists(config_data['pca_file']):
        raise IOError('Could not find PCA components file {}'.format(config_data['pca_file']))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = output_dir+output_file

    print('Loading PCs from {}'.format(config_data['pca_file']))
    with h5py.File(config_data['pca_file'], 'r') as f:
        pca_components = f[config_data['pca_path']][...]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = '{}.yaml'.format(os.path.splitext(config_data['pca_file'])[0])

    use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(pca_yaml)

    if use_fft:
        print('Using FFT...')

    with warnings.catch_warnings():
        #warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
        if config_data['cluster_type'] == 'nodask':
            apply_pca_local(pca_components=pca_components, h5s=h5s, yamls=yamls,
                            use_fft=use_fft, clean_params=clean_params,
                            save_file=save_file, chunk_size=config_data['chunk_size'],
                            mask_params=mask_params, fps=config_data['fps'],
                            missing_data=missing_data)

        else:
            client, cluster, workers, cache =\
             initialize_dask(cluster_type=config_data['cluster_type'],
                             nworkers=config_data['nworkers'],
                             cores=config_data['cores'],
                             processes=config_data['processes'],
                             memory=config_data['memory'],
                             wall_time=config_data['wall_time'],
                             queue=config_data['queue'],
                             scheduler='distributed',
                             timeout=config_data['timeout'],
                             cache_path=dask_cache_path)

            logger = logging.getLogger("distributed.utils_perf")
            logger.setLevel(logging.ERROR)

            apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                           use_fft=use_fft, clean_params=clean_params,
                           save_file=save_file, chunk_size=config_data['chunk_size'],
                           fps=config_data['fps'], client=client, missing_data=missing_data,
                           mask_params=mask_params, gui=True)

            if cluster is not None:
                try:
                    shutdown_dask(cluster.scheduler)
                except:
                    pass

    config_data['pca_file_scores'] = save_file+'.h5'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    try:
        with open(index_file, 'r') as f:
            index_params = yaml.safe_load(f)
        f.close()
        index_params['pca_path'] = config_data['pca_file_scores']

        with open(index_file, 'w') as f:
            yaml.safe_dump(index_params, f)

        f.close()
    except:
        print('moseq2-index not found, did not update paths')

    return 'PCA Scores have been successfully computed.'


def compute_changepoints_command(input_dir, config_file, output_dir, output_file, output_directory=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    pca_file_scores, pca_file_components, h5s, yamls, save_file = \
        setup_cp_command(input_dir, config_file, config_data, output_dir, output_file, output_directory)

    pca_components, changepoint_params, cluster, client, missing_data, mask_params = \
        load_pcs_for_cp(pca_file_components, config_data)

    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.ERROR)

    get_changepoints_dask(pca_components=pca_components, pca_scores=pca_file_scores,
                          h5s=h5s, yamls=yamls, changepoint_params=changepoint_params,
                          save_file=save_file, chunk_size=config_data['chunk_size'],
                          fps=config_data['fps'], client=client, missing_data=missing_data,
                          mask_params=mask_params, gui=True)

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

    return 'Model-free syllable changepoints have been successfully computed.'
