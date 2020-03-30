import os
import h5py
import logging
import pathlib
import datetime
import ruamel.yaml as yaml
from moseq2_pca.util import recursive_find_h5s, select_strel, initialize_dask, get_timestamp_path

def setup_train_pca(input_dir, config_data, output_dir, output_file, output_directory):
    dask_cache_path = os.path.join(pathlib.Path.home(), 'moseq2_pca')
    # find directories with .dat files that either have incomplete or no extractions

    if config_data['missing_data'] and config_data['use_fft']:
        raise NotImplementedError("FFT and missing data not implemented yet")

    params = config_data
    if output_directory is None:
        h5s, dicts, yamls = recursive_find_h5s(input_dir)
    else:
        h5s, dicts, yamls = recursive_find_h5s(output_directory)
    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    params['start_time'] = timestamp
    params['inputs'] = h5s

    if output_directory is None:
        # outputting pca folder in inputted base directory.
        output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), output_dir)
    else:
        output_dir = os.path.join(output_directory, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    if os.path.exists('{}.h5'.format(save_file)):
        print(f'The file {save_file}.h5 already exists.\nWould you like to overwrite it? [Y -> yes, else -> exit]\n')
        ow = input()
        if ow == 'Y':
            print('Deleting old pca.')
            os.remove(f'{save_file}.h5')
        else:
            return "Did not overwrite"

    config_store = '{}.yaml'.format(save_file)
    with open(config_store, 'w') as f:
        yaml.safe_dump(params, f)

    tailfilter = select_strel((config_data['tailfilter_shape'], config_data['tailfilter_size']))

    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'gaussfilter_time': config_data['gaussfilter_time'],
        'tailfilter': tailfilter,
        'medfilter_time': config_data['medfilter_time'],
        'medfilter_space': config_data['medfilter_space']
    }

    client, cluster, workers, cache = \
        initialize_dask(cluster_type=config_data['cluster_type'],
                        nworkers=config_data['nworkers'],
                        cores=config_data['cores'],
                        processes=config_data['processes'],
                        local_processes=config_data['local_processes'],
                        memory=config_data['memory'],
                        wall_time=config_data['wall_time'],
                        queue=config_data['queue'],
                        timeout=config_data['timeout'],
                        scheduler='distributed',
                        cache_path=dask_cache_path)

    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.ERROR)

    return h5s, client, cluster, workers, cache, clean_params, save_file

def setup_cp_command(input_dir, config_file, config_data, output_dir, output_file, output_directory):

    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    h5_timestamp_path = get_timestamp_path(h5s[0])

    if output_directory is None:
        output_dir = os.path.join(input_dir, output_dir)  # outputting pca folder in inputted base directory.
    else:
        output_dir = os.path.join(output_directory, output_dir)

    if config_data['pca_file_components'] is None:
        pca_file_components = os.path.join(output_dir, 'pca.h5')
        config_data['pca_file_components'] = pca_file_components
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
    else:
        pca_file_components = config_data['pca_file_components']

    if config_data['pca_file_scores'] is None:
        pca_file_scores = os.path.join(output_dir, 'pca_scores.h5')
    else:
        pca_file_scores = config_data['pca_file_scores']

    if not os.path.exists(config_data['pca_file_components']):
        raise IOError('Could not find PCA components file {}'.format(config_data['pca_file_components']))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = output_dir + output_file

    return pca_file_components, pca_file_scores, h5s, yamls, save_file

def load_pcs_for_cp(pca_file_components, config_data):

    dask_cache_path = os.path.join(pathlib.Path.home(), 'moseq2_pca')

    print('Loading PCs from {}'.format(pca_file_components))
    with h5py.File(pca_file_components, 'r') as f:
        pca_components = f[config_data['pca_path']][...]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = '{}.yaml'.format(os.path.splitext(config_data['pca_file_components'])[0])

    # todo detect missing data and mask parameters, then 0 out, fill in, compute scores...
    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.safe_load(f.read())

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

            if missing_data and not os.path.exists(config_data['pca_file_scores']):
                raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    changepoint_params = {
        'k': config_data['klags'],
        'sigma': config_data['sigma'],
        'peak_height': config_data['threshold'],
        'peak_neighbors': config_data['neighbors'],
        'rps': config_data['dims']
    }

    client, cluster, workers, cache = \
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
    return pca_components, changepoint_params, cluster, client, missing_data, mask_params

def get_pca_yaml_data(pca_yaml):
    # todo detect missing data and mask parameters, then 0 out, fill in, compute scores...
    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.safe_load(f.read())
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

    return use_fft, clean_params, mask_params, missing_data