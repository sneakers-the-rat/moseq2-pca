'''

Helper functions for reading files and directories in preparation for changepoint analysis or apply pca.

'''

import h5py
import ruamel.yaml as yaml
from os.path import join, exists, splitext
from moseq2_pca.util import select_strel, read_yaml

def get_pca_paths(config_data, output_dir):
    '''
    Helper function for changepoints_wrapper to perform data-path existence checks.
    Returns paths to saved pre-trained PCA components and PCA Scores files.

    Parameters
    ----------
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data

    Returns
    -------
    config_data (dict): updated config_data dict with the proper paths
    pca_file_components (str): path to trained pca file
    pca_file_scores (str): path to pca_scores file
    '''

    # Get path to pre-computed PCA file
    pca_file_components = join(output_dir, 'pca.h5')
    if 'pca_file_components' not in config_data:
        config_data['pca_file_components'] = pca_file_components
    elif config_data['pca_file_components'] is not None:
        pca_file_components = config_data['pca_file_components']

    if not exists(pca_file_components):
        raise IOError(f'Could not find PCA components file {pca_file_components}')

    # Get path to PCA Scores
    pca_file_scores = config_data.get('pca_file_scores', join(output_dir, 'pca_scores.h5'))
    config_data['pca_file_scores'] = pca_file_scores

    return config_data, pca_file_components, pca_file_scores

def load_pcs_for_cp(pca_file_components, config_data):
    '''
    Load computed Principal Components for Model-free Changepoint Analysis.

    Parameters
    ----------
    pca_file_components (str): path to pca h5 file to read PCs
    config_data (dict): config parameters

    Returns
    -------
    pca_components (str): path to pca components
    changepoint_params (dict): dict of relevant changepoint parameters
    cluster (dask Cluster): Dask Cluster object.
    client (dask Client): Dask Client Object
    workers (dask Workers): intialized workers or None if cluster_type = 'local'
    missing_data (bool): Indicates whether to use mask_params
    mask_params (dict): Mask parameters to use when computing CPs
    '''

    print(f'Loading PCs from {pca_file_components}')
    with h5py.File(pca_file_components, 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = splitext(pca_file_components)[0] + '.yaml'

    if exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.safe_load(f.read())

            missing_data = pca_config.get('missing_data', False)
            if missing_data:
                print('Detected missing data...')
                mask_params = {
                    'mask_height_threshold': pca_config['mask_height_threshold'],
                    'mask_threshold': pca_config['mask_threshold']
                }
            else:
                mask_params = None

            if missing_data and not exists(config_data['pca_file_scores']):
                raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    # Pack changepoint parameters
    changepoint_params = {
        'k': config_data['klags'],
        'sigma': config_data['sigma'],
        'peak_height': config_data['threshold'],
        'peak_neighbors': config_data['neighbors'],
        'rps': config_data['dims']
    }

    return pca_components, changepoint_params, missing_data, mask_params

def get_pca_yaml_data(pca_yaml):
    '''
    Reads PCA yaml file and returns enclosed metadata.

    Parameters
    ----------
    pca_yaml (str): path to pca.yaml

    Returns
    -------
    use_fft (bool): indicates whether to use FFT
    clean_params (dict): dict of image filtering parameters
    mask_params (dict): dict of mask parameters)
    missing_data (bool): indicates whether to use mask_params
    '''

    if exists(pca_yaml):
        # Load pca metadata file
        pca_config = read_yaml(pca_yaml)

        use_fft = pca_config.get('use_fft', False)
        # Check if PCA was trained with masked data
        missing_data = pca_config.get('missing_data', False)
        if use_fft:
            print('Will use FFT...')
        if missing_data:
            print('Detected missing data...')

        # Get tail filter
        tailfilter = select_strel(pca_config['tailfilter_shape'], tuple(pca_config['tailfilter_size']))

        # Pack filtering paraneters
        clean_params = {
            'gaussfilter_space': pca_config['gaussfilter_space'],
            'gaussfilter_time': pca_config['gaussfilter_time'],
            'tailfilter': tailfilter,
            'medfilter_time': pca_config['medfilter_time'],
            'medfilter_space': pca_config['medfilter_space'],
        }

        # Get masking parameters
        mask_params = {
            'mask_height_threshold': pca_config['mask_height_threshold'],
            'mask_threshold': pca_config['mask_threshold'],
            'min_height': pca_config['min_height'],
            'max_height': pca_config['max_height']
        }
    else:
        raise IOError(f'Could not find {pca_yaml}')

    return use_fft, clean_params, mask_params, missing_data