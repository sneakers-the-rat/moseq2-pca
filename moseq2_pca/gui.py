'''

GUI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions perform jupyter notebook specific preprocessing, loads in corresponding parameters from the
CLI functions, then call the corresponding wrapper function with the given input parameters.

'''

import warnings
import ruamel.yaml as yaml
from .cli import train_pca, apply_pca, compute_changepoints
from moseq2_pca.helpers.wrappers import train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper


def train_pca_command(progress_paths, output_dir, output_file):
    '''
    Train PCA through Jupyter notebook, and updates config file.

    Parameters
    ----------
    progress_paths (dict): dictionary containing notebook progress paths
    output_dir (str): path to output pca directory
    output_file (str): name of output pca file.

    Returns
    -------
    None
    '''

    # Get appropriate inputs
    input_dir = progress_paths['train_data_dir']
    config_file = progress_paths['config_file']

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in train_pca.params if not tmp.required}

    # merge default params with those in config
    config_data = {**default_params, **config_data}

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    config_data = train_pca_wrapper(input_dir, config_data, output_dir, output_file)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)


def apply_pca_command(progress_paths, output_file):
    '''
    Compute PCA Scores given trained PCA using Jupyter Notebook.

    Parameters
    ----------
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns
    -------
    (str): success string.
    '''

    # Get proper inputs
    input_dir = progress_paths['train_data_dir']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    output_dir = progress_paths['pca_dirname']

    # TODO: additional post-processing, intelligent mapping of metadata to group names, make sure
    # moseq2-model processes these files correctly

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in apply_pca.params if not tmp.required}

    # merge default params with those in config
    config_data = {**default_params, **config_data}

    config_data = apply_pca_wrapper(input_dir, config_data, output_dir, output_file)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    try:
        with open(index_file, 'r') as f:
            index_params = yaml.safe_load(f)

        index_params['pca_path'] = config_data['pca_file_scores']

        with open(index_file, 'w') as f:
            yaml.safe_dump(index_params, f)

    except:
        print('moseq2-index not found, did not update paths')

    return 'PCA Scores have been successfully computed.'


def compute_changepoints_command(input_dir, progress_paths, output_file):
    '''
    Compute Changepoint distribution using Jupyter Notebook.

    Parameters
    ----------
    input_dir (str): path to directory containing training data
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns
    -------
    (str): success string.
    '''

    config_file = progress_paths['config_file']
    output_dir = progress_paths['pca_dirname']

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in compute_changepoints.params
                      if not tmp.required}

    # merge default params with those in config
    config_data = {**default_params, **config_data}

    config_data = compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    return 'Model-free syllable changepoints have been successfully computed.'
