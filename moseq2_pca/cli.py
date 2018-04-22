from moseq2_pca.util import recursive_find_h5s, command_with_config, clean_frames,\
    select_strel, insert_nans
from moseq2_pca.viz import display_components, scree_plot
import click
import os
import ruamel.yaml as yaml
import datetime
import h5py
import tqdm
import numpy as np
import dask.array as da
import dask.array.linalg as lng
import dask
from dask.diagnostics import ProgressBar
from chest import Chest


@click.group()
def cli():
    pass


@cli.command(name='train-pca', cls=command_with_config['config_file'])
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--cluster-type', type=click.Choice(['local']),
              default='local', help='Cluster type')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store results')
@click.option('--gaussfilter-space', default=(1.5, 1), type=(float, float), help="Spatial filter for data (Gaussian)")
@click.option('--gaussfilter-time', default=0, type=float, help="Temporal filter for data (Gaussian)")
@click.option('--medfilter-space', default=[0], type=int, help="Median spatial filter", multiple=True)
@click.option('--medfilter-time', default=[0], type=int, help="Median temporal filter", multiple=True)
@click.option('--tailfilter-iters', default=1, type=int, help="Number of tail filter iterations")
@click.option('--tailfilter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tailfilter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--rank', default=50, type=int, help="Rank for compressed SVD (generally>>nPCS)")
@click.option('--output-file', default='pca', type=str, help='Name of h5 file for storing pca results')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--visualize-results', default=True, type=bool, help='Visualize results')
@click.option('--config-file', '-c', type=click.Path(), help="Path to configuration file")
def train_pca(input_dir, cluster_type, output_dir, gaussfilter_space,
              gaussfilter_time, medfilter_space, medfilter_time, tailfilter_iters,
              tailfilter_size, tailfilter_shape, rank, output_file,
              h5_path, chunk_size, visualize_results, config_file):
    # find directories with .dat files that either have incomplete or no extractions

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

    config_store = os.path.join(output_dir, '{}.yaml'.format(save_file))
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

    if cluster_type == 'local':

        cache = Chest()
        dsets = [h5py.File(h5)[h5_path] for h5 in h5s]
        arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in dsets]
        stacked_array = da.concatenate(arrays, axis=0).astype('float32')
        nfeatures = stacked_array.shape[1] * stacked_array.shape[2]

        if gaussfilter_time > 0 or np.any(np.array(medfilter_time) > 0):
            stacked_array = stacked_array.map_overlap(
                clean_frames, depth=(20, 0, 0), boundary='reflect', dtype='float32', **clean_params)
        else:
            stacked_array = stacked_array.map_blocks(clean_frames, dtype='float32', **clean_params)

        stacked_array = stacked_array.reshape(-1, nfeatures)
        nsamples, nfeatures = stacked_array.shape
        mean = stacked_array.mean(axis=0)
        u, s, v = lng.svd_compressed(stacked_array-mean, rank, 0)
        total_var = stacked_array.var(ddof=1, axis=0).sum()

        print('Calculation setup complete...')

        with ProgressBar():
            s, v, mean, total_var = dask.compute(s, v, mean, total_var,
                                                 cache=cache)

        print('Calculation complete...')

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

        if visualize_results:
            display_components(output_dict['components'],
                               path='{}_components'.format(save_file))
            scree_plot(output_dict['explained_variance_ratio'],
                       path='{}_scree'.format(save_file))

        with h5py.File('{}.h5'.format(save_file)) as f:
            for k, v in output_dict.items():
                f.create_dataset(k, data=v, compression='gzip', dtype='float32')

    else:
        raise NotImplementedError('Other cluster types not supported')


@cli.command(name='apply-pca')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--cluster-type', type=click.Choice(['local']),
              default='local', help='Cluster type')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store results')
@click.option('--output-file', default='pca_scores', type=str, help='Name of h5 file for storing pca results')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-timestamp-path', default='/metadata/timestamps', type=str, help='Path to timestamps in h5 files')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--pca-file', type=click.Path(exists=True), default=os.path.join(os.getcwd(), '_pca/pca.h5'), help='Path to PCA results')
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--fill-gaps', default=True, type=bool, help='Fill dropped frames with nans')
def apply_pca(input_dir, cluster_type, output_dir, output_file, h5_path, h5_timestamp_path, pca_path,
              pca_file, chunk_size, fill_gaps):
    # find directories with .dat files that either have incomplete or no extractions

    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    print('Loading PCs from {}'.format(pca_file))
    with h5py.File(pca_file, 'r') as f:
        pca_components = f[pca_path].value

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:
        for h5, yml in tqdm.tqdm(zip(h5s, yamls), total=len(h5s),
                                 desc='Computing scores'):
            with h5py.File(h5, 'r') as f:
                frames = f[h5_path].value.reshape(-1, 6400)
                timestamps = f[h5_timestamp_path].value / 1000.0

            scores = frames.dot(pca_components.T)
            scores, score_idx, _ = insert_nans(data=scores, timestamps=timestamps,
                                               fps=int(1 / np.mean(np.diff(timestamps))))

            with open(yml, 'r') as yml_f:
                yml_dat = yml_f.read()
                try:
                    data = yaml.load(yml_dat, Loader=yaml.RoundTripLoader)
                except yaml.constructor.ConstructorError:
                    data = yaml.load(yml_dat, Loader=yaml.Loader)

            uuid = data['uuid']
            f_scores.create_dataset('scores/{}'.format(uuid), data=scores,
                                    dtype='float32', compression='gzip')

            f_scores.create_dataset('scores_idx/{}'.format(uuid), data=score_idx,
                                    dtype='float32', compression='gzip')


if __name__ == '__main__':
    cli()
