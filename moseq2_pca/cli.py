from moseq2_pca.util import recursive_find_h5s, command_with_config,\
    select_strel, initialize_dask, recursively_load_dict_contents_from_group
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


@click.group()
def cli():
    pass


@cli.command(name='add-groups')
@click.argument('index_file', type=click.Path(exists=True, resolve_path=True))
@click.argument('pca_file', type=click.Path(exists=True, resolve_path=True))
def add_groups(index_file, pca_file):

    with open(index_file, 'r') as f:
        index = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    if 'groups' in index:
        print('Adding groups to pca file {}'.format(pca_file))
        with h5py.File(pca_file, 'a') as f:
            for k, v in index['groups'].items():

                print('Adding {}:{}'.format(k, v))
                new_key = 'groups/{}'.format(k)

                if new_key in f:
                    del f[new_key]

                f[new_key] = v
    else:
        raise IOError('Could not find key groups in index file {}'.format(index_file))


@cli.command(name='train-pca', cls=command_with_config('config_file'))
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--cluster-type', type=click.Choice(['local', 'slurm']),
              default='local', help='Cluster type')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store results')
@click.option('--gaussfilter-space', default=(1.5, 1), type=(float, float), help="Spatial filter for data (Gaussian)")
@click.option('--gaussfilter-time', default=0, type=float, help="Temporal filter for data (Gaussian)")
@click.option('--medfilter-space', default=[0], type=int, help="Median spatial filter", multiple=True)
@click.option('--medfilter-time', default=[0], type=int, help="Median temporal filter", multiple=True)
@click.option('--missing-data', is_flag=True, type=bool, help="Use missing data PCA")
@click.option('--missing-data-iters', default=10, type=int, help="Missing data PCA iterations")
@click.option('--mask-threshold', default=-16, type=float, help="Threshold for mask (missing data only)")
@click.option('--mask-height-threshold', default=5, type=float, help="Threshold for mask based on floor height")
@click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--tailfilter-iters', default=1, type=int, help="Number of tail filter iterations")
@click.option('--tailfilter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tailfilter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--use-fft', type=bool, is_flag=True, help='Use 2D fft')
@click.option('--rank', default=50, type=int, help="Rank for compressed SVD (generally>>nPCS)")
@click.option('--output-file', default='pca', type=str, help='Name of h5 file for storing pca results')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-mask-path', default='/frames_mask', type=str, help="Path to log-likelihood mask in h5 files")
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--visualize-results', default=True, type=bool, help='Visualize results')
@click.option('--config-file', '-c', type=click.Path(), help="Path to configuration file")
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=50, help="Number of workers")
@click.option('-t', '--threads', type=int, default=2, help="Number of threads per workers")
@click.option('-p', '--processes', type=int, default=4, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="4GB", help="RAM usage per workers")
@click.option('-w', '--wall-time', type=str, default="01:00:00", help="Wall time for workers")
@click.option('--timeout', type=float, default=5, help="Time to wait for workers to initialize before proceeding (minutes)")
def train_pca(input_dir, cluster_type, output_dir, gaussfilter_space,
              gaussfilter_time, medfilter_space, medfilter_time, missing_data, missing_data_iters, mask_threshold,
              mask_height_threshold, min_height, max_height, tailfilter_iters, tailfilter_size,
              tailfilter_shape, use_fft, rank, output_file, h5_path, h5_mask_path, chunk_size,
              visualize_results, config_file, queue, nworkers, threads, processes, memory, wall_time, timeout):

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

    client, cluster, workers, cache =\
        initialize_dask(cluster_type=cluster_type,
                        nworkers=nworkers,
                        threads=threads,
                        processes=processes,
                        memory=memory,
                        wall_time=wall_time,
                        queue=queue,
                        timeout=timeout)

    dsets = [h5py.File(h5, mode='r')[h5_path] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0).astype('float32')
    stacked_array[stacked_array < min_height] = 0
    stacked_array[stacked_array > max_height] = 0

    print('Processing {:d} total frames'.format(stacked_array.shape[0]))

    if missing_data:
        mask_dsets = [h5py.File(h5, mode='r')[h5_mask_path] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in mask_dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = da.logical_and(stacked_array_mask < mask_threshold,
                                            stacked_array > mask_height_threshold)
        # stacked_array_mask = dask.compute(stacked_array_mask)
    else:
        stacked_array_mask = None

    output_dict =\
        train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                       clean_params=clean_params, use_fft=use_fft,
                       rank=rank, cluster_type=cluster_type, min_height=min_height,
                       max_height=max_height, client=client, cluster=cluster,
                       iters=missing_data_iters, workers=workers, cache=cache)

    if visualize_results:
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


@cli.command(name='apply-pca', cls=command_with_config('config_file'))
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--cluster-type', type=click.Choice(['local', 'slurm', 'nodask']),
              default='local', help='Cluster type')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store results')
@click.option('--output-file', default='pca_scores', type=str, help='Name of h5 file for storing pca results')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-mask-path', default='/frames_mask', type=str, help="Path to log-likelihood mask in h5 files")
@click.option('--h5-timestamp-path', default='/metadata/timestamps', type=str, help='Path to timestamps in h5 files')
@click.option('--h5-metadata-path', default='/metadata/extraction', type=str, help='Path to metadata in h5 files')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--pca-file', type=click.Path(exists=True), default=os.path.join(os.getcwd(), '_pca/pca.h5'), help='Path to PCA results')
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--fill-gaps', default=True, type=bool, help='Fill dropped frames with nans')
@click.option('--fps', default=30, type=int, help='Fps (only used if no timestamps found)')
@click.option('--detrend-window', default=0, type=float, help="Length of detrend window (in seconds, 0 for no detrending)")
@click.option('--config-file', '-c', type=click.Path(), help="Path to configuration file")
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=20, help="Number of workers")
@click.option('-t', '--threads', type=int, default=2, help="Number of threads per workers")
@click.option('-p', '--processes', type=int, default=4, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="4GB", help="RAM usage per workers")
@click.option('-w', '--wall-time', type=str, default="01:00:00", help="Wall time for workers")
@click.option('--timeout', type=float, default=5, help="Time to wait for workers to initialize before proceeding (minutes)")
def apply_pca(input_dir, cluster_type, output_dir, output_file, h5_path, h5_mask_path, h5_timestamp_path,
              h5_metadata_path, pca_path, pca_file, chunk_size, fill_gaps, fps, detrend_window,
              config_file, queue, nworkers, threads, processes, memory, wall_time, timeout):
    # find directories with .dat files that either have incomplete or no extractions
    # TODO: additional post-processing, intelligent mapping of metadata to group names, make sure
    # moseq2-model processes these files correctly

    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    print('Loading PCs from {}'.format(pca_file))
    with h5py.File(pca_file, 'r') as f:
        pca_components = f[pca_path].value

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
                'medfilter_space': pca_config['medfilter_space']
            }

            mask_params = {
                'mask_height_threshold': pca_config['mask_height_threshold'],
                'mask_threshold': pca_config['mask_threshold']
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
                            h5_metadata_path=h5_metadata_path, h5_path=h5_path,
                            h5_mask_path=h5_mask_path, mask_params=mask_params,
                            h5_timestamp_path=h5_timestamp_path, fps=fps,
                            missing_data=missing_data)

        else:
            client, cluster, workers, cache =\
             initialize_dask(cluster_type=cluster_type,
                             nworkers=nworkers,
                             threads=threads,
                             processes=processes,
                             memory=memory,
                             wall_time=wall_time,
                             queue=queue,
                             scheduler='distributed',
                             timeout=timeout)
            apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                           use_fft=use_fft, clean_params=clean_params,
                           save_file=save_file, chunk_size=chunk_size,
                           h5_metadata_path=h5_metadata_path, h5_path=h5_path,
                           h5_timestamp_path=h5_timestamp_path, fps=fps,
                           client=client, missing_data=missing_data,
                           h5_mask_path=h5_mask_path, mask_params=mask_params)

            if workers is not None:
                cluster.stop_workers(workers)


@cli.command('compute-changepoints', cls=command_with_config('config_file'))
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store results')
@click.option('--output-file', default='changepoints', type=str, help='Name of h5 file for storing pca results')
@click.option('--cluster-type', type=click.Choice(['local', 'slurm']), default='local', help='Cluster type')
@click.option('--pca-file-components', type=click.Path(), default=os.path.join(os.getcwd(), '_pca/pca.h5'), help="Path to PCA components")
@click.option('--pca-file-scores', type=click.Path(), default=os.path.join(os.getcwd(), '_pca/pca_scores.h5'), help='Path to PCA results')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--neighbors', type=int, default=1, help="Neighbors to use for peak identification")
@click.option('--threshold', type=float, default=.5, help="Peak threshold to use for changepoints")
@click.option('-k', '--klags', type=int, default=6, help="Lag to use for derivative calculation")
@click.option('-s', '--sigma', type=float, default=3.5, help="Standard deviation of gaussian smoothing filter")
@click.option('-d', '--dims', type=int, default=600, help="Number of random projections to use")
@click.option('--fps', default=30, type=int, help='Fps (only used if no timestamps found)')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-mask-path', default='/frames_mask', type=str, help="Path to log-likelihood mask in h5 files")
@click.option('--h5-timestamp-path', default='/metadata/timestamps', type=str, help='Path to timestamps in h5 files')
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--config-file', '-c', type=click.Path(), help="Path to configuration file")
@click.option('--visualize-results', default=True, type=bool, help='Visualize results')
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=20, help="Number of workers")
@click.option('-t', '--threads', type=int, default=2, help="Number of threads per workers")
@click.option('-p', '--processes', type=int, default=4, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="4GB", help="RAM usage per workers")
@click.option('-w', '--wall-time', type=str, default="01:00:00", help="Wall time for workers")
@click.option('--timeout', type=float, default=5, help="Time to wait for workers to initialize before proceeding (minutes)")
def compute_changepoints(input_dir, output_dir, output_file, cluster_type, pca_file_components,
                         pca_file_scores, pca_path, neighbors, threshold, klags, sigma, dims, fps, h5_path,
                         h5_mask_path, h5_timestamp_path, chunk_size, config_file, visualize_results, queue,
                         nworkers, threads, processes, memory, wall_time, timeout):

    params = locals()
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    print('Loading PCs from {}'.format(pca_file_components))
    with h5py.File(pca_file_components, 'r') as f:
        pca_components = f[pca_path].value

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
                        threads=threads,
                        processes=processes,
                        memory=memory,
                        wall_time=wall_time,
                        queue=queue,
                        scheduler='distributed',
                        timeout=timeout)

    get_changepoints_dask(pca_components=pca_components, pca_scores=pca_file_scores,
                          h5s=h5s, yamls=yamls, changepoint_params=changepoint_params,
                          save_file=save_file, chunk_size=chunk_size,
                          h5_path=h5_path, h5_timestamp_path=h5_timestamp_path, fps=fps,
                          client=client, missing_data=missing_data,
                          h5_mask_path=h5_mask_path, mask_params=mask_params)

    if workers is not None:
        cluster.stop_workers(workers)

    if visualize_results:
        import numpy as np
        with h5py.File('{}.h5'.format(save_file), 'r') as f:
            cps = recursively_load_dict_contents_from_group(f, 'cps')
        block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])
        plt, _ = changepoint_dist(block_durs, headless=True)
        plt.savefig('{}_dist.png'.format(save_file))
        plt.savefig('{}_dist.pdf'.format(save_file))
        plt.close()


if __name__ == '__main__':
    cli()
