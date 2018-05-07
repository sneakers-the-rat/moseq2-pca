from moseq2_pca.util import recursive_find_h5s, command_with_config,\
    select_strel, initialize_dask
from moseq2_pca.viz import display_components, scree_plot
from moseq2_pca.pca.util import apply_pca_dask, apply_pca_local, train_pca_dask
import click
import os
import ruamel.yaml as yaml
import datetime
import h5py
import warnings
import dask.array as da
import dask
import tqdm


@click.group()
def cli():
    pass


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
@click.option('--mask-threshold', default=-16, type=float, help="Threshold for mask (missing data only)")
@click.option('--mask-height-threshold', default=5, type=float, help="Threshold for mask based on floor height")
@click.option('--tailfilter-iters', default=1, type=int, help="Number of tail filter iterations")
@click.option('--tailfilter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tailfilter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--use-fft', type=bool, is_flag=True, help='Use 2D fft')
@click.option('--rank', default=50, type=int, help="Rank for compressed SVD (generally>>nPCS)")
@click.option('--output-file', default='pca', type=str, help='Name of h5 file for storing pca results')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-mask-path', default='/frames_ll', type=str, help="Path to log-likelihood mask in h5 files")
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--visualize-results', default=True, type=bool, help='Visualize results')
@click.option('--config-file', '-c', type=click.Path(), help="Path to configuration file")
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=50, help="Number of workers")
@click.option('-t', '--threads', type=int, default=2, help="Number of threads per workers")
@click.option('-p', '--processes', type=int, default=4, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="4GB", help="RAM usage per workers")
@click.option('-w', '--wall-time', type=str, default="01:00:00", help="Wall time for workers")
def train_pca(input_dir, cluster_type, output_dir, gaussfilter_space,
              gaussfilter_time, medfilter_space, medfilter_time, missing_data, mask_threshold,
              mask_height_threshold, tailfilter_iters, tailfilter_size, tailfilter_shape, use_fft,
              rank, output_file, h5_path, h5_mask_path, chunk_size, visualize_results,
              config_file, queue, nworkers, threads, processes, memory, wall_time):

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

    client, cluster, workers, cache =\
        initialize_dask(cluster_type=cluster_type,
                        nworkers=nworkers,
                        threads=threads,
                        processes=processes,
                        memory=memory,
                        wall_time=wall_time,
                        queue=queue)

    dsets = [h5py.File(h5, mode='r')[h5_path] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0).astype('float32')

    if missing_data:
        mask_dsets = [h5py.File(h5, mode='r')[h5_mask_path] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=(chunk_size, -1, -1)) for dset in dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = ~da.logical_and(stacked_array_mask > mask_threshold,
                                            stacked_array > mask_height_threshold)
        # stacked_array_mask = dask.compute(stacked_array_mask)
    else:
        stacked_array_mask = None

    output_dict =\
        train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                       clean_params=clean_params, use_fft=use_fft,
                       rank=rank, cluster_type=cluster_type,
                       client=client, cluster=cluster, workers=workers, cache=cache)

    if visualize_results:
        plt = display_components(output_dict['components'], headless=True)
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
def apply_pca(input_dir, cluster_type, output_dir, output_file, h5_path, h5_timestamp_path,
              h5_metadata_path, pca_path, pca_file, chunk_size, fill_gaps, fps, detrend_window,
              config_file, queue, nworkers, threads, processes, memory, wall_time):
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

    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)
            if 'use_fft' in pca_config.keys():
                use_fft = pca_config['use_fft']
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
                            h5_timestamp_path=h5_timestamp_path, fps=fps)

        else:
            client, cluster, workers, cache =\
             initialize_dask(cluster_type=cluster_type,
                             nworkers=nworkers,
                             threads=threads,
                             processes=processes,
                             memory=memory,
                             wall_time=wall_time,
                             queue=queue,
                             scheduler='distributed')
            apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                           use_fft=use_fft, clean_params=clean_params,
                           save_file=save_file, chunk_size=chunk_size,
                           h5_metadata_path=h5_metadata_path, h5_path=h5_path,
                           h5_timestamp_path=h5_timestamp_path, fps=fps,
                           client=client)

            if workers is not None:
                cluster.stop_workers(workers)


if __name__ == '__main__':
    cli()
