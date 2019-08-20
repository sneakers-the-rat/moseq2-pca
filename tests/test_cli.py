import pytest
import os
import numpy as np
import h5py
import ruamel.yaml as yaml
from click.testing import CliRunner
import pathlib
from moseq2_pca.cli import clip_scores, add_groups, train_pca, apply_pca, compute_changepoints


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


def test_clip_scores():

    h5path = 'tests/test_files/test_scores.h5'
    clip_samples = '15'

    clip_params = [h5path, clip_samples]

    runner = CliRunner()
    result = runner.invoke(clip_scores, clip_params)
    outputfile = 'tests/test_files/test_scores_clip.h5'
    assert(os.path.exists(outputfile) == True)
    os.remove(outputfile)
    assert (result.exit_code == 0)

def test_add_groups():
    index_filepath = 'tests/test_files/test_index.yaml'
    pca_filepath = 'tests/test_files/test_scores.h5'

    runner = CliRunner()
    result = runner.invoke(add_groups, [index_filepath, pca_filepath])

    assert (result.exit_code == 0)


def test_train_pca(temp_dir):

    data_path = os.path.join(temp_dir, 'test_mouse.h5')
    yaml_path = os.path.join(temp_dir, 'test_mouse.yaml')

    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    fake_movie = np.tile(fake_mouse, (100, 1, 1))

    with h5py.File(data_path, 'w') as f:
        f.create_dataset('frames', data=fake_movie, compression='gzip', dtype='u1')
        f.create_dataset('frames_mask', data=fake_movie > 0, compression='gzip', dtype='bool')

    with open(yaml_path, 'w') as f:
        yaml.dump({'uuid': 'testing'}, f, Dumper=yaml.RoundTripDumper)

    train_params_local = ['-i', temp_dir,
                    '--cluster-type', 'local',
                    '-o', os.path.join(temp_dir, '_pca'),
                    '--gaussfilter-space',  1.5, 1,
                    '--gaussfilter-time', 0,
                    'medfilter-space', 0,
                    'medfilter-time', 0,
                    '--missing-data', ## FLAG
                    '--missing-data-iters', 10,
                    '--mask-threshold', -16,
                    '--mask-height-threshold', 5,
                    '--min-height', 10,
                    '--max-height', 100,
                    '--tailfilter-size',  9, 9,
                    '--tailfilter-shape', 'ellipse',
                    'use_fft', ## FLAG
                    '--recon-pcs', 10,
                    '--rank', 50,
                    '--output-file', 'pca',
                    '--visualize-results', True,
                    '--config-file', 'tests/test_files/test_conf.yaml',
                    '-d', os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                    '-n', 1,
                    '-c', 1,
                    '-p', 1,
                    '-m', "5GB",
                    '-w', '6:00:00',
                    '--timeout', 5]

    train_params_slurm = ['-i', temp_dir,
                          '--cluster-type', 'slurm',
                          '-o', os.path.join(temp_dir, '_pca'),
                          '--gaussfilter-space', 1.5, 1,
                          '--gaussfilter-time', 0,
                          'medfilter-space', 0,
                          'medfilter-time', 0,
                          '--missing-data',  ## FLAG
                          '--missing-data-iters', 10,
                          '--mask-threshold', -16,
                          '--mask-height-threshold', 5,
                          '--min-height', 10,
                          '--max-height', 100,
                          '--tailfilter-size', 9, 9,
                          '--tailfilter-shape', 'ellipse',
                          'use_fft',  ## FLAG
                          '--recon-pcs', 10,
                          '--rank', 50,
                          '--output-file', 'pca',
                          '--visualize-results', True,
                          '--config-file', 'tests/test_files/test_conf.yaml',
                          '-d', os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                          '-n', 1,
                          '-c', 1,
                          '-p', 1,
                          '-m', "5GB",
                          '-w', '6:00:00',
                          '--timeout', 5]

    runner = CliRunner()

    result = runner.invoke(train_pca,
                           train_params_local,
                           catch_exceptions=False)

    assert (os.path.exists(os.path.join(temp_dir, '_pca') == True))
    assert(result.exit_code == 0)

    result = runner.invoke(train_pca,
                           train_params_slurm,
                           catch_exceptions=False)

    assert (os.path.exists(os.path.join(temp_dir, '_pca2') == True))
    assert(result.exit_code == 0)


def test_apply_pca(temp_dir):

    data_path = os.path.join(temp_dir, 'test_mouse.h5')
    yaml_path = os.path.join(temp_dir, 'test_mouse.yaml')
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    fake_movie = np.tile(fake_mouse, (100, 1, 1))

    with h5py.File(data_path, 'w') as f:
        f.create_dataset('frames', data=fake_movie, compression='gzip', dtype='u1')
        f.create_dataset('frames_mask', data=fake_movie > 0, compression='gzip', dtype='bool')
        f.create_dataset('timestamps', data=np.arange(len(fake_movie)) * 30, compression='gzip', dtype='int32')
        f.create_dataset('/metadata/acquisition', data='acquisition')

    with open(yaml_path, 'w') as f:
        yaml.dump({'uuid': 'testing'}, f, Dumper=yaml.RoundTripDumper)

    apply_params_local = ['-i', temp_dir,
                          '-o', os.path.join(temp_dir, '_pca'),
                          '--output-file', 'pca_scores',
                          '--cluster-type', 'local',
                          '--fps', 30,
                          '--fill-gaps', True,
                          '--chunk-size', 4000,
                          '--fill-gaps', True,
                          '--pca-path', '/components',
                          '--pca-file', None,
                          '--h5-path', '/frames',
                           '-n', 1,
                          '-q', 'debug',
                          '--detrend-window', 0,
                           '--memory', '5GB',
                           '--visualize-results', True,
                          '--config-file', 'tests/test_files/test_conf.yaml',
                          '-d', os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                          '-n', 1,
                          '-c', 1,
                          '-p', 1,
                          '-m', "5GB",
                          '-w', '6:00:00',
                          '--timeout', 5]

    apply_params_slurm = ['-i', temp_dir,
                          '-o', os.path.join(temp_dir, '_pca'),
                          '--output-file', 'pca_scores',
                          '--cluster-type', 'slurm',
                          '--fps', 30,
                          '--fill-gaps', True,
                          '--chunk-size', 4000,
                          '--fill-gaps', True,
                          '--pca-path', '/components',
                          '--pca-file', None,
                          '--h5-path', '/frames',
                          '-n', 1,
                          '-q', 'debug',
                          '--detrend-window', 0,
                          '--memory', '5GB',
                          '--visualize-results', True,
                          '--config-file', 'tests/test_files/test_conf.yaml',
                          '-d', os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                          '-n', 1,
                          '-c', 1,
                          '-p', 1,
                          '-m', "5GB",
                          '-w', '6:00:00',
                          '--timeout', 5]

    runner = CliRunner()

    _ = runner.invoke(train_pca,
                      apply_params_local,
                      catch_exceptions=False)

    result = runner.invoke(apply_pca,
                           apply_params_slurm,
                           catch_exceptions=False)

    assert (os.path.exists(os.path.join(temp_dir, '_pca') == True))
    assert result.exit_code == 0

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca'),
                            '--pca-file', os.path.join(temp_dir, '_pca/pca.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--cluster-type', 'local'],
                           catch_exceptions=False)

    assert (os.path.exists(os.path.join(temp_dir, '_pca') == True))
    assert result.exit_code == 0

    _ = runner.invoke(train_pca,
                      ['-i', temp_dir, '-o',
                       os.path.join(temp_dir, '_pca2'),
                       '-n', 1,
                       '--memory', '5GB',
                       '--missing-data'],
                      catch_exceptions=False)

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca2'),
                            '--pca-file', os.path.join(temp_dir, '_pca2/pca.h5'),
                            '--cluster-type', 'nodask'],
                           catch_exceptions=False)

    assert (os.path.exists(os.path.join(temp_dir, '_pca2') == True))
    assert result.exit_code == 0

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca2'),
                            '--pca-file', os.path.join(temp_dir, '_pca2/pca.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--cluster-type', 'local'],
                           catch_exceptions=False)
    assert (os.path.exists(os.path.join(temp_dir, '_pca2') == True))
    assert result.exit_code == 0


def test_compute_changepoints(temp_dir):

    data_path = os.path.join(temp_dir, 'test_mouse.h5')
    yaml_path = os.path.join(temp_dir, 'test_mouse.yaml')
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    fake_movie = np.tile(fake_mouse, (100, 1, 1))

    with h5py.File(data_path, 'w') as f:
        f.create_dataset('frames', data=fake_movie, compression='gzip', dtype='u1')
        f.create_dataset('frames_mask', data=fake_movie > 0, compression='gzip', dtype='bool')
        f.create_dataset('timestamps', data=np.arange(len(fake_movie)) * 30, compression='gzip', dtype='int32')
        f.create_dataset('/metadata/acquisition', data='acquisition')

    with open(yaml_path, 'w') as f:
        yaml.dump({'uuid': 'testing'}, f, Dumper=yaml.RoundTripDumper)


    cc_params_local = ['-i', temp_dir,
                       '-o', os.path.join(temp_dir, '_pca'),
                       '--output-file', 'changepoints',
                       '--cluster-type', 'local',
                       '--pca-file-components', None,
                       '--pca-file-scores', None,
                       '--pca-path', '/components',
                       '--neighbors', 1,
                       '--threshold', .5,
                       '-k', 6,
                       '-s', 3.5,
                       '-d', 30,
                       '--config-file', 'tests/test_files/test_conf.yaml',
                       '-q', 'debug',
                       '--h5-path', '/frames',
                       '--h5-mask-path', 'frames_mask',
                       '--chunk-size', 4000
                       '-d', os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                       '-n', 1,
                       '-c', 1,
                       '-p', 1,
                       '-m', "5GB",
                       '-w', '6:00:00',
                       '--timeout', 5
                       '--fps', 30,
                       '--memory', '5GB',
                       '--visualize-results', True]
    runner = CliRunner()

    _ = runner.invoke(train_pca,
                      cc_params_local,
                      catch_exceptions=False)

    cc_params_nodask = ['-i', temp_dir,
                       '-o', os.path.join(temp_dir, '_pca'),
                       '--output-file', 'changepoints',
                       '--cluster-type', 'nodask',
                       '--pca-file-components', None,
                       '--pca-file-scores', None,
                       '--pca-path', '/components',
                       '--neighbors', 1,
                       '--threshold', .5,
                       '-k', 6,
                       '-s', 3.5,
                       '-d', 30,
                       '--config-file', 'tests/test_files/test_conf.yaml',
                       '-q', 'debug',
                       '--h5-path', '/frames',
                       '--h5-mask-path', 'frames_mask',
                       '--chunk-size', 4000
                       '-d', os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                       '-n', 1,
                       '-c', 1,
                       '-p', 1,
                       '-m', "5GB",
                       '-w', '6:00:00',
                       '--timeout', 5
                       '--fps', 30,
                       '--memory', '5GB',
                       '--visualize-results', True]

    _ = runner.invoke(apply_pca,
                      cc_params_nodask,
                      catch_exceptions=False)

    result = runner.invoke(compute_changepoints,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca'),
                            '--pca-file-components', os.path.join(temp_dir, '_pca/pca.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--pca-file-scores', os.path.join(temp_dir, '_pca/pca_scores.h5'),
                            '--cluster-type', 'local'],
                           catch_exceptions=False)

    assert (os.path.exists(os.path.join(temp_dir, '_pca') == True))
    assert(result.exit_code == 0)

    _ = runner.invoke(train_pca,
                      ['-i', temp_dir, '-o',
                       os.path.join(temp_dir, '_pca2'),
                       '--memory', '5GB',
                       '-n', 1,
                       '--missing-data'],
                      catch_exceptions=False)

    _ = runner.invoke(apply_pca,
                      ['-i', temp_dir,
                       '-o', os.path.join(temp_dir, '_pca2'),
                       '--pca-file', os.path.join(temp_dir, '_pca2/pca.h5'),
                       '--cluster-type', 'nodask'],
                      catch_exceptions=False)

    result = runner.invoke(compute_changepoints,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca2'),
                            '--pca-file-components', os.path.join(temp_dir, '_pca2/pca.h5'),
                            '--pca-file-scores', os.path.join(temp_dir, '_pca2/pca_scores.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--cluster-type', 'local'],
                           catch_exceptions=False)
    assert (os.path.exists(os.path.join(temp_dir, '_pca2') == True))
    assert(result.exit_code == 0)
