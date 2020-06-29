import os
import shutil
from pathlib import Path
import ruamel.yaml as yaml
from unittest import TestCase
from click.testing import CliRunner
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_pca.cli import clip_scores, train_pca, apply_pca, compute_changepoints

class TestCli(TestCase):

    def test_clip_scores(self):

        data_dir = 'data/'
        h5path = os.path.join(data_dir, 'test_scores.h5')
        clip_samples = '15'

        clip_params = [h5path, clip_samples]

        runner = CliRunner()
        result = runner.invoke(clip_scores, clip_params)
        outputfile = 'data/test_scores_clip.h5'

        assert(Path(outputfile).exists()), "Clipped scores file was not created"
        os.remove(outputfile)
        assert (result.exit_code == 0), "CLI function did not complete successfully"

    def test_train_pca(self):
        data_dir = 'data/'
        out_dir = Path(data_dir).joinpath('tmp_pca')

        train_params_local = ['-i', data_dir,
                              '--cluster-type', 'local',
                              '--missing-data',
                              '-o', str(out_dir)]

        # in case it asks for user input
        with TemporaryDirectory() as tmp:
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

        runner = CliRunner()

        result = runner.invoke(train_pca,
                               train_params_local,
                               catch_exceptions=False)

        assert (result.exit_code == 0), "CLI Command did not successfully complete"
        assert (out_dir.exists()), "pca directory was not successfully created"
        outfiles = [str(f.name) for f in list(out_dir.iterdir())]

        assert ('pca.h5' in outfiles and 'pca.yaml' in outfiles and \
                'pca_components.pdf' in outfiles and 'pca_scree.pdf' in outfiles), \
            'PCA files were not computed successfully'

        shutil.rmtree(str(out_dir))

    def test_apply_pca(self):
        data_dir = Path('data/')
        pca_yaml = data_dir.joinpath('_pca/pca.yaml')

        with open(str(pca_yaml), 'r') as f:
            pca_meta = yaml.safe_load(f)

        pca_meta['missing_data'] = True

        with open(str(pca_yaml), 'w') as f:
            yaml.safe_dump(pca_meta, f)

        outpath = Path('_pca')

        apply_params_local = ['-i', str(data_dir),
                              '-o', str(outpath),
                              '--output-file', 'pca_scores1',
                              ]

        runner = CliRunner()

        result = runner.invoke(apply_pca,
                               apply_params_local,
                               catch_exceptions=False)

        assert result.exit_code == 0, "CLI command did not successfully complete"
        assert data_dir.joinpath(outpath).is_dir(), "pca directory does not exist"
        assert data_dir.joinpath(outpath, 'pca_scores1.h5').is_file(), "pca scores were not correctly saved"

        with open(str(pca_yaml), 'r') as f:
            pca_meta = yaml.safe_load(f)

        pca_meta['missing_data'] = False

        with open(str(pca_yaml), 'w') as f:
            yaml.safe_dump(pca_meta, f)

        os.remove(os.path.join('data', outpath, 'pca_scores1.h5'))

    def test_compute_changepoints(self):
        data_path = Path('data/')
        outpath = Path('_pca')

        if not outpath.exists():
            outpath.mkdir()

        cc_params_local = ['-i', str(data_path),
                           '-o', str(outpath),
                           '--output-file', 'changepoints1']

        runner = CliRunner()

        result = runner.invoke(compute_changepoints,
                          cc_params_local,
                          catch_exceptions=False)

        assert result.exit_code == 0, "CLI command did not successfully complete"
        assert outpath.exists(), "simulated path was not generated"
        assert data_path.joinpath(outpath, 'changepoints1.h5').is_file(), "changepoints were not computed"
        assert data_path.joinpath(outpath, 'changepoints1_dist.pdf').is_file(), "changepoint pdf was not create"
        assert data_path.joinpath(outpath, 'changepoints1_dist.png').is_file(), "changepoint image was not create"

        data_path.joinpath(outpath, 'changepoints1.h5').unlink()
        data_path.joinpath(outpath, 'changepoints1_dist.pdf').unlink()
        data_path.joinpath(outpath, 'changepoints1_dist.png').unlink()
