import os
from pathlib import Path
from unittest import TestCase
from moseq2_pca.gui import train_pca_command, apply_pca_command, compute_changepoints_command

class TestGUI(TestCase):

    def test_train_pca_command(self):
        data_dir = 'data/'
        config_file = 'data/config.yaml'
        output_dir = 'data/tmp_pca'
        output_file = 'pca'

        train_pca_command(data_dir, config_file, output_dir, output_file)

        assert Path(output_dir).is_dir(), "PCA path was not created."

        out_files = list(Path(output_dir).iterdir())

        assert len(out_files) >= 6, 'PCA did not correctly generate all required files.'
        assert Path(output_dir).joinpath('pca.h5').is_file(), "PCA file was not created in the correct location"
        assert Path(output_dir).joinpath('pca.yaml').is_file(), "PCA metadata file is missing"
        assert Path(output_dir).joinpath('pca_components.pdf').is_file(), "PCA components image is missing"
        assert Path(output_dir).joinpath('pca_scree.pdf').is_file(), "PCA Scree plot is missing"

        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.removedirs(output_dir)


    def test_apply_pca_command(self):
        data_dir = Path('data/')
        index_file = 'data/test_index.yaml'
        config_file = 'data/config.yaml'
        outpath = Path('_pca')
        output_file = 'pca_scores2'

        if not data_dir.joinpath(outpath).is_dir():
            outpath.mkdir()

        apply_pca_command(str(data_dir), index_file, config_file, str(outpath), output_file)
        assert data_dir.joinpath(outpath, output_file+'.h5').is_file(), "Scores file was not created."

        data_dir.joinpath(outpath, output_file+'.h5').unlink()


    def test_compute_changepoints_command(self):
        data_dir = Path('data/')
        config_file = 'data/config.yaml'
        outpath = Path('_pca')
        output_file = 'changepoints2'

        compute_changepoints_command(str(data_dir), config_file, str(outpath), output_file)

        assert data_dir.joinpath(outpath).is_dir(), 'PCA Path was not found'
        assert data_dir.joinpath(outpath).joinpath(output_file+'.h5').is_file()
        assert data_dir.joinpath(outpath).joinpath(output_file+'_dist.pdf').is_file()
        assert data_dir.joinpath(outpath).joinpath(output_file+'_dist.png').is_file()

        data_dir.joinpath(outpath).joinpath(output_file + '.h5').unlink()
        data_dir.joinpath(outpath).joinpath(output_file + '_dist.pdf').unlink()
        data_dir.joinpath(outpath).joinpath(output_file + '_dist.png').unlink()