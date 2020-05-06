import h5py
import numpy as np
from unittest import TestCase
from moseq2_pca.util import recursively_load_dict_contents_from_group
from moseq2_pca.viz import display_components, scree_plot, changepoint_dist

class TestViz(TestCase):

    def test_display_components(self):
        # get components
        pca_path = 'data/_pca/pca.h5'
        with h5py.File(pca_path, 'r') as f:
            components = f['components'][()]
            plt, ax = display_components(components)
            assert (plt != None and ax != None)

    def test_scree_plot(self):
        # get explained_variance_ratio
        pca_path = 'data/_pca/pca.h5'
        with h5py.File(pca_path, 'r') as f:
            components = f['explained_variance_ratio'][()]
            plt = scree_plot(components)
            assert (plt != None)

    def test_changepoint_dist(self):
        # original params: cps, headless=False
        save_file = 'data/_pca/changepoints'

        with h5py.File(f'{save_file}.h5', 'r') as f:
            cps = recursively_load_dict_contents_from_group(f, 'cps')
        block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])

        assert block_durs.shape == (51,1)

        out = changepoint_dist(block_durs, headless=True)

        assert out != None