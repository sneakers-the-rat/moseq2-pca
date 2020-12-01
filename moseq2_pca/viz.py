'''

Visualization operations for plotting computed PCs, a Scree Plot, and the Changepoint PDF histogram.

'''

import click
import logging
import warnings
import numpy as np
import skimage.util
import seaborn as sns
from os.path import join
from scipy.stats import mode
import matplotlib.pyplot as plt

def plot_pca_results(output_dict, save_file, output_dir):
    '''
    Convenience function to graph and save Trained PCA results.

    Parameters
    ----------
    output_dict (dict): Dict object containing PCA training results
    save_file (str): Path to write images to.
    output_dir (str): Directory containing logger

    Returns
    -------
    None
    '''

    try:
        # Plotting PCA Components
        plt, _ = display_components(output_dict['components'], headless=True)
        plt.savefig(f'{save_file}_components.png')
        plt.savefig(f'{save_file}_components.pdf')
        plt.close()
    except Exception as e:
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('could not plot components')
        click.echo('You may find error logs here:', join(output_dir, 'train.log'))

    try:
        # Plotting Scree Plot
        plt = scree_plot(output_dict['explained_variance_ratio'], headless=True)
        plt.savefig(f'{save_file}_scree.png')
        plt.savefig(f'{save_file}_scree.pdf')
        plt.close()
    except Exception as e:
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('could not plot scree')
        click.echo('You may find error logs here:', join(output_dir, 'train.log'))


def display_components(components, cmap='gray', headless=False):
    '''
    Creates grid of computed Principal Components.

    Parameters
    ----------
    components (2D np.ndarray): components to graph
    cmap (str): color map to use
    headless (bool): trim first element in PC list

    Returns
    -------
    plt (plt.figure): figure to save/graph
    ax (plt.ax): figure axis variable
    '''

    # Get square image size
    im_size = int(np.sqrt(components.shape[1]))
    plotv = components.reshape((-1, im_size, im_size))
    plotv = skimage.util.montage(plotv)

    if headless:
        plt.switch_backend('agg')

    # Plot PCs
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(plotv, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    return plt, ax


def scree_plot(explained_variance_ratio, headless=False):
    '''
    Creates Scree plot describing principal components.

    Parameters
    ----------
    explained_variance_ratio (1D np.array): explained variance ratio of each principal component
    headless (bool): trim first element in PC list

    Returns
    -------
    plt (plt.figure): figure to save/graph
    '''

    csum = np.cumsum(explained_variance_ratio)*1e2

    if headless:
        plt.switch_backend('agg')

    sns.set_style('ticks')
    plt.plot(np.cumsum(explained_variance_ratio)*1e2)

    idx = np.where(csum >= 90)
    plt.ylim((0, 100))
    plt.xlim((0, len(explained_variance_ratio)))

    if len(idx[0]) > 0:
        idx = np.min(idx)
        plt.plot([idx, idx], [0, csum[idx]], 'k-')
        plt.plot([0, idx], [csum[idx], csum[idx]], 'k-')
        plt.title(f'{csum[idx]:0.2f}% in {idx + 1} pcs')

    plt.ylabel('Variance explained (percent)')
    plt.xlabel('nPCs')
    sns.despine()

    return plt


def changepoint_dist(cps, headless=False):
    '''
    Creates bar plot describing computed Changepoint Distribution.

    Parameters
    ----------
    cps (np.ndarray): changepoints to graph
    headless (bool): trim first element in PC list

    Returns
    -------
    plt (plt.figure): figure to save/graph
    ax (plt.ax): figure axis variable
    '''

    if cps.size > 0:

        if headless:
            plt.switch_backend('agg')

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.set_style('ticks')

        ax = sns.distplot(cps, kde_kws={'gridsize': 600}, bins=np.linspace(0, 10, 100))
        ax.set_xlim((0, 2))
        ax.set_xticks(np.linspace(0, 2, 11))

        s = f'Mean, median, mode (s) = {np.mean(cps):.4f} {np.median(cps):.4f}, {mode(cps)[0][0][0]:.4f}'
        plt.text(.5, 2, s, fontsize=12)
        plt.ylabel('P(block duration)')
        plt.xlabel('Block duration (s)')
        sns.despine()

        return plt, ax
    else:
        warnings.warn('No changepoints detected - check if mouse is present or moving')