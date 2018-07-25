import skimage.util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode


def display_components(components, cmap='gray', headless=False):

    im_size = int(np.sqrt(components.shape[1]))
    plotv = components.reshape((-1, im_size, im_size))
    plotv = skimage.util.montage(plotv)

    if headless:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(plotv, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    return plt, ax


def scree_plot(explained_variance_ratio, headless=False):

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
        plt.title('{:0.2f}% in {} pcs'.format(csum[idx], idx + 1))

    plt.ylabel('Variance explained (percent)')
    plt.xlabel('nPCs')
    sns.despine()

    return plt


def changepoint_dist(cps, headless=False):

    if headless:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.set_style('ticks')

    ax = sns.distplot(cps, kde_kws={'gridsize': 600}, bins=np.linspace(0, 10, 100))
    ax.set_xlim((0, 2))
    ax.set_xticks(np.linspace(0, 2, 11))

    s = "Mean, Median, Mode (s) = %s" % str((mean(cps), median(cps), mode(cps)[0][0][0]))
    plt.text(1.5, 1.5, s)
    plt.ylabel('P(block duration)')
    plt.xlabel('Block duration (s)')
    sns.despine()

    return plt, ax
