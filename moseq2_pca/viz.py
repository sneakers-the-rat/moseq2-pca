import skimage.util.montage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_components(components, cmap='gray', headless=False):

    im_size = int(np.sqrt(components.shape[1]))
    plotv = components.reshape((-1, im_size, im_size))
    plotv = skimage.util.montage.montage2d(plotv)

    if headless:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(plotv, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    return plt


def scree_plot(explained_variance_ratio, headless=False):

    csum = np.cumsum(explained_variance_ratio)*1e2
    idx = np.min(np.where(csum > 90))

    if headless:
        plt.switch_backend('agg')

    sns.set_style('ticks')
    plt.plot(np.cumsum(explained_variance_ratio)*1e2)
    plt.plot([idx, idx], [0, csum[idx]], 'k-')
    plt.plot([0, idx], [csum[idx], csum[idx]], 'k-')
    plt.ylim((0, 100))
    plt.xlim((0, len(explained_variance_ratio)))
    plt.title('{:0.2f}% in {} pcs'.format(csum[idx], idx))
    plt.ylabel('Variance explained (percent)')
    plt.xlabel('nPCs')
    sns.despine()

    return plt
