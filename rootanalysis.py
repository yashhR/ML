from rca import RCA
from setuptools import setup

# initialize RCA instance:
rca_fitter = RCA(4)

# fit it to stars
rca_fitter.fit(stars, star_positions)

# return PSF model at positions of interest
psfs = rca_fitter.estimate_psf(galaxy_positions)


setup(
    name='rca',
    version='2.0.2',
    description='Resolved Component Analysis',
    author='Morgan A. Schmitz, Fred Ngole',
    author_email='morgan.schmitz@cea.fr',
    url='https://github.com/CosmoStat/rca',
    packages=['rca'],
    install_requires=['numpy','scipy','modopt']
)

#HST

import numpy as np
import matplotlib.pyplot as plt
from rca import RCA
from rca import utils

np.random.seed(1331)

def plot_func(im, wind=False, cmap='gist_stern', title=''):
    if not wind:
        plt.imshow(im, cmap=cmap, interpolation='Nearest')
    else:
        vmin, vmax = wind
        plt.imshow(im, cmap=cmap, interpolation='Nearest', vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

    starcat = np.load('stars.npy')
    pos = np.load('pos.npy')
    print(starcat.shape, pos.shape)

    mask = np.ones(starcat.shape)
    mask[starcat<-1e10] = 0
    starcat[starcat<-1e10] = 0
    for j in np.random.choice(starcat.shape[0], 5, False):
    plot_func(starcat[j])
    print(j)

test_idx = np.random.choice(starcat.shape[0], 23, False)
train_idx = list(set(range(starcat.shape[0])) - set(test_idx))

train_stars, train_pos = starcat[train_idx], pos[train_idx]
test_stars, test_pos = starcat[test_idx], pos[test_idx]
train_mask = mask[train_idx]

train_stars = utils.rca_format(train_stars)
train_mask = utils.rca_format(train_mask)

n_comp = 4
rca_fitter = RCA(n_comp, verbose=True)

S, A = rca_fitter.fit(train_stars, train_pos, train_mask, psf_size=2.)
rca_fitter.quicksave('fitted_model')

    eigenPSFs = utils.reg_format(S)
    for epsf in eigenPSFs:
        plot_func(np.abs(epsf))

    reconstructed_stars = rca_fitter.validation_stars(utils.rca_format(test_stars), test_pos)
    for j in np.random.choice(test_stars.shape[0], 3, False):
    rec_star = reconstructed_stars[j]
    wind = (min(np.min(test_stars[j]),np.min(rec_star)),
            max(np.max(test_stars[j]),np.max(rec_star)))
    plot_func(test_stars[j], title='Test star', wind=wind)
    plot_func(rec_star, title='PSF model', wind=wind)
    plot_func(np.abs(test_stars[j]-rec_star), title='residual', wind=wind)
    print(j)

    MSE = np.mean(np.linalg.norm(test_stars - reconstructed_stars, axis=(1,2))**2 /
              np.linalg.norm(test_stars, axis=(1,2))**2)
    print(MSE)
