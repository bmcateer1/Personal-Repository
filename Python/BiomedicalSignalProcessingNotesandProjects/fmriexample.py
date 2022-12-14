# -*- coding: utf-8 -*-
"""fMRIexample.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/haribharadwaj/notebooks/blob/main/BME511/fMRIexample.ipynb

<a href="https://colab.research.google.com/github/haribharadwaj/notebooks/blob/main/BME511/fMRIexample.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# fMRI finger tapping task example

A finger tapping task is done with fMRI where the subject does alternating 24-second-long blocks of fixation and finger tapping. The blood oxygenation-dependent signal (BOLD signal) is sampled at 4 Hz (i.e., MRI TR = 250 ms). This yields 96 samples of fixation images and 96 samples of finger tapping images.

Note that although fMRI data are volumetric (i.e., 3 spatial dimensions), only a single slice is provided here by taking advantage of *a priori* knowledge about where activation is likely to be observed.
"""

import numpy as np
import pylab as pl

# Setting it so figs will be a bit bigger
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [5, 3.33]
plt.rcParams['figure.dpi']  = 120

"""## Structual MRI for overlay

fMRI data (i.e., BOLD data) are T2-weighted images. A structural image (T1-weighted imaged) is provided for overlay purposes in the file ```motor.png```.
"""

import imageio
I = imageio.imread('motor.png', as_gray=True)
I = np.float64(I)/255.
pl.imshow(I, cmap='gray', vmin=0, vmax=1)

"""## BOLD data

fMRI data for 96 samples of each of two conditions is available in ```bold_data.mat```. Let's start by examining the contents of thhis file.
"""

from scipy import io
dat = io.loadmat('bold_data.mat', squeeze_me=True)

dat.keys()

dat['fs']

dat['fix'].shape

dat['tap'].shape

F = dat['fix'].mean(axis=0) * 100
pl.imshow(F, cmap='RdBu_r', vmin=-10, vmax=10)
pl.colorbar(label='% Change')

T = dat['tap'].mean(axis=0) * 100
pl.imshow(T, cmap='RdBu_r', vmin=-10, vmax=10)
pl.colorbar(label='% Change')

"""## Permutation Statistics

As usual, under the null hypothesis $\mathcal{H}_0$, there is no difference between tehh BOLD signal anywhhere in the brain slice between the ```fix``` and ```tap``` conditions. So to generate examples of data that we'd get when $\mathcal{H}_0$ is true, you can swap the two conditions.
"""

basket = np.concatenate((dat['fix'], dat['tap']), axis=0)

basket.shape

Nperms = 100
maxvals = np.zeros(Nperms)  # To store the peak we get under the null
nsamps = dat['fix'].shape[0]
for k in range(Nperms):
    order = np.random.permutation(2 * nsamps)
    Fperm = basket[order[:nsamps], :, :].mean(axis=0)
    Tperm = basket[order[nsamps:], :, :].mean(axis=0)
    maxvals[k] = (Tperm - Fperm).max()

pl.hist(maxvals, bins=20)

"""## Get threshold from $\mathcal{H}_0$ examples"""

maxvals.sort()
alpha = 0.05
threshold = maxvals[np.int32((1 - alpha) * Nperms)]
print(f'The threshold for alpha = {alpha} is {threshold * 100} %')

cutoff = threshold * 100
D = T - F
overlay = np.zeros(D.shape)
overlay[D > cutoff] = D[D > cutoff].copy()

pl.imshow(overlay, cmap='RdBu_r', vmin=-10, vmax=10)
pl.colorbar(label='% change')

"""## Overlay structural image and thresholded functional data"""

pl.imshow(I, cmap='gray', vmin=0, vmax=1)
pl.imshow(overlay, cmap='RdBu_r', vmin=-10, vmax=10, alpha=0.6)