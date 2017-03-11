"""
Old script from previous version of code

Purpose: playing around with star detection algo from astropy
"""

import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
#from astropy.visualization import SqrtStretch
#from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder, CircularAperture
import numpy as np
import cv2

from os.path import join

from startrack.code import data_dir
import startrack.code_old as trails

plt.close("all")
do_clip_analysis = False

test_img = join(data_dir(), "test_img_stardetector.jpg")

y0, y1 = 1000, 2000
x0, x1 = 1500, 2500

img = cv2.cvtColor(trails.get_image_data(test_img)[y0:y1, x0:x1, :],\
                                                            cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(1,2, figsize=(16,8))
ax[0].imshow(img, cmap = "gray")
ax[1].imshow(img > 200)


#hdu = datasets.load_star_image()
mean, median, std = sigma_clipped_stats(img, sigma = 100.0)
daofind = DAOStarFinder(fwhm = 2.0, threshold = std)

data = img - median
sources = daofind(data)
positions = (sources['xcentroid'], sources['ycentroid'])
apertures = CircularAperture(positions, r=4.)
#norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='gray')# origin='lower', norm=norm)
apertures.plot(color='blue', lw=1.5, alpha=0.5)

if do_clip_analysis:
    sigmas = np.arange(0,10,0.1)
    means, medians, stds = [], [], []
    for sigma in sigmas:
        mean, median, std = sigma_clipped_stats(data, sigma = sigma)
        print sigma
        print mean, median, std
        means.append(mean), medians.append(median), stds.append(std)
        
    fig, ax = plt.subplots(1,3)
    ax[0].plot(sigmas, means)
    ax[0].set_xlabel("Sigma")
    ax[0].set_ylabel("Mean")
    ax[1].plot(sigmas, medians)
    ax[1].set_xlabel("Sigma")
    ax[1].set_ylabel("Median")
    ax[2].plot(sigmas, stds)
    ax[2].set_xlabel("Sigma")
    ax[2].set_ylabel("Stds")