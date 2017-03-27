# -*- coding: utf-8 -*-
"""
Development script 2: close gaps
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from cv2 import dilate, erode, goodFeaturesToTrack, calcOpticalFlowPyrLK
        
plt.close("all")
bg = np.random.randint(20, 40, (100, 100)) #noise background

kernel_size = 6

fig, ax = plt.subplots(2,2, figsize=(16,12))
img1 = deepcopy(bg)
img1[30:33, 30:33] = 200

img2 = deepcopy(bg)
img2[35:38, 35:38] = 170

trail1 = np.maximum(img1, img2)
ax[0,0].imshow(img1, cmap="gray")
ax[0,0].set_title("Img 1")
ax[0,1].imshow(img2, cmap="gray")
ax[0,1].set_title("Img 2")
ax[1,0].imshow(trail1, cmap="gray")
ax[1,0].set_title("Trail (max pix)")

thresh = 100
#flow = calcOpticalFlowFarneback(img1,img2, 0.5, 4, 8, 5, 5, 1.1, flags=OPTFLOW_FARNEBACK_GAUSSIAN)

m = np.logical_or(img1 > thresh, img2 > thresh).astype(np.uint8)
morph_kernel =  np.ones((kernel_size, kernel_size)).astype(np.uint8)

f = goodFeaturesToTrack(m, 5, 0.1, 1)
ax[1,1].imshow(m, cmap="gray")
ax[1,1].set_title("Mask")

