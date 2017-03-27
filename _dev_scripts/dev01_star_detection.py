"""
Development script 1: automatic star detection

Purpose: playing around with star detection algo

Loads test image (shipped with code), converts it to gray scale and applies star search 
algorithm to it. Outputs a plot of the image (cropped in ROI) including the detected 
sources.
"""

from matplotlib.pyplot import close, show
from astropy.stats import sigma_clipped_stats
import numpy as np
from photutils import DAOStarFinder, CircularAperture
import cv2

from os.path import join, basename
from startrack.io import dummy_image
from startrack import Image

### GLOBAL SETTINGS
from SETTINGS import SAVE_DIR, SAVEFIGS, DPI, FORMAT, OPTPARSE
SCRIPT_ID = basename(__file__).split("_")[0]

### OPTIONS
ROI = [1500, 1000, 2500, 2000] #[x0, y0, x1, y1]
FWHM = 2.0 # size of sources

if __name__=="__main__":
    close("all")
    figs = [] #append all figures supposed to be saved
    
    test_img_path = dummy_image()
    
    img = Image(test_img_path).crop(ROI)
    ax = img.show()
    
    gray = img.to_gray()
    
    corners =cv2. goodFeaturesToTrack(gray.data, 1000, 0.01, 10)
    corners = np.int0(corners)
    
    img = cv2.cvtColor(gray.data, cv2.COLOR_GRAY2BGR)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
        
    corner_img = Image(img)
    corner_img.show()
    #hdu = datasets.load_star_image()
    mean, median, std = sigma_clipped_stats(gray.data)
    daofind = DAOStarFinder(fwhm=2.0, threshold=std)
    
    data = gray.data - median
    sources = daofind(data)
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)
    #norm = ImageNormalize(stretch=SqrtStretch())
    #imshow(data, cmap='gray')# origin='lower', norm=norm)
    apertures.plot(color='y', lw=1.5, alpha=0.5, ax=ax)
    
    figs.append(ax.figure)
    
    ### IMPORTANT STUFF FINISHED    
    if SAVEFIGS:
        for k in range(len(figs)):
            figs[k].savefig(join(SAVE_DIR, "%s_out_%d.%s" %(SCRIPT_ID, (k+1), FORMAT)),
                                  format=FORMAT, dpi=DPI)
                          
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"