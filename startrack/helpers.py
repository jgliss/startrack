# -*- coding: utf-8 -*-
"""
Module containing all sorts of helper methods (copied from pyplis package)    
"""

import matplotlib.cm as colormaps
import matplotlib.colors as colors
from datetime import datetime, time, date

from numpy import mod, linspace, hstack, vectorize, asarray, meshgrid, int, floor, log10,\
    isnan
from cv2 import pyrUp

exponent = lambda num: int(floor(log10(abs(num))))

time_delta_to_seconds = vectorize(lambda x: x.total_seconds())

def to_datetime(value):
    """Method to evaluate time and / or date input and convert to datetime"""
    if isinstance(value, datetime):
        return value
    elif isinstance(value, date):
        return datetime.combine(value, time())
    elif isinstance(value, time):
        return datetime.combine(date(1900,1,1), value)
    else:
        raise ValueError("Conversion into datetime object failed for input: "
            "%s (type: %s)" %(value, type(value)))       
        
def isnum(val):
    """Checks if input is number (int or float) and not nan
    
    Parameters
    ---------------
    val : 
        object to be checked
    
    Returns
    ----------
    bool 
    """
    if isinstance(val, (int, float)) and not isnan(val):
        return True
    return False
    
def mesh_from_img(img_arr):
    """Create a mesh from an 2D numpy array (e.g. gray scale image)
    
    Parameters
    ---------------
    img_arr : arraay
        image data (2D numpy array)
        
    Returns
    ----------
    tuple
        2-element tuple containing mesh
    """
    if not img_arr.ndim == 2:
        raise ValueError("Invalid dimension for image: %s" %img_arr.ndim)
    (ny, nx) = img_arr.shape
    xvec = linspace(0, nx - 1, nx)
    yvec = linspace(0, ny - 1, ny)
    return meshgrid(xvec, yvec)
    
def sub_img_to_detector_coords(img_arr, shape_orig, pyrlevel,
                                                        roi_abs=[0, 0, 9999, 9999]):
    """Converts a shape manipulated image to original detecor coords
    
    Parameters
    ---------------
    img_arr : array
        the sub image array (e.g. corresponding to a  certain ROI and / or pyrlevel)
    :param tuple shape_orig: original image shape (detector dimension)
    :param int pyrlevel: the pyramid level of the sub image
    :param list roi_abs: region of interest (in absolute image coords) of the
        sub image
    
    Note
    ------
    
    Regions outside the ROI are set to 0
        
    """
    from numpy import zeros, float32
    new_arr = zeros(shape_orig).astype(float32)
    for k in range(pyrlevel):
        img_arr = pyrUp(img_arr)
    new_arr[roi_abs[1]:roi_abs[3], roi_abs[0] : roi_abs[2]] = img_arr
    return new_arr
 
def roi2rect(roi):
    """Convert roi coordinates into rectangle coords: start point, height and width

    Parameters
    ---------------
    roi : list
        region of interest, i.e. ``[x0, y0, x1, y1]``
    
    Returns
    ----------
    tuple
        4-element tuple ``(x0, y0, w, h)
    """
    return (roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1])
    
def check_roi(roi, shape=None):
    """Checks if input fulfills all criteria for a valid ROI
    
    Parameters
    ---------------
    roi : list
        the ROI candidate to be checked
    shape : tuple
        dimension of image for which the ROI is supposed to be checked (optional)
        
    Returns
    ----------
    bool
        
    """
    try:
        if not len(roi) == 4:
            raise ValueError("Invalid number of entries for ROI")
        if not all([x >= 0 for x in roi]):
            raise ValueError("ROI entries must be larger than 0")
        if not (roi[2] > roi[0] and roi[3] > roi[1]):
            raise ValueError("x1 and y1 must be larger than x0 and y0")
        if shape is not None:
            if any([y > shape[0] for y in [roi[1], roi[3]]]):
                raise ValueError("ROI out of bounds of input shape..")
            elif any([x > shape[1] for x in [roi[0], roi[2]]]):
                raise ValueError("ROI out of bounds of input shape..")
        return True
    except:
        return False

def subimg_shape(img_shape=None, roi=None, pyrlevel=0):
    """Get shape of subimg after cropping and size reduction
    
    Parameters
    ---------------
    img_shape : tuple
        original image shape
    roi : list
        region of interest in original image, if this is provided img_shape param will be 
        ignored and the final image size is determined based on a cropped image within 
        the roi
    pyrlevel : int
        scale space parameter (Gauss pyramide) for size reduction
    
    Returns
    ----------
    tuple
        2-element tuple corresponding to shape``(h, w)`` of sub image
    """
    if roi is None:
        if not isinstance(img_shape, tuple):
            raise TypeError("Invalid input type for image shape: need tuple")
        shape = list(img_shape)
    else:
        shape = [roi[3] - roi[1], roi[2] - roi[0]]
    
    if not pyrlevel > 0:   
        return tuple(shape)
    for k in range(len(shape)):
        num = shape[k]
        add_one = False
        for i in range(pyrlevel):
            r = mod(num, 2)
            num = num / 2
            if not r == 0:
                add_one = True
            #print [i, num, r, add_one]
        shape[k] = num
        if add_one:
            shape[k] += 1
    return tuple(shape)

    
def map_coordinates_sub_img(pos_x_abs, pos_y_abs, roi_abs=[0,0,9999,9999],
                                                     pyrlevel=0, inverse=False):
    """Maps original input coordinates onto sub image
    
    Parameters
    -------------- 
    pos_x_abs : (:obj:`int`, :obj:`array`)
        x coordinate(s) in absolute image coords
    pos_y_abs : (:obj:`int`, :obj:`array`)
        y coordinate(s) in absolute image coords
    roi_abs : list
        ROI in absolute image coordinates (i.e. ``[x0, y0, x1, y1]``)
    pyrlevel : int
        level of gauss pyramid 
    inverse : bool
        if True, do inverse transformation (default=False)
    
    Returns
    ----------
    tuple
        2-element tuple containing mapped coordinates
        
    """
    op = 2 ** pyrlevel
    x, y = asarray(pos_x_abs), asarray(pos_y_abs)
    x_offs, y_offs = roi_abs[0], roi_abs[1]
    if inverse:
        return x_offs + x * op, y_offs + y * op
    return (x - x_offs) / op, (y - y_offs) / op

def map_roi(roi_abs, pyrlevel_rel=0, inverse=False):
    """Maps a list containing start / stop coords onto size reduced image
    
    Parameters
    ---------------
    roi_abs : list
        list specifying rectangular ROI in absolute image coordinates (i.e. ``[x0, y0, x1, y1]``)
    pyrlevel_rel : int
        gauss pyramid level (relative, use negative numbers to go up)
    inverse : bool
        inverse mapping
    
    Returns
    ----------
    list
        roi coordinates mapped to size reduced image
    """
    (x0, x1), (y0, y1) = map_coordinates_sub_img([roi_abs[0], roi_abs[2]],
                                                 [roi_abs[1], roi_abs[3]], 
                                                 pyrlevel=pyrlevel_rel, 
                                                 inverse=inverse)
            
    return [int(num) for num in [x0, y0, x1, y1]]
def shifted_color_map(vmin, vmax, cmap = None):
    """Shift center of a diverging colormap to value 0
    
    Note
    -------
    
        This method was found `here <http://stackoverflow.com/questions/
        7404116/defining-the-midpoint-of-a-colormap-in-matplotlib>`_ 
        (last access: 17/01/2017). Thanks to `Paul H <http://stackoverflow.com/
        users/1552748/paul-h>`_ who provided it.
    
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and if you want the
    middle of the colormap's dynamic range to be at zero level
    
    Parameters
    ---------------
    vmin : float 
        lower end of data value range
    vmax : float
        upper end of data value range
    cmap : 
        colormap (if None, use default cmap: seismic)
    
    Returns
    ----------
    cmap
        shifted colormap
        
    """

    if cmap is None:
        cmap = colormaps.seismic
        
    midpoint = 1 - abs(vmax)/(abs(vmax) + abs(vmin))
    
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = linspace(0, 1, 257)

    # shifted index to match the data
    shift_index = hstack([
        linspace(0.0, midpoint, 128, endpoint=False), 
        linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    return colors.LinearSegmentedColormap('shiftedcmap', cdict)
        
if __name__ == "__main__":
    import numpy as np
    from cv2 import pyrDown
    arr = np.ones((512,256), dtype = float)
    roi =[40, 50, 122, 201]
    pyrlevel = 3
    
    crop = arr[roi[1]:roi[3],roi[0]:roi[2]]
    for k in range(pyrlevel):
        crop = pyrDown(crop)
    print crop.shape
    print subimg_shape(roi = roi, pyrlevel = pyrlevel)
    
    
    
    