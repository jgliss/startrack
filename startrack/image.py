# -*- coding: utf-8 -*-
"""Image base module"""
from matplotlib.pyplot import subplots
from astropy.io import fits
from numpy import ndarray, histogram, nan, isnan, uint8, ones
from os.path import abspath, splitext, basename, exists
from warnings import warn
from datetime import datetime
from cv2 import pyrDown, pyrUp, imread, dilate, cvtColor, COLOR_BGR2GRAY
from scipy.ndimage.filters import gaussian_filter, median_filter
from collections import OrderedDict as od
from copy import deepcopy

from .helpers import map_roi, check_roi

class ImgMetaError(Exception):
        pass

class Image(object):
    """ Image base class
    
    Based on :class:`Img` object of `pyplis <http://pyplis.readthedocs.io/en/latest/
    code_lib.html#module-pyplis.image>`__. 
    
    Image data is represented as :class:`numpy.ndarray` object and the image data is 
    stored in the attribute :attr:`data`.
    
    Image data is imported using :func:`cv2.imread`, or, for FITS files using astropy.
    
    The object includes several loading routines and basic image editing. 
    Image meta information can be provided on creation of this instance by
    providing valid meta keys and the corresponding values, i.e.::
        
        from startrack import Image
        png_image_file = "C:/Test/my_img_file.png"
        acq_time = datetime(2016, 10, 10, 13, 15, 12) #10/10/2016, 13:15:12
        exposure_time = 0.75 #s
        img = Image(png_image_file, start_acq=acq_time, texp=exposure_time)
        
    Meta information is stored in the dictionary ``self.meta`` and can be 
    printed using :func:`print_meta`. The two most important image meta 
    parameters are the acquisition time (``img.meta["start_acq"]``) and the
    exposure time (``img.meta["texp"]``). These two parameters have class own
    access methods (:func:`start_acq` and :func:`texp`).
    
    The class provides several image editing routines, of which the most 
    important ones (within this library) are (please see documentation of the 
    individual functions for more information):
        
        1. :func:`subtract_dark_image` (subtract a dark image)
        #. :func:`correct_dark_offset` (Correct for dark and offset. Models 
                a dark image based on one dark and one offset image using the 
                exposure time of this image, then uses 1. for subtraction)
        #. :func:`crop` (crop image within region of interest)
        #. :func:`apply_median_filter` (median filtering of image)
        #. :func:`add_gaussian_blurring` (Add blurring to image taking into 
                account current blurring amount)
        #. :func:`apply_gaussian_blurring` (applies gaussian filter to image)
        #. :func:`pyr_down` (reduce image size using gaussian pyramide)
        #. :func:`pyr_up` (increase image size using gaussian pyramide)
                
    All image editing steps performed using these functions are logged in the 
    dictionary ``self.edit_log``, it is therefore recommended to use the class
    own methods for these image editing routines (and not apply them manually 
    to the image data, e.g. by using ``cv2.pyrDown(img.data)`` for resizing or 
    ``img.data = img.data[y0:y1, x0:x1]`` for cropping a ROI ``[x0, x1, y0, y1]``
    ) in order to keep track of the changes applied.
    
    The image data is imported as :obj:`uint16`.
    
    Parameters
    ---------------
    inp : 
        input: can be, e.g file path to image or numpy array
    import_method : function
        custom image load method, must return tuple containing image data (2D or 3D 
        numpy array) and dictionary containing meta information (can be empty if read 
        routine does not import any meta information)
    **meta_info : 
        keyword args specifying meta data (see :attr:`meta` for valid keys)
    
        
    """
    _FITSEXT = [".fits", ".fit", ".fts"]
    
    def __init__(self, inp=None, import_method=None, **meta_info):
        if isinstance(inp, Image):
            return inp
            
        self._data = None #: the actual image data
        
        self.vign_mask = None
        
        # custom data import method (optional on class initialisation)
        self.import_method = import_method
        #Log of applied edit operations
        self.edit_log = od([("darkcorr"   ,   0), # boolean
                                     ("blurring"    ,   0), # int (width of kernel)
                                     ("median"    ,   0), # int (size of filter)
                                     ("crop"        ,   0), # boolean
                                     ("8bit"         ,   0), # boolean
                                     ("pyrlevel"   ,   0), # int (pyramide level)
                                     ("vigncorr"   ,   0), # boolean (vignette corrected)
                                     ("is_bin"      ,   0),
                                     ("is_inv"      ,   0),
                                     ("others"     ,   0)])# boolean 
        
        self._roi_abs = [0, 0, 9999, 9999] #will be set on image load
        
        self._header_raw = {}
        self.meta = od([("start_acq"     ,   datetime(1900, 1, 1)),#datetime(1900, 1, 1)),
                                 ("stop_acq"      ,   datetime(1900, 1, 1)),#datetime(1900, 1, 1)),
                                 ("texp"          ,   float(0.0)), # exposure time [s]
                                 ("bit_depth"     ,   nan), # pix bit depth
                                 ("path"          ,   ""),
                                 ("file_name"     ,   ""),
                                 ("file_type"     ,   "")])
                                
        
        try:
            temp = import_method(inp)            
            inp = temp[0]
            meta_info.update(temp[1])
        except:
            pass
          
        for k, v in meta_info.iteritems():
            if self.meta.has_key(k) and isinstance(v, type(self.meta[k])):
                self.meta[k] = v
            elif self.edit_log.has_key(k):
                self.edit_log[k] = v
        if inp is not None:                              
            self.load_input(inp)
        try:
            self.set_roi_whole_image()
        except:
            pass
    
    @property
    def data(self):
        """Get / set image data"""
        return self._data
    
    @data.setter
    def data(self, val):
        """Setter for image data"""
        self._data = val
        
    def set_data(self, inp):
        """Try load input"""
        try:
            self.load_input(inp)
        except Exception as e:
            print repr(e)
    
    def reload(self):
        """Try reload from file"""
        self.__init__(self.meta["path"])
        
    def load_input(self, inp):
        """Try to load input as numpy array and additional meta data"""
        try:
            if any([isinstance(inp, x) for x in [str, unicode]]) and\
                                                                exists(inp):
                self.load_file(inp)
            
            elif isinstance(inp, ndarray):
                self.data = inp
            else:
                raise
        except:
            raise IOError("Image data could not be imported, invalid input: %s"
                        %(inp))
    
    def histogram(self, num_bins=None):
        """Make histogram of current image"""
        if isnan(self.meta["bit_depth"]):
            print ("Error in " + self.__str__() + ".make_histogram\n "
                "No MetaData available => BitDepth could not be retrieved. "
                "Using 100 bins and img min/max range instead")
            hist, bins = histogram(self.data, 100)
            return hist, bins
        #print "Determining Histogram"
        hist, bins = histogram(self.data, 2**(self.meta["bit_depth"]),
                               [0, 2**(self.meta["bit_depth"])])
        return hist, bins
    
    def crop(self, roi_abs=[0, 0, 9999, 9999], new_img=False):
        """Cut subimage specified by rectangular ROI
        
        :param list roi_abs: region of interest (i.e. ``[x0, y0, x1, y1]``)
            in ABSOLUTE image coordinates. The ROI is automatically converted 
            with respect to current pyrlevel
        :param bool new_img: creates and returns a new image object and leaves 
            this one uncropped        
        :return:
            - Image, cropped image
        """
        if self.edit_log["crop"]:
            warn("Cropping image that was already cropped...")
        self.roi_abs = roi_abs #updates current roi_abs setting
        roi = self.roi #self.roi is @property method and takes care of ROI conv
        if self.ndim == 2:
            sub = self.data[roi[1]:roi[3], roi[0]:roi[2]]
        elif self.ndim == 3:
            sub = self.data[roi[1]:roi[3], roi[0]:roi[2], :]
        else:
            raise ValueError("Invalid image dimension")
        im = self
        if new_img:
            im = self.duplicate()
#        im._roi_abs = roi
        im.edit_log["crop"] = 1
        im.data = sub
        return im
    
    @property
    def pyrlevel(self):
        """Returns current gauss pyramid level (stored in ``self.edit_log``)"""
        return self.edit_log["pyrlevel"]
    
    @property 
    def roi(self):
        """Returns current roi (in consideration of current pyrlevel)"""
        roi_sub = map_roi(self._roi_abs, self.edit_log["pyrlevel"])
        return roi_sub
    
    @property
    def roi_abs(self):
        """Current ROI in absolute image coordinates
        
        Note
        ------
        
            use :func:`roi` to get ROI for current pyrlevel
            
        """
        return self._roi_abs
        
    @roi_abs.setter
    def roi_abs(self, val):
        """Updates current ROI"""
        if check_roi(val):
            self._roi_abs = val
            
    def set_roi_whole_image(self):
        """Set current ROI to whole image area based on shape of image data"""
        h, w = self.data.shape[:2]
    
        self._roi_abs = [0, 0, w * 2**self.pyrlevel, h * 2**self.pyrlevel]     
    
    def apply_median_filter(self, size_final=3):
        """Apply a median filter to 
        
        :param tuple shape (3,3): size of the filter        
        """
        diff = int(size_final - self.edit_log["median"])
        if diff > 0:
            self.data = median_filter(self.data, diff)
            self.edit_log["median"] += diff
          
    def blur(self, sigma, **kwargs):
        """Add gaussian blurring using :class:`scipy.ndimage.filters.gaussian_filter`
        
        :param int sigma: amount of blurring
        """
        self.data = gaussian_filter(self.data, sigma, **kwargs)
        self.edit_log["blurring"] += sigma   
    
    def to_pyrlevel(self, final_state=0):
        """Down / upscale image to a given pyramide level"""
        steps = final_state - self.edit_log["pyrlevel"]
        if steps > 0:
            return self.pyr_down(steps)
        elif steps < 0:
            return self.pyr_up(-steps)
            
    def pyr_down(self, steps=0):
        """Reduce the image size using gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrDown` 
        """
        if not steps:
            return
        #print "Reducing image size, pyrlevel %s" %steps
        for i in range(steps):
            self.data = pyrDown(self.data)
        self.edit_log["pyrlevel"] += steps
        return self
    
    def pyr_up(self, steps):
        """Increasing the image size using gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrUp` 
        """
        for i in range(steps):
            self.data = pyrUp(self.data)
        self.edit_log["pyrlevel"] -= steps  
        self.edit_log["others"] = 1
        return self

    def print_meta(self):
        """Print current image meta information"""
        for key, val in self.meta.iteritems():
            print "%s: %s\n" %(key, val)
               
    def duplicate(self):
        """Duplicate this image"""
        #print self.meta["file_name") + ' successfully duplicated'
        return deepcopy(self)
        
    def to_gray(self, cspace_conv=COLOR_BGR2GRAY):
        """Convert to gray scale image
        
        Note
        -------
        
        Creates new :class:`Image` object: this image remains unchanged
        
        Parameters
        ---------------
        cspace_conv :
            cv2 conversion arg 
            
        Returns
        ----------
        Image 
            new image object containing gray scale image 
        """
        new = self.duplicate()
        if self.is_gray: 
            warn("Image is already gray scale")
        else:
            new.data = cvtColor(self.data, COLOR_BGR2GRAY)
        return new
         
    def to_binary(self, threshold=None):
        """Convert image to binary image using threshold
        
        Note
        -------
        
        Creates new :class:`Image` object: this image remains unchanged
        
        Parameters
        ---------------
        threshold : float
            threshold, if None, use mean value of image data (gray)
            
        Returns
        -----------
        Img
            binary image (new instance)
        """
        if self.is_gray:
            img = self.duplicate()
        else:
            img = self.to_gray()
        if threshold is None:
            threshold = img.mean()
        img.data = (img.data > threshold).astype(uint8)
        img.edit_log["is_bin"] = True
        return img
    
    def invert(self):
        """Invert image
        
        Note
        -------
        
        Works currently only for gray scale images
        
        Returns
        -----------
        Img
            inverted image object (new instance)
        
        """
        if not self.is_gray:
            raise NotImplementedError("Currently only gray scale images can be inverted")
        elif self.is_binary:
            inv = ~self.data / 255.0
            new = self.duplicate()
            new.data = (inv).astype(uint8)
            return new
        else:
            raise NotImplementedError("Coming soon...")
            
            
    def dilate(self, kernel=None):
        """Apply morphological transformation Dilation to image
        
        Uses :func:`cv2.dilate` for dilation. The method requires specification
        of a smoothing kernel, if unspecified, a 9x9 neighbourhood is used
        
        Note
        -------
        
        This operation can only be performed to binary images, use 
        :func:`to_binary` if applicable.
        
        Parameters
        ---------------
        kernel : array
            kernel used for :func:`cv2.dilate`, if None a 9x9 array is used::
            
                kernel = np.ones((9,9), dtype=np.uint8)
        
        Returns
        ----------
        Img 
            dilated binary image
            
        """
        if not self.is_binary:
            raise AttributeError("Img needs to be binary, use method to_binary")
        if kernel is None:
            kernel = ones((9,9), dtype=uint8)
        new = self.duplicate()
        new.data = dilate(new.data, kernel=kernel)
        new.edit_log["others"] = True
        return new
        
    def normalise(self, blur=1):
        """Normalise this image
        
        Note
        -------
        
        This operation can only be performed to gray scale images, use 
        :func:`to_gray` if applicable.
        """
        if not self.is_gray():
            raise AttributeError("Only works for gray scale images")
        new = self.duplicate()
        if self.edit_log["blurring"] == 0 and blur != 0:
            new.add_gaussian_blurring(blur)
        new.data = new.data / new.data.max()
        return new
        
    def mean(self):
        """Returns mean value of current image data"""
        return self.to_gray().data.mean()
        
    def std(self):
        """Returns standard deviation of current image data"""
        return self.to_gray().data.std()
        
    def min(self):
        """Returns minimum value of current image data"""
        return self.to_gray().data.min()
    
    def max(self):
        """Returns maximum value of current image data"""
        return self.to_gray().data.max()
        
    def meta(self, meta_key):
        """Returns current meta data for input key"""
        return self.meta[meta_key]
    
    """DECORATORS"""    
    @property
    def start_acq(self):
        """Get image acquisition time"""
        if self.meta["start_acq"] == datetime(1900, 1, 1):
            raise ImgMetaError("Image acquisition time not set")
        return self.meta["start_acq"]
    
    @property
    def stop_acq(self):
        """Returns stop time of acquisition (if available)"""
        return self.meta["stop_acq"]
    
    @property
    def texp(self):
        """Get image acquisition time
        
        :returns: acquisition time if available (i.e. it deviates from the
            default 1/1/1900), else, raises ImgMetaError
        """
        if self.meta["texp"] == 0.0:
            raise ImgMetaError("Image exposure time not set")
        return self.meta["texp"]
    
    @property
    def shape(self):
        """Shape of image data"""
        return self.data.shape
        
    @property
    def ndim(self):
        """Dimension of image data"""
        return self.data.ndim
    
    @property 
    def dtype(self):
        """Data type of image data"""
        return self.data.dtype
    @property    
    def pyr_up_factor(self):
        """Factor to convert coordinates at current pyramid level into original size coordinates
        """
        return 2 ** self.edit_log["pyrlevel"]
        
    @property
    def is_gray(self):
        """Checks if image is gray image"""
        if self.data.ndim == 2:
            return True
        elif self.data.ndim == 3:
            return False
        else:
            raise Exception("Unexpected image dimension %s..." %self.data.ndim)
            
    @property
    def modified(self):
        """Check if this image is modified
        
        Returns
        ----------
        bool
        """
        if sum(self.edit_log.values()) > 0:
            return True
        return False

    def load_file(self, file_path):
        """Try to import file specified by input path
        
        Parameters
        ---------------
        file_path : str
            path to image file
        """
        ext = splitext(file_path)[-1]
        try:
            self.load_fits(file_path)
        except:
            self.data = imread(file_path)#.astype(self.dtype)
        self.meta["path"] = abspath(file_path)
        self.meta["file_name"] = basename(file_path)
        self.meta["file_type"] = ext
    
    def load_fits(self, file_path):
        """Import a FITS file 
        
        `FITS info <http://docs.astropy.org/en/stable/io/fits/>`_
        
        Parameters
        ---------------
        file_path : str
            path to FITS file
        """
        hdu = fits.open(file_path)
        head = hdu[0].header 
        self._header_raw = head
        self.data = hdu[0].data
        hdu.close()
        editkeys = self.edit_log.keys()
        metakeys = self.meta.keys()
        for key, val in head.iteritems():
            k = key.lower()
            if k in editkeys:
                self.edit_log[k] = val
            elif k in metakeys:
                self.meta[k] = val
        try:
            self._roi_abs = hdu[1].data["roi_abs"]
        except:
            pass
        try:
            self.vign_mask = hdu[2].data
            print "Fits file includes vignetting mask"
        except:
            pass
                
    def show(self, **kwargs):
        """Show image (wrapper for :func:`show_img`)"""
        return self.show_img(**kwargs)

    def show_img(self, ax=None, **kwargs):
        """Show image using matplotlib imshow
        
        Parameters
        ---------------
        ax : :obj:`Axes`, optional
            matplotlib axes instance
        **kwargs
            additional keyword args passed to :func:`imshow` (e.g. vmin, vmax)
            
        Returns
        ----------
        Axes
        """
        if ax is None:
            _, ax = subplots(1,1)
        
        ax.imshow(self.data, **kwargs)
        return ax
    
    def info(self):
        """Image info (prints string representation)"""
        print self.__str__()
        
    """MAGIC METHODS"""
    def __str__(self):
        """String representation"""
        gray = self.to_gray()
        s = "\n-----------\nstartrack Image\n-----------\n\n"
        s += "Min / Max intensity (gray): %s - %s\n" %(gray.min(), gray.max())
        s += "Mean intensity (gray): %s\n" %(gray.data.mean())
        s += "Shape: %s\n" %str(self.shape)
        s += "ROI (abs. coords): %s\n" %self.roi_abs
        s += "\nMeta information\n-------------------\n"
        for k, v in self.meta.iteritems():
            s += "%s: %s\n" %(k, v)
        s += "\nEdit log\n-----------\n"
        for k, v in self.edit_log.iteritems():
            s += "%s: %s\n" %(k, v)
        return s
            
    def __call__(self):
        """Return image numpy array on call"""
        return self.data
        
    def __add__(self, val):
        """Add data to this image
        
        Parameters
        ---------------
        other
            data to be added (e.g. :obj:`Image` or :obj:`array`)
        
        Returns
        ----------
        Image
            result image object
        """
        try: #other is Image instance
            im = self.duplicate()
            im.data = self.data + val.data
            return im
        except: # other is probably numpy array or number
            try:
                im = self.duplicate()
                im.data = self.data + val
                return im
            except:
                raise TypeError("Could not add value %s to image" %type(val))
        
            
    def __sub__(self, val):
        """Subtract data from this image
        
        Parameters
        ---------------
        other
            data to be subtracted (e.g. :obj:`Image` or :obj:`array`)
        
        Returns
        ----------
        Image
            result image object
        """
        try:
            im = self.duplicate()
            im.data = self.data - val.data
            return im
        except:
            try:
                im = self.duplicate()
                im.data = self.data - val
                return im
            except:
                raise TypeError("Could not subtract value %s from image" 
                                                                %type(val))
    
    def __mul__(self, val):
        """Multiply data to this image
        
        Parameters
        ---------------
        other
            data to be multiplied (e.g. :obj:`Image` or :obj:`array`)
        
        Returns
        ----------
        Image
            result image object
        """
        try:
            im = self.duplicate()
            im.data = self.data * val.data
            return im
        except:
            try:
                im = self.duplicate()
                im.data = self.data * val
                return im
            except:
                raise TypeError("Could not multilply image with value %s" 
                                                                %type(val))

    def __div__(self, val):
        """Divide this image with input data 
        
        Parameters
        ---------------
        other
            divisor (e.g. :obj:`Image` or :obj:`array`)
        
        Returns
        ----------
        Image
            result image object
        """
        try:
            im = self.duplicate()
            im.data = self.data / val.data
            return im
        except:
            try:
                im = self.duplicate()
                im.data = self.data / val
                return im
            except:
                raise TypeError("Could not divide image with value %s" 
                                                                %type(val))
            

