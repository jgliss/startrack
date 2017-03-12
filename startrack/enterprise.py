from os import mkdir
from os.path import join, exists, dirname, basename
from datetime import datetime
from warnings import warn
#__all__ = []
    
class TrailMaker(object):
    """Class for preparing a list of star trail images to create start trail time lapse
    
    This class loops over all images from which the star trail video is supposed to be created
    and processes them in a certain fashion (the idea is to provide several options to 
    create different effects). 
    
    Note
    ------
    
    No code yet ...
    
    Attributes
    ------------
    img_files : list
        list containing image file paths supposed to be processed
    start_idx : int
        start index for backwards adding of trails (if e.g. 10, then the first 9 images are only 
        loaded and saved without creating trails). 
    stop_idx : int
        last index considered for backwards adding.
    save_dir : str
        directory where processed images are stored.
        
    Parameters
    ---------------
    img_files : list
        list containing image file paths supposed to be processed
    start_idx : int
        start index for backwards adding of trails  
    stop_idx : :obj:`int`, optional
        last index considered for backwards adding. If None (default), then the trails are 
        created until the last image in the list
    save_dir : int, optional
        directory where processed images are stored. If None (default), then a new 
        directory *startrack_out* is created within the image directory
    """
    def __init__(self, img_files=[], start_idx=0, stop_idx=None, save_dir=None):
        
        if stop_idx is None:
            stop_idx = -1
    
        self._img_files =None
        self.img_files = img_files
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        
        if (save_dir is None or not exists(save_dir)) and exists(self.img_dir):
            save_dir = join(dirname(img_files[0]), "startrack_out")   
            if exists(save_dir):
                warn("Save directory %s already exists, images will be overwritten on re-run")
            else:
                mkdir(save_dir)
                
        self.save_dir = save_dir        
    
    @property 
    def img_files(self):
        """List containing image file paths"""
        return self._img_files
        
    @img_files.setter
    def img_files(self, val):
        if not isinstance(val, list):
            raise ValueError("Please provide a list with image file paths...")
        if not len(val) > 0:
            warn("No files available in list")
        self._img_files = val
        
    @property
    def img_dir(self):
        """Directory containing the images to be processed"""
        try:
            return dirname(self.img_files[0])
        except:
            return "Not available"
    
    @property
    def noi(self):
        """Number of images"""
        return len(self.img_files)
        
    def __str__(self):
        """String representation"""
        return ("startrack TrailMaker\n---------------------------------\n\n"
                   "Image source directory: %s\n"
                   "Number of images: %d\n"
                   "Save directory (processed results): %s\n"
                   "First image for trail processing: %d\n"
                   "Last image for trail processing: %d\n\n" 
                   "This could end up a very important class for preparing startrail images..."
                   %(self.img_dir, self.noi, self.save_dir, self.start_idx, self.stop_idx))