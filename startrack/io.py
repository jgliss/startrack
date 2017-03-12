"""
Input / output stuff
"""
from os.path import join, isfile
from os import listdir

def dummy_image():
    """Returns file path of dummy image"""
    import startrack
    return join(startrack.__dir__, "data", "test_img_stardetector.jpg")
    
def download_test_data(to_dir=None):
    """Download test dataset"""
    raise NotImplementedError("So much for the little training cruise . . .")
    
def get_image_files(img_dir, file_type=None):
    """Load all image file paths into list
    
    Parameters
    --------------
    img_dir : str
        image directory
    file_type : :obj:`str`, optional
        specify a certain file type
    
    Returns
    ----------
    list
        list containing image file paths
    """
    if file_type is None:
        return [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))]
    return [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) and 
               f.endswith(file_type)]
