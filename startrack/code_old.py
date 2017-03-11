# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 16:16:58 2016

@author: jg
"""
from os import listdir, mkdir
from os.path import join, isfile, basename, exists, dirname
from datetime import datetime
import shutil
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def get_paths(img_dir, file_type = "jpg", start_num = None, stop_num = None,\
                                                    num_pos = 1, delim = "_"):
            
    all_paths = [join(img_dir, f) for f in listdir(img_dir) if\
                        isfile(join(img_dir, f)) and f.endswith(file_type)]
    try:
        return [p for p in all_paths if start_num <= int(basename(p).\
                            split(".")[0].split("_")[num_pos]) <= stop_num]
    except:
        print "Failed to extract numbers from filenames, return all files"
        return all_paths
    
def get_image_data(fp):
    return np.array(cv2.imread(fp))

def make_mask(path, threshold = 20, dilate = 0, pyr_level = 0):
    """Create mask based on input image and threshold 
    
    The mask can be dilated using parameter dilate
    """
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)        
    for k in range(pyr_level):
        im = cv2.pyrDown(im)    
    mask = (im < threshold).astype(float)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 3)   
    return ~mask.astype(bool)

def prepare_folder_lrt_render(img_dir, file_type = "jpg"):
    render_dir = join(img_dir, "for_lrt")
    mkdir(render_dir)
    all_file_names = [f for f in listdir(img_dir) if\
                        isfile(join(img_dir, f)) and f.endswith(file_type)]
                        
    all_file_names.sort()
    counter = 1
    for file_name in all_file_names:
        new_name = "LRT_%05d.jpg" %counter
        shutil.copy(join(img_dir, file_name), join(render_dir, new_name))
        counter +=1
        
def process_startrails(img_dir, start_idx_blend = 0, mask_file = None,\
                mask_thresh = 20, mask_dilate = 3, std_thresh = None,\
                    std_blur = 4, save_series = False, save_base = "",\
                    run_test = False, naming = "original", **kwargs):
    """Function which processes a list of night sky images to create a startrail
    image"""
    if not naming in ["original", "lrt"]:
        raise ValueError("Use original or lrt as naming convention")
    file_paths = get_paths(img_dir, **kwargs)
    save_id = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
    if not exists(save_base):
        save_base = join(img_dir, "pystartrails_out")
        try:
            mkdir(save_base)
        except:
            pass
    #init last image
    last_img = get_image_data(file_paths[0])
    stack = np.zeros_like(last_img)
    height, width, _ = last_img.shape
    print height * width
    
    try:
        mask_init = make_mask(mask_file, mask_thresh, mask_dilate)
        used_mask = True
        print mask_init.shape
        """CHECK THIS STUFF"""
#==============================================================================
#     save = cv2.adaptiveThreshold(np.uint8(mask_init * 255), 255,\
#                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
#==============================================================================
        cv2.imwrite(join(save_base, "thresh_mask_%s.png" %save_id), np.uint8(mask_init * 255))
                                        
    except:
        mask_init = np.ones((height, width))
        used_mask = False

    if save_series:
        save_series_dir = join(save_base, ("for_LRT_%s" %save_id))
        try:
            mkdir(save_series_dir)
        except:
            pass
        
    f = open(join(save_base, ("settings_%s.txt" %save_id)), "w")
    f.write("Image base path: %s\n" %img_dir)
    f.write("First file: %s\n" %basename(file_paths[0]))
    f.write("Last file: %s\n" %basename(file_paths[-1]))
    f.write("First added file: %s\n" %basename(file_paths[start_idx_blend]))
    f.write("Threshold mask used: %s\n" %used_mask)
    f.write("\tThreshold mask file path: %s\n" %mask_file)
    f.write("\tThreshold mask lower threshold: %s\n" %mask_thresh)
    f.write("\tThreshold mask dilation: %s\n\n" %mask_dilate)
    f.write("Settings for detection of short term changes (i.e. pixel areas "
        "showing abrupt intensity variations changes between successive "
        "images\n")
    f.write("\tThresh standard deviation: %s\n" %std_thresh)
    f.write("\tThresh standard deviation (pre blur amount): %s\n" %std_blur)
    f.write("\tThresh standard deviation: %s\n" %std_thresh)
    f.close()
    if run_test:
        return mask_init
    stack = last_img
    if naming == "orignial":
        first_name =  basename(file_paths[0])
    else:
        first_name = "LRT_%05d.jpg" %1
        
    cv2.imwrite(join(save_series_dir, first_name), stack.astype(np.uint8))

    print "Total number of pixels: %s" %(width * height)
    print "Pixels considered for blending: %s " %sum(mask_init)
    for k in range(1, len(file_paths)):
        image = file_paths[k]
        print "Processing image file %s" %basename(image)
        image_new = get_image_data(image)
        if k < start_idx_blend:
            stack = image_new
        else:
            if std_thresh is not None:
                st = np.zeros((height, width, 2))
                st[:,:,0] = gaussian_filter(\
                            cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY), std_blur)
                st[:,:,1] = gaussian_filter(\
                            cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY),std_blur)
                cond = (st.std(axis = 2) < std_thresh)
                print "Remove additional %s pixels" %sum(cond)
                mask =  mask_init * cond
            else: 
                mask = mask_init
         
            stack = np.maximum(stack, image_new * mask[:, :, np.newaxis])
            
        if save_series:
            #output = Image.fromarray(stack.astype(np.uint8), mode = "RGB")
            if naming == "original":
                name = basename(image)
            else:
                name = "LRT_%05d.jpg" %(k+1)
            cv2.imwrite(join(save_series_dir, name), stack.astype(np.uint8))
        last_img = image_new
        
    cv2.imwrite(join(save_base, "result_%s.jpg" %save_id),\
                                            stack.astype(np.uint8))
    #stack = np.array(np.round(stack), dtype = np.uint8)
    return stack.astype(np.uint8)
    
def get_diff_image_gray(p0, p1):
    im0 = cv2.cvtColor(cv2.imread(p0), cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2GRAY)
    return (im1 -im0).astype(np.uint8)