# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: validomiX
#     language: python
#     name: validomix
# ---

import os
import shutil
import numpy as np
import glob
import itertools
import pandas as pd
import random
from skimage.filters import threshold_otsu
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tifffile import imsave
import matplotlib.pyplot as plt


def MyconvertFloatToChar(img):
    minimum_val = img.min()
    maximum_val = img.max()
    if (maximum_val - minimum_val) == 0.0:
        scale = 1.0
    else:
        scale = 255.0 / (maximum_val - minimum_val)
    img = (img - minimum_val) * scale
    img[img < 0.0] = 0.0
    img[img > 255] = 255
    return img.astype(np.uint8)


dir_path = "/pathto/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patch2/seg/*tif"
path_512 = "/pathto/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patch3/seg/"
path_128 = "/pathto/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patch1/seg/"
targetname = "/pathto/cyto-dl/data/all_cells_mask_test_dir/multiscale_all_cells_mask_v0/"

for filename in glob.glob(dir_path):
    # Directory prep
    print(filename.split('/')[-1])
    imgname = filename.split('/')[-1]
    id_temp = imgname.split('fms_id=')[1]
    fms_id = id_temp.split('_')[0]
    print(fms_id)
    fms_id_dir_path = targetname + fms_id 
    if not os.path.exists(fms_id_dir_path):
        os.makedirs(fms_id_dir_path)
    targetfilename = fms_id_dir_path + '/' + imgname
    # Image operations
    reader = AICSImage(filename) 
    IMG = reader.data
    IMG = IMG[0]
    print(IMG.shape)
    struct_img0 = IMG[0,:,:,:]
    struct_img1 = MyconvertFloatToChar(struct_img0)
    thre = threshold_otsu(struct_img1)
    bw = struct_img1 > thre
    
    img_512_path = path_512 + imgname
    img_512 = AICSImage(img_512_path).data
    img_512 = img_512[0]
    struct_img0_512 = img_512[0,:,:,:]
    struct_img1_512 = MyconvertFloatToChar(struct_img0_512)
    thre_512 = threshold_otsu(struct_img1_512)
    bw_512 = struct_img1_512 > thre_512
    
    img_128_path = path_128 + imgname
    img_128 = AICSImage(img_128_path).data
    img_128 = img_128[0]
    struct_img0_128 = img_128[0,:,:,:]
    struct_img1_128 = MyconvertFloatToChar(struct_img0_128)
    thre_128 = threshold_otsu(struct_img1_128)
    bw_128 = struct_img1_128 > thre_128
    
    bw_all = bw + bw_128 + bw_512
    
    out=bw_all.astype(np.uint8)
    out[out>0] = 255
    imsave(targetfilename, out)
