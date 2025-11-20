import argparse
import glob
import os
import numpy as np

from bioio import BioImage
from skimage.filters import threshold_otsu
from tifffile import imwrite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", required=True)
    outputdir = parser.parse_args().outputdir.rstrip("/")
    dir_path = outputdir + "/runtime_data/infer_movie_multiscale_patch2/seg/*tif"
    path_512 = outputdir + "/runtime_data/infer_movie_multiscale_patch3/seg/"
    path_128 = outputdir + "/runtime_data/infer_movie_multiscale_patch1/seg/"
    targetname = outputdir + "/output/"

    for filename in glob.glob(dir_path):
        # Directory prep
        print(filename.split('/')[-1])
        imgname = filename.split('/')[-1]
        #id_temp = imgname.split('fms_id=')[1]
        #fms_id = id_temp.split('_')[0]
        fms_id = imgname.split('_')[0] + '_' + imgname.split('_')[1]
        print(fms_id)
        fms_id_dir_path = targetname + fms_id 
        if not os.path.exists(fms_id_dir_path):
            os.makedirs(fms_id_dir_path)
        targetfilename = fms_id_dir_path + '/' + imgname
        # Image operations
        reader = BioImage(filename) 
        IMG = reader.data
        IMG = IMG[0]
        print(IMG.shape)
        struct_img0 = IMG[0,:,:,:]
        struct_img1 = MyconvertFloatToChar(struct_img0)
        thre = threshold_otsu(struct_img1)
        bw = struct_img1 > thre
        
        img_512_path = path_512 + imgname
        img_512 = BioImage(img_512_path).data
        img_512 = img_512[0]
        struct_img0_512 = img_512[0,:,:,:]
        struct_img1_512 = MyconvertFloatToChar(struct_img0_512)
        thre_512 = threshold_otsu(struct_img1_512)
        bw_512 = struct_img1_512 > thre_512
        
        img_128_path = path_128 + imgname
        img_128 = BioImage(img_128_path).data
        img_128 = img_128[0]
        struct_img0_128 = img_128[0,:,:,:]
        struct_img1_128 = MyconvertFloatToChar(struct_img0_128)
        thre_128 = threshold_otsu(struct_img1_128)
        bw_128 = struct_img1_128 > thre_128
        
        bw_all = bw + bw_128 + bw_512
        
        out=bw_all.astype(np.uint8)
        out[out>0] = 255
        imwrite(targetfilename, out)


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


if __name__ == "__main__":
    main()
