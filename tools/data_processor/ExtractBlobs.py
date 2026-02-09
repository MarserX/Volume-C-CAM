'''
Title           : ExtractBlobs.py
Description     : This script extract Largest Blob from images
Author          : Jinnew Chyi
Date Created    : 20170914
Date Modified   : 20170914
paramaters      : a binary image,a int number (no matter positive or negative)
return          : a binary image with only the largest N blob
python_version  : 2.7.12
Example         : return a binary image with only the largest blob:
                  binaryImage = ExtractNLargestBlobs(binaryImage, 1)
Example         : return a binary image with the 3 largest blobs:
                  binaryImage = ExtractNLargestBlobs(binaryImage, 3)
'''

from skimage import measure, morphology
import numpy as np

def ExtractNLargestBlobs(binaryImage, numberToExtract):
    if numberToExtract < 0 :
        numberToExtract = -numberToExtract
    try:
        labels = measure.label(binaryImage, connectivity=1)
        if labels.max() > 1:
            regions = measure.regionprops(labels)
            allAreas = [regions[i].area for i in range(labels.max())]
            allAreas.sort(reverse=True)
            binaryImage.dtype=np.bool
            if numberToExtract > labels.max():
                offset=int(labels.max())-1
            else:
                offset=int(numberToExtract)-1
            dst=morphology.remove_small_objects(binaryImage, min_size=allAreas[offset],connectivity=1)
            dst.dtype = np.uint8
            return dst
        else :
            return binaryImage
    except:
        return False