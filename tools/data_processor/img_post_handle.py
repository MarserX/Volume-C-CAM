import os
import cv2
from ExtractBlobs import ExtractNLargestBlobs

def Extract_Largest_One(img):
    img_out = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    img_out = ExtractNLargestBlobs(img_out, 1)
    img_out = cv2.dilate(img_out, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    img_out = ExtractNLargestBlobs(img_out, 1)
    return img_out


if __name__ == '__main__':
    src_dir = '/data3/masx/code/Volume-C-CAM/Volume-C-CAM-for-ACDC/myExperiment/results/EM_rounds_acdc/round_1/prediction/lpcam_filter_png'
    des_dir = '/data3/masx/code/Volume-C-CAM/Volume-C-CAM-for-ACDC/myExperiment/results/EM_rounds_acdc/round_1/prediction/lpcam_filter_png_post/'
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    img_list = os.listdir(src_dir)
    # img_list = os.listdir('E:/files/code/VNet-for-CZ/test_result2')
    for img_name in img_list:
        img_path = os.path.join(src_dir, img_name)
        img = cv2.imread(img_path)
        img_out = Extract_Largest_One(img)
        cv2.imwrite(des_dir + img_name, img_out)



