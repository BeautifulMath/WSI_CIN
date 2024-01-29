## Reference: 'Cell-type-specific nuclear morphology predicts genomic instability and prognosis in multiple cancer types (PathAI)'

import os
import numpy as np
from PIL import Image
import pandas as pd
from skimage import measure
from skimage.color import rgb2hsv, rgb2lab
from tqdm import tqdm
import cv2 

def compute_features_list(image_path):
    """This function computes morphology features and mean, std of intensity. 
    Args:
        image_path: nucleus image path (We assume nucleus positioned at the center and the surrounding areas filled with white background)
    Returns: A list with all the features for the given image
    """
    img = Image.open(image_path)
    image_rgb = np.array(img.convert('RGB')) 
    image_gray = np.array(img.convert('L')) 
    ret, label = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)
    image_hsv = rgb2hsv(image_rgb) # convert RGB to  HSV scale
    image_s = image_hsv[:, :, 1] * 255 # rescale (0~1 -> 0~255)
    image_lab = rgb2lab(image_rgb) # convert RGB to LAB scale
    image_a = image_lab[:, :, 1] + 128 # rescale (-127~127 -> 1~255)
    image_b = image_lab[:, :, 2] + 128 # rescale (-127~127 -> 1~255)
    
    props_gray = measure.regionprops(label,image_gray)
    regionmask = props_gray[0].image # binary background mask
    intensity_gray = props_gray[0].intensity_image  # gray scale 
    
    props_s = measure.regionprops(label,image_s)
    intensity_s = props_s[0].intensity_image  # saturation value 
    
    props_a = measure.regionprops(label,image_a)
    intensity_a = props_a[0].intensity_image  # A value 
    
    props_b = measure.regionprops(label,image_b)
    intensity_b = props_b[0].intensity_image  # B value 
    
    # intensity_features 
    int_gray_mean = np.mean(intensity_gray[regionmask]) 
    int_gray_sd = np.std(intensity_gray[regionmask]) 
    int_s_mean = np.mean(intensity_s[regionmask])
    int_s_sd =  np.std(intensity_s[regionmask])
    int_a_mean = np.mean(intensity_a[regionmask])
    int_a_sd = np.std(intensity_a[regionmask])
    int_b_mean = np.mean(intensity_b[regionmask])
    int_b_sd = np.std(intensity_b[regionmask])
    
    # morphology_features
    area = props_gray[0].area
    perimeter = props_gray[0].perimeter
    convex_area = props_gray[0].convex_area
    major_axis_length = props_gray[0].major_axis_length
    minor_axis_length = props_gray[0].minor_axis_length
    eccentricity = props_gray[0].eccentricity
    solidity = area / convex_area
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    total_feat = [area,major_axis_length,minor_axis_length,perimeter,circularity,eccentricity,solidity,int_gray_mean,int_gray_sd,int_s_mean,int_s_sd,int_a_mean,int_a_sd,int_b_mean,int_b_sd]
    return total_feat

features = ['area', 'major_axis_length','minor_axis_length', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'int_gray_mean', 'int_gray_sd', 'int_s_mean', 'int_s_sd', 'int_a_mean', 'int_a_sd', 'int_b_mean', 'int_b_sd']

image_folder = './nucleus'
feats_path = './features/nucleus/morphology' 
if not os.path.exists(feats_path):
    os.makedirs(feats_path)
wsi_list = os.listdir(image_folder)

for j in range(len(wsi_list)):
    img_folder = os.path.join(image_folder, wsi_list[j])
    nucleus_list = os.listdir(img_folder)
    row_list = []
    for i in tqdm(range(len(nucleus_list))):
        img_path = os.path.join(img_folder, nucleus_list[i])
        row_list.append(compute_features_list(img_path))

    feature_df = pd.DataFrame(row_list, columns = features)
    feature_df.to_csv(os.path.join(feats_path, wsi_list[j][:23] + '.csv'), index=False)
