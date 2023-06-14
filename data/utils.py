# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:03:19 2023

@author: Hurgroup
"""

import cv2
import numpy as np
# import json
import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# def create_mask(image, coordinates):
#     mask = np.zeros_like(image)
#     polygon = np.array(coordinates, dtype=np.int32).reshape((-1, 1, 2))
#     cv2.fillPoly(mask, [polygon], 255)
#     return mask

# def resize_image_and_mask(image, mask, new_size):
#     aspect_ratio = float(image.shape[1]) / float(image.shape[0])
#     new_width = int(aspect_ratio * new_size)
#     resized_image = cv2.resize(image, (new_width, new_size), interpolation=cv2.INTER_AREA)
#     resized_mask = cv2.resize(mask, (new_width, new_size), interpolation=cv2.INTER_AREA)
#     return resized_image, resized_mask
import numpy as np
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image)
    plt.show()
    
def resizing_image(image):
    resized = cv2.resize(image, (640, 480))
    return resized

def update_bbox(bbox, original_image_shape, resized_image_shape):
    #x_ratio = float(resized_image_shape[1]) / float(original_image_shape[1])
    #y_ratio = float(resized_image_shape[0]) / float(original_image_shape[0])
    x, y, w, h = bbox
    image_height = original_image_shape[0] / resized_image_shape[0]
    image_width = original_image_shape[1] / resized_image_shape[1]
    
    x = x / image_width
    y = y / image_height
    w = w / image_width
    h = h / image_height

    x = x + w / 2
    y = y + h / 2
    
    x = x / resized_image_shape[1]
    y = y / resized_image_shape[0]
    w = w / resized_image_shape[1]
    h = h / resized_image_shape[0]

    new_bbox = (x, y, w, h)
    return new_bbox

def crop_image(image, bounding_box):
    x, y, w, h = map(int, bounding_box)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def save_dataset(dataset, output_dir):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for i in range(len(dataset)):
        #a = random.random()
        #if a > 0.5:
        #    count += 1
        #    continue
        data = dataset[i]
        if data == None:
            continue
        else:
            image, bbox, information = data
            image_filename = f"{i+202412:04d}.jpg"
            # mask_filename = f"mask_{i:04d}.jpg"
            bbox_filename = f"{i+202412:04d}.txt"
    
            # if not os.path.exists(output_dir+'/images'):
            os.makedirs(output_dir+'/images', exist_ok = True)
            # if not os.path.exists(output_dir+'/labels'):
            os.makedirs(output_dir+'/labels', exist_ok = True)
                
            cv2.imwrite(os.path.join(output_dir+'/images', image_filename), image)
            # cv2.imwrite(os.path.join(output_dir+'/images', mask_filename), mask_to_save)
    
            # Save the bounding box data as a .json file
            with open(os.path.join(output_dir+'/labels', bbox_filename), "w") as f:
                # json.dump({"bbox": bbox}, f)
                if information['sugar_brix'] < 9.6:
                    f.write(f"{0} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_brix'] >= 9.6 and information['sugar_brix'] < 10.6:
                    f.write(f"{1} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_brix'] >= 10.6 and information['sugar_brix'] < 11.6:
                    f.write(f"{2} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_brix'] >= 11.6 and information['sugar_brix'] < 12.6:
                    f.write(f"{3} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_brix'] >= 12.6 and information['sugar_brix'] < 13.6:
                    f.write(f"{4} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_brix'] >= 13.6:
                    f.write(f"{5} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")

                '''
                if information['sugar_grade'] == 'A':
                    f.write(f"{0}  {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_grade'] == 'B':
                    f.write(f"{1}  {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                if information['sugar_grade'] == 'C':
                    f.write(f"{2}  {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                #f.write(f"{information['apple_kind']+'_'+information['sugar_grade']}  {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
                '''
def plot_preprocessed_with_bboxes(preprocessed_dir, num_images=10):
    image_files = sorted(os.listdir(os.path.join(preprocessed_dir, "images")))
    bbox_files = sorted(os.listdir(os.path.join(preprocessed_dir, "labels")))

    random_indices = random.sample(range(len(image_files)), num_images)
    
    for idx in range(len(random_indices)):
        # Read image
        image_path = os.path.join(preprocessed_dir, "images", image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read bbox data
        bbox_path = os.path.join(preprocessed_dir, "labels", bbox_files[idx])
        with open(bbox_path, "r") as f:
            bbox_data = f.read().strip().split()[1:]
            bbox = [float(b) for b in bbox_data]

        # Plot image and bbox
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()