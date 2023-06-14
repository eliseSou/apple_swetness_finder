# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:10:06 2023

@author:  Hurgroup
"""

import os
from sklearn.model_selection import train_test_split
from imgpreprocessing import SegmentationDataset
from torch.utils.data import DataLoader
from utils import save_dataset, plot_preprocessed_with_bboxes


def get_image_json_pairs(root_folder):
    image_files = {}
    json_files = {}

    for root, _, files in os.walk(root_folder):
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in [".jpg", ".jpeg"]:
                image_files[filename] = os.path.join(root, file)
            elif ext == ".json":
                json_files[filename] = os.path.join(root, file)

    pairs = []
    for name in image_files:
        if name in json_files:
            pairs.append((image_files[name], json_files[name]))

    return pairs
for i in ['apple']:
    root_folder = i
    image_json_pairs = get_image_json_pairs(root_folder)

# Split pairs into train and validation sets
    train_pairs, val_pairs = train_test_split(image_json_pairs, test_size=0.20, random_state=42)

# Create the train and validation datasets
    train_dataset_0 = SegmentationDataset(train_pairs, small=False, cutted=False, darker=False, brighter=False)
    #train_dataset_1 = SegmentationDataset(train_pairs, small=True, cutted=False, darker=False, brighter=False, small_size=50)
    #train_dataset_2 = SegmentationDataset(train_pairs, small=True, cutted=False, darker=False, brighter=False, small_size=25)
    #train_dataset_3 = SegmentationDataset(train_pairs, small=True, cutted=False, darker=False, brighter=False, small_size=10)
    #train_dataset_4 = SegmentationDataset(train_pairs, small=False, cutted=False, darker=True, brighter=False)
    #train_dataset_5 = SegmentationDataset(train_pairs, small=False, cutted=False, darker=False, brighter=True)
    #val_dataset = SegmentationDataset(val_pairs, small=False, cutted=False, darker=False, brighter=False)

    train_dataset = train_dataset_0



# # Create DataLoaders for train and validation datasets
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    print("Start Saving Train DataSet")
    save_dataset(train_dataset, "preprocessed_train")

    #print("Start Saving Validation DataSet")
    #save_dataset(val_dataset, "preprocessed_val")

'''
preprocessed_dir = "preprocessed_train"
plot_preprocessed_with_bboxes(preprocessed_dir, num_images=5)
'''