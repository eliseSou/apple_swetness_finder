import json
import cv2
from torch.utils.data import Dataset
from utils import update_bbox, resizing_image
import io

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class SegmentationDataset(Dataset):
    def __init__(self, image_json_pairs, small, cutted, darker, brighter, small_size=50, rightleft=False, updown=False, new_size=640, transform=None):
        self.image_json_pairs = image_json_pairs
        self.new_size = new_size
        self.transform = transform 
        self.small = small
        self.cutted = cutted
        self.small_size = small_size
        self.rightleft = rightleft
        self.updown = updown
        self.darker = darker
        self.brighter = brighter

    def __len__(self):
        return len(self.image_json_pairs)

    def __getitem__(self, idx):
        image_path, json_path = self.image_json_pairs[idx]
        
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        json_data = None
        file_loaded = False
        
        try:
            with io.open(json_path, "r", encoding="utf-8-sig") as file:
                json_data = json.load(file)
                image = cv2.imread(image_path)
            file_loaded = True
        except UnicodeDecodeError:
            return
            
        if file_loaded:
            if image is None:
                return
            flat_segmentation_coordinates = json_data["annotations"]["segmentation"]
            coordinates = [(int(round(flat_segmentation_coordinates[i])), int(round(flat_segmentation_coordinates[i+1]))) for i in range(0, len(flat_segmentation_coordinates), 2)]
            # mask = create_mask(image, coordinates)
            bbox = json_data["annotations"]["bbox"]
            apple_kind = json_data["collection"]["apple_kind"]
            sugar_grade = json_data["annotations"]["sugar_grade"]
            sugar_brix = json_data["collection"]["sugar_content_nir"]

            information = {"apple_kind":apple_kind, "sugar_grade":sugar_grade, "sugar_brix":sugar_brix}
            # cropped_image = crop_image(image, bbox)
            
            if self.small == True:
                image, bbox = make_small(image, bbox, self.small_size)
            #if self.cutted == True:
            #    image, bbox = make_cutted(image, bbox, self.rightleft, self.updown)

            if self.darker == True:
                image = make_darker(image)
                image = np.array(image)

            if self.brighter == True:
                image = make_brighter(image)
                image = np.array(image)
            
            # resized_image, resized_mask = resize_image_and_mask(image, mask, self.new_size)
            resized_image = resizing_image(image)
            resized_bbox = update_bbox(bbox, image.shape, resized_image.shape)
            if self.transform:
                # resized_image, resized_mask = self.transform(resized_image, resized_mask)
                resized_image = self.transform(resized_image)

            # # Convert to PyTorch tensors
            # resized_image = torch.from_numpy(resized_image.transpose((2, 0, 1))).float()
            # resized_mask = torch.from_numpy(resized_mask).long()
            
            # return resized_image, resized_mask, resized_bbox, information
            return resized_image, resized_bbox, information
        
            pass

import numpy as np
def make_small(image, bbox, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # 남는 공간을 검정색으로 채웁니다.
    
    bbox[0] = bbox[0] * (scale_percent / 100)
    bbox[1] = bbox[1] * (scale_percent / 100)
    bbox[2] = bbox[2] * (scale_percent / 100)
    bbox[3] = bbox[3] * (scale_percent / 100)


    black_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    x_offset = int((image.shape[1] - resized.shape[1]) / 2)
    y_offset = int((image.shape[0] - resized.shape[0]) / 2)
    black_image[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    bbox[0] = bbox[0] + x_offset
    bbox[1] = bbox[1] + y_offset

    # 결과 이미지를 저장합니다.
    return black_image, bbox


def make_brighter(input_image, brightness=100):
    '''
        input : 
            image (i dont know the type must be open with Image.open)
            if brightness is not specified is set to 100
        output : 
            two image : light and dark
    '''
    img = np.array(input_image)
    cv2.convertScaleAbs(img, img, 1, brightness)
    img_bright = Image.fromarray(np.uint8(img))
    return img_bright

def make_darker(input_image, brightness=100):

    img = np.array(input_image)
    cv2.convertScaleAbs(img, img, 1, -brightness)
    img_dark = Image.fromarray(np.uint8(img))
    return img_dark


def make_many(input_image):

    return 0