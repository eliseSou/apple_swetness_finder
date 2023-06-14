#!/usr/bin/env/python3
#!/usr/bin/python

from PIL import Image
import numpy as np
import cv2
import os


def fast_brightness(input_image, brightness=100):
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
    cv2.convertScaleAbs(img, img, 1, -brightness)
    img_dark = Image.fromarray(np.uint8(img))
    return img_bright, img_dark
    
def change_light_in_directory(directory, brightness=100):
    '''input : 
            string : directory, the name of the directory 
            integer : brightness, level of change, if not specified it is set to 100
        the function save the generated image in the same directory. 
    '''
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            light_image, dark_image  = fast_brightness(img)
            dark_filename = os.path.splitext(filename)[0] + "_dark" + os.path.splitext(filename)[1]
            light_filename = os.path.splitext(filename)[0] + "_light" + os.path.splitext(filename)[1]
            dark_image.save(os.path.join(directory, dark_filename))
            light_image.save(os.path.join(directory, light_filename))