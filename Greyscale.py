#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Blue

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

PIXEL_TO_UM = 2

def calculate_ring_average_grayscale(image, inner_radius, outer_radius):
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    mask_inner = (x - center_x) ** 2 + (y - center_y) ** 2 >= inner_radius ** 2
    mask_outer = (x - center_x) ** 2 + (y - center_y) ** 2 < outer_radius ** 2

    ring_mask = mask_inner & mask_outer
    ring_pixels = image[ring_mask]
    
    if len(ring_pixels) == 0:
        return 0  
    return np.mean(ring_pixels)

def plot_grayscale_profile(image_path, min_radius, max_radius, step):

    try:
        image = Image.open(image_path)
        grayscale_image = ImageOps.grayscale(image)
        image_np = np.array(grayscale_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    radii = []
    average_grayscales = []
    
    for radius in range(min_radius, max_radius + 1, step):
        avg_gray = calculate_ring_average_grayscale(image_np, radius - step, radius)
        average_radius_um = (radius - step / 2) * PIXEL_TO_UM
        radii.append(average_radius_um)
        average_grayscales.append(avg_gray)

        if radius % 50 == 0:
            print(f"Processed radius: {radius} pixels, Average grayscale: {avg_gray}")
    

    plt.figure(figsize=(10, 6))
    plt.plot(radii, average_grayscales, label='Average Grayscale', color='b')
    plt.xlabel('Radial Distance r (µm)')
    plt.ylabel('Average Grayscale')
    plt.title('Variation of Average Grayscale with Radial Distance r (Blue Food Color)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Variation of Average Grayscale with Radial Distance r (Blue Food Color)')
    plt.show()

plot_grayscale_profile('blueDoubeP.JPG', 25, 900, 1)


# In[16]:


# Red

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

PIXEL_TO_UM = 4

def calculate_ring_average_grayscale(image, inner_radius, outer_radius):
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    mask_inner = (x - center_x) ** 2 + (y - center_y) ** 2 >= inner_radius ** 2
    mask_outer = (x - center_x) ** 2 + (y - center_y) ** 2 < outer_radius ** 2

    ring_mask = mask_inner & mask_outer
    ring_pixels = image[ring_mask]
    
    if len(ring_pixels) == 0:
        return 0  
    return np.mean(ring_pixels)

def plot_grayscale_profile(image_path, min_radius, max_radius, step):

    try:
        image = Image.open(image_path)
        grayscale_image = ImageOps.grayscale(image)
        image_np = np.array(grayscale_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    radii = []
    average_grayscales = []
    
    for radius in range(min_radius, max_radius + 1, step):
        avg_gray = calculate_ring_average_grayscale(image_np, radius - step, radius)
        average_radius_um = (radius - step / 2) * PIXEL_TO_UM
        radii.append(average_radius_um)
        average_grayscales.append(avg_gray)

        if radius % 50 == 0:
            print(f"Processed radius: {radius} pixels, Average grayscale: {avg_gray}")
    

    plt.figure(figsize=(10, 6))
    plt.plot(radii, average_grayscales, label='Average Grayscale', color='r')
    plt.xlabel('Radial Distance r (µm)')
    plt.ylabel('Average Grayscale')
    plt.title('Variation of Average Grayscale with Radial Distance r (Red Food Color)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Variation of Average Grayscale with Radial Distance r (Red Food Color)')
    plt.show()

plot_grayscale_profile('redDoubleP.JPG', 12, 450, 1)


# In[14]:


import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

PIXEL_TO_UM_RED = 4
PIXEL_TO_UM_BLUE = 2

def calculate_ring_average_grayscale(image, inner_radius, outer_radius):
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    mask_inner = (x - center_x) ** 2 + (y - center_x) ** 2 >= inner_radius ** 2
    mask_outer = (x - center_x) ** 2 + (y - center_x) ** 2 < outer_radius ** 2

    ring_mask = mask_inner & mask_outer
    ring_pixels = image[ring_mask]
    
    if len(ring_pixels) == 0:
        return 0  
    return np.mean(ring_pixels)

def plot_combined_grayscale_profile(image_path_red, min_radius_red, max_radius_red, step_red, image_path_blue, min_radius_blue, max_radius_blue, step_blue):

    try:
        image_red = Image.open(image_path_red)
        grayscale_image_red = ImageOps.grayscale(image_red)
        image_np_red = np.array(grayscale_image_red)
        
        image_blue = Image.open(image_path_blue)
        grayscale_image_blue = ImageOps.grayscale(image_blue)
        image_np_blue = np.array(grayscale_image_blue)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    radii_red = []
    average_grayscales_red = []
    radii_blue = []
    average_grayscales_blue = []

    for radius in range(min_radius_red, max_radius_red + 1, step_red):
        avg_gray_red = calculate_ring_average_grayscale(image_np_red, radius - step_red, radius)
        average_radius_um_red = (radius - step_red / 2) * PIXEL_TO_UM_RED
        radii_red.append(average_radius_um_red)
        average_grayscales_red.append(avg_gray_red)

        if radius % 50 == 0:
            print(f"Processed radius (Red): {radius} pixels, Average grayscale: {avg_gray_red}")

    for radius in range(min_radius_blue, max_radius_blue + 1, step_blue):
        avg_gray_blue = calculate_ring_average_grayscale(image_np_blue, radius - step_blue, radius)
        average_radius_um_blue = (radius - step_blue / 2) * PIXEL_TO_UM_BLUE
        radii_blue.append(average_radius_um_blue)
        average_grayscales_blue.append(avg_gray_blue)

        if radius % 50 == 0:
            print(f"Processed radius (Blue): {radius} pixels, Average grayscale: {avg_gray_blue}")

    min_length = min(len(radii_red), len(radii_blue))
    combined_radii = radii_red[:min_length]
    combined_avg_grayscales = [(average_grayscales_red[i] + average_grayscales_blue[i]) / 2 for i in range(min_length)]

    plt.figure(figsize=(10, 6))
    plt.plot(combined_radii, combined_avg_grayscales, label='Combined Average Grayscale', color='purple')
    plt.xlabel('Radial Distance r (µm)')
    plt.ylabel('Average Grayscale')
    plt.title('Combined Average Grayscale with Radial Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig('Combined Average Grayscale with Radial Distance.png')
    plt.show()

plot_combined_grayscale_profile('redDoubleP.JPG', 12, 450, 1, 'blueDoubeP.JPG', 25, 900, 1)


# In[ ]:




