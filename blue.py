import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PIXEL_TO_UM_COLOR_WATER = 2.5
PIXEL_TO_UM_NO_WATER = 2.5

def calculate_ring_average_blue(image, inner_radius, outer_radius):
    height, width = image.shape[:2]
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    mask_inner = (x - center_x) ** 2 + (y - center_x) ** 2 >= inner_radius ** 2
    mask_outer = (x - center_x) ** 2 + (y - center_x) ** 2 < outer_radius ** 2

    ring_mask = mask_inner & mask_outer
    ring_pixels = image[ring_mask]
    
    if len(ring_pixels) == 0:
        return 0  
    return np.mean(ring_pixels[:, 2])

def plot_combined_blue_profile(image_path_color_water, min_radius_color_water, max_radius_color_water, step_color_water, image_path_no_water, min_radius_no_water, max_radius_no_water, step_no_water):

    try:
        image_color_water = Image.open(image_path_color_water)
        image_np_color_water = np.array(image_color_water)
        
        image_no_water = Image.open(image_path_no_water)
        image_np_no_water = np.array(image_no_water)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    radii_color_water = []
    average_blues_color_water = []
    radii_no_water = []
    average_blues_no_water = []

    for radius in range(min_radius_color_water, max_radius_color_water + 1, step_color_water):
        avg_blue_color_water = calculate_ring_average_blue(image_np_color_water, radius - step_color_water, radius)
        average_radius_um_color_water = (radius - step_color_water / 2) * PIXEL_TO_UM_COLOR_WATER
        radii_color_water.append(average_radius_um_color_water)
        average_blues_color_water.append(avg_blue_color_water)

        if radius % 50 == 0:
            print(f"Processed radius (Color Water): {radius} pixels, Average blue: {avg_blue_color_water}")

    for radius in range(min_radius_no_water, max_radius_no_water + 1, step_no_water):
        avg_blue_no_water = calculate_ring_average_blue(image_np_no_water, radius - step_no_water, radius)
        average_radius_um_no_water = (radius - step_no_water / 2) * PIXEL_TO_UM_NO_WATER
        radii_no_water.append(average_radius_um_no_water)
        average_blues_no_water.append(avg_blue_no_water)

        if radius % 50 == 0:
            print(f"Processed radius (No Water): {radius} pixels, Average blue: {avg_blue_no_water}")

    min_length = min(len(radii_color_water), len(radii_no_water))
    combined_radii = radii_color_water[:min_length]
    combined_avg_blues = [(average_blues_color_water[i] / average_blues_no_water[i]) for i in range(min_length)]

    plt.figure(figsize=(10, 6))
    plt.plot(combined_radii, combined_avg_blues, label='Color Water / No Water Average Blue', color='purple')
    plt.xlabel('Radial Distance r (µm)')
    plt.ylabel('Average Blue Intensity Ratio')
    plt.title('Color Water to No Water Average Blue Intensity Ratio with Radial Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig('Color Water to No Water Average Blue Intensity Ratio with Radial Distance.png')
    plt.show()

plot_combined_blue_profile('_DSC0002(2).JPG', 20, 900, 1, '_DSC0004(1).JPG', 20, 900, 1)






