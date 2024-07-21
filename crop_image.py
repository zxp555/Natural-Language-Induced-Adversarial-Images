from PIL import Image
import os
import glob

def crop_image_to_quarters(image_path):
    try:
        image_names = []
        # Open the image using Pillow
        image = Image.open(image_path)

        # Get the width and height of the image
        width, height = image.size

        # Calculate the dimensions for each quarter
        quarter_width = width // 2
        quarter_height = height // 2

        # Crop and save the top-left quarter
        top_left = image.crop((0, 0, quarter_width, quarter_height))
        top_left.save(f"{image_path}_1.png")
        image_names.append(f"{image_path}_1.png")

        # Crop and save the top-right quarter
        top_right = image.crop((quarter_width, 0, width, quarter_height))
        top_right.save(f"{image_path}_2.png")
        image_names.append(f"{image_path}_2.png")

        # Crop and save the bottom-left quarter
        bottom_left = image.crop((0, quarter_height, quarter_width, height))
        bottom_left.save(f"{image_path}_3.png")
        image_names.append(f"{image_path}_3.png")

        # Crop and save the bottom-right quarter
        bottom_right = image.crop((quarter_width, quarter_height, width, height))
        bottom_right.save(f"{image_path}_4.png")
        image_names.append(f"{image_path}_4.png")

        print("Image cropped into four quarters successfully.")
        
        return image_names
    except Exception as e:
        print(f"Error occurred while cropping the image: {e}")
        

# folder_path = 'natural_file/fail'

# image_files = glob.glob(os.path.join(folder_path, '*'))
# for i in image_files:
# crop_image_to_quarters("15_37_0.png")