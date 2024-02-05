from PIL import Image
import os
import re
from itertools import product

def create_row_of_four_images(folder_path):
    # Extracting serial numbers from the file names
    pattern = re.compile(r"serial(\d+)_")
    serials = set()
    for file in os.listdir(folder_path):
        match = pattern.search(file)
        if match:
            serials.add(match.group(1))

    # Create a row of four images for each serial number
    for serial in serials:
        images = []
        for file in os.listdir(folder_path):
            if f"serial{serial}_" in file:
                img_path = os.path.join(folder_path, file)
                images.append(Image.open(img_path))

        # Sort images by their score or index if needed
        # images.sort(key=lambda img: img.filename)

        # Ensure only the first four images are used
        images = images[:4]

        # Calculate total width and max height for the row
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)
        
        # Create a blank image for the row
        row_img = Image.new('RGB', (total_width, max_height))

        # Place images in the row
        current_x = 0
        for img in images:
            row_img.paste(img, (current_x, 0))
            current_x += img.size[0]

        # Save the row image
        row_img.save(os.path.join(folder_path, f"row_serial{serial}_row.png"))

    return f"Rows of four images created for serials: {', '.join(serials)}"

# Example usage, replace the folder path with the actual path where images are stored



# Example usage
folder_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/attention/attention_mulimodal/image/tp_down_mre"  # Replace with the actual folder path
create_row_of_four_images(folder_path)







from PIL import Image
import os
import re
from itertools import product
from PIL import Image, ImageDraw, ImageFont

def create_combined_image(folder_path):
    # Extracting serial numbers from the file names
    pattern = re.compile(r"serial(\d+)_")
    serials = set()
    for file in os.listdir(folder_path):
        match = pattern.search(file)
        if match:
            serials.add(match.group(1))

    # Initialize variables for the combined image
    combined_images = []
    max_width = 0
    total_height = 0

    # Create a row of four images for each serial number and record their dimensions
    for serial in serials:
        images = []
        for file in os.listdir(folder_path):
            if f"serial{serial}_" in file:
                img_path = os.path.join(folder_path, file)
                images.append(Image.open(img_path))

        # Ensure only the first four images are used
        images = images[:4]

        # Calculate total width and max height for the row
        row_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)

        # Update the maximum width and total height for the combined image
        max_width = max(max_width, row_width)
        total_height += max_height + 30  # Extra space for serial number text

        # Create a blank image for the row
        row_img = Image.new('RGB', (row_width, max_height + 30), "white")

        # Place images in the row
        current_x = 0
        for img in images:
            row_img.paste(img, (current_x, 30))
            current_x += img.size[0]

        # Add serial number text
        draw = ImageDraw.Draw(row_img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Serial {serial}", (0, 0, 0), font=font)

        combined_images.append(row_img)

    # Create a blank image for the combined image
    combined_img = Image.new('RGB', (max_width, total_height), "white")

    # Place each row in the combined image
    current_y = 0
    for img in combined_images:
        combined_img.paste(img, (0, current_y))
        current_y += img.size[1]

    # Save the combined image
    combined_img_path = os.path.join(folder_path, "combined_image.png")
    combined_img.save(combined_img_path)

    return f"Combined image saved at {combined_img_path}"

# Example usage, replace the folder path with the actual path where images are stored
folder_path = "/home/minkyoon/crohn/for_clam/attention/attention_mulimodal/image_endo/tn_down"
create_combined_image(folder_path)



