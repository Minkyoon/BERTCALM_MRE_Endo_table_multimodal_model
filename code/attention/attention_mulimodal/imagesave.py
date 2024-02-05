import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Read the CSV file
data = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# Filter the dataframe to only include rows where the accession_number is 1
filtered_data = data[data['accession_number'] == 1106]

# Sort the filtered data by the file path (assuming the file paths are in ascending order)

# Create the output directory if it doesn't exist
output_dir = '/home/minkyoon/crohn/for_clam/attention/attention_mulimodal/image/wholeimage/1106'
if not os.path.exists(output_dir):
    os.makedirs(output_dir , exist_ok=True)

# Loop over the file paths in the sorted dataframe
for filepath in filtered_data['filepath']:
    # Load the image
    print(filepath)
    image = np.load(filepath)
    
    # If the images are not in the right range or orientation, 
    
    # Display the image
    plt.imshow(image, cmap='gray')
    
    # Remove axes and white margins
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # Save the image to the specified folder
    output_path = os.path.join(output_dir, os.path.basename(filepath).replace('.npy', '.png'))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    
    
    
from PIL import Image
import os

# Define the directory where the images are stored
image_dir = '/home/minkyoon/crohn/for_clam/attention/attention_mulimodal/image/wholeimage/937'

# List all the image files in the directory
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
# Load the images
images = [Image.open(os.path.join(image_dir, image_file)) for image_file in image_files]

# Get the total width and height of the concatenated image
total_width = sum(image.width for image in images)
max_height = max(image.height for image in images)

# Create a blank image with the total width and height
concatenated_image = Image.new('RGB', (total_width, max_height))

# Paste each image into the concatenated image
x_offset = 0
for image in images:
    concatenated_image.paste(image, (x_offset, 0))
    x_offset += image.width

# Display the concatenated image
plt.imshow(concatenated_image)
plt.axis('off')
plt.show()



from PIL import Image
import os
import math
import matplotlib.pyplot as plt

# Define the directory where the images are stored
image_dir = '/home/minkyoon/crohn/for_clam/attention/attention_mulimodal/image/attention/tp_y1'

# List all the image files in the directory
#image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')] )
# Load the images
images = [Image.open(os.path.join(image_dir, image_file)) for image_file in image_files]

images=images[0:15]
# Determine the number of rows and columns for the grid
num_images = len(images)
num_cols = int(math.sqrt(num_images))
num_rows = math.ceil(num_images / num_cols)

# Get the width and height of a single image
image_width = images[0].width
image_height = images[0].height

# Create a blank image with the total width and height of the grid
grid_width = image_width * num_cols
grid_height = image_height * num_rows
grid_image = Image.new('RGB', (grid_width, grid_height))

# Paste each image into the grid
for i, image in enumerate(images):
    row = i // num_cols
    col = i % num_cols
    x_offset = col * image_width
    y_offset = row * image_height
    grid_image.paste(image, (x_offset, y_offset))

# Display the grid image in a larger size
plt.figure(figsize=(15, 15))
plt.imshow(grid_image)
plt.axis('off')
plt.show()
