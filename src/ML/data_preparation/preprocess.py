#---IMPORTS----------+
import os
import numpy as np
import cv2
from PIL import Image

#---FUNCTIONS--------+

def average_imgs(folder_path, folder_dest, show=False):
    """
    Takes in a path to a folder, and averages the pixelvalues of the images
    in the folder. The output image is saved and returned.

    This functions requires all images to be the same size

    @params:
        folder_path: <string> Full path to the folder where the images are
        show: <bool> Visualize image or note
        folder_dest <string> Path to the folder where the files should be saved
    @returns:
        None
    """
    # Read all files
    files = os.listdir(folder_path)
    imgs = [folder_path + file for file in files if file[-4:] == ".png"]

    # Group all the images based on first index, and image type (last 7 characters)
    img_grouping = {}
    for i in range(len(imgs)):
        temp_img = imgs[i].split("/")[-1]
        temp_key = (temp_img[0], temp_img[-7:])

        if temp_key in img_grouping:
            img_grouping[temp_key].append(imgs[i])
        else:
            img_grouping[temp_key] = [imgs[i]]

    # Convert into list
    img_grouping = [v for k, v in img_grouping.items()]
    
    # Averaging all the images, and saving them
    for i in range(len(img_grouping)):
        imgs = [np.array(Image.open(img)) for img in img_grouping[i]]
        arrs = [np.array(img) for img in imgs]
        avg_arr = np.mean(arrs, axis=0).astype(float)
        
        avg_img = Image.fromarray(avg_arr)
        avg_img = avg_img.convert("L") # Convert to grayscale
        # For RGD: ...convert("RGB")
        
        new_file_name = img_grouping[i][0].split("/")[-1]
        new_file_name = new_file_name[0:2] + new_file_name[4:]

        avg_img.save(folder_dest + new_file_name)
        
        if show:
            avg_img.show()


def clear_img_directory(folder_path):
    """
    Removes all images in a directory

    @arguments:
        folder_path: <string> Directory that should be cleared
    @returns:
        None
    """
    file_names = os.listdir(folder_path)
    imgs = [folder_path + file for file in file_names if file[-4:] == ".png"]
    for file in imgs:
        os.remove(file)


def combine_imgs(folder_path, folder_dest="", remove=True):
    """
    Combine multiple images into one bigger image.

    @arguments:
        folder_path: <string> Path to where the images are stored 
        folder_dest: <string> To which folder the combined images should be saved
        remove: <bool> Whether to remove the old images or not
    @returns:
        None
    """
    if folder_dest == "":
        folder_dest = folder_path
    
    # Read all files
    file_names = os.listdir(folder_path)
    imgs = [folder_path + file for file in file_names if file[-4:] == ".png"]

    # Group all images based on the first index
    img_grouping = {}
    for i in range(len(imgs)):
        temp_img = imgs[i].split("/")[-1]
        temp_key = temp_img[0]

        if temp_key in img_grouping:
            img_grouping[temp_key].append(imgs[i])
        else:
            img_grouping[temp_key] = [imgs[i]]

    # Convert into list
    img_grouping = [v for k, v in img_grouping.items()]
    img_grouping = sorted(img_grouping, key=lambda x: int(x[0].split("/")[-1].split('_')[0]))

    # Combine images and save them to folder_dest
    for i in range(len(img_grouping)):
        files = [f for f in img_grouping[i] if f.split("/")[-1].startswith(str(i)) and f.endswith(("ETA.png", "MET.png", "PT.png"))]
        files = sorted(files)
        images = [Image.open(f) for f in files]
         
        combined_image = np.hstack(images)
        combined_image =Image.fromarray(combined_image)
        combined_image.save(os.path.join(folder_dest, f"{i}_combined.png"))

    # Remove the old images
    if remove:
        for rm_file in imgs:
            os.remove(rm_file)


def lower_res(folder_path, folder_dest=""):
    """
    Lowers the resolution with 30% for all images in a folder
    @arguments:
        folder_path: <string> Path to where the images are stored
        folder_dest: <string> To which folder the combined images should be saved
    @returns:
        None
    """
    if folder_dest == "":
        folder_dest = folder_path

     # Read all files
    file_names = os.listdir(folder_path)
    imgs = [folder_path + file for file in file_names if file[-4:] == ".png"]

    for img in imgs:
        image = Image.open(img)

        current_width, current_height = image.size
        
        new_width = int(current_width * 0.4)
        new_height = int(current_height * 0.4)

        resized_image = image.resize((new_width, new_height), Image.BICUBIC)
        resized_image.save(img)
        image.close()
        

