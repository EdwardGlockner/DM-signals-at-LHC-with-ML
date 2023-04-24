#---IMPORTS----------+
import time
import os
import sys
import numpy as np
from PIL import Image


#---FIXING PATH------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/DataGeneration/data_preparation", "")


#---GLOBALS----------+
try:
    if sys.platform == "darwin": # macOS
        clear = lambda : os.system("clear")

    elif sys.platform == "win32" or sys.platform == "win64": # windows
        clear = lambda : os.system("cls")

    elif sys.platform == "linux" or sys.platform == "linux2": # linux
        clear = lambda : os.system("clear")

except OSError as e:
    print("Error identifying operating systems")

bar = "+-----------------"


#---FUNCTIONS--------+
def average_img(folder_path, show=False, save_file="Average.png"):
    """
    asdfasdf
    Requires all images to be the same size
    @params:
        folder_path: <string> Full path to the folder where the images are
        show: <bool> Visualize image or note
        save_file <string> Name of the saved file

    @returns:
        avg_img: <> asdf 
    """
    # Read all files
    files = os.listdir(folder_path)
    imgs = [folder_path + file for file in files if file[-8:] in ["ETA1.png", "ETA1.PNG", "ETA1.jpg", "ETA1.JPG"]]
    imgs = [np.array(Image.open(img)) for img in imgs]
    arrs = [np.array(img) for img in imgs]
    avg_arr = np.mean(arrs, axis=0).astype(float)
    
    avg_img = Image.fromarray(avg_arr)
    avg_img = avg_img.convert("L") # Convert to grayscale
    # For RGD: ...convert("RGB")
    avg_img.save(save_file)
    
    if show:
        avg_img.show()
    
    return avg_img 


#---MAIN----------+
def main():
    folder_path = dirname + "Data_Analysis/All_png/"
    start = time.time()
    average_img(folder_path, show=False)
    end = time.time()
    print(f"Time: {end-start}")


#---RUN CODE------+
if __name__ == "__main__":
    main()

