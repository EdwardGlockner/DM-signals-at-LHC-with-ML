#---IMPORTS--------------+
import sys
import os

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML","")

#---LOCAL IMPORTS--------+


#---GLOBALS--------------+
try:
    if sys.platform in ["darwin", "linux", "linux2"]: #macOS
        clear = lambda : os.system("clear")

    elif sys.platform in ["win32", "win64"]: #windows
        clear = lambda : os.system("cls")
    
    else:
        clear = ""

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"

#---FUNCTIONS------------+

#---MAIN-----------------+
def main():
    # Create all the paths to the directories
    folder_img_path = dirname + "src/ML/raw_data/images/"
    folder_dest_path = dirname + "src/ML/processed_data/images/"
    folder_target_path = dirname + "src/ML/raw_data/target_values/data_MSSM.csv" 

    # Preprocessing and creating datasets
    preprocess(folder_img_path, folder_dest_path) 

    re, image_shape = get_sets(folder_dest_path, folder_target_path) 
    #eventually: re, cl, image_shape = get_sets(....
    # Train the classification model

    # Train the regression model
    train_regression(re, image_shape) 

    """
    cl, re =  get_sets(folder_img_path, folder_target_path, img_height, img_width)
    X_val_cl, y_val_cl = cl[4], cl[5]
    X_val_re, y_val_re = re[4], re[5]

    train_classification(cl[0], cl[1], cl[2], cl[3])
    train_regression(re[0], re[1], re[2], re[3])
    """


#---RUN CODE-------------+
if __name__ == "__main__":
    main()

