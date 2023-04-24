#---IMPORTS--------------+
import sys
import os

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML","")

#---LOCAL IMPORTS--------+
from classification_bCNN import *
from regression_CNN import *
from data_prep import *

#---GLOBALS--------------+
try:
    if sys.platform in ["darwin", "linux", "linux2"]: #macOS
        clear = lambda : os.system("clear")

    elif sys.platform == "win32" or sys.platform == "win64": #windows
        clear = lambda : os.system("cls")
    
    else:
        clear = ""

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"


#---FUNCTIONS------------+
def preprocess(folder_img_path):
    """
    asdfasdf

    @arguments:
        None
    @returns:
        None
    """
    lower_res_script = ""
    combine_script = ""
    lower_res(lower_res_script, folder_img_path)
    combine_imgs(combine_script, folder_img_path)

    average_imgs(folder_img_path)

def get_sets(folder_img_path, folder_target_path, img_height, img_width):
    """
    asdfasdf

    @arguments:
        None
    @returns:
        None
    """
    img_arr = read_images(folder_img_path, img_height, img_width)
    target_vals = regression_create_targets(folder_target_path) 
    class_labels = classification_create_labels(folder_img_path)

    
    if len(img_arr) != len(target_vals):
        print("Error in function <get_sets>. Arrays img_arr and target_vals are not the same size.")
        return None

    if len(img_arr) != len(class_labels):
        print("Erorr in function <get_sets>. Arrays img_arr and class_labels are not the same size.")

    X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl = \
            shuffle_and_create_sets(img_arr, class_labels)

    X_train_re, y_train_re, X_test_re, y_test_re, X_val_re, y_val_re = \
            shuffle_and_create_sets(img_arr, target_vals)
    
    return [X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl], \
            [X_train_re, y_train_re, X_test_re, y_test_re, X_val_re, y_val_re]


def train_classification():
    """
    asdfasdf

    @arguments:
        None
    @returns:
        None
    """
    pass


def train_regression():
    """
    asdfasdf

    @arguments:
        None
    @returns:
        None
    """
    pass


def run_all():
    """
    asdfasdf

    @arguments:
        None
    @returns:
        None
    """

#---MAIN-----------------+
def main():
    folder_img_path = ""
    folder_target_path = ""
    img_height = ""
    img_width = ""
    
    preprocess(folder_img_path) 
    cl, re =  get_sets(folder_img_path, folder_target_path, img_height, img_width)


#---RUN CODE-------------+
if __name__ == "__main__":
    main()

