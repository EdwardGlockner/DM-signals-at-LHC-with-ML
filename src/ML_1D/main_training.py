#---IMPORTS--------------+
import sys
import os

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
sys.path.insert(1, os.path.join(dirname, "src/data_preparation/data_prep_1D"))
sys.path.insert(1, os.path.join(dirname, "src/ML_1D/ml_models"))

#---LOCAL IMPORTS--------+
from classification_bCNN import classification_bCNN
from regression_CNN import regression_CNN 
from data_prep_1D import create_sets_from_csv, shuffle_and_create_sets

#---GLOBALS--------------+
try:
    if sys.platform in ["darwin", "linux", "linux2"]: #macOS
        clear = lambda: os.system("clear")

    elif sys.platform in ["win32", "win64"]: #windows
        clear = lambda: os.system("cls")
    
    else:
        clear = lambda: None

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"


#---FUNCTIONS------------+
def preprocess():
    """

    """
    pass


def get_sets(folder_csv_path):
    """

    """
    X_data, targets, labels = create_sets_from_csv(folder_csv_path) 
    X_train, y_train_cl, y_train_re, X_test, y_test_cl,\
            y_test_re, X_val, y_val_cl, y_val_re =\
            shuffle_and_create_sets(X_data, labels, targets, print_shapes=False)

    input_shape = (X_train.shape[1], X_train.shape[2])

    return [X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re], input_shape

def train_classification(data_sets, input_shape):
    """

    """
    if len(data_sets) != 6:
        print(f"Error in function <train_classification>. Excpected 6 datasets, got {len(data_sets)}")

        
    X_train, y_train, X_test, y_test, X_val, y_val = data_sets 
    model = classification_bCNN(X_train, y_train, X_test, y_test, input_shape, "classification_bCNN") 
    model.compile()
    model.train()


def train_regression(data_sets, input_shape):
    """

    """
    if len(data_sets) != 6:
        print(f"Error in function <train_regression>. Excpected 6 datasts, got {len(data_sets)}")

    X_train, y_train, X_test, y_test, X_val, y_val = data_sets
    model = regression_CNN(X_train, y_train, X_test, y_test, input_shape, "regression_CNN")
    model.compile()
    model.train()


#---MAIN-----------------+
def main():
    clear()
    folder_csv_path = dirname + "src/data_preparation/data_prep_1D/raw_data_all.csv"
    data_sets, input_shape = get_sets(folder_csv_path) 
    X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re = data_sets
    cl_data_set = [X_train, y_train_cl, X_test, y_test_cl, X_val, y_val_cl]
    re_data_set = [X_train, y_train_re, X_test, y_test_re, X_val, y_val_re]

    train_classification(cl_data_set, input_shape)
   # train_regression(re_data_set, input_shape) 
#---RUN CODE-------------+
if __name__ == "__main__":
    main()

