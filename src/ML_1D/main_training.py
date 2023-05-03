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
#from classification_bCNN import classification_bCNN
#from regression_CNN import regression_CNN 
from data_prep_1D import get_models, read_csvs_to_data_set      
#from preprocess_1D import    

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
def preprocess():
    pass


def get_sets(folder_csv_path):
    model_names = get_models(folder_csv_path)
    read_csvs_to_data_set(folder_csv_path, model_names)

def train_classification():
    pass

def train_regression():
    pass

#---MAIN-----------------+
def main():
    clear()
    folder_csv_path = dirname + "src/data_preparation/data_prep_1D"
    get_sets(folder_csv_path) 

#---RUN CODE-------------+
if __name__ == "__main__":
    main()

