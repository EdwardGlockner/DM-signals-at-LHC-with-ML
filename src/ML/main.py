#---IMPORTS--------------+
import sys
import os


#---LOCAL IMPORTS--------+
from classification_bCNN import *
from regression_CNN import *
from classification_data_prep import *
from regression_data_prep import * 


#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML","")


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
def classification():
    """
    asdfasdf

    @arguments:
        None

    @returns:
        None
    """
    pass


def regression():
    """
    asdfasdf

    @arguments:
        None

    @returns:
        None
    """
    pass


#---MAIN-----------------+
def main():
    pass


#---RUN CODE-------------+
if __name__ == "__main__":
    main()

