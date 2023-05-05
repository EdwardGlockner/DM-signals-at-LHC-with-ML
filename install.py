import os
import sys
import getopt


def usage():
    print("\nUsage:")
    print("install.py -v python_version | -h help\n")
    print("-v \t Options: Any python version installed in the system.")
    print("Specifies if the program should be run in training mode, or training and validation mode.")
    print("Python version is by default set to the python path\n")
    print("-h \t Help: Get information on how to run the file\n")
    
def arg_parse(argv):
    """

    """
    python_version = ""
    try:
        opts, args = getopt.getopt(argv, "v:", ["python_version"])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        if opt == "-v":
            python_version = arg

    return python_version 


def main(python_version):
    if not python_version:
        os.system("pip install -r requirements.txt")
    else:
        os.system(python_version + " -m pip install -r requirements.txt")


if __name__ == "__main__":
    python_version = arg_parse(sys.argv[1:])
    main(python_version)

