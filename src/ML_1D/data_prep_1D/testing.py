

from data_prep_1D import *
import os
import sys
import numpy as np

dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D/data_prep_1D","")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



def main():
    sneutrino_jet_path = dirname + "/Storage_data/MSSM_sneutrino_jet/norm_amp_array/raw_data_all.csv"
    sneutrino_jet_path2 = dirname + "/Storage_data/MSSM_sneutrino_jet/data_all.csv"
    neutralino_jet_path = dirname + "/Storage_data/MSSM_neutralino_jet/norm_amp_array/raw_data_all.csv"
    neutralino_jet_path2 = dirname + "/Storage_data/MSSM_neutralino_jet/data_all.csv"
    sneutrino_z_path = dirname + "/Storage_data/MSSM_neutralino_z/norm_amp_array/raw_data_all.csv"
    neutralino_z_path = dirname + "/Storage_data/MSSM_neutralino_z/norm_amp_array/raw_data_all.csv"
        
    np.set_printoptions(threshold=np.inf)
    bar = "**************************************************"
    input_data, masses, models = create_sets_from_csv(sneutrino_jet_path, neutralino_jet_path)
    """
    print(len(input_data))
    print(len(masses))
    print(len(models))
    print(bar)
    print(masses)
    print(bar)
    print(models)
    print(bar)
    
    ###########################################
    print(input_data.shape)
    print(input_data[0].shape)
    print(input_data[0][0].shape)
    print(type(input_data[0][0][0]))
    print(input_data[0,:,2], input_data[0,:,2].shape)
    """
    ###########################################
    cl, re = shuffle_and_create_sets(input_data, models, masses, print_shapes=True)



if __name__ == "__main__":
    main()
