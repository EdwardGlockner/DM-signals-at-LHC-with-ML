import os
import csv
import random


def merge_csv_files(file1, file2, output_file_1, output_file_2, output_file_3):
    # Read data from the first file
    with open(file1, 'r') as csv_file1:
        csv_reader1 = csv.reader(csv_file1)
        data1 = list(csv_reader1)

    # Read data from the second file
    with open(file2, 'r') as csv_file2:
        csv_reader2 = csv.reader(csv_file2)
        data2 = list(csv_reader2)

    # Concatenate the rows
    merged_data = data1 + data2
    # Shuffle the merged data
    random.shuffle(merged_data)

    # Calculate the number of rows for each output file
    total_rows = len(merged_data)
    rows_1 = int(total_rows * 0.64)  # 64% of the rows
    rows_2 = int(total_rows * 0.16)  # 16% of the rows
    rows_3 = total_rows - rows_1 - rows_2  # Remaining 20% of the rows

    # Split the shuffled data into three subsets
    subset_1 = merged_data[:rows_1]
    subset_2 = merged_data[rows_1:rows_1+rows_2]
    subset_3 = merged_data[rows_1+rows_2:]

    # Write the subsets to separate files
    with open(output_file_1, 'w', newline='') as file_1:
        csv_writer = csv.writer(file_1)
        csv_writer.writerows(subset_1)

    with open(output_file_2, 'w', newline='') as file_2:
        csv_writer = csv.writer(file_2)
        csv_writer.writerows(subset_2)

    with open(output_file_3, 'w', newline='') as file_3:
        csv_writer = csv.writer(file_3)
        csv_writer.writerows(subset_3)

    print("CSV files merged, shuffled, and split successfully!")


def split_csv(file_path, output_file_1, output_file_2, output_file_3):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    first_elements = [row[0] for row in data]
    second_elements = [row[1] for row in data]
    other_elements = [row[2:] for row in data]

    with open(output_file_1, 'w', newline='') as first_file:
        csv_writer = csv.writer(first_file)
        csv_writer.writerows([[element] for element in first_elements])

    with open(output_file_2, 'w', newline='') as second_file:
        csv_writer = csv.writer(second_file)
        csv_writer.writerows([[element] for element in second_elements])

    with open(output_file_3, 'w', newline='') as other_file:
        csv_writer = csv.writer(other_file)
        csv_writer.writerows(other_elements)

    print("CSV files created successfully!")

def main():

    dirname = os.getcwd()

    sneutrino_monoz = dirname + "/raw_data/sneutrino_monoz/data_Sneutrino_monoz.csv"
    neutralino_monoz = dirname + "/raw_data/neutralino_monoz/data_Neutralino_monoz.csv"
    sneutrino_jet = dirname + "/raw_data/sneutrino_jet/data_Sneutrino_jet.csv"
    neutralino_jet = dirname + "/raw_data/neutralino_jet/data_Neutralino_jet.csv"

    merge_csv_files(sneutrino_monoz, neutralino_monoz, "64monoz.csv", "16monoz.csv", "20monoz.csv")
    merge_csv_files(sneutrino_jet, neutralino_jet, "64jet.csv", "16jet.csv", "20jet.csv")

    split_csv("64monoz.csv", "monoz_mass_training", "monoz_model_training", "monoz_input_training")
    split_csv("16monoz.csv", "monoz_mass_validation", "monoz_model_validation", "monoz_input_validation")
    split_csv("20monoz.csv", "monoz_mass_testing", "monoz_model_testing", "monoz_input_testing")

    split_csv("64jet.csv", "jet_mass_training", "jet_model_training", "jet_input_training")
    split_csv("16jet.csv", "jet_mass_validation", "jet_model_validation", "jet_input_validation")
    split_csv("20jet.csv", "jet_mass_testing", "jet_model_testing", "jet_input_testing")


if __name__ == "__main__":
    main()
