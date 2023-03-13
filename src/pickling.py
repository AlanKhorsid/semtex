import csv
import os

from util import pickle_save, pickle_save_in_folder

# specify the path to the cea_target.csv file
cea_target_file = "/Users/alankhorsid/Documents/semtex/datasets/HardTablesR1/DataSets/HardTablesR1/Test/target/cea_target.csv"

# specify the path to the target folder
tables_folder = "/Users/alankhorsid/Documents/semtex/datasets/HardTablesR1/DataSets/HardTablesR1/Test/tables"


# read the cea_target file and store the information in a dictionary
cea_target_info = {}
with open(cea_target_file, "r") as f:
    # The cea_target file has this format: filename, i, j
    reader = csv.reader(f)
    for row in reader:
        filename = row[0]
        i = int(row[1])
        j = int(row[2])
        cea_target_info[(filename, i, j)] = 1

# loop through the keys in the cea_target_info dictionary
counter = 0
all_cells = []
for key in cea_target_info.keys():
    filename = key[0]
    i = key[1]
    j = key[2]

    # construct the path to the csv file
    csv_path = os.path.join(tables_folder, f"{filename}.csv")

    # open the csv file and retrieve the cell at (i,j)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        cell = rows[i][j]

        # pickle_save_in_folder(cell, "all_test_cells_folder")

        all_cells.append(cell)
        # print the cell
        # print(cell)

        # increment the counter
        counter += 1

print(f"Processed {counter} cells")
print(f"Number of unique cells: {len(all_cells)}")
# pickle_save(all_cells)
