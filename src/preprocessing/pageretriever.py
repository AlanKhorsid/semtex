import csv
import os


# specify the path to the directory containing the csv files
csv_dir = "/Users/alankhorsid/Documents/semtex/datasets/WikidataTables2023R1/DataSets/Test/tables"

# specify the path to the cea_gt file
cea_gt_file = "/Users/alankhorsid/Documents/semtex/datasets/WikidataTables2023R1/DataSets/Test/targets/cea_targets.csv"

# read the cea_gt file and store the information in a dictionary
cea_gt_info = {}
with open(cea_gt_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        filename = row[0]
        i = int(row[1])
        j = int(row[2])
        cea_gt_info[(filename, i, j)] = None

# loop through the keys in the cea_gt_info dictionary
for key in cea_gt_info.keys():
    filename = key[0]
    i = key[1]
    j = key[2]

    # construct the path to the csv file
    csv_path = os.path.join(csv_dir, f"{filename}.csv")

    # open the csv file and retrieve the cell at (i,j)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        cell = rows[i][j]

    # collect all cells in one neline separated text file
    with open("all-cells-test-2023.txt", "a") as f:
        f.write(cell + "\n")
