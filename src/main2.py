from classes2 import TableCollection
from util2 import get_csv_rows, open_tables, pickle_load, pickle_save, progress
from pathlib import Path

ROOTPATH = Path(__file__).parent.parent

tables = open_tables("validation", year="2023", spellcheck="bing")
# tables: TableCollection = pickle_load("open_tables_validation_2023_bing", is_dump=True)

tables.fetch_candidates()
tables.fetch_info()
tables.fetch_statement_entities()
x = 1

# # pickle_save(tables, "tables_validation_bing")
# tables: TableCollection = pickle_load("tables_test_bing", is_dump=True)

# # tables.limit_to(10)

# cea_results = []
# num_correct_annotations = 0
# num_submitted_annotations = 0
# num_ground_truth_annotations = 0

# cta_rows = get_csv_rows(f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/gt/cta_gt.csv")
# x = []

# with progress:
#     for file, table in progress.track(tables.tables.items(), description="DOGBOOSTING"):
#         cea_res, correct, submitted, ground_truth = table.dogboost()

#         # col_ids = set([row[1] for row in cea_res])
#         # for col_id in col_ids:
#         #     try:
#         #         cta_row = next((row for row in cta_rows if row[0] == file and row[1] == f"{col_id}"), None)
#         #         candidates = [row[2] for row in cea_res if row[1] == col_id]
#         #         x.append(
#         #             {
#         #                 "file": file,
#         #                 "col_id": col_id,
#         #                 "candidates": candidates,
#         #                 "cta": int(cta_row[2].split("/")[-1][1:]),
#         #             }
#         #         )
#         #     except:
#         #         pass

#         cea_res = [(file, f"{row[0]}", f"{row[1]}", f"http://www.wikidata.org/entity/Q{row[2].id}") for row in cea_res]
#         cea_results.extend(cea_res)

#         num_correct_annotations += correct
#         num_submitted_annotations += submitted
#         num_ground_truth_annotations += ground_truth

# precision = num_correct_annotations / num_submitted_annotations if num_submitted_annotations > 0 else 0
# recall = num_correct_annotations / num_ground_truth_annotations if num_ground_truth_annotations > 0 else 0
# f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# print("Precision: ", precision)
# print("Recall: ", recall)
# print("F1: ", f1)

from cta import cta_no_query

x = pickle_load("cea_results_test")

with open("cta_results.csv", "w") as f:
    f.write('"Table ID","Column ID","Annotation IRI"\n')
    for row in x:
        cta_guess = cta_no_query(row["candidates"])
        if cta_guess is None:
            continue

        f.write(f'"{row["file"]}","{row["col_id"]}","http://www.wikidata.org/entity/Q{cta_guess}"\n')

# # writeo cea_results to file
# with open("cea_results.csv", "w") as f:
#     f.write('"Table ID","Row ID","Column ID","Entity IRI"\n')
#     for row in cea_results:
#         f.write(f'"{row[0]}","{row[1]}","{row[2]}","{row[3]}"\n')
