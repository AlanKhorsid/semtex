from classes2 import TableCollection
from util2 import get_csv_rows, open_tables, pickle_load, pickle_save, progress
from pathlib import Path

ROOTPATH = Path(__file__).parent.parent

# # tables = open_tables("test", year="2023", spellcheck="bing")
# # pickle_save(tables, "open_tables_test_2023_bing")
# tables: TableCollection = pickle_load("open_tables_test_2023_bing", is_dump=True)

# tables.fetch_candidates()
# tables.fetch_info()
# tables.fetch_statement_entities()

# # pickle_save(tables, "tables_validation_2023_bing")
tables: TableCollection = pickle_load("tables_test_2023_progress_1840", is_dump=True)

cea_results = []
num_correct_annotations = 0
num_submitted_annotations = 0
num_ground_truth_annotations = 0

# cta_rows = get_csv_rows(f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/gt/cta_gt.csv")
# x = []
cea_predictions = []
cpa_predictions = []
cta_predictions = []

with progress:
    for i, (file, table) in enumerate(progress.track(tables.tables.items(), description="DOGBOOSTING")):
        table_cea_predictions, table_cpa_predictions, table_cta_predictions = table.dogboost()
        cea_predictions.extend([[file] + pred for pred in table_cea_predictions])
        cpa_predictions.extend([[file] + pred for pred in table_cpa_predictions])
        cta_predictions.extend([[file] + pred for pred in table_cta_predictions])

        if i % 100 == 0 and i > 1840:
            pickle_save(tables, f"tables_test_2023_progress_{i}")


# write cea_predictions to file
with open("cea_results.csv", "w") as f:
    f.write('"Table ID","Row ID","Column ID","Entity IRI"\n')
    for table, row, col, entity in cea_predictions:
        f.write(f'"{table}","{row}","{col}","{entity}"\n')

# writeo cpa_predictions to file
with open("cpa_results.csv", "w") as f:
    f.write('"Table ID","Column ID 1","Column ID 2","Property IRI"\n')
    for table, col_1, col_2, prop in cpa_predictions:
        f.write(f'"{table}","{col_1}","{col_2}","{prop}"\n')

# write cta_predictions to file
with open("cta_results.csv", "w") as f:
    f.write('"Table ID","Column ID","Annotation IRI"\n')
    for table, col, entity in cta_predictions:
        f.write(f'"{table}","{col}","{entity}"\n')
