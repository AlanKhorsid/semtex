from util2 import open_tables, pickle_load, pickle_save

validation_tables = open_tables("test", spellcheck="bing")

# validation_tables.limit_to(5)

x = 1

# validation_tables.fetch_candidates()
# validation_tables.fetch_info()
# validation_tables.fetch_statement_entities()

# num_correct_annotations = 0
# num_submitted_annotations = 0
# num_ground_truth_annotations = 0

# for file, table in validation_tables.tables.items():
#     correct, submitted, ground_truth = table.dogboost()
#     num_correct_annotations += correct
#     num_submitted_annotations += submitted
#     num_ground_truth_annotations += ground_truth

# precision = num_correct_annotations / num_submitted_annotations if num_submitted_annotations > 0 else 0
# recall = num_correct_annotations / num_ground_truth_annotations if num_ground_truth_annotations > 0 else 0
# f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# x = 1
