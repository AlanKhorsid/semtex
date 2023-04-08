from util2 import open_tables

validation_tables = open_tables("validation")
test_tables = open_tables("test")

validation_tables.fetch_candidates()
test_tables.fetch_candidates()

validation_tables.fetch_info()
test_tables.fetch_info()
x = 1
