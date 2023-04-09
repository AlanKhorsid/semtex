from util2 import open_tables
from dateutil.parser import parse as parse_date
from datetime import date, timedelta

validation_tables = open_tables("validation", spellcheck="bing")

# validation_tables.limit_to(5)

validation_tables.fetch_candidates()
validation_tables.fetch_info()
validation_tables.fetch_statement_entities()

for file, table in validation_tables.tables.items():
    table.dogboost()
