import os
from great_expectations.data_context import FileDataContext
import great_expectations as ge
import pandas as pd

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


context_path = os.path.abspath(f"{project_root_dir}/services/gx") ## ????----------------
os.makedirs(context_path, exist_ok=True)

context = FileDataContext.create(project_root_dir=context_path)



suite_name = "initial_validation"
# create the suite
try:
    suite = context.add_expectation_suite(suite_name)
except:
    # If suite already exists, get the existing suite
    suite = context.get_expectation_suite(suite_name)

data = pd.read_csv(f'{project_root_dir}/data/samples/sample.csv')
df_ge = ge.from_pandas(data)


df_ge.expect_column_values_to_not_be_null("Open")
df_ge.expect_column_mean_to_be_between("High", min_value=10, max_value=300)
df_ge.expect_column_values_to_match_strftime_format("Date", "YYYY-MM-DD")
for column in df_ge.columns:
    if 'News' in column.lower():
        df_ge.expect_column_min_to_be_between(column, min_value=0, max_value=None)

context.save_expectation_suite(suite, suite_name)
