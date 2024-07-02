import os
from great_expectations.data_context import FileDataContext
import great_expectations as ge
import pandas as pd

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root_dir, 'data', 'samples', 'sample.csv')

data = pd.read_csv(f'{project_root_dir}/data/samples/sample.csv')

context_path = os.path.abspath(f"{project_root_dir}/services") 
context = FileDataContext.create(project_root_dir=context_path)

data_source = context.sources.add_or_update_pandas(
                                        name='batch_ds'
                                        )
data_asset = data_source.add_csv_asset(
                                       name='batch_asset',
                                       filepath_or_buffer=data_path)

suite_name = "initial_validation"
try:
    suite = context.add_expectation_suite(suite_name)
except:
    suite = context.get_expectation_suite(suite_name)

batch_request = data_asset.build_batch_request()

validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)


validator.expect_column_values_to_not_be_null("Open")
validator.expect_column_mean_to_be_between("High", min_value=10, max_value=300)
validator.expect_column_values_to_match_strftime_format("Date", "%Y-%m-%d")
for column in data.columns:
    if 'News' in column:
        validator.expect_column_min_to_be_between(column, min_value=0, max_value=None)

validator.save_expectation_suite()
validation_result = validator.validate()


if not validation_result["success"]:
    raise Exception("Data validation failed")
else:
    print("Data validation passed")

