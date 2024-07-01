import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.data_context.types.resource_identifiers import ValidationResultIdentifier
import os
import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest


project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def validate_initial_data(df, context_path=f"{project_root_dir}/services/gx", suite_name='initial_validation'):
    print(context_path)
    context = ge.data_context.DataContext(context_path)

    # Save the DataFrame to a CSV file to use it with Great Expectations
    sample_file_path = os.path.join(context_path, "sample.csv")
    df.to_csv(sample_file_path, index=False)

    # Define a batch request
    batch_request = RuntimeBatchRequest(
        datasource_name="default_filesystem_datasource",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="sample_data",  # You can name it anything
        runtime_parameters={"path": sample_file_path},
        batch_identifiers={"default_identifier_name": "default_identifier"},
    )

    # Get the validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # Validate the data
    results = validator.validate()

    # Check if validation passed
    if not results["success"]:
        raise Exception("Data validation failed")

    return results

# Usage

df = pd.read_csv(os.path.join(project_root_dir, 'data', 'samples', 'sample.csv'))

try:
    results = validate_initial_data(df)
    print("Data validation passed")
except Exception as e:
    print(str(e))
