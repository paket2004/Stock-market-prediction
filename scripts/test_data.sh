#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

take_data_sample() {
    echo "Taking data sample..."
    python3 "${ROOT_DIR}/src/data.py"
    cp /path/to/original_data.csv ./data_sample.csv
}


validate_data_sample() {
    echo "Validating data sample..."
    # Replace this with your actual validation logic
    # Example: python validate_data.py ./data_sample.csv
    # This example assumes a Python script `validate_data.py` that validates the data
    python3 "${ROOT_DIR}/tests/validate_initial_data.py"

    return $?
}

# Function to version the data sample

version_data_sample() {
    echo "Versioning data sample..."
    git add "${ROOT_DIR}/data_sample.csv" && git commit -m "Versioning data sample"
}




take_data_sample


if validate_data_sample; then

    version_data_sample
    echo "Data sample validated and versioned."
else
    echo "Data sample is invalid. Skipping versioning."
fi

