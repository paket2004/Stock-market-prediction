#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
VERSION=""


take_data_sample() {
    echo "Taking data sample..."
    VERSION=$(python3 "${ROOT_DIR}/src/data.py")
    echo "Data sample verion $version"
}


validate_data_sample() {
    echo "Validating data sample..."
    python3 "${ROOT_DIR}/tests/validate_initial_data.py"

    return $?
}


version_data_sample() {
    echo "Versioning data sample..."
    git add "${ROOT_DIR}/data_sample.csv"
    git commit -m "Versioning data sample v$VERSION"
    git tag "v$VERSION"
    git push origin --tags
}




take_data_sample


if validate_data_sample; then

    version_data_sample
    echo "Data sample validated and versioned."
else
    echo "Data sample is invalid. Skipping versioning."
fi

