#!/bin/bash

cd "$(dirname "$0")/.."

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Installing requirements in the active virtual environment..."
else
    echo "No virtual environment detected. Please activate your virtual environment first."
    exit 1
fi

pip install -r requirements.txt

