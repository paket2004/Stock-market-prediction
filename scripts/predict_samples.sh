#!/bin/bash
echo "Start the script"

# root directory of the project guarantees predict_samples.sh works 
# independently of the folder it was called from
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# main config path
CONFIG_FILE="${ROOT_DIR}/configs/config.yaml"

PREDICT_FILE="${ROOT_DIR}/src/predict.py"

# read a yaml value
read_yaml() {
    python -c "
import yaml
import sys

config_file = sys.argv[1]
key = sys.argv[2]

with open(config_file, 'r') as f:
    data = yaml.safe_load(f)

value = data.get(key, '')" "$CONFIG_FILE" "$1"
}


# write in a yaml value
write_yaml() {
    python -c "
import yaml
import sys

config_file = sys.argv[1]
key = sys.argv[2]
value = int(sys.argv[3])

with open(config_file, 'r') as f:
    data = yaml.safe_load(f)

data[key] = value

with open(config_file, 'w') as f:
    yaml.safe_dump(data, f)
" "$CONFIG_FILE" "$1" "$2"
}
# read initial example_version
initial_version=$(read_yaml 'example_version')
echo 'Initial version: $initial_version'

for version in {0..4}
do
    echo "Testing version $version..."
    write_yaml 'example_version' "$version"
    echo "Displayed prediction results for version $version"
    python "$PREDICT_FILE"

    sleep 1
done

# restore the initial example_version
write_yaml "example_version" "$initial_version"
echo "The initial version was restored"
