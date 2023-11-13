#!/bin/bash
echo "Checking for Python"
if command -v python3 &>/dev/null; then
    echo "Python 3 is installed."
    python3 --version
else
    echo "Python 3 is not installed. Please install Python 3."
    exit 1
fi

cd data

echo "Augmenting Data"

python3 augment_data.py

echo "DONE"

cd preprocessing

echo "Preprocessing Images"

python3 preprocess.py

echo "DONE"

cd ../../..

exit 0