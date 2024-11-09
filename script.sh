#!/bin/bash

# Create a Python virtual environment named 'env'
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install requirements from the requirements.txt file
pip install -r requirements.txt

# Run the main.py
python main.py --dataset=TRIANGLES --train_num=200 --val_num=100 --workers_num=4 --each_class_train=10 

# Deactivate the virtual environment after script execution
deactivate
