#!/bin/bash

python -m gridder.make_experiments -o /tmp/ -e experimental_config.py
bash run.sh
