#load_args

import argparse
import importlib
parser = argparse.ArgumentParser()
parser.add_argument("--args", type=str, default='args', help="specify which args file")

p = parser.parse_args()
name = 'arguments.' + p.args
args = importlib.import_module(name)
