#load_args

import argparse
import importlib
parser = argparse.ArgumentParser()
parser.add_argument("--args", type=str, default='args', help="specify which args file")

p, unk = parser.parse_known_args()
print("args:", p)
print("unknown args:", unk)
name = 'arguments.' + p.args
args = importlib.import_module(name)
