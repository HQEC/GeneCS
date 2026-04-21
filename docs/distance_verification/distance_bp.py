from genecs.deform import *
import pickle
import argparse
from contextlib import redirect_stdout, redirect_stderr

parser = argparse.ArgumentParser(description="GeneLS Lattice Surgery Settings")
parser.add_argument('--file_name', '-f', help='Name of the code.', default="BB_144_12_12_full_Z_0")
parser.add_argument('--weight_threshold', '-w', help='Threshold of the logical operator weight.', default=12, type=int)
parser.add_argument('--trail_num', '-t', help='Number of trails for BP-OSD.', default=100, type=int)
args = parser.parse_args()

file_name = args.file_name
weight_threshold = args.weight_threshold
num_trails = args.trail_num


with open(f"deform_code/{file_name}.pkl", "rb") as f:
    data = pickle.load(f)

qcode = data["qcode"]

lz = qcode.lz
lx = qcode.lx
hz = qcode.hz.toarray()
hx = qcode.hx.toarray()

output_path = 'deform_distance' + '/' + f'{file_name}_bposd' + '.txt'

with open(output_path, "w") as f:   
    with redirect_stdout(f), redirect_stderr(f):
        heuristic_distance_test(hz, hx, lz, lx, weight_threshold=weight_threshold, num_trails=num_trails)

'''
python3 distance_bp.py -f BB_72_X_[0] -w 6 -t 100
'''