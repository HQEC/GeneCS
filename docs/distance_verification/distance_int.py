from genecs.deform import *
import pickle
import argparse
from contextlib import redirect_stdout, redirect_stderr

parser = argparse.ArgumentParser(description="GeneLS Lattice Surgery Settings")
parser.add_argument('--file_name', '-f', help='Name of the code.', default="BB_144_12_12_full_Z_0")
parser.add_argument('--weight_threshold', '-w', help='Threshold of the logical operator weight.', default=12, type=int)
args = parser.parse_args()

file_name = args.file_name
weight_threshold = args.weight_threshold

with open(f"deform_code/{file_name}.pkl", "rb") as f:
    data = pickle.load(f)

qcode = data["qcode"]

lz = qcode.lz
lx = qcode.lx
hz = qcode.hz.toarray()
hx = qcode.hx.toarray()
k = lz.shape[0]

output_path = 'deform_distance' + '/' + f'{file_name}_int' + '.txt'

best_logical_Z, best_logical_Z_weight, best_logical_X, best_logical_X_weight = integer_distance_test(hz, hx, lz, lx, weight_threshold=weight_threshold)

with open(output_path, "w") as f:   
    with redirect_stdout(f), redirect_stderr(f):
        for i in range(k):
            print(f"Logical qubit {i}: best Z weight = {best_logical_Z_weight[i]}")
            print(f"Best logical Z operator for qubit {i}: {best_logical_Z[i]}")
            print(f"Logical qubit {i}: best X weight = {best_logical_X_weight[i]}")
            print(f"Best logical X operator for qubit {i}: {best_logical_X[i]}")      

        min_weight = min(min(best_logical_Z_weight.values()), min(best_logical_X_weight.values()))
        if min_weight >= weight_threshold:
            print(f"All logical operators have weight above the threshold of {weight_threshold}!")
        else:
            print(f"Some logical operators have weight below the threshold of {weight_threshold}...")

'''
python3 distance_int.py -f BB_72_X_[0] -w 6
'''