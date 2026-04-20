from genecs.utils import *
from genecs.fun import *
from genecs.deform import *
from bposd.css import css_code
import pickle
from contextlib import redirect_stdout, redirect_stderr
import argparse
import time
import pandas as pd

parser = argparse.ArgumentParser(description="GeneLS Lattice Surgery Settings")
parser.add_argument('--file_name', '-f', help='Name of the code.', default="BB_144_12_12")
parser.add_argument('--logical_type', '-lt', help='Type of the logical operator to be measured.', default="Z", choices=["Z", "X"])
parser.add_argument('--logical_idx', '-li', help='Index of the logical operator to be measured.', default=0, type=int)
parser.add_argument('--qubit_degree', '-qd', help='Maximum degree of qubits in the graph.', default=14, type=int)
parser.add_argument('--check_degree', '-cd', help='Maximum degree of checks in the graph.', default=8, type=int)
parser.add_argument('--sample_num', '-s', help='Number of samples for congestion-aware expansion.', default=100, type=int)
args = parser.parse_args()

file_name = args.file_name
logical_type = args.logical_type
logical_idx = args.logical_idx
qubit_degree = args.qubit_degree
check_degree = args.check_degree
sample_num = args.sample_num

t0 = time.time()
######################## Construct Ancilla system Graph ########################

# load the code
with open(f"./benchmark/{file_name}.pkl", "rb") as f:
    data = pickle.load(f)

output_path = 'deform_data' + '/' + f'{file_name}_full_{logical_type}_{logical_idx}' + '.txt'
output_file = open(file=output_path, mode="a", encoding='utf-8')

qcode = data["qcode"]

with open(output_path, "w") as f:   
    with redirect_stdout(f), redirect_stderr(f):
        print("Testing original code...")
        qcode.test()
        print("\n")

lz = qcode.lz; lx = qcode.lx
hz = qcode.hz.toarray(); hx = qcode.hx.toarray()
k = lz.shape[0]; n = lz.shape[1]

if logical_type == "Z":
    logical_op = lz[logical_idx,:]
    other_h = hx
elif logical_type == "X":
    logical_op = lx[logical_idx,:]
    other_h = hz

# Step 1: Construct vertex of graph and the bijection fuction
G, qubit_to_vertex = create_graph_GeneLS(logical_op)

# Step 2: Construct the perfect matching
G, perfect_matchings = construct_perfect_matching_GeneLS(G, qubit_to_vertex, logical_op, other_h)

# Step 3: Congestion-aware expansion of graph
# 2 reserved for vertex checks, 2 reserved for perfect matching or thickening per edge
G_base, history, cycle_basis, groups = congestion_aware_expander_sweep_GeneLS(G, sample_num, edge_key="exp", max_edge_degree=qubit_degree-4)

# Step 4: Thickening
beta = cheeger_lower_bound(G_base)
if abs(1 - beta) < 1e-6:
    beta = 1.0
L = max(math.ceil(1 / beta), len(groups))
print("Thickening layer =", L, file=output_file)
print(f"Redundancy = {(L - min(math.ceil(1 / beta), len(groups))) / L * 100:.1f} %", file=output_file)
G_thick = thicken_graph(G_base, L)

# Step 5: Cellulation
new_cycle_basis = cellulation_GeneLS(G_thick, groups, cycle_basis, d=check_degree)
add_interlayer_rectangle_cycles_GeneLS(G_thick, G_base, L, new_cycle_basis)

print("\n", file=output_file)
new_qubit_num = G_thick.number_of_edges()
new_check_num = len(new_cycle_basis) + G_thick.number_of_nodes()
print("Number of new qubits:", G_thick.number_of_edges(), file=output_file)
print("Number of new chekcs:", len(new_cycle_basis) + G_thick.number_of_nodes(), file=output_file)

data = {
    "G_thick": G_thick,
    "perfect_matching": perfect_matchings, 
    "cycle_basis": new_cycle_basis, 
    "HX": hx,
    "HZ": hz,
    "logical_type": logical_type,
    "logical_idx": logical_idx,
    "logical_op": logical_op,
}

with open(f"deform_graph/{file_name}_full_{logical_type}_{logical_idx}.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

######################## Code Deformation ########################

# Construct the parity check matrix for derformed code
E = G_thick.number_of_edges()
hz = enlarge_parity_check(hz, E)
hx = enlarge_parity_check(hx, E)

# add new vertex check
if logical_type == "Z":
    hz, _ = add_vertex_checks(hz, G_thick)
elif logical_type == "X":
    hx, _ = add_vertex_checks(hx, G_thick)

# modify original check involved in perfect matching 
if logical_type == "Z":
    hx = add_perfect_matching_checks(hx, G_thick, perfect_matchings)
elif logical_type == "X":
    hz = add_perfect_matching_checks(hz, G_thick, perfect_matchings)

# add cycle checks to fix gauge degrees of freedom
if logical_type == "Z":
    hx = append_cycle_basis_checks(G_thick, hx, new_cycle_basis, hx.shape[1])
elif logical_type == "X":
    hz = append_cycle_basis_checks(G_thick, hz, new_cycle_basis, hz.shape[1])

t1 = time.time()

check_degree, qubit_degree = compute_max_degrees(hx, hz)
print("\n", file=output_file)
print("Max check degree:", check_degree, file=output_file)
print("Max qubit degree:", qubit_degree, file=output_file)

print("\n", file=output_file)
print("Compilation time:", t1-t0, file=output_file)

# check logical operator now in stabilizer group
if logical_type == "Z":
    logical_ext = extend_logical(logical_op, E)  
    print("\n", file=output_file)
    print("Is measured logical operator in stabilizer group?", is_in_row_space_GF2(hz, logical_ext), file=output_file)
elif logical_type == "X":
    logical_ext = extend_logical(logical_op, E)  
    print("\n", file=output_file)
    print("Is measured logical operator in stabilizer group?", is_in_row_space_GF2(hx, logical_ext), file=output_file)

qcode=css_code(hx,hz)
with open(output_path, "a") as f:   
    with redirect_stdout(f), redirect_stderr(f):
        print("Testing deformed code...")
        qcode.test()
        print("\n")

deform_code = {
    "qcode": qcode,
}

with open(output_path, "a") as f:   
    with redirect_stdout(f), redirect_stderr(f):
        compute_global_distribution(hx, hz)
        print("\n")


with open(f"deform_code/{file_name}_full_{logical_type}_{logical_idx}.pkl", "wb") as f:
    pickle.dump(deform_code, f)

# Save selected values as CSV
df = pd.DataFrame([{
    'file_name': file_name,
    ' qubit_num': new_qubit_num,
    ' check_num': new_check_num,
    ' qubit_deg': qubit_degree,
    ' check_deg': check_degree,
    ' Comp_time': t1-t0, 
}])
df.to_csv(f'opt_full.csv', mode='a', header=not pd.io.common.file_exists(f'opt_full.csv'), index=False)

'''
python3 opt_full.py -f BB_144 -lt X -li 0 -s 100 -qd 12 -cd 12
'''

