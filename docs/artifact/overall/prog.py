from genecs.fun import *
import argparse

parser = argparse.ArgumentParser(description="GeneLS Lattice Surgery Settings")
parser.add_argument('--file_name', '-f', help='Name of the code.', default="BB_144_12_12")
parser.add_argument('--logical_type', '-lt', help='Type of the logical operator to be measured.', default="Z", choices=["Z", "X"])
parser.add_argument('--logical_idx', '-li', help='Index of the logical operator to be measured.', nargs='+', type=int, default=[0])
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

innercode_synthesis(file_name, logical_type, logical_idx, qubit_degree, check_degree, sample_num, benchmark_path='../../benchmark', saving_path='./bin')
