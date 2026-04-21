from genecs.fun import *
import argparse

parser = argparse.ArgumentParser(description="GeneLS Lattice Surgery Settings")
parser.add_argument('--file_name_1', '-f1', help='Name of the first code.', default="BB_144_12_12")
parser.add_argument('--file_name_2', '-f2', help='Name of the second code.', default="BB_144_12_12")
parser.add_argument('--logical_type_1', '-lt1', help='Type of the logical operator to be measured for the first code.', default="Z", choices=["Z", "X"])
parser.add_argument('--logical_type_2', '-lt2', help='Type of the logical operator to be measured for the second code.', default="Z", choices=["Z", "X"])
parser.add_argument('--logical_idx_1', '-li1', help='Index of the logical operator to be measured for the first code.', nargs='+', type=int, default=[0])
parser.add_argument('--logical_idx_2', '-li2', help='Index of the logical operator to be measured for the second code.', nargs='+', type=int, default=[0])
parser.add_argument('--qubit_degree_1', '-qd1', help='Maximum degree of qubits in the graph for the first code.', default=14, type=int)
parser.add_argument('--qubit_degree_2', '-qd2', help='Maximum degree of qubits in the graph for the second code.', default=14, type=int)
parser.add_argument('--check_degree_1', '-cd1', help='Maximum degree of checks in the graph for the first code.', default=8, type=int)
parser.add_argument('--check_degree_2', '-cd2', help='Maximum degree of checks in the graph for the second code.', default=8, type=int)
parser.add_argument('--sample_num', '-s', help='Number of samples for congestion-aware expansion.', default=100, type=int)
parser.add_argument('--adapter_size', '-a', help='Size of the adapter.', default=12, type=int)
args = parser.parse_args()

file_name_1 = args.file_name_1
file_name_2 = args.file_name_2
file_name = file_name_1 + '+' + file_name_2
logical_type_1 = args.logical_type_1
logical_type_2 = args.logical_type_2
logical_idx_1 = args.logical_idx_1
logical_idx_2 = args.logical_idx_2
qubit_degree_1 = args.qubit_degree_1
qubit_degree_2 = args.qubit_degree_2
check_degree_1 = args.check_degree_1
check_degree_2 = args.check_degree_2
sample_num = args.sample_num
adapter_size = args.adapter_size

crosscode_synthesis(file_name_1, file_name_2, logical_type_1, logical_type_2, logical_idx_1, logical_idx_2, qubit_degree_1, qubit_degree_2, check_degree_1, check_degree_2, sample_num, adapter_size, benchmark_path= '../../benchmark', saving_path = './bin')