import scipy.sparse as sp
import numpy as np
from mip import Model, xsum, minimize, BINARY
from ldpc import bposd_decoder

##### Code Deformation #####

def enlarge_parity_check(H, E):
    """
    Append E zero columns to a parity-check matrix H.
    H: scipy sparse matrix, shape (r, n)
    Returns: sparse matrix of shape (r, n + E)
    """
    r, n = H.shape
    zero_block = sp.csr_matrix((r, E), dtype=H.dtype)
    return sp.hstack([H, zero_block], format="csr")

def edge2qubit(G, n):
    """
    Create a mapping from graph edges to qubit indices.

    G: NetworkX graph (G_thick)
    n: number of original qubits (columns in H)

    Returns:
        edge_to_qubit: dict mapping (u, v, key) -> qubit index
    """
    edge_list = list(G.edges(keys=True))
    edge_to_qubit = {
        e: n + i
        for i, e in enumerate(edge_list)
    }
    return edge_to_qubit

# vertex check type is the same as logical operator type
def add_vertex_checks(H, G):
    """
    Add vertex Z/X-checks to H.

    For each node v in G:
      - Z/X acts on all edge-qubits incident to v
      - If G.nodes[v] has data['qubit'] = q, also act on qubit q

    H: scipy.sparse matrix
    G: NetworkX graph (G_thick)
    edge_to_qubit: dict mapping (u, v, key) -> qubit index

    Returns:
        H_new: scipy.sparse matrix with extra rows appended
    """

    E = G.number_of_edges()

    n = H.shape[1] - E
    edge_list = list(G.edges(keys=True))
    edge_to_qubit = {
        e: n + i
        for i, e in enumerate(edge_list)
    }

    _, n_total = H.shape

    rows = []
    cols = []
    data = []

    current_row = 0

    m = H.shape[0]
    idx = 0
    vertex2check = {}

    for v in G.nodes:

        vertex2check[v] = m + idx
        idx += 1

        # 1) incident edge-qubits
        for u, w, k in G.edges(v, keys=True):
            # normalize edge orientation
            if (u, w, k) in edge_to_qubit:
                e = (u, w, k)
            else:
                e = (w, u, k)

            q = edge_to_qubit[e]
            rows.append(current_row)
            cols.append(q)
            data.append(1)

        # 2) optional node-attached original qubit
        q_node = G.nodes[v].get("qubit")
        if q_node is not None:
            rows.append(current_row)
            cols.append(q_node)
            data.append(1)

        current_row += 1

    num_new_rows = current_row
    vertex = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(num_new_rows, n_total),
        dtype=H.dtype
    )

    H_new = sp.vstack([H, vertex], format="csr")
    return H_new, vertex2check

def extend_logical(L, E):
    """
    Append E zeros to a logical Z operator.
    L can be a scipy sparse row or a numpy array.
    """
    if hasattr(L, "toarray"):
        # scipy sparse row -> dense 1D array
        L = L.toarray().ravel()
    else:
        L = np.asarray(L).ravel()

    return np.concatenate([L, np.zeros(E, dtype=int)])

def is_in_row_space_GF2(H, v):
    """
    Check if vector v is in the row space of H over GF(2).

    H: scipy.sparse matrix, shape (r, n)
    v: 1D numpy array of length n (0/1)

    Returns:
        True if v ∈ rowspace(H), else False
    """
    # Convert rows of H to bitsets
    rows = []
    for i in range(H.shape[0]):
        row = H.getrow(i).toarray().ravel() % 2
        x = 0
        for j, bit in enumerate(row):
            if bit:
                x |= (1 << j)
        rows.append(x)

    # Convert v to bitset
    v_bits = 0
    for j, bit in enumerate(v % 2):
        if bit:
            v_bits |= (1 << j)

    # Compute rank of rows
    def gf2_rank(vectors):
        pivots = {}
        for vec in vectors:
            x = vec
            while x:
                b = x.bit_length() - 1
                if b in pivots:
                    x ^= pivots[b]
                else:
                    pivots[b] = x
                    break
        return len(pivots)

    rank_rows = gf2_rank(rows)
    rank_aug = gf2_rank(rows + [v_bits])

    return rank_aug == rank_rows

# perfect matching check type is different logical operator type
def add_perfect_matching_checks(H_ext, G_thick, perfect_matchings):
    """
    For each existing Z/X-check row s:
      if s in perfect_matchings:
        for each matched pair (a,b):
          find the MultiGraph edge between (a,b) whose data satisfies:
             d.get("kind") == "matching" and d.get("check") == s
          then set H_ext[s, edge_qubit_index] ^= 1   (GF2)
    Returns: updated H_ext (csr)
    """

    E = G_thick.number_of_edges()
    n = H_ext.shape[1] - E
    edge_list = list(G_thick.edges(keys=True))
    edge_to_qubit = {
        e: n + i
        for i, e in enumerate(edge_list)
    }

    # make it writable
    H = H_ext.tolil(copy=True)

    for s, pairs in perfect_matchings.items():
        # if s is not a valid row index, you may want to skip or raise
        if s < 0 or s >= H.shape[0]:
            continue

        for a, b in pairs:
            # find the matching edge (a,b, key) in the MultiGraph
            found = None
            if G_thick.has_edge(a, b):
                # G_thick[a][b] is dict: key -> attr_dict
                for k, d in G_thick[a][b].items():
                    if d.get("kind") == "matching" and d.get("check") == s:
                        found = (a, b, k)
                        break

            # undirected safety (usually not needed if has_edge(a,b) works)
            if found is None and G_thick.has_edge(b, a):
                for k, d in G_thick[b][a].items():
                    if d.get("kind") == "matching" and d.get("check") == s:
                        found = (b, a, k)
                        break

            if found is None:
                raise KeyError(f"Cannot find matching edge for check {s} between nodes ({a},{b})")

            # map that graph-edge to the new edge-qubit column
            if found in edge_to_qubit:
                q = edge_to_qubit[found]
            else:
                # orientation swap fallback
                u, v, k = found
                q = edge_to_qubit[(v, u, k)]

            # GF(2) add: toggle that entry
            H[s, q] = H[s, q] + 1

    return H.tocsr()

# cycle check type is different logical operator type
def cycle_to_row(cycle, edge_to_qubit, N_total):
    row = np.zeros(N_total, dtype=np.uint8)

    for u, v, k in cycle:
        if (u, v, k) in edge_to_qubit:
            q = edge_to_qubit[(u, v, k)]
        elif (v, u, k) in edge_to_qubit:
            q = edge_to_qubit[(v, u, k)]
        else:
            raise KeyError(f"Edge {(u,v,k)} not in edge_to_qubit")

        row[q] ^= 1   # GF(2)

    return row

def append_cycle_basis_checks(G_thick, H_ext, cycle_basis, N_total):
    
    E = G_thick.number_of_edges()
    n = H_ext.shape[1] - E
    edge_list = list(G_thick.edges(keys=True))
    edge_to_qubit = {
        e: n + i
        for i, e in enumerate(edge_list)
    }

    rows = []
    for cycle in cycle_basis:
        rows.append(cycle_to_row(cycle, edge_to_qubit, N_total))

    New = np.vstack(rows)          # shape: (#cycles, N_total)
    H_new = sp.vstack([H_ext, sp.csr_matrix(New)], format="csr")
    return H_new

##### Code Distance #####

def distance_test(stab,logicOp):
	# number of qubits
	n = stab.shape[1]
	# number of stabilizers
	m = stab.shape[0]

	# maximum stabilizer weight
	wstab = np.max([np.sum(stab[i,:]) for i in range(m)])
	# weight of the logical operator
	# wlog = np.count_nonzero(logicOp)
	wlog = logicOp.nnz
	# how many slack variables are needed to express orthogonality constraints modulo two
	num_anc_stab = int(np.ceil(np.log2(wstab)))
	num_anc_logical = int(np.ceil(np.log2(wlog)))
	# total number of variables
	num_var = n + m*num_anc_stab + num_anc_logical

	model = Model()
	model.verbose = 0
	x = [model.add_var(var_type=BINARY) for i in range(num_var)]
	model.objective = minimize(xsum(x[i] for i in range(n)))

	# orthogonality to rows of stab constraints
	for row in range(m):
		weight = [0]*num_var
		supp = np.nonzero(stab[row,:])[0]
		for q in supp:
			weight[q] = 1
		cnt = 1
		for q in range(num_anc_stab):
			weight[n + row*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
		model+= xsum(weight[i] * x[i] for i in range(num_var)) == 0

	# odd overlap with logicOp constraint
	supp = np.nonzero(logicOp)[1]
	weight = [0]*num_var
	for q in supp:
		weight[q] = 1
	cnt = 1
	for q in range(num_anc_logical):
			weight[n + m*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
	model+= xsum(weight[i] * x[i] for i in range(num_var)) == 1

	model.optimize()

	opt_val = sum([x[i].x for i in range(n)])
	sol = [x[i].x for i in range(n)]
	return int(opt_val), sol


def integer_distance_test(hz, hx, lz, lx, weight_threshold=12):
     
    k = lz.shape[0]; n = lz.shape[1]
    best_logical_X = {i: None for i in range(k)}
    best_logical_Z = {i: None for i in range(k)}
    best_logical_X_weight = {i: float('inf') for i in range(k)}
    best_logical_Z_weight = {i: float('inf') for i in range(k)}

    for i in range(k):
        w, sol = distance_test(hx,lx[i,:])
        best_logical_Z[i] = sol
        best_logical_Z_weight[i] = w
        print(f"Logical qubit {i}: best Z weight = {best_logical_Z_weight[i]}")
        print(f"Best logical Z operator for qubit {i}: {best_logical_Z[i]}")

    for i in range(k):
        w, sol = distance_test(hz,lz[i,:])
        best_logical_X[i] = sol
        best_logical_X_weight[i] = w
        print(f"Logical qubit {i}: best X weight = {best_logical_X_weight[i]}")
        print(f"Best logical X operator for qubit {i}: {best_logical_X[i]}")        

    lowest_Z_weight = min(best_logical_Z_weight.values())
    lowest_X_weight = min(best_logical_X_weight.values())
    lowest_w = min(lowest_Z_weight, lowest_X_weight)
    if lowest_w < weight_threshold:
        print(f"\nDistance test: Found logical operator with weight {lowest_w} < threshold {weight_threshold}.")

    return best_logical_Z, best_logical_Z_weight, best_logical_X, best_logical_X_weight


def heuristic_distance_test(hz, hx, lz, lx, weight_threshold=12, num_trails=1000):

    # setup BP-OSD decoder parameters
    my_bp_method = "ms"
    my_max_iter = 10000
    my_osd_method = "osd_cs"
    my_osd_order = 20
    my_ms_scaling_factor = 0

    k = lz.shape[0]

    best_logical_X = {i: None for i in range(k)}
    best_logical_Z = {i: None for i in range(k)}
    best_logical_X_weight = {i: float('inf') for i in range(k)}
    best_logical_Z_weight = {i: float('inf') for i in range(k)}

    for trial in range(num_trails):

        logical_op_idx = trial % k
        
        # Logical Z Operator
        random_logical_op = lx[logical_op_idx,:].toarray()
        HZ1 = np.vstack((hx,random_logical_op))
        syndrome = np.zeros(HZ1.shape[0],dtype=int)
        syndrome[-1]=1

        bpdZ=bposd_decoder(
            HZ1,
            error_rate=0.01,
            max_iter=my_max_iter, #the maximum number of iterations for BP)
            bp_method=my_bp_method,
            ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
            osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
            osd_order=my_osd_order #the osd search depth
            )
        bpdZ.decode(syndrome)
        low_weight_logical = bpdZ.osdw_decoding
        wt = np.count_nonzero(low_weight_logical)
        if wt<best_logical_Z_weight[logical_op_idx] and wt>0:
            best_logical_Z[logical_op_idx] = low_weight_logical
            best_logical_Z_weight[logical_op_idx] = wt

        # Logical X Operator
        random_logical_op = lz[logical_op_idx,:].toarray()
        HX1 = np.vstack((hz,random_logical_op))
        syndrome = np.zeros(HX1.shape[0],dtype=int)
        syndrome[-1]=1

        bpdX=bposd_decoder(
            HX1,
            error_rate=0.01,
            max_iter=my_max_iter, #the maximum number of iterations for BP)
            bp_method=my_bp_method,
            ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
            osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
            osd_order=my_osd_order #the osd search depth
            )
        bpdX.decode(syndrome)
        low_weight_logical = bpdX.osdw_decoding
        wt = np.count_nonzero(low_weight_logical)
        if wt<best_logical_X_weight[logical_op_idx] and wt>0:
            best_logical_X[logical_op_idx] = low_weight_logical
            best_logical_X_weight[logical_op_idx] = wt

    for i in range(k):
        print(f"Logical qubit {i}: best Z weight = {best_logical_Z_weight[i]}")
        print(f"Best logical Z operator for qubit {i}: {best_logical_Z[i]}")
        print(f"Logical qubit {i}: best X weight = {best_logical_X_weight[i]}")
        print(f"Best logical X operator for qubit {i}: {best_logical_X[i]}")

    lowest_Z_weight = min(best_logical_Z_weight.values())
    lowest_X_weight = min(best_logical_X_weight.values())
    lowest_w = min(lowest_Z_weight, lowest_X_weight)
    if lowest_w < weight_threshold:
        print(f"\nHeuristic distance test: Found logical operator with weight {lowest_w} < threshold {weight_threshold}.")
    else:
        print("\nAll logical operators have enough distance!")


##### Code Merge #####    

def find_d_nodes_tree(G_thick, D):
    """
    Find a tree with exactly D nodes inside the set of nodes whose
    node attribute contains the key 'qubit'.

    Parameters
    ----------
    G_thick : nx.MultiGraph
        Input graph.
    D : int
        Number of vertices in the desired tree.

    Returns
    -------
    nodes_set : set
        Set of D nodes in the tree.
    edges : list
        List of edges in the form (u, v, key).
    """

    # 1. Select nodes containing the 'qubit' attribute
    qubit_nodes = {n for n, data in G_thick.nodes(data=True) if "qubit" in data}

    if len(qubit_nodes) < D:
        raise ValueError("Adapter size exceeds the logical operator size.")

    # 2. Induced subgraph
    G_sub = G_thick.subgraph(qubit_nodes)

    # 3. BFS to construct a tree with D nodes
    start = next(iter(G_sub.nodes))

    visited = {start}
    tree_nodes = [start]
    tree_edges = []

    queue = [start]

    while queue and len(tree_nodes) < D:
        u = queue.pop(0)

        for v, edge_dict in G_sub.adj[u].items():

            if v in visited:
                continue

            # choose one edge key (since MultiGraph may have multiple)
            key = next(iter(edge_dict))

            visited.add(v)
            tree_nodes.append(v)
            tree_edges.append((u, v, key))
            queue.append(v)

            if len(tree_nodes) == D:
                break

    if len(tree_nodes) < D:
        raise ValueError("Could not find a connected tree with D nodes.")

    return set(tree_nodes), tree_edges

def incidence_matrix(nodes, edges):
    """
    Construct the incidence matrix of a subtree.

    Parameters
    ----------
    nodes : set
        Nodes in the subtree.
    edges : list
        List of edges in form (u, v, key).

    Returns
    -------
    B : numpy.ndarray
        Incidence matrix (|E| x |V|)
    column_dict : dict
        column_index -> node
    row_dict : dict
        row_index -> (u, v, key)
    """

    node_list = list(nodes)

    # column mapping
    column_dict = {i: node for i, node in enumerate(node_list)}
    node_to_col = {node: i for i, node in column_dict.items()}

    # row mapping
    row_dict = {i: edge for i, edge in enumerate(edges)}

    B = np.zeros((len(edges), len(node_list)), dtype=int)

    for r, (u, v, k) in row_dict.items():
        cu = node_to_col[u]
        cv = node_to_col[v]

        B[r, cu] = 1
        B[r, cv] = 1

    return B, column_dict, row_dict


def combine_parity_check(H1, H2):
    """
    Combine two parity check matrices acting on different qubits
    into a block diagonal matrix.
    """

    m1, n1 = H1.shape
    m2, n2 = H2.shape

    zero12 = sp.csr_matrix((m1, n2))
    zero21 = sp.csr_matrix((m2, n1))

    top = sp.hstack([H1, zero12])
    bottom = sp.hstack([zero21, H2])

    H = sp.vstack([top, bottom])

    return H


def add_adapter(H, P_1, P_2, vertex2check_1, vertex2check_2, column_dict_1, column_dict_2, m1, n1, m2, n2, D):
    
    """
    Add adapter edges between two codes by using permutation matrices P_1 and P_2
    """

    for i in range(D):

        check_idx_1 = vertex2check_1[column_dict_1[i]]
        H[check_idx_1, n1+n2+P_1[i].argmax()]=1

        check_idx_2 = vertex2check_2[column_dict_2[i]]+m1
        H[check_idx_2, n1+n2+P_2[i].argmax()]=1

    return H

def add_adapter_cycle(H, T_1, T_2, edge2qubit_1, edge2qubit_2, row_dict_1, row_dict_2, m1, n1, m2, n2, D):
    
    """
    Add adapter cycles between two codes by using transformation matrices T_1 and T_2
    """

    _, n = H.shape
    zeros = sp.csr_matrix((D, n))
    H = sp.vstack([H, zeros])

    for i in range(D):

        # connect to adapter
        H[m1+m2+i, n1+n2+i]=1
        H[m1+m2+i, n1+n2+(i+1)%D]=1
        
        # connect to ancilla qubits on both sides

        idx_ls_1 = np.nonzero(T_1[i])[0].tolist()
        for idx in idx_ls_1:
            edge_1 = row_dict_1[idx]
            try:
                qubit_1 = edge2qubit_1[edge_1]
            except KeyError:
                qubit_1 = edge2qubit_1[(edge_1[1], edge_1[0], edge_1[2])]
            H[m1+m2+i, qubit_1]=1

        idx_ls_2 = np.nonzero(T_2[i])[0].tolist()
        for idx in idx_ls_2:
            edge_2 = row_dict_2[idx]
            try:
                qubit_2 = edge2qubit_2[edge_2]
            except KeyError:
                qubit_2 = edge2qubit_2[(edge_2[1], edge_2[0], edge_2[2])]
            H[m1+m2+i, n1+qubit_2]=1

    return H