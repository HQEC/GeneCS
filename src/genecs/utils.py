import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import vstack
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

##### Lower bound of Cheeger constant #####

def cheeger_lower_bound(
    G
):
    """
    Compute the lower bound of the Cheeger constant h(G) >= lambda_2 / 2
    using the combinatorial Laplacian.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph

    Returns
    -------
    Cheeger : float
        Lower bound lambda_2 / 2
    """

    if not nx.is_connected(G):
        return 0.0

    # adjacency matrix (sparse)
    A = nx.adjacency_matrix(G, dtype=float)

    # degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)

    # Laplacian
    L = D - A

    # compute smallest eigenvalues
    eigvals, eigvecs = spla.eigsh(
        L,
        k=2,
        which="SA"   # smallest algebraic
    )

    # sort
    eigvals = np.sort(eigvals)

    lambda_2 = eigvals[1]
    Cheeger = lambda_2 / 2

    return Cheeger

##### Graph visualization #####

def visual_base_graph(G):
    # Visualization
    matching_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("kind") == "matching"
    ]

    expander_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("kind") == "expander"
    ]

    pos = nx.spring_layout(G, seed=1)

    plt.figure(figsize=(6, 6))

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=600,
        node_color="lightblue",
        edgecolors="black"
    )

    # draw non-matching edges (if any)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=expander_edges,
        edge_color="green",
        width=1,
        alpha=0.5
    )

    # draw matching edges (highlighted)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=matching_edges,
        edge_color="red",
        width=1,
        alpha=0.5
    )

    # node labels = node labels
    labels = {v: v for v in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.axis("off")
    plt.title("Base Graph | Green: Expander Edges, Red: Perfect-Matching Edges")
    plt.show()

##### Graph Congestion #####

def canon_edge(u, v, k):
    return (u, v, k) if u <= v else (v, u, k)

def cycle_basis_congestion(cycle_basis, undirected=True, verbose=True, topk=10):
    edge_count = Counter()
    edge_to_cycles = defaultdict(list)

    for ci, cycle in enumerate(cycle_basis):
        if undirected:
            edges_in_cycle = {canon_edge(u, v, k) for u, v, k in cycle}
        else:
            edges_in_cycle = {tuple(e) for e in cycle}

        for e in edges_in_cycle:
            edge_count[e] += 1
            edge_to_cycles[e].append(ci)

    congestion = max(edge_count.values(), default=0)
    busiest_edges = [e for e, c in edge_count.items() if c == congestion]

    if verbose:
        print(f"Total cycles in basis = {len(cycle_basis)}")
        print(f"Unique edges counted   = {len(edge_count)}")
        print(f"Congestion             = {congestion}")
        print(f"Edges achieving max congestion (show up to {topk}):")
        for e in busiest_edges[:topk]:
            print(f"  edge={e}, participates_in_cycles={edge_to_cycles[e]}")

    return congestion, edge_count, edge_to_cycles

##### Sanity check #####

def cycle_to_vector(cycle, edge_to_idx, E):
    x = np.zeros(E, dtype=int)
    for u, v, k in cycle:
        if (u, v, k) in edge_to_idx:
            e = (u, v, k)
        elif (v, u, k) in edge_to_idx:
            e = (v, u, k)
        else:
            raise KeyError(f"Edge {(u, v, k)} not found in edge_to_idx")

        x[edge_to_idx[e]] ^= 1

    return x

def check_cycle_in_null_space(G, cycle_basis):

    edges = list(G.edges(keys=True))
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    E = len(edges)

    nodes = list(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    V = len(nodes)

    B = np.zeros((V, E), dtype=int)

    for (u, v, k), j in edge_to_idx.items():
        B[node_to_idx[u], j] ^= 1
        B[node_to_idx[v], j] ^= 1

    for i, cycle in enumerate(cycle_basis):
        x = cycle_to_vector(cycle, edge_to_idx, E)
        y = (B @ x) % 2
        assert np.all(y == 0), f"Cycle {i} is NOT in null space"

def cycle_to_bitset(cycle, edge_to_idx, E):
    """Return Python int whose binary representation is the F2 edge-incidence vector."""
    x = 0
    for u, v, k in cycle:
        # try both orientations (undirected graph)
        if (u, v, k) in edge_to_idx:
            e = (u, v, k)
        elif (v, u, k) in edge_to_idx:
            e = (v, u, k)
        else:
            raise KeyError(f"Edge {(u, v, k)} not found in edge_to_idx")

        j = edge_to_idx[e]
        x ^= (1 << j)   # mod 2

    return x

def independent_cycle(G, cycle_basis):
    """
    Returns:
      rank: dimension of span over F2 (max independent set size)
      indep_indices: indices of cycles that form an independent subset
    """

    edges = list(G.edges(keys=True))
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    E = len(edges)

    vecs = [cycle_to_bitset(cyc, edge_to_idx, E) for cyc in cycle_basis]

    # pivot_bit -> (pivot_vector, original_index)
    pivots = {}
    indep_indices = []

    for idx, v in enumerate(vecs):
        x = v
        while x:
            b = x.bit_length() - 1  # highest set bit position
            if b in pivots:
                x ^= pivots[b][0]
            else:
                pivots[b] = (x, idx)
                indep_indices.append(idx)
                break
        # if x becomes 0, it's dependent and gets skipped

    rank = len(pivots)
    return rank

def cycle_space_dimension_formula(G, cycle_basis):
    V = G.number_of_nodes()
    print(f"V = {V}")
    E = G.number_of_edges()
    print(f"E = {E}")
    C = len(cycle_basis)
    print(f"C = {C}")
    return C == E - V + 1

def check_edge_nonoverlapping(groups, cycle_basis):
    cycle_edge_sets = [
        {canon_edge(u, v, k) for u, v, k in cyc}
        for cyc in cycle_basis
    ]
    for g in groups:
        used = set()
        for idx in g:
            es = cycle_edge_sets[idx]
            assert used.isdisjoint(es), f"Edge overlap detected in group: {g}"
            used |= es

def compute_global_distribution(hx, hz):
    # ---- X check row weights ----
    hx_row_weights = np.asarray(hx.getnnz(axis=1)).ravel()

    # ---- Z check row weights ----
    hz_row_weights = np.asarray(hz.getnnz(axis=1)).ravel()

    # ---- Global qubit participation ----
    # Stack vertically to combine X and Z checks
    H_total = vstack([hx, hz])

    total_col_weights = np.asarray(H_total.getnnz(axis=0)).ravel()

    # Distributions
    hx_row_dist = Counter(hx_row_weights)
    hz_row_dist = Counter(hz_row_weights)
    col_dist = Counter(total_col_weights)

    print("\n========== CSS Code Structure ==========\n")

    print("X-check Weight Distribution:")
    for weight in sorted(hx_row_dist):
        print(f"  Weight {int(weight)}: {hx_row_dist[weight]} X-checks")

    print()

    print("Z-check Weight Distribution:")
    for weight in sorted(hz_row_dist):
        print(f"  Weight {int(weight)}: {hz_row_dist[weight]} Z-checks")

    print()

    print("Qubit Participation Distribution (X + Z combined):")
    for weight in sorted(col_dist):
        print(f"  {col_dist[weight]} qubits are involved in {int(weight)} total checks")

    print("\n========================================\n")

def compute_max_degrees(hx, hz):
    # Stack HX and HZ
    H_total = vstack([hx, hz])
    
    # Max check degree = max row weight
    row_weights = np.asarray(H_total.getnnz(axis=1)).ravel()
    max_check_degree = int(row_weights.max())
    
    # Max qubit degree = max column weight
    col_weights = np.asarray(H_total.getnnz(axis=0)).ravel()
    max_qubit_degree = int(col_weights.max())
    
    return max_check_degree, max_qubit_degree
