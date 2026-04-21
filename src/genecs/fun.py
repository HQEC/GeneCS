import networkx as nx
import numpy as np
import random
import math
import copy
import os
from bposd.css import css_code
from itertools import count
import itertools
from genecs.utils import *
from genecs.deform import *
from genecs.adapter import *
from collections import defaultdict, deque
from contextlib import redirect_stdout, redirect_stderr
import pickle
import time
import pandas as pd

##### Step 1: Construct vertex of graph and the bijection fuction #####

def create_graph_GeneCS(logical_op):
    """
    Create a graph from the logical operator.
    Parameters
    ----------
    logical_op : scipy.sparse matrix
        The logical operator in sparse matrix format.
    Returns
    ----------
    G : networkx.MultiGraph
        The graph representation of the logical operator.
    qubit_to_vertex : dict
        A dictionary mapping qubits to vertices in the graph.
    """

    _, cols = logical_op.nonzero()
    qubits = cols.tolist()

    G = nx.MultiGraph()

    for i in range(len(qubits)):
        G.add_node(i, qubit=qubits[i])

    qubit_to_vertex = {
        data['qubit']: v
        for v, data in G.nodes(data=True)
    }

    return G, qubit_to_vertex

##### Step 2.1: Construct the perfect matching #####

def randomized_perfect_matching_GeneCS(vertices, rng=None):
    """
    vertices: list of graph nodes (must have even length)
    rng: random.Random instance (optional) for reproducibility
    returns: list of edges [(v1,v2), (v3,v4), ...]
    """
    if len(vertices) % 2 != 0:
        raise ValueError("Vertex list must have even length")

    rng = rng or random.Random(0)
    vs = list(vertices)          # copy
    rng.shuffle(vs)              # uniform random permutation
    return [(vs[i], vs[i+1]) for i in range(0, len(vs), 2)]

def construct_perfect_matching_GeneCS(G, qubit_to_vertex, logical_op, h, rng=None):
    """
    Construct a perfect matching from the logical operator.
    Parameters
    ----------
    G : networkx.MultiGraph
        The graph representation of the logical operator.
    qubit_to_vertex : dict
        A dictionary mapping qubits to vertices in the graph.
    logical_op : scipy.sparse matrix
        The logical operator in sparse matrix format.
    rng : random.Random instance (optional)
        Random number generator for reproducibility.
    Returns
    ----------
    matching : list of tuples
        A list of edges representing the perfect matching.
    """

    _, cols = logical_op.nonzero()
    qubits = cols.tolist()

    # Iterate every check row s and compute overlap positions (local anti-commute sites)
    K = {}  # map: s -> sorted list of qubits q where hx[s,q]=1 and lz[q]=1

    for s in range(h.shape[0]):
        row = h[s, :]                      # numpy row (1D or 2D)
        row_support = np.where(row != 0)[0] # nonzero column indices
        overlap = [q for q in row_support if q in qubits]
        K[s] = overlap  # keep empty lists too if you want

    perfect_matchings = {}   # s -> list of edges (vertex-level)

    for s, qubits_ in K.items():
        if len(qubits_) == 0:
            continue

        # 1. qubit -> vertex
        vertices = [qubit_to_vertex[q] for q in qubits_]

        # 2. perfect matching on these vertices
        edges = randomized_perfect_matching_GeneCS(vertices)

        perfect_matchings[s] = edges

    for s, edges in perfect_matchings.items():
        for u, v in edges:
            G.add_edge(u, v, key=f"match_{s}", check=s, kind="matching")

    return G, perfect_matchings

##### Step 2.2: Add expander graph #####

def add_edge_GeneCS(G, target_degree, rng=None, edge_key="exp"):
    """
    Try to add exactly ONE edge for a FIXED target_degree.
    If impossible under current constraints, return (G, None).

    Returns:
        (G, (u, v, key)) if an edge is added
        (G, None)        if no valid edge exists for this target_degree
    """
    if rng is None:
        rng = random.Random()

    if G.number_of_nodes() < 2:
        return G, None

    d = target_degree

    degrees = dict(G.degree())
    deficit = {v: max(0, d - deg) for v, deg in degrees.items()}
    low_nodes = [v for v, dv in deficit.items() if dv > 0]

    def edge_exists(u, v):
        return G.has_edge(u, v)

    # ---- special case: only one low-degree node ----
    if len(low_nodes) == 1:
        u = low_nodes[0]
        candidates = [
            v for v, deg in degrees.items()
            if v != u and deg == d and (not edge_exists(u, v))
        ]
        if not candidates:
            # IMPORTANT: end of this "d-round"
            return G, None

        v = rng.choice(candidates)
        G.add_edge(u, v, key=edge_key)
        return G, (u, v, edge_key)

    # ---- normal case ----
    if len(low_nodes) >= 2:
        pairs = []
        weights = []

        for u, v in itertools.combinations(low_nodes, 2):
            if not edge_exists(u, v):
                w = deficit[u] * deficit[v]
                if w > 0:
                    pairs.append((u, v))
                    weights.append(w)

        if not pairs:
            # IMPORTANT: end of this "d-round"
            return G, None

        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for (u, v), w in zip(pairs, weights):
            acc += w
            if r <= acc:
                G.add_edge(u, v, key=edge_key)
                return G, (u, v, edge_key)

        # numerical fallback
        u, v = pairs[-1]
        G.add_edge(u, v, key=edge_key)
        return G, (u, v, edge_key)

    # no deficits at all for this target_degree -> nothing to do in this round
    return G, None

def build_expander_GeneCS(
    G0,
    cheeger_threshold,
    seed=0,
    edge_key="exp",
    start_from="max",   # "max" or "minplus"
    verbose=False,
):
    """
    G0 is NOT modified.
    """
    G = G0.copy()
    rng = random.Random(seed)
    history = []
    step = 0
    n = G.number_of_nodes()
    max_target_degree = n - 1

    current_lb = cheeger_lower_bound(G)

    # initialize target degree
    degs = dict(G.degree())
    if start_from == "max":
        target_degree = max(degs.values())
    elif start_from == "minplus":
        target_degree = min(degs.values()) + 1
    else:
        raise ValueError("start_from must be 'max' or 'minplus'.")

    while current_lb < cheeger_threshold:

        if verbose:
            print(f"[step {step}] target_degree={target_degree}, cheeger_lb={current_lb}")

        G, result = add_edge_GeneCS(G, target_degree, rng=rng, edge_key=edge_key)

        # ---- key change: if no edge can be added for this target_degree,
        #      we end this round and increase the target_degree ----
        if result is None:
            if target_degree >= max_target_degree:
                # cannot increase target_degree any further in a simple graph
                if verbose:
                    print(
                        f"[stop] target_degree reached max {max_target_degree}, "
                        "no more edges possible."
                    )
                break

            target_degree += 1
            continue

        u, v, key = result

        step += 1
        current_lb = cheeger_lower_bound(G)

        history.append({
            "step": step,
            "u": u,
            "v": v,
            "edge_key": key,
            "target_degree": target_degree,
            "cheeger_lb": current_lb,
        })

    return G, history

def build_expander_sweep_GeneCS(
    G0,
    cheeger_threshold,
    seeds_num,
    edge_key="exp",
    verbose=False,
):
    """
    Try multiple random seeds and return the graph that reaches
    the Cheeger threshold with the fewest added edges.

    Parameters
    ----------
    G0 : nx.MultiGraph
        Initial graph (will NOT be modified).
    cheeger_threshold : float
        Target Cheeger lower bound.
    max_seeds : int
        Number of random seeds to try (0, 1, ..., max_seeds-1).
    edge_key : str
        Key used for expander edges.
    verbose : bool
        Print progress information.

    Returns
    -------
    best_G : nx.MultiGraph
        Graph achieving the threshold with the fewest added edges.
    best_history : list
        Edge-addition history for best_G.
    best_seed : int
        Random seed that achieved the best result.
    """
    best_G = None
    best_history = None
    best_seed = None
    best_edges = None

    for seed in range(seeds_num):
        if verbose:
            print(f"\n[seed {seed}] start")

        # IMPORTANT: always start from a fresh copy
        G = G0.copy()

        G_final, history = build_expander_GeneCS(
            G,
            cheeger_threshold=cheeger_threshold,
            seed=seed,
            edge_key=edge_key,
            start_from="max",
            verbose=verbose,
        )

        num_edges = len(history)

        if verbose:
            print(
                f"[seed {seed}] edges added = {num_edges}"
            )

        if best_edges is None or num_edges < best_edges:
            best_G = G_final
            best_history = history
            best_seed = seed
            best_edges = num_edges

            if verbose:
                print(
                    f"[seed {seed}] NEW BEST ({best_edges} edges)"
                )

    return best_G, best_history, best_seed

##### Step 3: Calculating the cycle basis and their partition #####

def _as_multigraph_GeneCS(G):
    """Ensure MultiGraph and copy (so we can mutate safely)."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return G.copy()
    # Convert simple Graph -> MultiGraph (keys will be 0)
    return nx.MultiGraph(G)


def _find_degree_one_vertex_GeneCS(G):
    for v, deg in G.degree():
        if deg == 1:
            return v
    return None


def _find_degree_two_vertex_GeneCS(G):
    for v, deg in G.degree():
        if deg == 2:
            return v
    return None


def _find_self_loop_edge_at_GeneCS(G, v):
    # Returns (u,v,key) where u==v
    for u, w, k in G.edges(v, keys=True):
        if u == w == v:
            return (u, w, k)
    return None


def _incident_edges_excluding_loops_GeneCS(G, v):
    # Return list of (u,w,key) with v incident, excluding self-loop at v
    edges = []
    for u, w, k in G.edges(v, keys=True):
        if u == w == v:
            continue
        edges.append((u, w, k))
    return edges


def _other_endpoint_GeneCS(edge, v):
    u, w, k = edge
    return w if u == v else u

def _vertex_load_GeneCS(G, v, edge_load):
    inc = list(G.edges(v, keys=True))
    if not inc:
        return 0.0
    s = 0
    for a, b, k in inc:
        s += edge_load.get(canon_edge(a, b, k), 0)
    return s / len(inc)

def _max_load_vertex_GeneCS(G, edge_load, rng):
    best_v = None
    best_L = None
    for v in G.nodes():
        L = _vertex_load_GeneCS(G, v, edge_load)
        if best_v is None or L > best_L:
            best_v, best_L = v, L
        elif L == best_L and rng.random() < 0.5:
            best_v = v
    return best_v

def _find_short_cycle_multigraph_GeneCS(G, max_len, rng, edge_load):
    """
    Find a simple cycle as list of edges (u,v,key).
    Priority:
      1) self-loop (length 1)
      2) parallel edges between same (u,v) (length 2)
      3) BFS-based short cycle (<= max_len) in underlying simple structure
    """
    # 1) self-loop
    for u, v, k in G.edges(keys=True):
        if u == v:
            return [(u, v, k)]

    # 2) parallel edges -> 2-cycle
    # For each unordered pair, if >=2 keys, return two distinct edges
    for u, v in G.edges():
        if u == v:
            continue
        keys = list(G[u][v].keys())  # dict: key -> attr
        if len(keys) >= 2:
            return [(u, v, keys[0]), (u, v, keys[1])]

    # 3) BFS short cycle in underlying graph
    start = _max_load_vertex_GeneCS(G, edge_load, rng)

    # We'll run BFS from a few starts (often 1 is enough)
    for start in [start]:
        parent = {start: None}
        parent_edge = {start: None}  # store edge (p, x, key) used to reach x
        depth = {start: 0}

        # queue
        q = [start]
        qi = 0

        while qi < len(q):
            x = q[qi]
            qi += 1
            if depth[x] >= max_len:
                continue

            # iterate neighbors with keys
            for y in G.neighbors(x):
                # pick one representative key for adjacency x-y for exploration
                # (keys only matter for distinguishing edges; for cycle existence, any key works)
                keys_xy = list(G[x][y].keys())
                if not keys_xy:
                    continue
                k_xy = keys_xy[0]

                if y not in parent:
                    parent[y] = x
                    parent_edge[y] = (x, y, k_xy)
                    depth[y] = depth[x] + 1
                    q.append(y)
                else:
                    # found a non-tree edge x-y that forms a cycle, avoid using the tree-parent edge
                    if parent[x] == y or parent[y] == x:
                        continue

                    # reconstruct node paths from x and y to root
                    path_x = []
                    cur = x
                    while cur is not None:
                        path_x.append(cur)
                        cur = parent[cur]
                    path_y = []
                    cur = y
                    while cur is not None:
                        path_y.append(cur)
                        cur = parent[cur]

                    # find LCA
                    set_x = set(path_x)
                    lca = None
                    for node in path_y:
                        if node in set_x:
                            lca = node
                            break
                    if lca is None:
                        continue

                    # build node cycle: x -> ... -> lca, then lca -> ... -> y, then edge y-x
                    # nodes along x to lca
                    nodes_x_to_lca = []
                    cur = x
                    while cur != lca:
                        nodes_x_to_lca.append(cur)
                        cur = parent[cur]
                    nodes_x_to_lca.append(lca)

                    # nodes along y to lca (excluding lca), then reverse to go lca -> ... -> y
                    nodes_y_to_lca = []
                    cur = y
                    while cur != lca:
                        nodes_y_to_lca.append(cur)
                        cur = parent[cur]
                    # cycle nodes in order:
                    cycle_nodes = nodes_x_to_lca + list(reversed(nodes_y_to_lca))

                    # length check (cycle edge count = len(cycle_nodes)-1 + 1)
                    cycle_len = len(cycle_nodes)  # because we add closing edge
                    if cycle_len > max_len:
                        continue

                    # convert node cycle to edge list using parent_edge for tree edges + add closing edge
                    edges = []
                    # along cycle_nodes consecutive pairs
                    for a, b in zip(cycle_nodes[:-1], cycle_nodes[1:]):
                        # This edge is on the BFS tree somewhere; retrieve by walking from deeper node
                        # We stored parent_edge[child] = (parent, child, key)
                        # Need to pick correct direction: find which is parent of which
                        if parent[a] == b:
                            # a is child, b is parent
                            e = parent_edge[a]
                        elif parent[b] == a:
                            e = parent_edge[b]
                        else:
                            # In rare cases, consecutive nodes could be connected by non-tree adjacency
                            # Pick any key
                            keys_ab = list(G[a][b].keys())
                            e = (a, b, keys_ab[0])
                        edges.append(e)

                    # closing edge between last node and first node: (cycle_nodes[-1], cycle_nodes[0])
                    a = cycle_nodes[-1]
                    b = cycle_nodes[0]
                    keys_ab = list(G[a][b].keys())
                    if not keys_ab:
                        continue
                    edges.append((a, b, keys_ab[0]))

                    return edges

    raise RuntimeError("Failed to find a short cycle; graph might be tiny/degenerate.")


def weakly_fundamental_cycle_basis_GeneCS(G_input, rng=None):
    """
    Freedman–Hastings Algorithm A (randomized).
    Input: any graph (can be MultiGraph w/ self-loops + multi-edges).
    Output: list of cycles; each cycle is list of edges (u,v,key).
    """

    def _add_cycle_load(cycle_edges):
        for (u, v, k) in cycle_edges:
            eid = canon_edge(u, v, k)
            edge_load[eid] = edge_load.get(eid, 0) + 1

    rng = rng or random.Random()
    G = _as_multigraph_GeneCS(G_input)

    virt_id = count(0)               # unique IDs for virtual edges
    virt_map = {}                    # virt_key -> (e1, e2) in original graph for expansion

    edge_load = {}
    for u, v, k in G.edges(keys=True):
        edge_load[canon_edge(u, v, k)] = 0

    def expand_cycle_edges_GeneCS(cycle_edges):
        """Replace any virtual edge by its original 2-edge path, recursively if needed."""
        out = []
        for (u, v, k) in cycle_edges:
            if isinstance(k, tuple) and k and k[0] == "virt" and k in virt_map:
                e1, e2 = virt_map[k]
                out.extend(expand_cycle_edges_GeneCS([e1, e2]))
            else:
                out.append((u, v, k))
        return out

    def A(Gcur):
        # Base case
        if Gcur.number_of_edges() == 0:
            return []
        
        # print("in the A function")

        # 1) degree-1 vertex
        v1 = _find_degree_one_vertex_GeneCS(Gcur)
        if v1 is not None:
            # print("Found degree-1 vertex:", v1)
            Gnext = Gcur.copy()
            Gnext.remove_node(v1)
            return A(Gnext)

        # 2) degree-2 vertex
        v2 = _find_degree_two_vertex_GeneCS(Gcur)
        if v2 is not None:
            # print("Found degree-2 vertex:", v2)
            loop = _find_self_loop_edge_at_GeneCS(Gcur, v2)
            if loop is not None:
                # Case A: self-loop at v2
                # print("Found self-loop at degree-2 vertex:", loop)
                _add_cycle_load([loop])
                Gnext = Gcur.copy()
                u, v, k = loop
                Gnext.remove_edge(u, v, key=k)
                return [ [loop] ] + A(Gnext)

            # Case B: contract v2 with its two incident (non-loop) edges
            inc = _incident_edges_excluding_loops_GeneCS(Gcur, v2)
            if len(inc) != 2:
                raise RuntimeError(f"degree-2 vertex {v2} did not have exactly 2 non-loop incidences")

            e1, e2 = inc[0], inc[1]
            x = _other_endpoint_GeneCS(e1, v2)
            y = _other_endpoint_GeneCS(e2, v2)

            Gnext = Gcur.copy()
            # remove v2 (and its edges)
            Gnext.remove_node(v2)

            # add virtual edge (x,y)
            kid = next(virt_id)
            virt_key = ("virt", v2, kid)
            Gnext.add_edge(x, y, key=virt_key, virt=True)

            L1 = edge_load.get(canon_edge(*e1), 0)
            L2 = edge_load.get(canon_edge(*e2), 0)
            edge_load[canon_edge(x, y, virt_key)] = max(L1, L2)

            # record expansion: virtual edge -> original two edges
            virt_map[virt_key] = (e1, e2)

            basis = A(Gnext)

            # expand virtual edges in returned cycles
            basis_expanded = []
            for cyc in basis:
                basis_expanded.append(expand_cycle_edges_GeneCS(cyc))
            return basis_expanded

        # print("All vertices have degree >= 3, finding short cycle...")
        # 3) min degree >= 3: find short cycle, remove random edge
        V = max(Gcur.number_of_nodes(), 2)
        max_len = 2 * math.ceil(math.log2(V)) + 1

        C = _find_short_cycle_multigraph_GeneCS(Gcur, max_len=max_len, rng=rng, edge_load=edge_load)
        _add_cycle_load(C)
        e = max(C, key=lambda ee: edge_load.get(canon_edge(*ee), 0))

        Gnext = Gcur.copy()
        u, v, k = e
        if Gnext.has_edge(u, v, key=k):
            Gnext.remove_edge(u, v, key=k)
        else:
            # undirected: try swapped endpoints
            Gnext.remove_edge(v, u, key=k)

        return [C] + A(Gnext)

    # run
    return A(G)

def edge_load_GeneCS(cycle_basis):
    """
    cycle_basis: List[List[Edge]]
    Edge: (u, v, label)
    """
    load = defaultdict(int)

    for cycle in cycle_basis:
        for edge in cycle:
            load[edge] += 1

    return dict(load)

def spanning_forest_GeneCS(G_base, cycle_basis):
    """
    G_base: networkx.MultiGraph or MultiDiGraph
    cycle_basis: List[List[(u, v, key)]]
    """

    # 1. copy graph
    G_forest = copy.deepcopy(G_base)

    # Baseline: component count must NOT increase beyond this
    base_components = nx.number_connected_components(G_forest)

    # 2. compute load
    load = defaultdict(int)
    for cycle in cycle_basis:
        for edge in cycle:
            load[edge] += 1

    # 3. process each cycle
    for cycle in cycle_basis:
        # candidate edges: in this cycle AND still in G_forest
        candidates = []
        for (u, v, k) in cycle:
            if G_forest.has_edge(u, v, k):
                candidates.append((u, v, k))

        if not candidates:
            continue

        # Try removing edges in descending load order
        candidates_sorted = sorted(candidates, key=lambda e: load[e], reverse=True)

        # remove it
        for (u, v, k) in candidates_sorted:
            # Remove temporarily
            G_forest.remove_edge(u, v, k)

            # Check if component count increased
            if nx.number_connected_components(G_forest) > base_components:
                # Undo removal (this edge is a bridge/cut edge in current graph)
                G_forest.add_edge(u, v, key=k)
                continue

            # Accept removal
            break

    return G_forest

def dynamic_cycle_basis_GeneCS(
    B,
    T,
    e,
    edge_load,
    rng=random.Random(),
    load_aware=True,
):
    """
    Dynamic maintenance of cycle basis and spanning forest
    after adding a single edge e=(u,v,key).

    Parameters
    ----------
    B : list
        Current cycle basis (list of cycles, each cycle is list of edges).
    T : nx.MultiGraph
        Current spanning forest.
    e : tuple
        New edge (u, v, key).
    edge_load : dict
        edge -> load (number of cycles containing the edge).
    rng : random.Random
    load_aware : bool
        Whether to remove edge from path using max-load rule.

    Returns
    -------
    B_new : list
    T_new : nx.MultiGraph
    """
    u, v, k = e

    # Case 1: u and v are NOT connected in T
    if not nx.has_path(T, u, v):
        T.add_edge(u, v, key=k)
        # ensure edge has load 0
        eid = canon_edge(u, v, k)
        edge_load.setdefault(eid, 0)
        return B, T, edge_load

    # Case 2: u and v already connected -> fundamental cycle
    path_nodes = nx.shortest_path(T, u, v)

    # extract edges along the path
    path_edges = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        keys = list(T[a][b].keys())
        if not keys:
            raise RuntimeError("Invalid spanning forest structure.")
        path_edges.append((a, b, keys[0]))

    # fundamental cycle
    C_e = path_edges + [e]
    B_new = B + [C_e]

    # update edge loads
    for ee in C_e:
        eid = canon_edge(*ee)
        edge_load[eid] = edge_load.get(eid, 0) + 1

    # choose which edge to remove from the path
    if load_aware:
        e_remove = max(
            path_edges,
            key=lambda ee: edge_load.get(canon_edge(*ee), 0)
        )
    else:
        e_remove = rng.choice(path_edges)

    # update forest
    T.add_edge(u, v, key=k)
    x, y, kk = e_remove
    if T.has_edge(x, y, key=kk):
        T.remove_edge(x, y, key=kk)
    else:
        T.remove_edge(y, x, key=kk)
    return B_new, T, edge_load

# static cycle basis partition function
def static_partition_GeneCS(cycle_basis, max_edge_degree=1):
    """
    Partition cycles into groups.

    If max_edge_degree = 1:
        cycles in each group share no common edges (original behavior).

    If max_edge_degree > 1:
        each edge may appear in at most max_edge_degree cycles within a group.

    Parameters
    ----------
    cycle_basis : list
        List of cycles; each cycle is list of (u,v,key).
    max_edge_degree : int
        Maximum allowed usage of an edge inside one group.

    Returns
    -------
    groups : list[list[int]]
    group_edge_loads : list[dict]
        For each group: edge -> usage count in that group.
    """

    # Precompute edge sets per cycle
    cycle_edge_sets = [
        {canon_edge(u, v, k) for u, v, k in cyc}
        for cyc in cycle_basis
    ]

    remaining = list(range(len(cycle_basis)))
    groups = []
    group_edge_loads = []

    while remaining:
        Ri = []
        Ri_load = {}   # edge -> count in this group

        # reverse greedy (same structure as your original)
        for idx in reversed(remaining):
            es = cycle_edge_sets[idx]

            # check feasibility under edge-degree constraint
            feasible = True
            for e in es:
                if Ri_load.get(e, 0) >= max_edge_degree:
                    feasible = False
                    break

            if feasible:
                Ri.append(idx)
                for e in es:
                    Ri_load[e] = Ri_load.get(e, 0) + 1

        remaining = [i for i in remaining if i not in Ri]

        groups.append(Ri)
        group_edge_loads.append(Ri_load)

    return groups, group_edge_loads

def dynamic_partition_GeneCS(
    cycle_basis,
    groups,
    group_edge_loads,
    max_edge_degree=1,
):
    """
    Dynamically insert the LAST cycle in cycle_basis into partition groups.

    If max_edge_degree = 1:
        non-overlap partition (original behavior).

    If max_edge_degree > 1:
        each edge may appear in at most max_edge_degree cycles
        inside a single group.

    Parameters
    ----------
    cycle_basis : list
        Full cycle basis AFTER appending the new cycle.
    groups : list[list[int]]
        Existing partition groups.
    group_edge_loads : list[dict]
        For each group: edge -> usage count.
    max_edge_degree : int
        Maximum allowed usage of an edge in a group.

    Returns
    -------
    groups : updated groups
    group_edge_loads : updated loads
    """

    new_idx = len(cycle_basis) - 1
    new_cycle = cycle_basis[new_idx]

    new_edges = {canon_edge(u, v, k) for (u, v, k) in new_cycle}

    # Try to place into existing groups
    for i in range(len(groups)):
        feasible = True

        for e in new_edges:
            if group_edge_loads[i].get(e, 0) >= max_edge_degree:
                feasible = False
                break

        if feasible:
            groups[i].append(new_idx)
            for e in new_edges:
                group_edge_loads[i][e] = group_edge_loads[i].get(e, 0) + 1
            return groups, group_edge_loads

    # If not placed, create new group
    groups.append([new_idx])
    new_load = {}
    for e in new_edges:
        new_load[e] = 1
    group_edge_loads.append(new_load)

    return groups, group_edge_loads

def bfs_spanning_tree(G, root=None):
    """
    Build a BFS spanning tree from a MultiGraph, preserving edge keys.

    Parameters:
        G : nx.MultiGraph
            Input graph (edges have keys)
        root : optional node
            Starting node for BFS

    Returns:
        T : nx.MultiGraph
            Spanning tree with edges (u, v, key)
    """
    if root is None:
        root = next(iter(G.nodes))

    visited = set([root])
    queue = deque([root])

    tree_edges = []

    while queue:
        u = queue.popleft()

        # iterate neighbors with edge keys
        for v, edge_dict in G.adj[u].items():
            if v not in visited:
                # pick ANY one edge key between u and v
                key = next(iter(edge_dict.keys()))

                visited.add(v)
                queue.append(v)

                tree_edges.append((u, v, key))

    # build tree
    T = nx.MultiGraph()
    T.add_nodes_from(G.nodes)

    for u, v, key in tree_edges:
        T.add_edge(u, v, key=key)

    return T

def congestion_aware_expander_GeneCS(
    G0,
    seed=0,
    edge_key="exp",
    start_from="max",   # "max" or "minplus"
    max_edge_degree=1,   # for cycle partitioning
    verbose=False,
):
    """
    G0 is NOT modified.
    """
    G = G0.copy()
    rng = random.Random(seed)
    history = []
    step = 0
    n = G.number_of_nodes()
    max_target_degree = n - 1

    eps = 1e-5
    cheeger_lb = max(cheeger_lower_bound(G), eps)
    # calculate the layer need to boost Cheeger constant after perfect matching
    layer_Cheeger = math.ceil(1 / cheeger_lb)

    # calculate the layer need to decongest cycles after perfect matching
    cycle_basis = weakly_fundamental_cycle_basis_GeneCS(G, rng)
    cycle_origin = len(cycle_basis)
    # G_forest = spanning_forest_GeneCS(G, cycle_basis)
    G_forest = bfs_spanning_tree(G)
    is_conn = nx.is_connected(G_forest)
    edge_load = edge_load_GeneCS(cycle_basis)
    groups, group_edge_loads = static_partition_GeneCS(cycle_basis, max_edge_degree=max_edge_degree)
    layer_cycle = len(groups)

    # initialize target degree
    degs = dict(G.degree())
    if start_from == "max":
        target_degree = max(degs.values())
    elif start_from == "minplus":
        target_degree = min(degs.values()) + 1
    else:
        raise ValueError("start_from must be 'max' or 'minplus'.")

    while layer_cycle < layer_Cheeger:

        if verbose:
            print(f"[step {step}], layer_cycle={layer_cycle}, layer_Cheeger={layer_Cheeger}")

         # Periodically rebuild the spanning forest (every k steps)
        # to maintain a low-diameter structure (≈ O(log V)).
        if step > 0 and step % 50 == 0:
            G_forest = bfs_spanning_tree(G)

        G, result = add_edge_GeneCS(G, target_degree, rng=rng, edge_key=edge_key)
        
        # ---- key change: if no edge can be added for this target_degree,
        #      we end this round and increase the target_degree ----
        if result is None:
            if target_degree >= max_target_degree:
                # cannot increase target_degree any further in a simple graph
                if verbose:
                    print(
                        f"[stop] target_degree reached max {max_target_degree}, "
                        "no more edges possible."
                    )
                break

            target_degree += 1
            continue

        u, v, key = result

        step += 1
        
        eps = 1e-5
        cheeger_lb = max(cheeger_lower_bound(G), eps)
        layer_Cheeger = math.ceil(1 / cheeger_lb)

        cycle_basis, G_forest, edge_load = dynamic_cycle_basis_GeneCS(cycle_basis, G_forest, result, edge_load)
        groups, group_edge_loads = dynamic_partition_GeneCS(cycle_basis, groups, group_edge_loads, max_edge_degree=max_edge_degree)
        layer_cycle = len(groups)

        history.append({
            "step": step,
            "u": u,
            "v": v,
            "edge_key": key,
            "target_degree": target_degree,
            "layer_cycle": layer_cycle,
            "layer_Cheeger": layer_Cheeger,
        })

    max_load = max(max(d.values()) for d in group_edge_loads)

    return G, history, cycle_basis, groups, max_load

# def congestion_aware_expander_GeneCS_alt(
#     G0,
#     seed=0,
#     edge_key="exp",
#     start_from="max",   # "max" or "minplus"
#     max_edge_degree=1,   # for cycle partitioning
#     verbose=False,
# ):
#     """
#     G0 is NOT modified.
#     """
#     G = G0.copy()
#     rng = random.Random(seed)
#     history = []
#     step = 0
#     n = G.number_of_nodes()
#     max_target_degree = n - 1

#     # calculate the layer need to boost Cheeger constant after perfect matching
#     layer_Cheeger = math.ceil(1 / cheeger_lower_bound(G))

#     # calculate the layer need to decongest cycles after perfect matching
#     cycle_basis = weakly_fundamental_cycle_basis_GeneCS(G, rng)
#     cycle_origin = len(cycle_basis)
#     G_forest = spanning_forest_GeneCS(G, cycle_basis)
#     is_conn = nx.is_connected(G_forest)
#     edge_load = edge_load_GeneCS(cycle_basis)
#     groups, group_edge_loads = static_partition_GeneCS(cycle_basis, max_edge_degree=max_edge_degree)
#     layer_cycle = len(groups)

#     # initialize target degree
#     degs = dict(G.degree())
#     if start_from == "max":
#         target_degree = max(degs.values())
#     elif start_from == "minplus":
#         target_degree = min(degs.values()) + 1
#     else:
#         raise ValueError("start_from must be 'max' or 'minplus'.")

#     while layer_cycle < layer_Cheeger:

#         if verbose:
#             print(f"[step {step}], layer_cycle={layer_cycle}, layer_Cheeger={layer_Cheeger}")

#         G, result = add_edge_GeneCS(G, target_degree, rng=rng, edge_key=edge_key)
        
#         # ---- key change: if no edge can be added for this target_degree,
#         #      we end this round and increase the target_degree ----
#         if result is None:
#             if target_degree >= max_target_degree:
#                 # cannot increase target_degree any further in a simple graph
#                 if verbose:
#                     print(
#                         f"[stop] target_degree reached max {max_target_degree}, "
#                         "no more edges possible."
#                     )
#                 break

#             target_degree += 1
#             continue

#         u, v, key = result

#         step += 1
#         print(step)
        
#         layer_Cheeger = math.ceil(1 / cheeger_lower_bound(G))

#         cycle_basis = weakly_fundamental_cycle_basis_GeneCS(G, rng)
#         groups, group_edge_loads = static_partition_GeneCS(cycle_basis, max_edge_degree=max_edge_degree)
#         layer_cycle = len(groups)

#         history.append({
#             "step": step,
#             "u": u,
#             "v": v,
#             "edge_key": key,
#             "target_degree": target_degree,
#             "layer_cycle": layer_cycle,
#             "layer_Cheeger": layer_Cheeger,
#         })

#     max_load = max(max(d.values()) for d in group_edge_loads)

#     return G, history, cycle_basis, groups, max_load

def congestion_aware_expander_sweep_GeneCS(
    G0,
    seeds_num,
    edge_key="exp",
    max_edge_degree=1,
    verbose=False,
):
    """
    Try multiple random seeds and return the graph that reaches the balance of expansion and congestion with the fewest added edges.

    Parameters
    ----------
    G0 : nx.MultiGraph
        Initial graph (will NOT be modified).
    max_seeds : int
        Number of random seeds to try (0, 1, ..., max_seeds-1).
    edge_key : str
        Key used for expander edges.
    verbose : bool
        Print progress information.

    Returns
    -------
    best_G : nx.MultiGraph
        Graph achieving the threshold with the fewest added edges.
    best_history : list
        Edge-addition history for best_G.
    best_seed : int
        Random seed that achieved the best result.
    """
    best_G = None
    best_history = None
    best_cycle_basis = None
    best_groups = None
    best_seed = None
    best_layers = None
    best_load = None
    best_qubit = None

    count = 0

    for seed in range(seeds_num):

        if verbose:
            print(f"\n[seed {seed}] start")

        # IMPORTANT: always start from a fresh copy
        G = G0.copy()

        try:
            G_base, history, cycle_basis, groups, max_load = congestion_aware_expander_GeneCS(G, seed=seed, edge_key=edge_key, start_from="max", max_edge_degree=max_edge_degree)
        except Exception:
            continue
        layers = max(math.ceil(1 / cheeger_lower_bound(G_base)), len(groups))

        num_edges = len(history)

        if verbose:
            print(
                f"[seed {seed}] edges added = {num_edges}"
            )

        count = count + 1

        if best_layers is None or layers < best_layers or (layers == best_layers and max_load < best_load and count < 1000) or (layers == best_layers and max_load <= best_load and num_edges < best_qubit):

            best_G = G_base
            best_history = history
            best_cycle_basis = cycle_basis
            best_groups = groups
            best_seed = seed
            best_layers = layers
            best_load = max_load
            best_qubit = num_edges
            if verbose:
                print(
                    f"[seed {seed}] NEW BEST ({best_layers} layers)"
                )

    return best_G, best_history, best_cycle_basis, best_groups

##### Step 4: Graph thickening #####

def thicken_graph(G_base: nx.MultiGraph, L: int):
    """
    Construct an L-layer thickening of G_base.

    Rules:
    - Layer 0 nodes: same name + same metadata as G_base
    - Layer l>=1 nodes: renamed as f"{v}_{l}", NO metadata
    - Intra-layer edges: copied from G_base, metadata = {"kind": "thickening"}
    - Inter-layer edges: connect v_l to v_{l+1}, metadata = {"kind": "thickening"}
    """
    G_thick = nx.MultiGraph()

    def node_at_layer(v, l):
        return v if l == 0 else f"{v}_{l}"

   # ---- layer 0: exact copy ----
    for v, vdata in G_base.nodes(data=True):
        G_thick.add_node(v, **copy.deepcopy(vdata))

    for u, v, k, edata in G_base.edges(keys=True, data=True):
        G_thick.add_edge(
            u, v, key=k, **copy.deepcopy(edata)
        )

    # ---- layers l >= 1: nodes only ----
    for l in range(1, L):
        for v in G_base.nodes():
            G_thick.add_node(node_at_layer(v, l))

    # ---- intra-layer edges for l >= 1 ----
    for l in range(1, L):
        for u, v, k in G_base.edges(keys=True):
            G_thick.add_edge(
                node_at_layer(u, l),
                node_at_layer(v, l),
                key=(k, l),
                kind="thickening"
            )

    # ---- add inter-layer edges ----
    for l in range(L - 1):
        for v in G_base.nodes():
            G_thick.add_edge(
                node_at_layer(v, l),
                node_at_layer(v, l + 1),
                key=(v, l+1),
                kind="thickening"
            )

    return G_thick

##### Step 5: Cellulation #####

def reorder_edge_cycle_GeneCS(edge_cycle):
    """
    Reorder an unordered list of edges (u,v,key)
    that form a simple cycle into traversal order.
    """
    # vertex -> incident edges
    adj = {}
    for e in edge_cycle:
        u, v, _ = e
        adj.setdefault(u, []).append(e)
        adj.setdefault(v, []).append(e)

    # sanity: degree 2 at each vertex
    for v, inc in adj.items():
        if len(inc) != 2:
            raise ValueError(f"Vertex {v} has degree {len(inc)} in cycle")

    start_edge = edge_cycle[0]
    u0, v0, _ = start_edge

    ordered = [start_edge]
    used = {start_edge}

    curr_vertex = v0
    prev_vertex = u0

    while True:
        # find unused edge incident to curr_vertex
        next_edge = None
        for e in adj[curr_vertex]:
            if e not in used:
                next_edge = e
                break

        if next_edge is None:
            break

        used.add(next_edge)
        ordered.append(next_edge)

        a, b, _ = next_edge
        next_vertex = b if a == curr_vertex else a

        prev_vertex, curr_vertex = curr_vertex, next_vertex

        if curr_vertex == u0:
            break

    if len(ordered) != len(edge_cycle):
        raise RuntimeError("Failed to reorder edge cycle")

    return ordered

def edge_cycle_to_vertex_cycle_GeneCS(ordered_edge_cycle):
    """
    Convert an ordered edge-cycle into an ordered vertex list.
    """
    u0, v0, _ = ordered_edge_cycle[0]
    verts = [u0, v0]

    for (u, v, _) in ordered_edge_cycle[1:]:
        last = verts[-1]
        if u == last:
            verts.append(v)
        elif v == last:
            verts.append(u)
        else:
            raise RuntimeError("Edge order inconsistent")

    # remove duplicate closing vertex
    if verts[0] == verts[-1]:
        verts.pop()

    return verts

def cellulation_edges_from_vertex_cycle_GeneCS(vertex_cycle):
    """
    Given an ordered vertex cycle [v0, v1, ..., v_{N-1}],
    generate cellulation edges according to:
      (1, N-1), (N-1, 2), (2, N-2), ...
    Returns list of unordered vertex pairs.
    """

    N = len(vertex_cycle)
    if N <= 3:
        return []

    edges = []
    i = 1
    j = N - 1

    while i < j-1:
        edges.append((vertex_cycle[i], vertex_cycle[j]))
        i += 1
        if i < j-1:
            edges.append((vertex_cycle[j], vertex_cycle[i]))
            j -= 1

    return edges

def cellulation_triangle_indices_GeneCS(N):
    """
    Return triangle cycles (as index triples) induced by the cellulation rule:
      diagonals: (1,N-1), (N-1,2), (2,N-2), (N-2,3), ...
    Triangles returned in the corresponding order:
      (0,1,N-1), (1,N-1,2), (N-1,2,N-2), (2,N-2,3), ...

    For N<=3: []  (no cellulation)
    For N=4: [(0,1,3), (1,3,2)]  (two triangles)
    """
    if N <= 3:
        return []

    tris = []

    # Start with endpoints on the "left" and "right" side of the polygon
    left = 1
    right = N - 1

    # First diagonal (1, N-1) creates triangle (0,1,N-1)
    tris.append((0, left, right))

    # Now we alternately advance left and retreat right, creating triangles
    # that use the most recent diagonal endpoint pair plus the next vertex.
    # This matches: (1,N-1),(N-1,2),(2,N-2),(N-2,3),...
    while left + 1 < right:
        # step: diagonal (right, left+1)  -> triangle (left, right, left+1)
        left2 = left + 1
        tris.append((left, right, left2))
        left = left2
        if left + 1 >= right:
            break

        # step: diagonal (left, right-1) -> triangle (right, left, right-1)
        right2 = right - 1
        tris.append((right, left, right2))
        right = right2

    return tris

def lift_node_GeneCS(u, l):
    return u if l == 0 else f"{u}_{l}"

def find_edge_key_GeneCS(G, u, v):
    """
    Given endpoints u, v in a MultiGraph G,
    return the key of ANY edge between u and v.
    """
    return next(iter(G[u][v].keys()))

def find_edge_triangle_GeneCS(cycle_list, triangle, l, idx):
    """
    cycle_list: list of (u, v, key), undirected
    triangle: (A, B, C)

    Returns:
        list of length 3, corresponding to edges:
        [(A,B), (B,C), (C,A)]

        Each element is either:
          - (u, v, key)          if edge exists in cycle_list
          - (u, v, None)         if edge does NOT exist in cycle_list
    """
    A, B, C = triangle

    # build lookup from unordered pair -> (u,v,key)
    lookup = {}
    for u, v, k in cycle_list:
        lookup[frozenset((u, v))] = (u, v, k)

    result = []
    for u, v in [(A, B), (B, C), (C, A)]:
        e = lookup.get(frozenset((u, v)))
        if e is not None:
            u, v, k = e
            k_l = k if l == 0 else (k, l)
            result.append((lift_node_GeneCS(u, l), lift_node_GeneCS(v, l), k_l))
        else:
            # the missing edge is in cellulation
            result.append((lift_node_GeneCS(u, l), lift_node_GeneCS(v, l), f"cellulation_{l}_{idx}"))

    return result

def split_list_GeneCS(input_list, d):
    """
    Split a list using a repeating pattern:
    - Take d elements → put into second_list
    - Take 1 element  → put into first_list
    - Repeat until all elements are processed
    """

    first_list = []
    second_list = []

    index = 0
    length = len(input_list)

    while index < length:
        # Take d elements and put them into second_list
        second_list.extend(input_list[index:index + d])
        index += d

        if index >= length:
            break

        # Take 1 element and put it into first_list
        first_list.append(input_list[index])
        index += 1

    return first_list, second_list

def normalize_edge_GeneCS(u, v):
    """Make edge direction-independent"""
    return tuple(sorted((u, v)))


def merge_triangles_by_redundant_edges_GeneCS(tri_cycles, redundant_edges, l):
    """
    Merge triangles if they share redundant edges.
    """
    lifted_redundant = []
    for u, v in redundant_edges:
        u_l = lift_node_GeneCS(u, l)
        v_l = lift_node_GeneCS(v, l)
        lifted_redundant.append((u_l, v_l))


    # Normalize redundant edges
    redundant_set = set(normalize_edge_GeneCS(u, v) for u, v in lifted_redundant)

    # Map edge -> triangles containing it
    edge_to_triangles = defaultdict(list)

    for i, tri in enumerate(tri_cycles):
        for (u, v, _) in tri:
            edge_to_triangles[normalize_edge_GeneCS(u, v)].append(i)

    # Build adjacency between triangles
    adjacency = defaultdict(set)

    for edge in redundant_set:
        triangles = edge_to_triangles.get(edge, [])
        for i in range(len(triangles)):
            for j in range(i + 1, len(triangles)):
                t1 = triangles[i]
                t2 = triangles[j]
                adjacency[t1].add(t2)
                adjacency[t2].add(t1)

    # Find connected components
    visited = set()
    components = []

    for i in range(len(tri_cycles)):
        if i not in visited:
            stack = [i]
            component = []

            while stack:
                node = stack.pop()
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)
                stack.extend(adjacency[node])

            components.append(component)

    # Merge triangles inside each component
    merged_polygons = []

    for component in components:
        edge_count = defaultdict(int)

        for tri_index in component:
            for edge in tri_cycles[tri_index]:
                u, v, label = edge
                key = normalize_edge_GeneCS(u, v)
                edge_count[(key, label)] += 1

        # Keep edges that appear only once
        merged_edges = [
            (u, v, label)
            for ((u, v), label), count in edge_count.items()
            if count == 1
        ]

        merged_polygons.append(merged_edges)

    return merged_polygons

def cellulation_GeneCS(G_thick, cycle_groups, cycle_basis, d=3):
    """
    For each layer l and each cycle in that layer:
      - reorder edge-cycle
      - convert to vertex-cycle
      - generate cellulation edges
      - add them to G_thick with kind="cellulation"
    """

    new_cycle_basis = []

    count = 0
    for l, cycles in enumerate(cycle_groups):
        for idx in cycles:
            edge_cycle = cycle_basis[idx]
            # print("\\")
            # print(f"Processing layer {l}, cycle {edge_cycle}")

            # 1. reorder edges
            ordered_edges = reorder_edge_cycle_GeneCS(edge_cycle)
            # print("ordered edges", ordered_edges)

            # 2. edges -> vertices
            vertex_cycle = edge_cycle_to_vertex_cycle_GeneCS(ordered_edges)
            # print("vertex cycle", vertex_cycle)

            # 3. generate cellulation edges
            cell_edges = cellulation_edges_from_vertex_cycle_GeneCS(vertex_cycle)
            cell_edges, redundant_edges = split_list_GeneCS(cell_edges, d-3)


            # -------------------------------
            # Case A: no cellulation needed
            # -------------------------------
            if not cell_edges:
                lifted_cycle = []
                for u, v, k in ordered_edges:
                    u_l = lift_node_GeneCS(u, l)
                    v_l = lift_node_GeneCS(v, l)
                    if l == 0:
                        k_l = k
                    else:
                        k_l = (k, l)
                    lifted_cycle.append((u_l, v_l, k_l))
                new_cycle_basis.append(lifted_cycle)
                continue

            # --------------------------------
            # Case B: triangulate the cycle
            # --------------------------------

            # (i) add cellulation edges to graph
            for u, v in cell_edges:
                count += 1
                u_l = lift_node_GeneCS(u, l)
                v_l = lift_node_GeneCS(v, l)

                # allow parallel edges; key is fixed string
                G_thick.add_edge(u_l, v_l, key=f"cellulation_{l}_{idx}")
            
            # (ii) build triangle cycles
            # We assume cellulation_edges_from_vertex_cycle()
            # returns diagonals in the order described in the paper:
            # (1, N-1), (N-1, 2), (2, N-2), ...
            # Reconstruct triangles explicitly
            # The polygon is vertex_cycle = [v0, v1, ..., v_{N-1}]
            poly = list(vertex_cycle)
            idx_triangles = cellulation_triangle_indices_GeneCS(len(poly))

            new_cycle_temp = []

            for triangle in idx_triangles:
                
                i,j,k = triangle

                a = poly[i]
                b = poly[j]
                c = poly[k]
                triangle = (a, b, c)

                tri_cycle = find_edge_triangle_GeneCS(ordered_edges, triangle, l, idx)
                new_cycle_temp.append(tri_cycle)

            merged_polygons = merge_triangles_by_redundant_edges_GeneCS(new_cycle_temp, redundant_edges, l)
            new_cycle_basis = new_cycle_basis + merged_polygons

    # print("number of edges added:", count)

    return new_cycle_basis


def add_interlayer_rectangle_cycles_GeneCS(
    G_thick,
    G_base,
    L,
    new_cycle_basis
):
    """
    Add rectangle cycles induced by inter-layer edges

    Each rectangle cycle is:
      (u_l, v_l) -> (v_l, v_{l+1}) -> (v_{l+1}, u_{l+1}) -> (u_{l+1}, u_l)

    Parameters
    ----------
    G_thick : nx.MultiGraph
    G_base : nx.MultiGraph
    L : int
        number of layers
    new_cycle_basis : list
        list of edge-based cycles; will be appended to
    """

    # 1. edges of base graph
    base_edges = list(G_base.edges(keys=True))

    # 2. iterate over adjacent layers
    count = 0
    for l in range(L - 1):
        for u, v, k in base_edges:
            count += 1
            # lift nodes
            u_l  = u if l == 0 else f"{u}_{l}"
            v_l  = v if l == 0 else f"{v}_{l}"
            u_lp = f"{u}_{l+1}"
            v_lp = f"{v}_{l+1}"

            # find keys for the 4 edges
            k_v_vert = find_edge_key_GeneCS(G_thick, v_l, v_lp)
            k_u_vert = find_edge_key_GeneCS(G_thick, u_lp, u_l)

            k_l = k if l == 0 else (k, l)

            rectangle_cycle = [
                (u_l,  v_l,  k_l),
                (v_l,  v_lp, k_v_vert),
                (v_lp, u_lp, (k, l+1)),
                (u_lp, u_l,  k_u_vert),
            ]

            new_cycle_basis.append(rectangle_cycle)

    return new_cycle_basis

##### Main Synthesis Function #####

def innercode_synthesis(file_name, logical_type, logical_idx, qubit_degree, check_degree, sample_num, benchmark_path= './benchmark', saving_path = './bin'):

    t0 = time.time()
    # load the code
    with open(f"{benchmark_path}/{file_name}.pkl", "rb") as f:
        data = pickle.load(f)

    check_dir_list = ["deform_data", "deform_code", "deform_graph"]
    for dir_ in check_dir_list:
        check_dir = os.path.join(saving_path, dir_)
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

    output_path = saving_path + '/' + 'deform_data' + '/' + f'{file_name}_{logical_type}_{logical_idx}' + '.txt'
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
        logical_op = lz[logical_idx[0], :].copy()
        other_h = hx
    elif logical_type == "X":
        logical_op = lx[logical_idx[0], :].copy()
        other_h = hz

    for idx in logical_idx[1:]:
        if logical_type == "Z":
            logical_op = logical_op + lz[idx, :]
        else:
            logical_op = logical_op + lx[idx, :]

    logical_op.data %= 2
    logical_op.eliminate_zeros()

    # Step 1: Construct vertex of graph and the bijection fuction
    G, qubit_to_vertex = create_graph_GeneCS(logical_op)

    # Step 2: Construct the perfect matching
    G, perfect_matchings = construct_perfect_matching_GeneCS(G, qubit_to_vertex, logical_op, other_h)

    # Step 3: Congestion-aware expansion of graph
    # 2 reserved for vertex checks, 2 reserved for perfect matching or thickening per edge
    G_base, _, cycle_basis, groups = congestion_aware_expander_sweep_GeneCS(G, sample_num, edge_key="exp", max_edge_degree=qubit_degree-4)

    # Step 4: Thickening
    beta = cheeger_lower_bound(G_base)
    if abs(1 - beta) < 1e-6:
        beta = 1.0
    L = max(math.ceil(1 / beta), len(groups))
    print("Thickening layer =", L, file=output_file)
    print(f"Redundancy = {(L - min(math.ceil(1 / beta), len(groups))) / L * 100:.1f} %", file=output_file)
    G_thick = thicken_graph(G_base, L)

    # Step 5: Cellulation
    new_cycle_basis = cellulation_GeneCS(G_thick, groups, cycle_basis, d=check_degree)
    new_cycle_basis = add_interlayer_rectangle_cycles_GeneCS(G_thick, G_base, L, new_cycle_basis)

    print("\n", file=output_file)
    new_qubit_num = G_thick.number_of_edges()
    new_check_num = len(new_cycle_basis) + G_thick.number_of_nodes()
    print("Number of new qubits:", G_thick.number_of_edges())
    print("Number of new qubits:", G_thick.number_of_edges(), file=output_file)
    print("Number of new chekcs:", len(new_cycle_basis) + G_thick.number_of_nodes())
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

    with open(f"{saving_path}/deform_graph/{file_name}_{logical_type}_{logical_idx}.pkl", "wb") as f:
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
    print("Max qubit degree:", qubit_degree)
    print("Max qubit degree:", qubit_degree, file=output_file)
    print("Max check degree:", check_degree)
    print("Max check degree:", check_degree, file=output_file)

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

    with open(f"{saving_path}/deform_code/{file_name}_{logical_type}_{logical_idx}.pkl", "wb") as f:
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

    df.to_csv(f'{saving_path}/output.csv', mode='a', header=not pd.io.common.file_exists(f'{saving_path}/output.csv'), index=False)


def crosscode_synthesis(file_name_1, file_name_2, logical_type_1, logical_type_2, logical_idx_1, logical_idx_2, qubit_degree_1, qubit_degree_2, check_degree_1, check_degree_2, sample_num, adapter_size, benchmark_path= './benchmark', saving_path = './bin'):
    
    if logical_type_1 != logical_type_2:
        print("Error, measuring different type logicals.")

    check_dir_list = ["deform_data", "deform_code", "deform_graph"]
    for dir_ in check_dir_list:
        check_dir = os.path.join(saving_path, dir_)
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

    output_path = saving_path + '/' + 'deform_data' + '/' + f'{file_name_1}_{logical_type_1}_{logical_idx_1}_AND_{file_name_2}_{logical_type_2}_{logical_idx_2}' + '.txt'
    output_file = open(file=output_path, mode="a", encoding='utf-8')

    t0 = time.time()
    ######################## Construct Ancilla System Graph for the First Code ########################

    # load the code
    with open(f"{benchmark_path}/{file_name_1}.pkl", "rb") as f:
        data_1 = pickle.load(f)

    qcode_1 = data_1["qcode"]

    with open(output_path, "w") as f:   
        with redirect_stdout(f), redirect_stderr(f):
            print("Testing original code 1...")
            qcode_1.test()
            print("\n")

    lz_1 = qcode_1.lz; lx_1 = qcode_1.lx
    hz_1 = qcode_1.hz.toarray(); hx_1 = qcode_1.hx.toarray()
    k = lz_1.shape[0]; n = lz_1.shape[1]

    if logical_type_1 == "Z":
        logical_op_1 = lz_1[logical_idx_1[0], :].copy()
        other_h_1 = hx_1
    elif logical_type_1 == "X":
        logical_op_1 = lx_1[logical_idx_1[0], :].copy()
        other_h_1 = hz_1

    for idx in logical_idx_1[1:]:
        if logical_type_1 == "Z":
            logical_op_1 = logical_op_1 + lz_1[idx, :]
        else:
            logical_op_1 = logical_op_1 + lx_1[idx, :]

    logical_op_1.data %= 2
    logical_op_1.eliminate_zeros()

    # Step 1: Construct vertex of graph and the bijection fuction
    G_1, qubit_to_vertex_1 = create_graph_GeneCS(logical_op_1)

    # Step 2: Construct the perfect matching
    G_1, perfect_matchings_1 = construct_perfect_matching_GeneCS(G_1, qubit_to_vertex_1, logical_op_1, other_h_1)

    # Step 3: Congestion-aware expansion of graph
    # 2 reserved for vertex checks, 2 reserved for perfect matching or thickening per edge
    G_base_1, _, cycle_basis_1, groups_1 = congestion_aware_expander_sweep_GeneCS(G_1, sample_num, edge_key="exp", max_edge_degree=qubit_degree_1-4)

    # Step 4: Thickening
    beta_1 = cheeger_lower_bound(G_base_1)
    if abs(1 - beta_1) < 1e-6:
        beta_1 = 1.0
    L_1 = max(math.ceil(1 / beta_1), len(groups_1))
    G_thick_1 = thicken_graph(G_base_1, L_1)

    # Step 5: Cellulation
    new_cycle_basis_1 = cellulation_GeneCS(G_thick_1, groups_1, cycle_basis_1, d=check_degree_1)
    new_cycle_basis_1 = add_interlayer_rectangle_cycles_GeneCS(G_thick_1, G_base_1, L_1, new_cycle_basis_1)

    data = {
        "G_thick": G_thick_1,
        "perfect_matching": perfect_matchings_1, 
        "cycle_basis": new_cycle_basis_1, 
        "HX": hx_1,
        "HZ": hz_1,
        "logical_type": logical_type_1,
        "logical_idx": logical_idx_1,
        "logical_op": logical_op_1,
    }

    with open(f"{saving_path}/deform_graph/{file_name_1}_merge_{logical_type_1}_{logical_idx_1}.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    ######################## Code Deformation for the First Code ########################

    nodes_1, edges_1 = find_d_nodes_tree(G_thick_1, adapter_size)
    G_tree_1, column_dict_1, row_dict_1 = incidence_matrix(nodes_1, edges_1)
    T_1,P_1 = skip_tree(G_tree_1)

    # Construct the parity check matrix for derformed code
    E_1 = G_thick_1.number_of_edges()
    n_1_ori = hz_1.shape[1]
    hz_1 = enlarge_parity_check(hz_1, E_1)
    hx_1 = enlarge_parity_check(hx_1, E_1)

    # add new vertex check
    if logical_type_1 == "Z":
        hz_1, vertex2check_1 = add_vertex_checks(hz_1, G_thick_1)
    elif logical_type_1 == "X":
        hx_1, vertex2check_1 = add_vertex_checks(hx_1, G_thick_1)

    edge2qubit_1 = edge2qubit(G_thick_1, n_1_ori)

    # modify original check involved in perfect matching 
    if logical_type_1 == "Z":
        hx_1 = add_perfect_matching_checks(hx_1, G_thick_1, perfect_matchings_1)
    elif logical_type_1 == "X":
        hz_1 = add_perfect_matching_checks(hz_1, G_thick_1, perfect_matchings_1)

    # add cycle checks to fix gauge degrees of freedom
    if logical_type_1 == "Z":
        hx_1 = append_cycle_basis_checks(G_thick_1, hx_1, new_cycle_basis_1, hx_1.shape[1])
    elif logical_type_1 == "X":
        hz_1 = append_cycle_basis_checks(G_thick_1, hz_1, new_cycle_basis_1, hz_1.shape[1])

    ######################## Construct Ancilla System Graph for the Second Code ########################

    # load the code
    with open(f"{benchmark_path}/{file_name_2}.pkl", "rb") as f:
        data_2 = pickle.load(f)

    qcode_2 = data_2["qcode"]

    with open(output_path, "a") as f:   
        with redirect_stdout(f), redirect_stderr(f):
            print("Testing original code 2...")
            qcode_2.test()
            print("\n")

    lz_2 = qcode_2.lz; lx_2 = qcode_2.lx
    hz_2 = qcode_2.hz.toarray(); hx_2 = qcode_2.hx.toarray()
    k = lz_2.shape[0]; n = lz_2.shape[1]

    if logical_type_2 == "Z":
        logical_op_2 = lz_2[logical_idx_2[0], :].copy()
        other_h_2 = hx_2
    elif logical_type_2 == "X":
        logical_op_2 = lx_2[logical_idx_2[0], :].copy()
        other_h_2 = hz_2

    for idx in logical_idx_2[1:]:
        if logical_type_2 == "Z":
            logical_op_2 = logical_op_2 + lz_2[idx, :]
        else:
            logical_op_2 = logical_op_2 + lx_2[idx, :]

    logical_op_2.data %= 2
    logical_op_2.eliminate_zeros()

    # Step 1: Construct vertex of graph and the bijection fuction
    G_2, qubit_to_vertex_2 = create_graph_GeneCS(logical_op_2)

    # Step 2: Construct the perfect matching
    G_2, perfect_matchings_2 = construct_perfect_matching_GeneCS(G_2, qubit_to_vertex_2, logical_op_2, other_h_2)

    # Step 3: Congestion-aware expansion of graph
    # 2 reserved for vertex checks, 2 reserved for perfect matching or thickening per edge
    G_base_2, _, cycle_basis_2, groups_2 = congestion_aware_expander_sweep_GeneCS(G_2, sample_num, edge_key="exp", max_edge_degree=qubit_degree_2-4)

    # Step 4: Thickening
    beta_2 = cheeger_lower_bound(G_base_2)
    if abs(1 - beta_2) < 1e-6:
        beta_2 = 1.0
    L_2 = max(math.ceil(1 / beta_2), len(groups_2))
    G_thick_2 = thicken_graph(G_base_2, L_2)

    # Step 5: Cellulation
    new_cycle_basis_2 = cellulation_GeneCS(G_thick_2, groups_2, cycle_basis_2, d=check_degree_2)
    new_cycle_basis_2 = add_interlayer_rectangle_cycles_GeneCS(G_thick_2, G_base_2, L_2, new_cycle_basis_2)

    data = {
        "G_thick": G_thick_2,
        "perfect_matching": perfect_matchings_2, 
        "cycle_basis": new_cycle_basis_2, 
        "HX": hx_2,
        "HZ": hz_2,
        "logical_type": logical_type_2,
        "logical_idx": logical_idx_2,
        "logical_op": logical_op_2,
    }

    with open(f"{saving_path}/deform_graph/{file_name_2}_merge_{logical_type_2}_{logical_idx_2}.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    ######################## Code Deformation for the First Code ########################

    nodes_2, edges_2 = find_d_nodes_tree(G_thick_2, adapter_size)
    G_tree_2, column_dict_2, row_dict_2 = incidence_matrix(nodes_2, edges_2)
    T_2,P_2 = skip_tree(G_tree_2)

    # Construct the parity check matrix for derformed code
    E_2 = G_thick_2.number_of_edges()
    n_2_ori = hz_2.shape[1]
    hz_2 = enlarge_parity_check(hz_2, E_2)
    hx_2 = enlarge_parity_check(hx_2, E_2)

    # add new vertex check
    if logical_type_2 == "Z":
        hz_2, vertex2check_2 = add_vertex_checks(hz_2, G_thick_2)
    elif logical_type_2 == "X":
        hx_2, vertex2check_2 = add_vertex_checks(hx_2, G_thick_2)

    edge2qubit_2 = edge2qubit(G_thick_2, n_2_ori)

    # modify original check involved in perfect matching 
    if logical_type_2 == "Z":
        hx_2 = add_perfect_matching_checks(hx_2, G_thick_2, perfect_matchings_2)
    elif logical_type_2 == "X":
        hz_2 = add_perfect_matching_checks(hz_2, G_thick_2, perfect_matchings_2)

    # add cycle checks to fix gauge degrees of freedom
    if logical_type_2 == "Z":
        hx_2 = append_cycle_basis_checks(G_thick_2, hx_2, new_cycle_basis_2, hx_2.shape[1])
    elif logical_type_2 == "X":
        hz_2 = append_cycle_basis_checks(G_thick_2, hz_2, new_cycle_basis_2, hz_2.shape[1])

    ######################## Merge Two Codes ########################

    m1_z, n1_z = hz_1.shape
    m2_z, n2_z = hz_2.shape
    m1_x, n1_x = hx_1.shape
    m2_x, n2_x = hx_2.shape

    hz = combine_parity_check(hz_1, hz_2)
    hx = combine_parity_check(hx_1, hx_2)

    hz = enlarge_parity_check(hz, adapter_size)
    hx = enlarge_parity_check(hx, adapter_size)

    # add Adapter edges
    if logical_type_1 == "Z":
        hz = add_adapter(hz, P_1, P_2, vertex2check_1, vertex2check_2, column_dict_1, column_dict_2, m1_z, n1_z, m2_z, n2_z, adapter_size)
    elif logical_type_1 == "X":
        hx = add_adapter(hx, P_1, P_2, vertex2check_1, vertex2check_2, column_dict_1, column_dict_2, m1_x, n1_x, m2_x, n2_x, adapter_size)

    # add Adapter cycles
    if logical_type_1 == "Z":
        hx = add_adapter_cycle(hx, T_1, T_2, edge2qubit_1, edge2qubit_2, row_dict_1, row_dict_2, m1_x, n1_x, m2_x, n2_x, adapter_size)
    elif logical_type_1 == "X":
        hz = add_adapter_cycle(hz, T_1, T_2, edge2qubit_1, edge2qubit_2, row_dict_1, row_dict_2, m1_z, n1_z, m2_z, n2_z, adapter_size)

    t1 = time.time()

    new_qubit_num = G_thick_1.number_of_edges()+G_thick_2.number_of_edges()+adapter_size
    new_check_num = len(new_cycle_basis_1) + G_thick_1.number_of_nodes()+len(new_cycle_basis_2) + G_thick_2.number_of_nodes()+adapter_size
    print("Number of new qubits:", G_thick_1.number_of_edges()+G_thick_2.number_of_edges()+adapter_size)
    print("Number of new qubits:", G_thick_1.number_of_edges()+G_thick_2.number_of_edges()+adapter_size, file=output_file)
    print("Number of new checks:", len(new_cycle_basis_1) + G_thick_1.number_of_nodes()+len(new_cycle_basis_2) + G_thick_2.number_of_nodes()+adapter_size)
    print("Number of new checks:", len(new_cycle_basis_1) + G_thick_1.number_of_nodes()+len(new_cycle_basis_2) + G_thick_2.number_of_nodes()+adapter_size, file=output_file)

    check_degree, qubit_degree = compute_max_degrees(hx, hz)
    print("\n", file=output_file)
    print("Max qubit degree:", qubit_degree)
    print("Max qubit degree:", qubit_degree, file=output_file)
    print("Max check degree:", check_degree)
    print("Max check degree:", check_degree, file=output_file)

    print("\n", file=output_file)
    print("Compilation time:", t1-t0, file=output_file)

    qcode=css_code(hx,hz)
    with open(output_path, "a") as f:   
        with redirect_stdout(f), redirect_stderr(f):
            print("Testing merged code...")
            qcode.test()
            print("\n")

    deform_code = {
        "qcode": qcode,
    }

    with open(output_path, "a") as f:   
        with redirect_stdout(f), redirect_stderr(f):
            compute_global_distribution(hx, hz)
            print("\n")

    with open(f"{saving_path}/deform_code/{file_name_1}_{file_name_2}_merged.pkl", "wb") as f:
        pickle.dump(deform_code, f)

    file_name = file_name_1 + '+' + file_name_2

    # Save selected values as CSV
    df = pd.DataFrame([{
        'file_name': file_name,
        ' qubit_num': new_qubit_num,
        ' check_num': new_check_num,
        ' qubit_deg': qubit_degree,
        ' check_deg': check_degree,
        ' Comp_time': t1-t0, 
    }])

    df.to_csv(f'{saving_path}/output.csv', mode='a', header=not pd.io.common.file_exists(f'{saving_path}/output.csv'), index=False)