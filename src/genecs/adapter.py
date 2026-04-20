import numpy as np
from collections import defaultdict, deque

def repetition_parity_check(n):

    H = np.zeros((n, n), dtype=int)

    for i in range(n-1):
        H[i, i] = 1
        H[i, i+1] = 1

    # last row: wrap-around
    H[n-1, 0] = 1
    H[n-1, n-1] = 1

    return H

def check_32_sparse(T):
    row_nnz = T.sum(axis=1)  
    col_nnz = T.sum(axis=0)   

    ok_rows = (row_nnz <= 3).all()
    ok_cols = (col_nnz <= 2).all()

    return ok_rows and ok_cols, row_nnz, col_nnz

def skip_tree(G):

    m,n = G.shape

    # build edge list
    edges = []
    for e in range(m):
        u,v = np.where(G[e]==1)[0]
        edges.append((u,v))

    # adjacency
    adj=[[] for _ in range(n)]
    for i,(u,v) in enumerate(edges):
        adj[u].append((v,i))
        adj[v].append((u,i))

    # BFS spanning tree
    parent=[-1]*n
    parent_edge=[-1]*n
    children=[[] for _ in range(n)]
    visited=[False]*n

    from collections import deque
    q=deque([0])
    visited[0]=True

    tree_edges=[]

    while q:
        v=q.popleft()
        for u,eid in adj[v]:
            if not visited[u]:
                visited[u]=True
                parent[u]=v
                parent_edge[u]=eid
                children[v].append(u)
                tree_edges.append(eid)
                q.append(u)

    tree_pos={e:i for i,e in enumerate(tree_edges)}

    # SkipTree labeling
    Label=[None]*n
    Index=0

    def LabelFirst(v):
        nonlocal Index
        Label[Index]=v
        Index+=1
        for c in children[v]:
            LabelLast(c)

    def LabelLast(v):
        nonlocal Index
        for c in children[v]:
            LabelFirst(c)
        Label[Index]=v
        Index+=1

    LabelFirst(0)

    # permutation matrix
    P=np.zeros((n,n),dtype=int)
    for l,v in enumerate(Label):
        P[v,l]=1

    # Ttilde
    Ttilde=np.zeros((n,n-1),dtype=int)

    def path_edges(u,v):
        anc=set()
        x=u
        while x!=-1:
            anc.add(x)
            x=parent[x]

        path=[]
        y=v
        while y not in anc:
            path.append(parent_edge[y])
            y=parent[y]

        lca=y

        x=u
        while x!=lca:
            path.append(parent_edge[x])
            x=parent[x]

        return path

    for l in range(n):

        v=Label[l]
        u=Label[(l+1) % n]

        path=path_edges(v,u)

        for e in path:
            Ttilde[l,tree_pos[e]]=1

    # build T
    T=np.zeros((n,m),dtype=int)

    for e,pos in tree_pos.items():
        T[:,e]=Ttilde[:,pos]

    return T%2,P%2