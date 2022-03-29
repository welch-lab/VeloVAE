import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from .model_util import knn_transition_prob

#######################################################################
# Prior Knowledge of the transition graph
#######################################################################
graph_default = {'pancreas':
                    {
                        'Ductal':['Ngn3 low EP'],\
                        'Ngn3 low EP':['Ngn3 high EP'],\
                        'Ngn3 high EP':['Pre-endocrine'],\
                        'Pre-endocrine':['Alpha','Beta','Delta','Epsilon'],\
                        'Alpha':[],\
                        'Beta':[],\
                        'Delta':[],\
                        'Epsilon':[]
                    },
                 'dentategyrus':
                    {
                        'Radial Glia-like':['nIPC', 'Astrocytes'],
                        'Astrocytes':[],
                        'nIPC':['Neuroblast'],
                        'Neuroblast':['Granule immature'],
                        'Granule immature': ['Granule mature'],
                        'Granule mature':[],
                        'Mossy':[],
                        'Cck-Tox':[],
                        'OPC':['OL'],
                        'OL':[],
                        'GABA':[],
                        'Endothelial':[],
                        'Microglia':[],
                        'Cajal Retzius':[]
                    },
                 'retina':
                    {
                        '0':['1'],
                        '1':['2','3','4','5'],
                        '2':[],
                        '3':[],
                        '4':[],
                        '5':[]
                    },
                 'mouse_brain':
                     {
                         'RG, Astro, OPC':['IPCs'],
                         'IPCs':['V-SVZ','Ependymal cells'],
                         'Ependymal cells':[],
                         'V-SVZ':['Upper Layer', 'Subplate'],
                         'Upper Layer':[],
                         'Subplate':['Deeper Layer'],
                         'Deeper Layer':[]
                     },
                 'braindev':
                     {
                         'Neural tube':['Neural crest'],
                         'Neural crest':['Mesenchyme', 'Radial glia'],
                         'Mesenchyme':['Fibroblast'],
                         'Radial glia':['Neuroblast', 'Ependymal', 'Glioblast'],
                         'Neuroblast':['Neuron'],
                         'Glioblast':['Oligodendrocyte'],
                         'Ependymal':[],
                         'Fibroblast':[],
                         'Neuron':[],
                         'Oligodendrocyte':[]
                     }
                }



init_types = {
                'pancreas':['Ductal'],
                'dentategyrus':['Radial Glia-like', 'OPC', 'GABA', 'Mossy', 'Endothelial', 'Cck-Tox', 'Microglia', 'Cajal Retzius'],
                'retina':['0'],
                'mouse_brain':['RG, Astro, OPC'],
                'braindev':['Neural tube']
             }

#######################################################################
# Functions to encode string-type data as integers
#######################################################################
def encodeType(cell_types_raw):
    """
    Use integer to encode the cell types
    Each cell type has one unique integer label.
    """
    #Map cell types to integers 
    label_dic = {}
    label_dic_rev = {}
    for i, type_ in enumerate(cell_types_raw):
        label_dic[type_] = i
        label_dic_rev[i] = type_
        
    return label_dic, label_dic_rev
    
def encodeGraph(graph_raw, init_types_raw, label_dic):
    """
    Encode the transition graph using integers
    Each cell type has one unique integer label.
    """
    graph_enc = {}
    for type_ in graph_raw.keys():
        graph_enc[label_dic[type_]] = [label_dic[child] for child in graph_raw[type_]]
    init_types_enc = [label_dic[x] for x in init_types_raw]
    
    return graph_enc, init_types_enc
    
def decodeGraph(graph, init_types, label_dic_rev):
    """
    Decode the transition graph from integers to the type name
    """
    graph_dec = {}
    for type_ in graph.keys():
        graph_dec[label_dic_rev[type_]] = [label_dic_rev[child] for child in graph[type_]]
    init_types_dec = [label_dic_rev[x] for x in init_types]
    
    return graph_enc, init_types_enc
    
def encodeTypeGraph(key, cell_types_raw):
    """
    Fetch a default transition graph and convert the string into integer encoding.
    key: name of the dataset
    cell_types_raw: an array of cell types of the string type
    """
    if(not key in graph_default):
        print('Unknown dataset!')
        return {},{},{},[]

    Ntype = len(cell_types_raw)

    #Map cell types to integers 
    label_dic = {}
    label_dic_rev = {}
    for i, type_ in enumerate(cell_types_raw):
        label_dic[type_] = i
        label_dic_rev[i] = type_

    #build the transition graph as a dictionary
    graph_enc = {}
    for type_ in graph_default[key].keys():
        graph_enc[label_dic[type_]] = [label_dic[child] for child in graph_default[key][type_]]
    init_types_enc = [label_dic[x] for x in init_types[key]]

    return label_dic, label_dic_rev, graph_enc, init_types_enc



def str2int(cell_labels_raw, label_dic):
    return np.array([label_dic[cell_labels_raw[i]] for i in range(len(cell_labels_raw))])
    
def int2str(cell_labels, label_dic_rev):
    return np.array([label_dic_rev[cell_labels[i]] for i in range(len(cell_labels))])

def recoverTransitionTimeRec(t_trans, ts, prev_type, graph):
    """
    Applied to the branching ODE
    Recursive helper function of recovering transition time.
    """
    if(len(graph[prev_type])==0):
        return
    for cur_type in graph[prev_type]:
        t_trans[cur_type] += t_trans[prev_type]
        ts[cur_type] += t_trans[cur_type]
        recoverTransitionTimeRec(t_trans, ts, cur_type, graph)
    return

def recoverTransitionTime(t_trans, ts, graph, init_type):
    """
    Applied to the branching ODE
    Recovers the transition and switching time from the relative time.
    """
    t_trans_orig = deepcopy(t_trans)
    ts_orig = deepcopy(ts)
    for x in init_type:
        ts_orig[x] += t_trans_orig[x]
        recoverTransitionTimeRec(t_trans_orig, ts_orig, x, graph)
    return t_trans_orig, ts_orig

def merge_nodes(graph, parents, n_nodes, loop, v_outside):
    """
    Merge nodes into a super node and change the graph accordingly
    """
    #Create a new map from 
    v_map = {}
    v_to_loop = {} #maps any vertex outside the loop to the vertex in the loop with the maximum weight
    loop_to_v = {}
    i_new = 0
    for i in range(n_nodes):
        if(not i in loop):
            v_map[i] = i_new
            i_new = i_new + 1
    for i in loop:
        v_map[i] = i_new
    
    weight_to_loop = {}
    weight_from_loop = {}
    for x in loop:
        for y in v_outside:
            if(not np.isinf(graph[x,y])): #edge to the loop
                if(not y in v_to_loop):
                    v_to_loop[y] = x
                    weight_to_loop[y] = graph[x,y] - graph[x,parents[x]]
                elif(graph[x,y] - graph[x,parents[x]] > weight_to_loop[y]):
                    v_to_loop[y] = x
                    weight_to_loop[y] = graph[x,y] - graph[x,parents[x]]
            if(not np.isinf(graph[y,x])): #edge from the loop
                if(not y in loop_to_v):
                    loop_to_v[y] = x
                    weight_from_loop[y] = graph[y,x]
                elif(graph[y,x] > weight_from_loop[y]):
                    loop_to_v[y] = x
                    weight_from_loop[y] = graph[y,x]
    
    graph_new = np.ones((n_nodes-len(loop)+1,n_nodes-len(loop)+1)) * (-np.inf)
    vc = v_map[loop[0]]
    for x in v_outside:
        for y in v_outside:
            graph_new[v_map[x],v_map[y]] = graph[x,y]
    
    for x in v_outside:
        if(x in v_to_loop):
            graph_new[vc,v_map[x]] = weight_to_loop[x]
        if(x in loop_to_v):
            graph_new[v_map[x],vc] = weight_from_loop[x]
    
    return v_map, v_to_loop, loop_to_v, graph_new

def adj_matrix_to_list(A):
    n_type = A.shape[0]
    adj_list = {}
    for i in range(n_type):
        adj_list[i] = []
    for i in range(n_type):
        idx_noninf = np.where(~(np.isinf(A[i])))[0]
        for par in idx_noninf:
            adj_list[par].append(i)
    return adj_list

def check_connected(adj_list, root, n_nodes):
    """
    Check if a directed graph is connected
    """
    checked = np.array([False for i in range(n_nodes)])
    loop = []
    
    queue = [root]
    checked[root] = True
    while(len(queue)>0):
        ptr = queue.pop(0)
        for child in adj_list[ptr]:
            if(not checked[child]):
                queue.append(child)
                checked[child] = True
        
    return np.all(checked)

def get_loop(trace_back, n_nodes, start):
    """
    trace_back is a dictionary mapping each node to its antecedent
    """
    loop = []
    v_outside = []
    ptr = start
    try:
        while(not trace_back[ptr]==start):
            loop.append(trace_back[ptr])
            ptr = trace_back[ptr]
        loop.append(start)
    except KeyError:
        print("Key Error.")
        print(trace_back)
    
    for i in range(n_nodes):
        if(not i in loop):
            v_outside.append(i)
    return np.flip(loop), v_outside

def check_loop(adj_list, n_nodes):
    loop = []
    
    for node in range(n_nodes):
        checked = np.array([False for i in range(n_nodes)])
        queue = [node]
        trace_back = {}
        while(len(queue)>0):
            ptr = queue.pop(0)
            if(checked[ptr]):
                loop, v_outside = get_loop(trace_back, n_nodes, ptr)
                return loop, v_outside
            else:
                checked[ptr] = True
            for child in adj_list[ptr]:
                trace_back[child] = ptr
                queue.append(child)
            
    return np.array([]), np.array(range(n_nodes))

def edmond_chu_liu(graph, r):
    """
    graph: a 2-d array representing an adjacency matrix
    Notice that graph[i,j] is the edge from j to i
    """
    #print(f"root: {r}")
    #print(graph)
    n_type = graph.shape[0]
    #step 1: remove any edge to the root
    graph[r] = -np.inf
    
    #step 2: find the best incident edge to each node except for r
    adj_list_pruned = {}
    
    parents = []
    for i in range(n_type):
        idx_noninf = np.where(~(np.isinf(graph[i])))[0]
        if(len(idx_noninf)>0):
            max_val = np.max(graph[i][idx_noninf])
            parents.append(np.where(graph[i]==max_val)[0][0])
        else:
            parents.append((i+np.random.choice(n_type)) % n_type)
    parents = np.array(parents)
    
    for i in range(n_type):
        adj_list_pruned[i] = []
    for i in range(n_type):
        if(not i==r):
            adj_list_pruned[parents[i]].append(i)

    #step 3: cycle detection using BFS
    loop, v_outside = check_loop(adj_list_pruned, n_type)

    #step 4: recursive call
    mst = np.zeros((n_type, n_type))
    if(len(loop)>0):
        v_map, v_to_loop, loop_to_v, graph_merged = merge_nodes(graph, parents, n_type, loop, v_outside)
        
        vc = v_map[loop[0]]
        mst_merged = edmond_chu_liu(graph_merged, v_map[r]) #adjacency matrix
        
        #edges outside the loop
        for x in v_outside:
            for y in v_outside:
                mst[x,y] = mst_merged[v_map[x], v_map[y]]
        
        #edges within the loop
        for i in range(len(loop)-1):
            mst[loop[i+1], loop[i]] = 1
        mst[loop[0], loop[-1]] = 1
        
        #edges from the loop
        for x in v_outside:
            if(mst_merged[v_map[x],vc]>0):
                mst[x,loop_to_v[x]] = mst_merged[v_map[x],vc]
        
        #There's exactly one edge to the loop
        source_to_loop = np.where(mst_merged[vc]>0)[0][0] #node in the merged mst
        for x in v_outside:
            if(v_map[x] == source_to_loop):
                source_to_loop = x
                break
        target_in_loop = v_to_loop[source_to_loop]
        mst[target_in_loop, source_to_loop] = 1
        #break the loop
        idx_in_loop = np.where(loop==target_in_loop)[0][0]
        mst[target_in_loop, loop[(idx_in_loop-1)%len(loop)]] = 0
    else:
        for i in range(n_type):
            if(i==r):
                mst[i,i] = 1
            else:
                mst[i, parents[i]] = 1
    return mst
#######################################################################
# Transition Graph
#######################################################################
class TransGraph():
    def __init__(self, adata, tkey, embed_key, cluster_key, train_idx=None, k=5, res=0.005):
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if train_idx is None else adata.obs[cluster_key][train_idx].to_numpy()
        self.t = adata.obs[tkey].to_numpy() if train_idx is None else adata.obs[tkey][train_idx].to_numpy()
        self.z = adata.obsm[embed_key] if train_idx is None else adata.obsm[embed_key][train_idx]
        
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encodeType(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])
        
        #Partition the graph
        print("Graph Partition")
        sc.pp.neighbors(adata, n_neighbors=k, key_added="lineage")
        sc.tl.louvain(adata, resolution=res, key_added="partition", neighbors_key="lineage")
        partition_labels = adata.obs["partition"].to_numpy() if train_idx is None else adata.obs["partition"][train_idx].to_numpy()
        lineages = np.unique(partition_labels)
        self.n_lineage = len(lineages)
        count = np.zeros((self.n_type, self.n_lineage))
        
        for i in (self.cell_types):
            for j, lin in enumerate(lineages):
                count[i, j] = np.sum((self.cell_labels==i)&(partition_labels==lin))
        self.partition = np.argmax(count, 1)
        print("Number of partitions: ",len(lineages))
    
    def compute_transition_deterministic(self, adata, n_par=2, dt=(0.01,0.03), k=5, soft_assign=True):
        """
        Compute a type-to-type transition based a cell-to-cell transition matrix
        """
        #Estimate initial time
        t_init = np.zeros((self.n_type))
        for i in (self.cell_types):
            t_init[i] = np.quantile(self.t[self.cell_labels==i], 0.02)
        #Compute cell-type transition probability
        print("Computing type-to-type transition probability")
        range_t = np.quantile(self.t, 0.99) - np.quantile(self.t, 0.01)
        P_raw = knn_transition_prob(self.t, 
                                    self.z, 
                                    self.t, 
                                    self.z, 
                                    self.cell_labels, 
                                    self.n_type, 
                                    [dt[0]*range_t, dt[1]*range_t], 
                                    k,
                                    soft_assign)
        psum = P_raw.sum(1)
        P = P_raw/psum.reshape(-1,1)

        for i in range(P.shape[0]):
            P[i,i] = 0
            idx_sort = np.argsort(P[i])
            for j in range(P.shape[0]-n_par):
                P[i,idx_sort[j]] = 0
            
        psum = P.sum(1)
        P = P/psum.reshape(-1,1)
        self.w = P
        
        #For each partition, get the MST
        print("Obtaining the MST in each partition")
        out = np.zeros((self.n_type, self.n_type))
        for l in range(self.n_lineage):
            vs_part = np.where(self.partition==l)[0]
            graph_part = np.log(P[vs_part][:, vs_part])
            adj_list = adj_matrix_to_list(graph_part)
            root = vs_part[np.argmin(t_init[vs_part])]
            root = np.where(vs_part==root)[0][0]
            
            if(check_connected(adj_list, root, len(vs_part))):
                mst_part = edmond_chu_liu(graph_part, root)
            else:    
                print("Warning: graph is disconnected! Using the fully-connected graph instead.")
                graph_part = np.log(P_raw[vs_part][:, vs_part])
                mst_part = edmond_chu_liu(graph_part, root)
                
            
            for i,x in enumerate(vs_part):
                for j,y in enumerate(vs_part):
                    out[x,y] = mst_part[i,j]
        return out
