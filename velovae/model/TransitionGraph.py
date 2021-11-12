import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
import igraph
from sklearn.neighbors import NearestNeighbors

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

#######################################################################
# Transition Graph
#######################################################################
class TransGraph():
    def __init__(self, cell_types_raw, graph_name=None, graph=None, init_types=None):
        if(graph_name is not None):
            self.label_dic, self.label_dic_rev, self.graph, self.init_types = encodeTypeGraph(graph_name, cell_types_raw)
        elif( (graph is not None) and (init_types is not None) ):
            cell_types_raw = list(graph.keys())
            self.label_dic, self.label_dic_rev = encodeType(cell_types_raw)
            self.graph, self.init_types = encodeGraph(graph, init_types, self.label_dic)
        else:
            self.label_dic, self.label_dic_rev = encodeType(cell_types_raw)
            self.resetGraph()
        
            
        
        self.Ntype = len(cell_types_raw)
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.Ntype)])
        self.A = None
        self.At = None
        self.g_cluster = None
        self.cluster_sizes = None
        self.P = None
        
    
    def resetGraph(self):
        self.graph = {}
        for i in self.cell_types:
            self.graph[i] = []
        self.init_types = self.cell_types
        
    def str2int(self, cell_labels_raw):
        return np.array([self.label_dic[cell_labels_raw[i]] for i in range(len(cell_labels_raw))])
    
    def int2str(self, cell_labels):
        return np.array([self.label_dic_rev[cell_labels[i]] for i in range(len(cell_labels))])
    
    
    
    
    
    
    
    #################################################
    # The following parts are deprecated.
    # Originally there was a plan to devise an algorithm
    #   to learn the transition graph. But this was
    #   vetoed later, as we switched to the transition
    #   ODE model.
    #################################################
    def getDepth(self):
        def updateDepth(graph, cur_type):
            dlist = []
            if(len(graph[cur_type]) > 0):
                for child in graph[cur_type]:
                    dlist.append(updateDepth(graph, child))
                    return np.max(dlist) + 1
            else:
                return 1
            
        dlist = []
        for cur_type in self.init_types:
            dlist.append(updateDepth(self.graph, cur_type))
        return np.max(dlist)
    
    def recoverTransitionTime(self, t_trans, ts):
        return recoverTransitionTime(t_trans, ts, self.graph, self.init_types)
    
    def _buildKNNGraph(self, X, t, k=30):
        """
        X: expression data
        t: time
        """
        Xt = t.reshape(-1,1) #reshape the data to match KNN interface
        knn_t = NearestNeighbors(n_neighbors=k, metric=dist_t)
        knn_t = knn_t.fit(Xt)
        knn_s = NearestNeighbors(n_neighbors=k)
        knn_s = knn_s.fit(X)
        self.At = knn_t.kneighbors_graph(Xt)
        self.A = knn_s.kneighbors_graph(X, mode='distance')
        
    def _clusterGraph(self,cell_labels):
        """
        Build a complete cell type transition graph from KNN adjacency matrices
        A: Weighted Adjacency matrix based on data similarity
        At: Adjacency matrix based on time
        """
        if(self.A is None or self.At is None):
            print('Adjacency matrix not detected! Please run buildKNNGraph first!')
        
        N = self.A.shape[0]
        graph = igraph.Graph(directed=True)
        graph.add_vertices(N)
        vout, vin = self.At.nonzero()
        
        graph.add_edges(list(zip(vout,vin)))
        
        #Add weights
        for edge in graph.es:
            s, t = edge.source, edge.target
            edge["weight"] = np.exp(-0.01*self.A[s,t])
        #Collapse all nodes with the same cell type into a single node and get the cluster graph
        vc = igraph.VertexClustering(graph, membership=cell_labels)
        self.g_cluster = vc.cluster_graph(combine_edges="sum")
        self.cluster_sizes = np.array(vc.sizes())
        
    def _transScore(self, eij, cluster_sizes, k):
        """
        Computes the likelihood that cluster i has transition to j.
        This is based on the theory of PAGA:
        
        Wolf, F.A., Hamey, F.K., Plass, M. et al. 
        PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells. 
        Genome Biol 20, 59 (2019). https://doi.org/10.1186/s13059-019-1663-x
        """
        N = np.sum(self.cluster_sizes)
        #Null hypothesis
        mu = k*cluster_sizes.reshape(-1,1)*cluster_sizes/N #[N cluster x N cluster]
        sigma = np.sqrt(k*cluster_sizes.reshape(-1,1)*(cluster_sizes/(N-1))*((N-cluster_sizes-1)/(N-1)))
        
        P = eij/mu
        
        return P
    
    def transScore(self, X, t, cell_labels, k=30):
        #Build a KNN graph
        self._buildKNNGraph(X,t,k)
        
        #Collapse the cell-level graph to cluster level
        N = len(t)
        self._clusterGraph(cell_labels)
        
        #Compute the transition score
        eij = self.g_cluster.es["weight"]
        edges = self.g_cluster.get_edgelist()
        rows = [x[0] for x in edges]
        cols = [x[1] for x in edges]
        E = csr_matrix((eij,(rows, cols))).toarray()
        P = self._transScore(E, self.cluster_sizes, k)
        
        #Prune the score matrix
        thred = np.median(P)*0.1
        P[P<thred] = 0
        self.P = P - P.T
        
        
    def getGraph(self, thred=0.0):
        """
        Compute the near-optimal transition graph.
        The algorithm is based on Edmond's algorithm for finding a minimum spanning tree 
        in a directed graph. If the graph is disconnected, minimum spanning trees are constructed
        within each disconnected part.
        """
        if(self.P is None):
            print('Transition matrix not detected! Please run transScore first!')
        self.resetGraph()
        #Find the optimal transition graph
        for j in range(self.P.shape[1]):
            i = np.argmax(self.P[:,j])
            if(self.P[i,j]>thred):
                self.graph[i].append(j)
        
        #Find initial types
        has_parent = np.zeros((self.Ntype))
        for i in self.graph:
            for j in self.graph[i]:
                has_parent[j] = 1
        self.init_types = np.where(has_parent<1)[0]
        
    def printGraph(self):
        print('Initial Types: ', [self.label_dic_rev[i] for i in self.init_types])
        for i in self.graph:
            print(self.label_dic_rev[i],' -> ',[self.label_dic_rev[j] for j in self.graph[i]])
