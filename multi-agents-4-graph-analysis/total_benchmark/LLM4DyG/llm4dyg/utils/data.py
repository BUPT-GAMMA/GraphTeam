import numpy as np
import pandas as pd
import os
import torch
from libwon.utils import setup_seed
from collections import Counter
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,"../../data")

def turn3dm(edges):
    es = []
    for i,e in enumerate(edges):
        e = torch.cat([e.T, torch.LongTensor([i]).expand(e.shape[1],1)], dim=1)
        es.append(e)
    es = torch.cat(es, dim = 0)
    return es

def turn_undirect(edges):
    set_e = set()
    for u,v,t in edges:
        if (u,v,t) not in set_e and (v,u,t) not in set_e:
            set_e.add((u,v,t))
    es = torch.stack([torch.LongTensor(e) for e in list(set_e)])
    return es

import networkx as nx    
import numpy as np
from libwon.utils import setup_seed
def get_sbm_graph(N, pin, pout, C, directed):
    sizes = [N] * C
    in_prob = pin
    out_prob = pout
    probs = np.zeros((len(sizes), len(sizes)))
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            probs[i, j] = in_prob if i == j else out_prob
    G = nx.stochastic_block_model(sizes, probs, directed = directed)
    edges = [e for e in G.edges()]
    return np.array(edges) # [E, 2]

def get_er_graph(N, p, directed = False):
    G = nx.erdos_renyi_graph(N, p, directed = directed)
    edges = [e for e in G.edges()]
    return np.array(edges) # [E, 2]

def get_er_graphs(T = 3, N = 2, p = 0.5, directed = False, seed = 0):
    setup_seed(seed)
    es = []
    for t in range(T):
        e = get_er_graph(N, p, directed)
        if len(e):
            time = np.array([t] * e.shape[0]).reshape(e.shape[0],1)
            es.append(np.concatenate([e, time], axis = -1))
    es = np.concatenate(es, axis = 0)
    return es

def get_sbm_graphs(T = 3, N = 2, p = 0.5, C = 3, directed = False, seed = 0):
    setup_seed(seed)
    es = []
    for t in range(T):
        e = get_sbm_graph(N, p, C, directed)
        if len(e):
            time = np.array([t] * e.shape[0]).reshape(e.shape[0],1)
            es.append(np.concatenate([e, time], axis = -1))
    es = np.concatenate(es, axis = 0)
    return es
class DyGraphGenER:
    def sample_dynamic_graph(self, T = 3, N = 2 , p = 0.5, directed = False, seed = 0):
        es = get_er_graphs(T, N, p, directed, seed)
        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))
        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        return info
    
import numpy as np
class DyGraphGenERCon:
    def sample_dynamic_graph(self, T = 3, N = 2 , p = 0.5, directed = False, seed = 0):
        es = get_er_graphs(1, N, p, directed, seed)
        es[:, 2] = np.random.randint(0, T, es.shape[0])
        es = list(es)
        es = sorted(es, key = lambda x: (x[2], x[0]))
        es = np.array(es)
        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))
        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        return info

class DyGraphGenSBMCon:
    def sample_dynamic_graph(self, T = 3, N = 2 , p = 0.5, C = 2, directed = False, seed = 0):
        setup_seed(seed)
        es = get_sbm_graph(N//C, p, p/2, C, directed)
        es = np.concatenate([es, np.zeros((es.shape[0], 1))], axis = -1).astype(int)
        # import pdb;pdb.set_trace()
        es[:, 2] = np.random.randint(0, T, es.shape[0])
        es = list(es)
        es = sorted(es, key = lambda x: (x[2], x[0]))
        es = np.array(es)
        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))
        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        return info
    

from igraph import Graph


class DyGraphGenFFCon:
    def sample_dynamic_graph(self, T = 3, N = 2 , p = 0.5, directed = False, seed = 0):
        setup_seed(seed)
        # es = get_sbm_graph(N//C, p, p/2, C, directed)
        es = Graph.Forest_Fire(N, fw_prob = p).get_edgelist()
        es = np.array(es)
        es = np.concatenate([es, np.zeros((es.shape[0], 1))], axis = -1).astype(int)
        # import pdb;pdb.set_trace()
        es[:, 2] = np.random.randint(0, T, es.shape[0])
        es = list(es)
        es = sorted(es, key = lambda x: (x[2], x[0]))
        es = np.array(es)
        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))
        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        return info
    
class DyGraphGenSBM:
    def sample_dynamic_graph(self, T = 3, N = 2 , p = 0.5, C = 3, directed = False, seed = 0):
        es = get_sbm_graphs(T, N, p, C, directed, seed)
        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))
        classes = []
        for i in range(C):
            for j in range(N):
                classes.append(i)
        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "classes": classes,
                "T": T,
                "N":N,
                "p":p,
                "C":C,
                "directed":directed,
                "seed":seed
                }
        return info
        
    
class DyGraphGen:
    def __init__(self, dataset = "enron"):
        if dataset == "enron":
            datafile = os.path.join(dataroot, "enron10/adj_time_list.npy")
            data = np.load(datafile, allow_pickle=True)
            edge_index = [torch.LongTensor(np.array(g.nonzero())) for g in data]
        elif dataset == "dblp":
            datafile = os.path.join(dataroot, "dblp/adj_time_list.npy")
            data = np.load(datafile, allow_pickle=True)
            edge_index = [torch.LongTensor(np.array(g.nonzero())) for g in data]
            
        elif dataset == "flights":
            datafile = os.path.join(dataroot, "Flights/adj_time_list.npy")
            data = np.load(datafile, allow_pickle=True)
            edge_index = [torch.LongTensor(np.array(g.nonzero())) for g in data]
        else:
            raise NotImplementedError(f"{dataset} not implemented")
        self.edge_index = edge_index
    
    def sample_dynamic_graph(self, T = 3, N = 3, seed = 0, undirect = True, **kwargs):
        edge_index = self.edge_index
        setup_seed(seed)
        
        # select time 
        allt = len(edge_index)
        t_start = np.random.choice(np.arange(allt - T - 1))
        t_end = t_start + T
        print(f"sampling time interval [{t_start},{t_end}]")

        # select nodes
        edge3d = turn3dm(edge_index[t_start:t_end])
        if undirect: edge3d = turn_undirect(edge3d)
        edge3d = edge3d.numpy()
        node_set = list(set(edge3d[:,:2].flatten()))
        nodes = set(np.random.choice(node_set, N, replace=False))
        
        # select subgraph
        df = pd.DataFrame(edge3d, columns = "n1 n2 t".split())
        df = df.query("n1 in @nodes or n2 in @nodes").copy()
        org_node_set = set(list(df["n1 n2".split()].values.flatten()))
        node_map = {n:i for i,n in enumerate(list(org_node_set))}
        df['n1'] = df["n1"].apply(lambda x :node_map[x])
        df['n2'] = df["n2"].apply(lambda x :node_map[x])
        edges = df.to_numpy()
        
        # get subgraph info
        num_nodes = len(set(edges[:, :2].flatten()))
        num_edges = len(edges)
        num_time = len(set(edges[:, 2].flatten()))
        ego_nodes = [ node_map[x] for x in list(nodes)]
        info = {"edge_index": edges.tolist(), "num_nodes":num_nodes, "num_edges":num_edges, "num_time": num_time, "ego_nodes":ego_nodes, "T": T, "N":N, "seed":seed,'p':None}
        return info
    

import networkx as nx
import random
def generate_dyg_ff(n =3, m =1, p =0.3, f=0.1, timesteps=5):
    """
    n = 3  # Number of initial nodes
    m = 1    # Number of initial edges for each node
    p = 0.3  # Probability of adding a new node during each step
    f = 0.1  # Probability of connecting a new node to an existing node (fire probability)
    timesteps = 5  # Number of time steps
    return edges [(n1,n2,t)]
    """
    G = nx.barabasi_albert_graph(n, m)

    edges = [(n1,n2,0) for n1,n2 in G.edges()]
    for t in range(timesteps):
        new_node = len(G.nodes())  # ID for the new node
        G.add_node(new_node)

        
        # Select a random node to connect to
        target_node = random.choice(list(G.nodes()))
        
        # With probability f, connect to an existing node
        if random.random() < f:
            new_edge = (new_node, target_node, t)
            G.add_edge(new_node, target_node)
            edges.append(new_edge)
        # With probability (1-f), connect to a node within a radius of 1
        else:
            neighbors = list(G.neighbors(target_node))
            if neighbors:
                neighbor = random.choice(neighbors)
                G.add_edge(new_node, neighbor)
                new_edge = (new_node, target_node, t)
                edges.append(new_edge)
    return edges