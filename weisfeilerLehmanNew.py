"""
Created on Fri Apr 14 2017

@author: g.nikolentzos

"""

import copy
import sys

import networkx as nx
import numpy as np
from grakel import GraphKernel
#from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
#from utils2 import graph_from_networkx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
from GK_WL import GK_WL

from utils import get_vocab, learn_model_and_predict, load_file, preprocessing


def create_graphs_of_words(docs, window_size):
    """ 
    Create graphs of words

    """
    graphs = list()
    sizes = list()
    degs = list()

    for doc in docs:
        G = nx.Graph()
        for i in range(len(doc)):
            if doc[i] not in G.nodes():
                G.add_node(doc[i])
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    G.add_edge(doc[i], doc[j])

        graphs.append(G)
        sizes.append(G.number_of_nodes())
        degs.append(2.0*G.number_of_edges()/G.number_of_nodes())

    print("Average number of nodes:", np.mean(sizes))
    print("Average degree:", np.mean(degs))

    return graphs


def spgk(sp_g1, sp_g2, norm1, norm2):
    """ 
    Compute spgk kernel

    """
    if norm1 == 0 or norm2==0:
        return 0
    else:
        kernel_value = 0
        for node1 in sp_g1:
            if node1 in sp_g2:
                kernel_value += 1
                for node2 in sp_g1[node1]:
                    if node2 != node1 and node2 in sp_g2[node1]:
                        kernel_value += (1.0/sp_g1[node1][node2]) * (1.0/sp_g2[node1][node2])

        kernel_value /= (norm1 * norm2)
        #print(kernel_value)
        
        return kernel_value


def build_kernel_matrix(graphs, depth):
    """ 
    Build kernel matrices

    """
    N = len(graphs)
    #print(N)

    sp = list()
    norm = list()

    print("\nGraph preprocessing progress:")
    for g in tqdm(graphs):
        current_sp = dict(nx.all_pairs_dijkstra_path_length(g, cutoff=depth))
        sp.append(current_sp)

        sp_g = nx.Graph()
        for node in current_sp:
            for neighbor in current_sp[node]:
                if node == neighbor:
                    sp_g.add_edge(node, node, weight=1.0)
                else:
                    sp_g.add_edge(node, neighbor, weight=1.0/current_sp[node][neighbor])

        M = nx.to_numpy_matrix(sp_g)
        norm.append(np.linalg.norm(M,'fro'))

    K = np.zeros((N, N))

    print("\nKernel computation progress:")
    for i in tqdm(range(N)):
        for j in range(i, N):
            K[i,j] = spgk(sp[i], sp[j], norm[i], norm[j])
            K[j,i] = K[i,j]

    return K

  




def main():
    """ 
    Main function

    """
    if len(sys.argv) != 5:
        print('Wrong number of arguments!!! Run as follows:')
        print('spgk.py <filenamepos> <filenameneg> <windowsize> <depth>')
    else:
        filename_pos = sys.argv[1]
        filename_neg = sys.argv[2]
        window_size = int(sys.argv[3])
        depth = int(sys.argv[4])

        docs_pos = load_file(filename_pos)
       # print(docs_pos)
        docs_pos = preprocessing(docs_pos)
       # print(docs_pos)
        labels_pos = []
        for i in range(len(docs_pos)):
            labels_pos.append(1)

        docs_neg = load_file(filename_neg)
        docs_neg = preprocessing(docs_neg)
        labels_neg = []
        for i in range(len(docs_neg)):
            labels_neg.append(0)

        docs = docs_pos
        docs.extend(docs_neg)
        labels = labels_pos
        labels.extend(labels_neg)
        labels = np.array(labels)

        vocab = get_vocab(docs)
        print("Vocabulary size: ", len(vocab))
        

    #     G_train_nx = create_graphs_of_words(docs,window_size) 
    #    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
    #     G_test_nx = create_graphs_of_words(docs,window_size)
    #     G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))
        
        graphs = create_graphs_of_words(docs, window_size)
        #print(graphs)
  
        

        #gk_wl = GK_WL()
        #print(graphs)
        #K = gk_wl.compare_list(graphs,1, node_label=True)
        #print(self.graphs)
        graph_list = graphs
        h=1
        
        n = len(graph_list)
        lists = [0] * n
        k = [0] * (h + 1)
        n_nodes = 0
        n_max = 0
        node_label=True


        # Compute adjacency lists and n_nodes, the total number of
        # nodes in the dataset.
        for i in range(n):
            
            #lists[i] = graph_list[i].adjacency_list()
            n_nodes = n_nodes + graph_list[i].number_of_nodes()

            # Computing the maximum number of nodes in the graphs. It
            # will be used in the computation of vectorial
            # representation.
            if(n_max < graph_list[i].number_of_nodes()):
                n_max = graph_list[i].number_of_nodes()

        phi = np.zeros((n_max, n), dtype=np.uint64)

        # INITIALIZATION: initialize the nodes labels for each graph
        # with their labels or with degrees (for unlabeled graphs)

        labels = [0] * n
        label_lookup = {}
        label_counter = 0

        # label_lookup is an associative array, which will contain the
        # mapping from multiset labels (strings) to short labels
        # (integers)

        if node_label is True:
            for i in range(n):
                l_aux = nx.get_node_attributes(graph_list[i],
                                               'node_label').values()
                # It is assumed that the graph has an attribute
                # 'node_label'
                labels[i] = np.zeros(len(l_aux), dtype=np.int32)

                for j in range(len(l_aux)):
                    if not (l_aux[j] in label_lookup):
                        label_lookup[l_aux[j]] = label_counter
                        labels[i][j] = label_counter
                        label_counter += 1
                    else:
                        labels[i][j] = label_lookup[l_aux[j]]
                    # labels are associated to a natural number
                    # starting with 0.
                    phi[labels[i][j], i] += 1
        else:
            for i in range(n):
                labels[i] = np.array(graph_list[i].degree().values())
                for j in range(len(labels[i])):
                    phi[labels[i][j], i] += 1

        # Simplified vectorial representation of graphs (just taking
        # the vectors before the kernel iterations), i.e., it is just
        # the original nodes degree.
        #self.vectors = 
        #vectors = np.copy(phi.transpose())

        k = np.dot(phi.transpose(), phi)
        print(k)
        # MAIN LOOP
        it = 0
        new_labels = copy.deepcopy(labels)

        while it < h:
            # create an empty lookup table
            label_lookup = {}
            label_counter = 0

            phi = np.zeros((n_nodes, n), dtype=np.uint64)
            for i in range(n):
                for v in range(len(lists[i])):
                    # form a multiset label of the node v of the i'th graph
                    # and convert it to a string

                    long_label = np.concatenate((np.array([labels[i][v]]),
                                                 np.sort(labels[i]
                                                 [lists[i][v]])))
                    long_label_string = str(long_label)
                    # if the multiset label has not yet occurred, add it to the
                    # lookup table and assign a number to it
                    if not (long_label_string in label_lookup):
                        label_lookup[long_label_string] = label_counter
                        new_labels[i][v] = label_counter
                        label_counter += 1
                    else:
                        new_labels[i][v] = label_lookup[long_label_string]
                # fill the column for i'th graph in phi
                aux = np.bincount(new_labels[i])
                phi[new_labels[i], i] += aux[new_labels[i]]

            k += np.dot(phi.transpose(), phi)
            labels = copy.deepcopy(new_labels)
            it = it + 1

        # Compute the normalized version of the kernel
        k_norm = np.zeros(k.shape)
        print(k_norm)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])

  
   
        K_train = k_norm

        labels_train = labels

        K_test = k_norm
        labels_test = labels
        
        clf = SVC(kernel='precomputed')
        clf.fit(K_train, labels_train) 
        labels_predicted = clf.predict(K_test)
        acc = accuracy_score(labels_test, labels_predicted)
        # Computes and prints the classification accuracy
        #acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", str(round(acc*100, 2)) + "%")
if __name__ == "__main__":
        main()
