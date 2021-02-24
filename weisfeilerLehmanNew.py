"""
Created on Fri Apr 14 2017

@author: g.nikolentzos

"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from grakel.utils import graph_from_networkx
from tqdm import tqdm
from utils import load_file, preprocessing, get_vocab, learn_model_and_predict, get_vocab1,load_file1, preprocessing1
#from utilsNew import get_vocab1
from sklearn.svm import SVC
#from utils2 import graph_from_networkx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from grakel.kernels import Kernel
#from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
from grakel.kernels import WeisfeilerLehman, VertexHistogram
#from grakel.kernels.vertex_histogram import VertexHistogram
from grakel.datasets import fetch_dataset
#from GK_WL import GK_WL

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
        
        return kernel_value

def create_graphs_of_words1(docs, vocab, window_size):
    graphs = list()
    sizes = list()
    degs = list()

    for idx,doc in enumerate(docs):
        G = nx.Graph()
        for i in range(len(doc)):
            if doc[i] not in G.nodes():
                G.add_node(doc[i])
                G.nodes[doc[i]]['foo'] = vocab[doc[i]]
        for i in range(len(doc)):
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    G.add_edge(doc[i], doc[j])
        
        graphs.append(G)
    
    return graphs
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


    # print("\nKernel computation progress:")
    # for i in tqdm(range(N)):
    #     for j in range(i, N):
    #         # K[i,j] = spgk(sp[i], sp[j], norm[i], norm[j])
    #         # K[j,i] = K[i,j]
    #         gk = WeisfeilerLehman(n_iter=1,base_graph_kernel=VertexHistogram, normalize=False)
    #         # Construct kernel matrices
    #         K[i,j] = gk.fit_transform(sp_g[i,j])
    #     return K


    


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

    #     docs_pos = load_file(filename_pos)
    #    # print(docs_pos)
    #     docs_pos = preprocessing(docs_pos)
    #    # print(docs_pos)
    #     labels_pos = []
    #     for i in range(len(docs_pos)):
    #         labels_pos.append(1)

    #     docs_neg = load_file(filename_neg)
    #     docs_neg = preprocessing(docs_neg)
    #     labels_neg = []
    #     for i in range(len(docs_neg)):
    #         labels_neg.append(0)

    #     docs = docs_pos
    #     docs.extend(docs_neg)
    #     labels = labels_pos
    #     labels.extend(labels_neg)
    #     labels = np.array(labels)
        # Read and pre-process train data
        train_data, y_train = load_file1(filename_pos)
        train_data = preprocessing1(train_data)

        # Read and pre-process test data
        test_data, y_test = load_file1(filename_neg)
        test_data = preprocessing1(test_data)

        #train_data, test_data, y_train, y_test = train_test_split(docs, labels, test_size=0.1, random_state=42)
        vocab = get_vocab1(train_data,test_data)
        print("Vocabulary size: ", len(vocab))
        
        #print(labels)
        # Create graph-of-words representations
        G_train_nx = create_graphs_of_words1(train_data, vocab, window_size) 
        G_test_nx = create_graphs_of_words1(test_data, vocab, window_size)
        # print("Example of graph-of-words representation of document")
        # nx.draw_networkx(G_train_nx[3], with_labels=True)
        #G_train_nx = create_graphs_of_words(docs,window_size) 
        G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='foo'))
        G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='foo'))
    #   G_test_nx = create_graphs_of_words(docs,window_size)
    #   G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))
        
        #graphs = create_graphs_of_words(docs, window_size)
        # K = build_kernel_matrix(graphs, depth)
        # Loads the MUTAG dataset
        #print(docs)
        # Initialize a Weisfeiler-Lehman subtree kernel
        gk = WeisfeilerLehman(n_iter=1, normalize=False, base_graph_kernel=VertexHistogram)

        # Construct kernel matrices
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)

        # Train an SVM classifier and make predictions
        clf = SVC(kernel='precomputed')
        clf.fit(K_train, y_train) 
        y_pred = clf.predict(K_test)

        # Evaluate the predictions
        print("Accuracy:", accuracy_score(y_pred, y_test))

        # #print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        # #working part
        # MUTAG = fetch_dataset("MUTAG", verbose=False)
        # print(MUTAG)
        # G, y = MUTAG.data, MUTAG.target
        # G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)

        # # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
        # gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
        # K_train = gk.fit_transform(G_train)
        # K_test = gk.transform(G_test)

        # # Uses the SVM classifier to perform classification
        # clf = SVC(kernel="precomputed")
        # clf.fit(K_train, y_train)
        # y_pred = clf.predict(K_test)

        # # Computes and prints the classification accuracy
        # acc = accuracy_score(y_test, y_pred)
        # print("Accuracy:", str(round(acc*100, 2)) + "%")
        # #print(G) 
        # # print(G_train)
        # #print(y)
        # # print(labels)
        # # Splits the dataset into a training and a test set
        # # print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")






if __name__ == "__main__":
    print("test")
    main()