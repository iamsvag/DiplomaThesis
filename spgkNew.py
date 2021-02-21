"""
Created on Fri Apr 14 2017

@author: g.nikolentzos

"""

import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
from utils import load_file, preprocessing, get_vocab, learn_model_and_predict
from sklearn.svm import SVC
#from utils2 import graph_from_networkx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath

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
        print(graphs)
        K = build_kernel_matrix(graphs, depth)
        print(K)

        # # Splits the dataset into a training and a test set
        # G_train, G_test, y_train, y_test = train_test_split(K, labels, test_size=0.1, random_state=42)

        # # Uses the shortest path kernel to generate the kernel matrices
        # gk = ShortestPath(normalize=True)
        # print(gk)

        # K_train = gk.fit_transform(G_train)
        # print(K_train)
        # K_test = gk.transform(G_test)

        # Uses the SVM classifier to perform classification
        # clf = SVC(kernel="precomputed")
        # clf.fit(K_train, y_train)
        # y_pred = clf.predict(K_test)


        K_train = K

        labels_train = labels

        K_test = K
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