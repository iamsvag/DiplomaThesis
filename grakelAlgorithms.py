

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from grakel.utils import graph_from_networkx
from tqdm import tqdm
from utils import load_file, preprocessing, get_vocab, learn_model_and_predict, get_vocab1
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from grakel.kernels import Kernel
from grakel.kernels import ShortestPath , PyramidMatch
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.datasets import fetch_dataset
from grakel import Graph
from timeit import default_timer as timer
from gpcharts import figure

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
    print(vocab)
    for idx,doc in enumerate(docs):
        G = nx.Graph()
        for i in range(len(doc)):
            if doc[i] not in G.nodes():
                G.add_node(doc[i])
                G.nodes[doc[i]]['label'] = vocab[doc[i]]
                
                
        for i in range(len(doc)):
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    G.add_edge(doc[i], doc[j])
        
        graphs.append(G)
    
    return graphs



def create_author_graph_of_words(docs, voc, window_size):
    graphs = []
    for doc in docs:
        edges = {}
        unique_words = set()
        for i in range(len(doc)):
            unique_words.add(doc[i])
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    unique_words.add(doc[j])
                    edge_tuple1 = (doc[i], doc[j])
                    if edge_tuple1 in edges:
                        edges[edge_tuple1] += 1
                    else:
                        edges[edge_tuple1] = 1
        node_labels = {word:voc[word] for word in unique_words}
        g = Graph(edges, node_labels=node_labels)
        graphs.append(g)

    return graphs


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
        docs_pos = preprocessing(docs_pos)
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
        train_data, test_data, y_train, y_test = train_test_split(docs, labels, test_size=0.33, random_state=42)
        vocab = get_vocab1(train_data,test_data)
        print("Vocabulary Size: ", len(vocab))
        
       
        
        # Create graph-of-words representations
        G_train = create_author_graph_of_words(train_data, vocab, window_size) 
        G_test = create_author_graph_of_words(test_data, vocab, window_size)
        
        strings = ["WeisfeilerLehman","ShortestPath","PyramidMatch"]


        print("\nKernel computation progress:")
        for i in range(3):
            start = timer()
            if strings[i] == "WeisfeilerLehman":
                print("\nAccuracy Calculation with Wiesfeiler-Lehman Algorith")
                # Initialize a Weisfeiler-Lehman subtree kernel
                gk = WeisfeilerLehman(n_iter=4, normalize=False, base_graph_kernel=VertexHistogram)
                
            elif  strings[i] == "ShortestPath":
                print("\nAccuracy Calculation with ShortestPath Algorith")
                # Initialize a Shortest path 
                gk = ShortestPath(n_jobs=None, normalize=False, verbose=False, with_labels=False, algorithm_type="auto")
            else:
                print("\nAccuracy Calculation with Pyramid Match Algorith")
                gk = PyramidMatch(n_jobs=None, normalize=False, verbose=False, with_labels=False, L=4, d=6)
            # Construct kernel matrices
            K_train = gk.fit_transform(G_train)
            K_test = gk.transform(G_test)

            #SVM classifier Training & Predictions 
            clf = SVC(kernel='precomputed')
            clf.fit(K_train, y_train) 
            y_pred = clf.predict(K_test)

            #Evaluation of predictions
            acc =  accuracy_score(y_pred, y_test)
            print("\nAccuracy:", str(round(acc*100, 2)) + "%" )
            end = timer()
            execTime = end - start
            print("\n Execution Time:")
            print(execTime,"Seconds") 

            
        #a log scale example
        fig4 = figure(title='Dataset Size Execution Time',ylabel='Seconds')
        xVals = ['Dataset Size',100,1800,1900,2000]
        yVals = [['Shortest Path', 'Weisfeiler-Lehman','PyramidMatch'],[0,0,0],[100,200,100],[100,50,50],[500,100,200]]
        fig4.plot(xVals,yVals,logScale=False)
        
        # #a log scale example
        # fig4 = figure(title='Dataset Size Execution Time',ylabel='Accuracy')
        # xVals = ['Dataset Size',100,1800,1900,2000]
        # yVals = [['Shortest Path', 'Weisfeiler-Lehman','PyramidMatch'],[0,0,0],[100,200,100],[100,50,50],[500,100,200]]
        # fig4.plot(xVals,yVals,logScale=False)
        
if __name__ == "__main__":
    main()