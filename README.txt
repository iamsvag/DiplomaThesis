last update 19/10
### Requirements
Code is written in Python 3.6 and requires:
* NetworkX 2.0
* tqdm 4
* scikit-learn 0.18


Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/archive/p/word2vec/


### Running the method
Use the following command: 

```
$ python spgk.py filepos fileneg window_size depth
```
python3 spgk.py data/subjectivity.pos data/subjectivity.neg 2 1

where "filepos" and "fileneg" point to the positive examples and negative examples respectively, "window_size" is the size of the sliding window, and "depth" is the maximum length of shortest paths that are taken into account


### Examples
Example command: 

python spgk.py data/subjectivity.pos data/subjectivity.neg 2 1

1. load files (positive & negative)
2. clean docs
3. labels_pos = for positive => list with "1"
4. labels_neg = for negative => list with "0"
5. docs = positive + negative (extend)
6. labels = labels_pos + labels_neg (extend)
7. load labels array ton the numpy library
8. vocab = get_vocab() => get unique words of given doc
9. create_graphs_of_words: 
  1. create a graph per line in a given `doc`
  2. for each word in line of doc => 
    1. create a node, if the word hasn't already been in the graph
    2. create an edge between the word and it's neihgbors. take into account the repeated words 
     
  3. at the end each element of the list `graphs` holds the graph of each line in `doc` 
10. build_kernel_matrix:
  1. find shortest path of each graph element
  2. each element of the sp list contains the dict which contains the neihgbors of each node in `depth`
  3. sp_g: contains the same graph but weighted
  4. K[i,j]: for each sp_g() compares the elements with each other and return the similarity percentage 
  

