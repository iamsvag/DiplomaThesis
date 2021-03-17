This repository has been made for the code of the experimental part of my Diploma Thesis.
University of Patras , Computer Engineer and Informatics Department


### Requirements
Code is written in Python 3.6 and requires:
* NetworkX 2.0
* tqdm 4
* scikit-learn 0.18


### Running the method
Use the following commands: 

```
$jsonTocsv.py
$csvToText.py
$ python grakelAlgorithms.py filepos fileneg window_size depth
```


where "filepos" and "fileneg" point to the positive examples and negative examples respectively, "window_size" is the size of the sliding window, and "depth" is the maximum length of shortest paths that are taken into account

