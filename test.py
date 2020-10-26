import sys
import csv 
import numpy as np
import networkx as nx
from tqdm import tqdm
from utils import load_file, preprocessing, get_vocab, learn_model_and_predict

def main():
    fields=[]
    rows =[]
    filename_pos = sys.argv[1]
    #print(filename_pos)
    #filename_neg = sys.argv[2]
    window_size = int(sys.argv[2])
    depth = int(sys.argv[3])
      #  reading csv file 
    with open(filename_pos, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile)
        fields = next(csvreader) 
       # print(fields)
        for row in csvreader: 
            rows.append(row) 
       # print(rows[0])

        line = rows[0]
       # print(line)

            # get number of columns
        #for line in csvfile.readlines():
        #array = line.split(',')
       # print(line[1])
       # print("---------------------")
        #print(rows[0][1])
        docs_pos = preprocessing(rows[:][1])
        print(docs_pos)
        #print(array)
        #text = array[1]
       # print(text)
    
main()