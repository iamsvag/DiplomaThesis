import csv 
import numpy as np
import networkx as nx
import pandas as pd

with open('positive.pos', "w") as my_output_file:
  data = pd.read_csv('positive.csv',engine="python")
  df = pd.DataFrame(data) 
  for index, row in df.iterrows(): 
    csv_row = "%s %s %s %s %s%s" %(row["key"].replace('\n', '').replace('\r', ''),row["title"].replace('\n', '').replace('\r', ''),row["description"].replace('\n', '').replace('\r', ''),row["type"].replace('\n', '').replace('\r', ''),row["text"].replace('\n', '').replace('\r', ''),"\n")
    my_output_file.write(csv_row)
my_output_file.close()