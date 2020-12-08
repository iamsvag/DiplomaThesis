import csv 
import numpy as np
import networkx as nx
import pandas as pd
import sys
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

with open('negative500.neg', "w") as my_output_Neg_file:
 with open('positive500.pos', "w") as my_output_Pos_file:
  data = pd.read_csv('csv500.csv',usecols=[0,1,2,4,7],engine="python")

  df = pd.DataFrame(data)

  for index, row in df.iterrows():
    
    if row[3] == "Bug":
      csv_row = "%s %s %s %s %s%s" %(row[0].replace('\n', '').replace('\r', ''),row["title"].replace('\n', '').replace('\r', ''),row["description"].replace('\n', '').replace('\r', ''),row["type"].replace('\n', '').replace('\r', ''),row["text"].replace('\n', '').replace('\r', ''),"\n")
      my_output_Pos_file.write(csv_row)
    else:
      csv_row = "%s %s %s %s %s%s" %(row[0].replace('\n', '').replace('\r', ''),row["title"].replace('\n', '').replace('\r', ''),row["description"].replace('\n', '').replace('\r', ''),row["type"].replace('\n', '').replace('\r', ''),row["text"].replace('\n', '').replace('\r', ''),"\n")
      my_output_Neg_file.write(csv_row)

 my_output_Pos_file.close()
my_output_Neg_file.close()