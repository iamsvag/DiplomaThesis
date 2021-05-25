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

with open('negtest.neg', "w") as my_output_Neg_file:
 with open('postest.pos', "w") as my_output_Pos_file:
  data = pd.read_csv('healthcare-dataset-stroke-data.csv',usecols=[0,6,3],engine="python")

  df = pd.DataFrame(data)

  for index, row in df.iterrows():
    print(row)
    if row[0] == "Male":
      print("mpika")
      csv_row = "%s %s %s%s" %(row["gender"].replace('\n', '').replace('\r', ''),row["smoking_status"].replace('\n', '').replace('\r', ''),row["work_type"].replace('\n', '').replace('\r', ''),"\n")
      my_output_Pos_file.write(csv_row)
    else:
      print("ksanampika")
      csv_row = "%s %s %s%s" %(row["gender"].replace('\n', '').replace('\r', ''),row["smoking_status"].replace('\n', '').replace('\r', ''),row["work_type"].replace('\n', '').replace('\r', ''),"\n")
      my_output_Neg_file.write(csv_row)

 my_output_Pos_file.close()
my_output_Neg_file.close()
