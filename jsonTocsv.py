import json
import csv 
with open("testjson1.json", "r") as infile: 
       json_object = json.load(infile) 
employee_data = json_object['issues'] 
data_file = open('csvFew.csv', 'w') 
csv_writer = csv.writer(data_file) 
count= 0
for emp in employee_data: 
    if count == 0: 
                
        # Writing headers of CSV file 
        header = emp.keys() 
        csv_writer.writerow(header) 
        count += 1
       
                
    csv_writer.writerow(emp.values()) 
data_file.close() 
