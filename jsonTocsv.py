import json
import csv 
with open("issues_55k.json", "r") as infile: 
       json_object = json.load(infile) 
employee_data = json_object['issues'] 
data_file = open('csvneg.csv', 'w') 
csv_writer = csv.writer(data_file) 
count= 0
for emp in employee_data: 
    if count == 0: 
                
        # Writing headers of CSV file 
        header = emp.keys() 
        csv_writer.writerow(header) 
        count += 1
        break
                
    csv_writer.writerow(emp.values()) 
data_file.close() 
#json_object = json.dumps(''dictionary'', indent = 4) 
    #print(data)
    #for p in data['people']:
       # print('Name: ' + p['name'])
       # print('Website: ' + p['website'])
       # print('From: ' + p['from'])
      #  print('')