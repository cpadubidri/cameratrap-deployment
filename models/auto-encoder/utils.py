import json
import csv

class Configuration():
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_json()

    def load_json(self):
        with open(self.filepath, 'r') as json_file:
            data = json.load(json_file)
            for key, value in data.items():
                setattr(self, key, value)
    
    def load_datapath(self):
        rows = []
        # Open the CSV file
        with open(self.datafilepath, mode='r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            next(csv_reader)
            
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Append the row to the list
                rows.append(row[0]+'image.png')
        
        return rows




if __name__=="__main__":
    conf = Configuration('assets/config/config.json')

    paths = conf.load_datapath()
    print((paths[:4]))


