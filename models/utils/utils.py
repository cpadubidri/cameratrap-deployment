import json


class Configuration():
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_json()

    def load_json(self):
        with open(self.filepath, 'r') as json_file:
            data = json.load(json_file)
            for key, value in data.items():
                setattr(self, key, value)


import logging
import sys

def starlogger(name,filename='training.log',console=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if console:
        logger.addHandler(console_handler)

    return logger