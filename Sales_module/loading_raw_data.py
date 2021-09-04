import pandas as pd
from Sales_module.logger import log_class
import os

class Loading_raw:
    """Class for loading raw data from local source
    """
    def __init__(self):
        self.folder='./Log_file/'
        self.filename='raw_data_logs.txt'
        
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder,self.filename)
    
    def load_train(self):
        """
        Methode: load_train

        input: None
        output: return train_data as pandas DataFrame
        on fail: return None and log error

        version: 1.0

        """
        try:
            self.log_object.create_log_file('Loading training data set from the local source into pandas DataFrame')
            train_data = pd.read_csv('Train.csv')
            return train_data
        except Exception as e:
            self.log_object.create_log_file("Fail while loading raw train data from local: " + str(e))
            raise e
    
    def load_test(self):
        """
        Methode: load_test

        input: None
        output : return test_data as pandas DataFrame
        on fail: return None and log error

        version: 1.0
        """
        try:
            self.log_object.create_log_file('Loading training data set from the local source into pandas DataFrame')
            test_data = pd.read_csv('Test.csv')
            return test_data
        except Exception as e:
            self.log_object.create_log_file("Fail while loading raw test data from local: " + str(e))
            raise e