
import pandas as pd
from Sales_module.logger import log_class
import os
from Sales_module.loading_raw_data import Loading_raw

class Features:
    def __init__(self):
        self.folder = './Log_file/'
        self.filename = 'features_logs.txt'
        self.df_object = Loading_raw()
        
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)
            
    def train_numerical_features(self):
        """
        Method: train_numerical_features
        It will return all numerical features from the dataset
        return: a list of numerical features columns name
        on failer: return None, log error

        Version: 0.1
        """
        try:
            df=self.df_object.load_train()
            numerical=[i for i in df.columns if df[i].dtypes=='int' or df[i].dtypes=='float']
            self.log_object.create_log_file(f'Numerical features in Train data set: {numerical}')
            
            return numerical
        except Exception as e:
            self.log_object.create_log_file("Error in returning Numerical Features list " + str(e))
        
    
    
    def train_categorical_features(self):
        """
        Method: train_categorical_features
        It will return all categorical features from the  train dataset
        return: a list of categorical features columns name
        on failer: return None, log error

        Version: 0.1
        """
        try:
            df=self.df_object.load_train()
            categorical=[i for i in df.columns if df[i].dtypes=='O']
            self.log_object.create_log_file(f'Categorical features in Train data set: {categorical}')
            
            return categorical
        except Exception as e:
            self.log_object.create_log_file("Error in returning Categorical Features list from train data" + str(e))
    
    def test_numerical_features(self):
        """
        Method: test_numerical_features
        It will return all numerical features from the  test dataset
        return: a list of numerical features columns name
        on failer: return None, log error

        Version: 0.1
        """
        try:
            df=self.df_object.load_test()
            numerical=[i for i in df.columns if df[i].dtypes=='int' or df[i].dtypes=='float']
            self.log_object.create_log_file(f'Numerical features in Test data set: {numerical}')
            
            return numerical
        except Exception as e:
            self.log_object.create_log_file("Error in returning Numerical Features list test data " + str(e))
        
    def test_categorical_features(self):
        """
        Method: test_categorical_features
        It will return all categorical features from the  test dataset
        return: a list of categorical features columns name
        on failer: return None, log error

        Version: 0.1
        """
        try:
            df=self.df_object.load_test()
            categorical=[i for i in df.columns if df[i].dtypes=='O']
            self.log_object.create_log_file(f'Categorical features in Test data set: {categorical}')
            
            return categorical
        except Exception as e:
            self.log_object.create_log_file("Error in returning Categorical Features list from test data" + str(e))