import pandas as pd
from Sales_module.loading_raw_data import Loading_raw
from Sales_module.logger import log_class
import os
import shutil
from Sales_module.dataframe_features import Features



class Validation:
    def __init__(self):
        self.folder='./Log_file/'
        self.filename = 'raw_validation_logs.txt'
        self.df_object = Loading_raw()
        self.feature_object = Features()
        
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)
    
    
# training standard deviation checking for zero value
    def train_data_stdzero(self):
        """
        Method to check if the standard deviation of the numerical features is zero or not.
        If the standard deviation is zero then we are dropping that feature.
        Methode: train_data_stdzero()
        input: None
        output: train_data with no zero standard deviation features
        on failure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('Check if standard deviation of the numericals features is zero or not.')
            self.log_object.create_log_file('Loading train dataset form the previous class method')
            
            train_data=self.df_object.load_train()
            numerical_features=self.feature_object.train_numerical_features()
            
            for feature in numerical_features:
                if train_data[feature].std()==0:
                    train_data.drop(columns=feature,axis=1, inplace=True)
            self.log_object.create_log_file("Zero std columns has been removed")
            return train_data
        except Exception as e:
            self.log_object.create_log_file('Error in method train_data_stdzero: ' + str(e))
    
# testing standard deviation checking for zero value
    def test_data_stdzero(self):
        """
        Method to check if the standard deviation of the numerical features is zero or not.
        If the standard deviation is zero then we are dropping that feature.
        Methode: test_data_stdzero()
        input: None
        output: test_data with no zero standard deviation features
        on failure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('Check if standard deviation of the numericals features is zero or not.')
            self.log_object.create_log_file('Loading test dataset form the previous class method')
            
            test_data=self.df_object.load_test()
            numerical_features=self.feature_object.test_numerical_features()
            
            for feature in numerical_features:
                if test_data[feature].std()==0:
                    test_data.drop(columns=feature,axis=1,inplace=True)
            self.log_object.create_log_file("Zero std columns has been removed")
            return test_data
        except Exception as e:
            self.log_object.create_log_file('Error in method test_data_stdzero: ' + str(e))
    
    

#checking if there is any column where all values are misisng
    def train_whole_missing_values(self):
        """
        Methode to check if there is any column in training data set where a column is having complete missing data. 
        If present then we are dropping that columns
        Method: train_whole_missing_values()
        input: None
        output: train_data with no columns having complete missing data
        on failure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('''Checking if there is any column in training data set where a column is 
                                               having complete missing data. If present then we are dropping that columns''')
            self.log_object.create_log_file('Loading training data set after removing zero standard deviation features')
            train_data=self.train_data_stdzero()
            
            for i in train_data.columns:
                if train_data[i].isnull().sum()==len(train_data[i]):
                    train_data.drop(columns=i,axis=1,inplace=True)
                else:
                    pass
            self.log_object.create_log_file("Columns having complete missing data has been removed")
            return train_data
        except Exception as e:
            self.log_object.create_log_file('Error in method train_whole_missing_values: ' + str(e))
    

#checking if there is any column where all values are misisng
    def test_whole_missing_values(self):
        """
        Methode to check if there is any column in test data set where a column is having complete missing data.
        If present then we are dropping that columns
        Method: test_whole_missing_values()
        input: None
        output: test_data with no columns having complete missing data
        on failure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('''Checking if there is any column in testing data set where a column is having 
                                                complete missing data. If present then we are dropping that columns''')
            self.log_object.create_log_file('Loading testing data set after removing zero standard deviation features')
            test_data = self.test_data_stdzero()
            
            for i in test_data.columns:
                if test_data[i].isnull().sum() == len(test_data[i]):
                    test_data.drop(columns=i,axis = 1,inplace=True)
                else:
                    pass
            self.log_object.create_log_file("Columns having complete missing data has been removed")
            return test_data
        except Exception as e:
            self.log_object.create_log_file('Error in method test_whole_missing_values: ' + str(e))