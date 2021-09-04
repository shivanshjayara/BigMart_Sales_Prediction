import warnings
warnings.filterwarnings('ignore')
from Sales_module.encoding import Encoder
import os
import pandas as pd
import numpy as np
from Sales_module.logger import log_class
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


class Data_scaling:
    def __init__(self):
        self.folder = './Log_file/'
        self.filename = 'splitting_and_scaling.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder,self.filename)
        
    
    def train_test_split(self):
        """
        Method: train_test_split
        Description: This method is used to split the data into train and test set.
        Parameters: None
        Return: train_data,test_data,train_label,test_label
        on fail: None, log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Start splitting dataset into training and test set")
            train = pd.read_csv('./Final_data_set/train_data.csv')
        
            X = train.drop(columns = 'Item_Outlet_Sales')
            Y = train['Item_Outlet_Sales']
        
            x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.09,random_state=101)
            self.log_object.create_log_file("Training and test set split completed")
        
            return x_train,x_test,y_train,y_test
        except Exception as e:
            self.log_object.create_log_file("Error in splitting dataset into training and test set "+ str(e))
    

    
    def scaling(self):
        """
        Method: scaling
        Description: This method is used to scale the data.
        Parameters: None
        Return: scaled training data, scaled test data
        on fail: None, log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Start scaling the data")
            x_train, x_test, y_train, y_test = self.train_test_split()
    
            std = StandardScaler() # creating instance of StandardScaler class
            x_train_std = std.fit_transform(x_train)
            x_test_std=std.transform(x_test)
            self.log_object.create_log_file("Data scaling completed")
        
            with open('standard_scaler.pkl','wb') as file:
                pickle.dump(std,file)
            self.log_object.create_log_file("scaling model saved")
        
            return x_train_std, x_test_std, y_train, y_test
        except Exception as e:
            self.log_object.create_log_file("Error in scaling the data "+ str(e))