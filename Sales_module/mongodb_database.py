import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from Sales_module.data_transformation import Data_transform
from Sales_module.logger import log_class
import os
import pymongo
import json

class Train_mongodb:
    """
    Class for inserting data from csv file to mongodb database
    Parameters:
    -----------
    dbname: name of database
    collection: name of collection
    """
    def __init__(self,dbname,collection):
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.dbname = self.client[dbname]
        self.collection = self.dbname[collection]      
        
        self.obj_trans = Data_transform()  # creating instance of Data Transformation Class
        
        self.folder='./Log_file/'
        self.filename='train_db_operation.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename) # creting log txt file
        
    def insert_data(self):
        """
        Method: insert_data
        Function: insert data from csv file to mongodb database
        input: None
        output: csv file
        on_error: log error

        version 1.0
        """
        try:
            df=self.obj_trans.train_withoutmissing()
            records= list(json.loads(df.T.to_json()).values())
            self.collection.insert_many(records)
            self.save_to_csv()
        except Exception as e:
            self.log_object.create_log_file("Error in insert train data in mongoDB server " + str(e))
            
    def save_to_csv(self):
        """
        Method: save_to_csv
        Function: save data from mongodb database to csv file
        input: None
        output: csv file
        on_error: log error

        version 1.0
        """
        try:
            result=self.collection.find()
            df = pd.DataFrame(result).drop(columns = '_id')
                
            if not os.path.isdir('./Data_from_database/'):
                os.mkdir('./Data_from_database/')
                
            df.to_csv('./Data_from_database/'+'train_data.csv',index=False)
            self.log_object.create_log_file("Training data saved to csv file")
            
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in getting data from mongoDB server " + str(e))
    
    
    
class Test_mongodb:
    """
    class: Test_MongoDB
    Parameters:
    -----------
    dbname: name of database
    collection: name of collection
    """
    def __init__(self,dbname,collection):
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.dbname=self.client[dbname]
        self.collection=self.dbname[collection]      
        self.obj_trans=Data_transform()
        
        self.folder='./Log_file/'
        self.filename='test_db_operation.txt'
        
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)
        
    def insert_data(self):
        """
        Method: insert_data
        Function: insert data from csv file to mongodb database
        input: None
        output: csv file
        on_error: log error

        version 1.0
        """
        try:
            df=self.obj_trans.test_withoutmissing()
            records= list(json.loads(df.T.to_json()).values())
            self.collection.insert_many(records)
            self.save_to_csv()
        except Exception as e:
            self.log_object.create_log_file("Error in insert test data in mongoDB server " + str(e))
        
    def save_to_csv(self):
        """
        Method: save_to_csv
        Function: save data from mongodb database to csv file
        input: None
        output: csv file
        on_error: log error

        version 1.0
        """
        try:
            self.log_object.create_log_file("Getting Test data from MongoDB Server")
            result=self.collection.find()
            df = pd.DataFrame(result).drop(columns = '_id')
                
            if not os.path.isdir('./Data_from_database/'):
                os.mkdir('./Data_from_database/')
                
            df.to_csv('./Data_from_database/'+'test_data.csv',index=False)
            self.log_object.create_log_file("Test data saved to csv file")
            self.log_object.create_log_file('''After inserting train and test data into new csv from database, 
                                                                                       previous transformed folder has been deleted''')
            
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in getting data from mongoDB server " + str(e))