from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from Sales_module.logger import log_class
import os


class Score():
    def __init__(self):
        self.folder = './Log_file/'
        self.filename = 'score_and_errors.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)

    
    def adj_r2(self,x,y,r2):
        """
        Methode: adj_r2
        Description: Caculate the adjusted r2 score
        Input: x, y, r2
        Output: adj_r2
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Calculating Adjusted r2")
            
            train_adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
            self.log_object.create_log_file("Calculated Adjusted r2: --Done")
            return train_adj_r2
        except Exception as e:
            self.log_object.create_log_file("Error in calcualating Adjusted r2 socre " + str(e))
    
    
    def evaluation_r2_score(self,act,pred):
        """
        Methode: evaluation_r2_score
        Description: Caculate the r2 score
        Input: actual values, predicted values
        Output: r2
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Calculating r2 score")
            r2_sc = r2_score(act,pred)
            self.log_object.create_log_file("Calculated r2 score: --Done")
            return r2_sc
        except Exception as e:
            self.log_object.create_log_file("Error in calcualating r2 socre " + str(e))
        
    
    
    def mae(self,act,pred):
        """
        Methode : mae
        Description: Caculate the mean absolute error
        Input: actual values, predicted values
        Output: mae
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Calculating mean absolute error")
            mae = mean_absolute_error(act,pred)
            self.log_object.create_log_file("Calculated mean absolute error: --Done")
            return mae
        except Exception as e:
            self.log_object.create_log_file("Error in calcualating mean absolute error " + str(e))

    
   
   
    def rmse(self,act,pred):
        """
        Methode: rmse
        Description: Caculate the root mean squared error
        Input: actual values, predicted values
        Output: rmse
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Calculating root mean squared error")
            mse = mean_squared_error(act,pred)
            rmse = np.sqrt(mse)
            self.log_object.create_log_file("Calculated root mean squared error: --Done")
            return rmse
        except Exception as e:
            self.log_object.create_log_file("Error in calcualating root mean squared error " + str(e))
    
    
    
    def cv_score(self,obj,X,Y):
        """
        Methode: cv_score
        Description: Caculate the cross validation score
        Input: obj, X, Y
        Output: cv_score
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Calculating cross validation score")
            cv_Score = cross_val_score(obj,X,Y,cv = 10,n_jobs = -1)
            self.log_object.create_log_file("Calculated cross validation score: --Done")
            return round(np.mean(cv_Score),2) # returning mean of cv_score
        except Exception as e:
            self.log_object.create_log_file("Error in calcualating cross validation score " + str(e))