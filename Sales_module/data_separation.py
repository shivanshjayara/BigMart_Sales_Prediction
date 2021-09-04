import warnings
warnings.filterwarnings('ignore')
from Sales_module.encoding import Encoder
import os
import pandas as pd
import numpy as np
from Sales_module.logger import log_class

class Separate_data:
    def __init__(self):
        self.folder = './Log_file/'
        self.filename = 'separate_data.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)
        
        self.coder_obj=Encoder()
        
        if not os.path.isdir('./Final_data_set/'):
            os.mkdir('./Final_data_set/')
        
    def separate(self):
        '''Defining a method that will separate the train and test data set.
        Further we are removing the outliers from the training data.
        Then saving the data set to the local system
        
        Methode: separate
        Parameter: None
        Return: Traning data set ready to be used for the model
        on failure: log error message

        Version: 1.0
        '''
        try:
            self.log_object.create_log_file("Strating Sepaaration of traning data")
            df=self.coder_obj.drop_columns()

            train_data=df[df['source']=='train']
            test_data=df[df['source']=='test']
            test_data.reset_index(drop=True,inplace=True)
            
            test_data.drop(columns=['Item_Outlet_Sales','source'],inplace=True)
            train_data.drop(columns='source',inplace=True)
            
            test_data.to_csv('./Final_data_set/'+'test_data.csv',index=False)
            return train_data
        except Exception as e:
            self.log_object.create_log_file("Error in Separation of traning data " + str(e))

    def outliers_removal(self):
        """
        Methode: outlier_removal
        Description: methode for removing outliars from the data set
        Parameter: None
        Return: Preproced data set saved to local system
        on falure: log error message

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Starting Outlier removal")
            df=self.separate()
            for i in ['Item_Visibility']:
                q1 = df[i].quantile(0.25)
                q3 = df[i].quantile(0.75)
                iqr = q3-q1
                lower = q1-(1.5*iqr)
                upper = q3+(1.5*iqr)
                df = df[(df[i]>lower) & (df[i]<upper)]
            df.to_csv('./Final_data_set/'+'train_data.csv',index=False)
            self.log_object.create_log_file("Outlier removal completed")
            self.log_object.create_log_file("Saved the data set to local system")
        
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in Outlier removal " + str(e))