
import warnings
warnings.filterwarnings('ignore')
from Sales_module.logger import log_class
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Sales_module.dataframe_features import Features
import os
from Sales_module.data_preprocessing import Preprocessing

class Encoder():
    def __init__(self):
        self.obj_pre = Preprocessing()
        self.obj_feature = Features()
        self.folder = './Log_file/'
        self.filename = 'encoding.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)
        
    def label_encoding(self):
        """
        Methode: label_encoding
        Function to Encoding the categorical variables
        on fail: log error

        version: 1.0
        """
        try:
            df=self.obj_pre.not_edible_item()
            categorical=[i for i in df.columns if df[i].dtypes=='O']
            cat_col=[]
            col_del=[]
            for i in categorical:
                if (i!='source') and (i!='Item_Identifier') and (i!='Item_Type'):
                    cat_col.append(i)
                else:
                    col_del.append(i)
            
#             le = LabelEncoder()

#             for col in cat_col:
#                 df[col] = le.fit_transform(df[col])
                
            df = pd.get_dummies(df, columns=cat_col,drop_first=True)
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in Label Encoding " + str(e))
    
    
    def drop_columns(self):
        """
        Methode: drop_columns
        Function to drop the columns which are not required
        on fail: log error

        Version: 1.0
        """
        try:
            df=self.label_encoding()
            df.drop(columns=['Item_Identifier','Item_Type','Outlet_Establishment_Year'],inplace=True,axis=1)
            
#             df.to_csv('combine_data_after_label_encoding.csv',index=False)
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in Drop Columns " + str(e))

