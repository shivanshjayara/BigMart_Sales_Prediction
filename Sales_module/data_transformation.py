
import pandas as pd
import numpy as np
from Sales_module.logger import log_class
from Sales_module.loading_raw_data import Loading_raw
from Sales_module.raw_validation import Validation
import os
from Sales_module.dataframe_features import Features
import warnings
warnings.filterwarnings('ignore')

class Data_transform:
    def __init__(self):
        self.folder = './Log_file/'
        self.filename = 'data_transformation.txt'
        self.val_object = Validation()
        self.feature = Features()
        
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder,self.filename)
        
    def combine_data(self):
        """
        Method: combine_data
        This method combines the train and test data set

        input: None
        output: combined data set
        on error: return None and log error

        Version: 1.0    

        """
        try:
            self.log_object.create_log_file('''Defining a function that will combining both train data and test data together 
                                                                                                   into a single dataframe''')
            self.log_object.create_log_file('''Creating another feature for train and test named "source" which will store the 
                                                                                                values as "train" and "test" ''')
    
            train_data = self.val_object.train_whole_missing_values()
            train_data['source'] = 'train' # add new column for idenitfy the train data
            
            test_data = self.val_object.test_whole_missing_values()
            test_data['source'] = 'test' # add new column for idenitfy the test data

            df = pd.concat([train_data,test_data], axis=0, ignore_index=True)
            self.log_object.create_log_file('Combining train and test data set Successful')
            return df
        except Exception as e:
            self.log_object.create_log_file('Error in combining the train and test data set' + str(e))
        
        
    def fill_missing(self):
        """
        Method: fill_missing
        This method will fill the missing values in the data set

        input: None
        output: data set with missing values filled
        on error: return None and log error

        Version: 1.0
        """
        self.log_object.create_log_file('Filling the missing values in the training data')
        try:
            df = self.combine_data() 
            self.log_object.create_log_file("Data set Combine Successful")
        except Exception as e:
            self.log_object.create_log_file('Error in combining the train and test data set' + str(e))
        
        numerical=self.feature.train_numerical_features()
        categorical=self.feature.train_categorical_features()
        
        missing_num_features = [i for i in numerical if df[i].isnull().sum()>0]
        missing_cat_features = [i for i in numerical if df[i].isnull().sum()>0]
        
        self.log_object.create_log_file(f'Numerical features having missing values are: {missing_num_features}')
        self.log_object.create_log_file(f'Categorical features having missing values are: {missing_cat_features}')

        #filling numerical missing values
        try:
            self.log_object.create_log_file('Filling numerical missing values in the training data: Started')
            self.log_object.create_log_file('Creating a pivot table displaying the mean weight of every respective items')
            item_avg_weight = df.groupby(["Item_Identifier"])["Item_Weight"].mean()
            self.log_object.create_log_file(f"Missing values in Item_weight column before filling:{df['Item_Weight'].isnull().sum()}")

            miss_bool = df['Item_Weight'].isnull()
            df.loc[miss_bool,'Item_Weight'] = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
            self.log_object.create_log_file('Missing values in Item_Weight column is filled ')
            self.log_object.create_log_file(f"Missing values in Item_weight column after filling: {df['Item_Weight'].isnull().sum()}")
            self.log_object.create_log_file("Missing Value Imputation Completed For Numerical Features.")
        except Exception as e:
            self.log_object.create_log_file('Error in filling numerical missing values in the training data:\n' + str(e))

        try:
            self.log_object.create_log_file('Filling categorical missing values in the training data: Started')
            #filling categorical missing values
            mode_outlet_size = df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda a: a.mode()[0]))
            df_1 = df[df['Outlet_Type']=='Grocery Store']
            df_2 = df[df['Outlet_Type']=='Supermarket Type1']
            df_3 = df[df['Outlet_Type']=='Supermarket Type2']
            df_4 = df[df['Outlet_Type']=='Supermarket Type3']
            
            groc=mode_outlet_size.loc['Outlet_Size','Grocery Store']
            super1=mode_outlet_size.loc['Outlet_Size','Supermarket Type1']
            super2=mode_outlet_size.loc['Outlet_Size','Supermarket Type2']
            super3=mode_outlet_size.loc['Outlet_Size','Supermarket Type3']
        
            self.log_object.create_log_file(f"Missing values in Outlet_Size column before filling:{df['Outlet_Size'].isnull().sum()}")       
            df_1['Outlet_Size'].fillna(groc,inplace=True)
            df_2['Outlet_Size'].fillna(super1,inplace=True)
            df_3['Outlet_Size'].fillna(super2,inplace=True)
            df_4['Outlet_Size'].fillna(super3,inplace=True)       
            df2=pd.concat([df_1,df_2,df_3,df_4]).sort_index(axis=0)
            self.log_object.create_log_file(f'''Missing values in Outlet_Size column afetr filling na is
                                                                                           {df2['Outlet_Size'].isnull().sum()}''')
            self.log_object.create_log_file("Missing Value Imputation Completed For Categorical Features.")
        except Exception as e:
            self.log_object.create_log_file('Error in filling categorical missing values in the training data' + str(e))
#         if not os.path.isdir('./Transformed_data/'):
#             os.mkdir('./Transformed_data/')
            
#         df2.to_csv('./Transformed_data/'+'combine_data_after_filling_missing_values.csv',index=False,header=True)
#         self.log_object.create_log_file('Combine data set without any missing values is saved in local system')
        
        return df2 # return the data set with missing values filled
    
    def train_withoutmissing(self):
        """
        Method: train_withoutmissing
        This method will seperate the training data from combined data set.

        input: None
        output: Training data set without any missing values
        on error: return None and log error

        Version: 1.0
        """
        try:

            self.log_object.create_log_file("Filling missing value on combined data set: Started")
            df=self.fill_missing()
            self.log_object.create_log_file("Filling missing value on combined data set: Completed")
            
            train_data = df[df['source']=='train']
            self.log_object.create_log_file('Creating a new column "source" in train dataset to give that data set an identity')
            self.log_object.create_log_file("Traning data Separated: Completed")
            
            return train_data
        except Exception as e:
            self.log_object.create_log_file('Error in separting the train data from the combine dataset' + str(e))
 
    
    
    def test_withoutmissing(self):
        """
        Method: test_withoutmissing
        This method will seperate the test data from combined data set.

        input: None
        output: Test data set without any missing values
        on error: return None and log error

        Version: 1.0
        """
        try:
            
            self.log_object.create_log_file('Define a function that will separte the test data from the combine dataset')
            self.log_object.create_log_file("Filling missing value on combined data set: Started")
            df=self.fill_missing()
            self.log_object.create_log_file("Filling missing value on combined data set: Completed")
     
            test_data = df[df['source']=='test']
            self.log_object.create_log_file('Creating a new column "source" in test dataset to give that data set an identity')
            self.log_object.create_log_file("Test data Separated: Completed")
        
            test_data.reset_index(drop=True,inplace=True)
            test_data.drop(columns='Item_Outlet_Sales',inplace=True)
            
            return test_data
        except Exception as e:
            self.log_object.create_log_file('Error in separting the test data from the combine dataset' + str(e))

