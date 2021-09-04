
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
from Sales_module.logger import log_class

class Preprocessing:
    def __init__(self):
        self.train = pd.read_csv('./Data_from_database/train_data.csv')
        self.test = pd.read_csv('./Data_from_database/test_data.csv')
        self.folder = './Log_file/'
        self.filename = 'data_preprocessing.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object=log_class(self.folder,self.filename)
    
    def combining_data(self):
        """
        Methode: combining_data
        Description: Combining the train and test data.
        out: combined data of train and test data
        on fail: return None, log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Combining Traning and test data:- Started")
            df=pd.concat([self.train,self.test], axis=0,ignore_index=True)
            self.log_object.create_log_file("Combining Traning and test data - Done")
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in combining data " + str(e))
    
    def item_visibility(self):
        '''Some of the item visibility is shown 0. But this is not possible because 
           if item is present then there must be some visibility. May be the item has kept far away from the customer
           due to which it is not actually not visible.
           So we will gonna take the average visibility of the item w.r.t. item identifier.
           So here we will replace the 0 visibility with the average value using pivot table.
           
           output: preprocessed data frame by remove Item_visibility column
           on fail: log error
           
           Version: 1.0
        '''
        try:
            self.log_object.create_log_file("Start filling 0 to Item-visibility:- Started")
            df=self.combining_data()
            visibility_avg = df.pivot_table(values='Item_Visibility', index='Item_Identifier')
            
            miss_bool = (df['Item_Visibility'] == 0)
            
            df.loc[miss_bool,'Item_Visibility'] = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
            self.log_object.create_log_file("Filling 0 to Item-visibility - Done")
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in filling 0 to Item-visibility " + str(e))


    def new_item_identifier_and_years_column(self):
        """
        Methode: new_item_identifier_and_years_column
        Description: Creating new item identifier and years column.
        out: preprocessed data frame
        on fail: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Creating new item identifier and years column: - Started")
            df=self.item_visibility()
            
            df['New_item_type'] = df['Item_Identifier'].apply(lambda a:a[:2])
            df['Outlet_years'] = 2021-df['Outlet_Establishment_Year']
            self.log_object.create_log_file("Creating new item identifier and years column - Done")
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in creating new item identifier and years column " + str(e))
        
    def mapping_fat_content(self):
        """
        Methode: mapping_fat_content
        Description: Creating fat content mapping.
        out: preprocessed data frame
        on fail: log error

        Version: 1.0
        """
        try:
            self.log_object.create_log_file("Creating mapping fat content- Started")
            df=self.new_item_identifier_and_years_column()
            
            fat_content={'Low Fat':'low', 'Regular':'reg', 'LF':'low', 'reg':'reg', 'low fat':'low'}
            df['Item_Fat_Content']=df['Item_Fat_Content'].map(fat_content)
            self.log_object.create_log_file("Creating mapping fat content- Done")
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in mapping fat content " + str(e))

    
    def not_edible_item(self):
        '''Some of the Item type are are household items and they are not consumable but they are assigned with some fat content. 
           So here we are separating them and creating another category named as : "non-edible" 
           
           output: preprocessed data frame by removing 'NC' from item fat content column
           on fail: log error

           Version: 1.0
        '''
        try:
            self.log_object.create_log_file("Creating mapping fat content- Started")
            df=self.mapping_fat_content()
            df['Item_Fat_Content'].loc[(df['New_item_type']=='NC')]='non_edible'
            return df
        except Exception as e:
            self.log_object.create_log_file("Error in mapping fat content " + str(e))