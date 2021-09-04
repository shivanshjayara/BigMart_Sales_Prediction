from flask import Flask, request, render_template,jsonify
import sklearn
from sklearn.preprocessing import StandardScaler
from Sales_module.encoding import Encoder
from Sales_module.logger import log_class
from Sales_module.mongodb_database import Train_mongodb,Test_mongodb
from Sales_module.loading_raw_data import Loading_raw
from Sales_module.data_transformation import Data_transform
from Sales_module.data_preprocessing import Preprocessing
from Sales_module.scaling_and_splitting import Data_scaling
from Sales_module.data_separation import Separate_data
from Sales_module.dataframe_features import Features
from Sales_module.raw_validation import Validation
from Sales_module.model_tuning_and_training import Parameter_tuning
from Sales_module.scores_and_errors import Score
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/train',methods=['POST'])
def training():
    if (request.method == 'POST'):
        operation = request.json['operation']
        if (operation.lower() == 'training'):
            load_obj=Loading_raw()
            load_obj.load_train()
            load_obj.load_test()

            # feature_obj=features()
            '''Validating all the raw data, both training as well as testing.
            Here we will check for zero standard deviation and column which is fully null'''
            validation_obj=Validation()
            validation_obj.train_whole_missing_values()
            validation_obj.test_whole_missing_values()

            '''Before inserting into data frame data transformation is required so 
                that database will able to store the data properly'''
            transform_obj=Data_transform()
            transform_obj.train_withoutmissing()
            transform_obj.test_withoutmissing()

            ''''MongoDB database we will be using for storing the data.
                Two separate table are created for train data set and test data set'''
            mongotrain_obj=Train_mongodb('sales_db','train_data_table')
            mongotrain_obj.insert_data()
            mongotest_obj=Test_mongodb('sales_db','test_data_table')
            mongotest_obj.insert_data()

            '''Data from database will be extracted and then we will perform preprocessing operation
               before doing any model creation'''
            preprocess_obj=Preprocessing()
            preprocess_obj.not_edible_item()

            '''There are some attributes which are categorical in nature. So in this class we will converting them into numerical bits'''
            encoder_obj=Encoder()
            encoder_obj.drop_columns()

            '''Both train and test data will be separated in this class and outliers will be removed form the trainig data set'''
            separate_obj=Separate_data()
            separate_obj.outliers_removal()

            '''Feature scaling by using standard scaler library will be used here'''
            scaling_obj=Data_scaling()
            scaling_obj.scaling()

            '''Model tunning by using Randomized Search cv will be  used here tunning the parameter before model creation.
                Then model will be created.
                4 models are created Linear regression, Gradient Boost, XGBoost, Randomforest.
                Which ever will give the best evaluation score and less RMSE that will be sleceted as the best model in separate folde'''
            trainmodel_obj=Parameter_tuning()
            result=trainmodel_obj.model_result()

            best_model_name=result[0]
            rmse = result[1]

            return jsonify(f'Best_model: {best_model_name} with RMSE:{round(rmse,2)}')

if __name__ == '__main__':
    app.run(port=5000, debug=True)