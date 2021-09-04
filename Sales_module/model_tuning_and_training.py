from math import e
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
from Sales_module.scaling_and_splitting import Data_scaling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn. ensemble import GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from Sales_module.scores_and_errors import Score


class Parameter_tuning:
    def __init__(self):
        self.obj_scale=Data_scaling()
        self._model_list=[]
        self.obj_score=Score()
        self.train = pd.read_csv('./Final_data_set/train_data.csv')
        self.X = self.train.drop(columns = 'Item_Outlet_Sales')
        self.Y = self.train['Item_Outlet_Sales']
        self.dict_model={}
        
        self.folder = './Log_file/'
        self.filename = 'model_tune_train.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder,self.filename)
        
        
    def parameters(self):
        """
        Method: parameters
        Description: This method is used to define the parameters for the model
        Parameters: None
        Return: parameters for individual models

        Version: 1.0
        """
        lr_parameters={'fit_intercept':[False,True],
                       'normalize':[False,True],
                       'copy_X':[False,True],
                       'positive':[False,True]
                       }     
        gb_parameters = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 
                         'n_estimators':[100,250,500,750,1000,1250,1500,1750],
                         'max_depth':[2,3,4,5,6,7],
                         'min_samples_split':[2,4,6,8,10,20,40,60,100],
                         'min_samples_leaf':[1,3,5,7,9],
                         'max_features':[2,3,4,5,6,7],
                         'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]
                        }
        rf_parameters={'n_estimators': [100,200,300,400,500,600],
                       'max_features': ['auto', 'sqrt'],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4],
                       'bootstrap': [True, False]
                      }
        xgb_parameters={"learning_rate":[0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
                        "max_depth":[ 3, 4, 5, 6, 8, 10, 12, 15],
                        "min_child_weight":[ 1, 3, 5, 7 ],
                        "gamma":[ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                        "colsample_bytree":[ 0.3, 0.4, 0.5 , 0.7 ]
                       }
        return lr_parameters, gb_parameters, rf_parameters, xgb_parameters
        
        
    def lr_tuning(self):
        """
        Method: lr_tuning
        Description: This method is used to tune the parameters for the linear regression model
        Parameters: None
        Return: Best hyperparameters for the linear regression model adn tuned model

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('Starting Tuning Linear Regression Model')
            lr_parameters = self.parameters()[0]
            x_train, x_test, y_train, y_test = self.obj_scale.scaling()
            lr_reg = LinearRegression()
            
            random_lr = RandomizedSearchCV(estimator = lr_reg,
                                        param_distributions=lr_parameters,
                                        cv = 5,
                                        scoring='neg_root_mean_squared_error',
                                        n_iter=10,
                                        n_jobs=-1,
                                        verbose=2,
                                        random_state=101
                                        )
            self.log_object.create_log_file('Start Tuning on Randomized Search CV...')
            random_lr.fit(x_train,y_train) 
            self.log_object.create_log_file('Tuning on Randomized Search CV Completed')       
            best_param = random_lr.best_params_
            
            # saving best parameters obtain from Search
            self.log_object.create_log_file(f"""Best Parameters for Linear Regression Model Obtain from Search:- \n
                                                positive: {str(best_param['positive'])} \n 
                                                normalize: {str(best_param['normalize'])} \n
                                                fit_interecpt: {str(best_param['fit_intercept'])} \n
                                                copy_X:  {str(best_param['copy_X'])} \n""")  
         
            lr_model = LinearRegression(positive = best_param['positive'],
                                        normalize = best_param['normalize'],
                                        fit_intercept = best_param['fit_intercept'] ,
                                        copy_X = best_param['copy_X']) 

            self.log_object.create_log_file('Starting Training on Best Parameters of Linear Regression Model')
            lr_model.fit(x_train,y_train)
            self.log_object.create_log_file('Training on Best Parameters of Linear Regression Model Completed')
            try:
                r2=lr_model.score(x_train,y_train)
                self.log_object.create_log_file(f"R2 Score on Training Data: {str(r2)}")
                train_adj_r2=self.obj_score.adj_r2(x_train,y_train,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Training Data: {str(train_adj_r2)}")
                test_adj_r2=self.obj_score.adj_r2(x_test,y_test,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Testing Data: {str(test_adj_r2)}")
                y_pred = lr_model.predict(x_test)
                evaluation_score=self.obj_score.evaluation_r2_score(y_test,y_pred)
                self.log_object.create_log_file(f"R2 Score on Testing Data: {str(evaluation_score)}")
                mae=self.obj_score.mae(y_test,y_pred)
                self.log_object.create_log_file(f"Mean Absolute Error(mae) on Testing Data: {str(mae)}")
                rmse=self.obj_score.rmse(y_test,y_pred)
                self.log_object.create_log_file(f"Root Mean Squared Error(rsme) on Testing Data: {str(rmse)}")
                cv_score=self.obj_score.cv_score(lr_model,self.X,self.Y)
                self.log_object.create_log_file(f"Cross Validation Score(CV) on Training Data: {str(cv_score)}")
            except Exception as e:
                self.log_object.create_log_file(f"Error in calculating Score for Training Data: {str(e)}")
            
            self.dict_model['linear_regression']=[lr_model,(evaluation_score+rmse)/2]
        
        except Exception as e:
            self.log_object.create_log_file(f"Error on tuning Linear Regression model: \n {str(e)}")   
    
    
    
    def gb_tuning(self):
        """
        Method: gb_tuning
        Description: This method is used to tune the parameters for the gradient boosting model
        Parameters: None
        Return: Best hyperparameters for the gradient boosting model adn tuned model

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('Starting Tuning Gradient Boosting Model')
            gb_parameters = self.parameters()[1]
            x_train, x_test, y_train, y_test = self.obj_scale.scaling()
            gb_reg = GradientBoostingRegressor()
            
            random_gb = RandomizedSearchCV(estimator=gb_reg,
                                        param_distributions=gb_parameters,
                                        cv=5,
                                        scoring='neg_root_mean_squared_error',
                                        n_iter=10,
                                        n_jobs=-1,
                                        verbose=2,
                                        random_state=101
                                        )
            self.log_object.create_log_file('Start Tuning on Randomized Search CV...')
            random_gb.fit(x_train,y_train)
            self.log_object.create_log_file('Tuning on Randomized Search CV Completed')
            best_param = random_gb.best_params_
            
            # saving best parameters obtain from Search
            self.log_object.create_log_file(f"""Best Parameters for Gradient Boosting Model Obtain from Search:- \n
                                                learning_rate: {str(best_param['learning_rate'])} \n
                                                subsample: {str(best_param['subsample'])} \n
                                                min_samples_split: {str(best_param['min_samples_split'])} \n
                                                min_samples_leaf: {str(best_param['min_samples_leaf'])} \n
                                                max_depth: {str(best_param['max_depth'])} \n
                                                max_features: {str(best_param['max_features'])} \n
                                                n_estimators: {str(best_param['n_estimators'])}""")

            gb_model = GradientBoostingRegressor(subsample = best_param['subsample'],
                                                n_estimators = best_param['n_estimators'],
                                                min_samples_split = best_param['min_samples_split'],
                                                min_samples_leaf = best_param['min_samples_leaf'],
                                                max_features = best_param['max_features'],
                                                max_depth = best_param['max_depth'],
                                                learning_rate = best_param['learning_rate'])
            self.log_object.create_log_file('Starting Training on Best Parameters of Gradient Boosting Model')
            gb_model.fit(x_train,y_train)
            self.log_object.create_log_file('Training on Best Parameters of Gradient Boosting Model Completed')
            try:
                r2 = gb_model.score(x_train,y_train)
                self.log_object.create_log_file(f"R2 Score on Training Data: {str(r2)}")
                train_adj_r2=self.obj_score.adj_r2(x_train,y_train,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Training Data: {str(train_adj_r2)}")
                test_adj_r2=self.obj_score.adj_r2(x_test,y_test,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Testing Data: {str(test_adj_r2)}")
                y_pred = gb_model.predict(x_test)
                evaluation_score=self.obj_score.evaluation_r2_score(y_test,y_pred)
                self.log_object.create_log_file(f"Evaluation Score on Testing Data: {str(evaluation_score)}")
                mae=self.obj_score.mae(y_test,y_pred)
                self.log_object.create_log_file(f"Mean Absolute Error(mae) on Testing Data: {str(mae)}")
                rmse=self.obj_score.rmse(y_test,y_pred)
                self.log_object.create_log_file(f"Root Mean Squared Error(rsme) on Testing Data: {str(rmse)}")
                cv_score=self.obj_score.cv_score(gb_model,self.X,self.Y)
                self.log_object.create_log_file(f"Cross Validation Score(CV) on Training Data: {str(cv_score)}")
            except Exception as e:
                self.log_object.create_log_file(f"Error in calculating Score for Training Data: {str(e)}")
            
            self.dict_model['gradient_boost']=[gb_model,(evaluation_score+rmse)/2]
        
        except Exception as e:
            self.log_object.create_log_file(f"Error on tuning Gradient Boosting model: \n {str(e)}")

                
    def rf_tuning(self):
        """
        Method: rf_tuning
        Description: This method is used to tune the parameters for the random forest model
        Parameters: None
        Return: Best hyperparameters for the random forest model adn tuned model

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('Starting Tuning Random Forest Model')
            rf_parameters = self.parameters()[2]
            x_train, x_test, y_train, y_test = self.obj_scale.scaling()
            
            rf_reg = RandomForestRegressor()
            random_rf = RandomizedSearchCV(estimator=rf_reg,
                                        param_distributions=rf_parameters,
                                        cv=5,
                                        scoring='neg_root_mean_squared_error',
                                        n_iter=10,
                                        n_jobs=-1,
                                        verbose=2,
                                        random_state=101
                                        )
            self.log_object.create_log_file('Start Tuning on Randomized Search CV...')
            random_rf.fit(x_train,y_train)
            self.log_object.create_log_file('Tuning on Randomized Search CV Completed')
            best_param = random_rf.best_params_

            # saving best hyperparameters obtain from Search
            self.log_object.create_log_file(f"""Best Parameters for Random Forest Model Obtain from Search:- \n
                                                n_estimators: {str(best_param['n_estimators'])} \n
                                                min_samples_split: {str(best_param['min_samples_split'])} \n
                                                min_samples_leaf: {str(best_param['min_samples_leaf'])} \n
                                                max_features: {str(best_param['max_features'])} \n
                                                max_depth: {str(best_param['max_depth'])} \n
                                                bootatrap: {str(best_param['bootstrap'])}""")
            
            rf_model = RandomForestRegressor(n_estimators = best_param['n_estimators'],
                                            min_samples_split = best_param['min_samples_split'],
                                            min_samples_leaf = best_param['min_samples_leaf'],
                                            max_features = best_param['max_features'],
                                            max_depth = best_param['max_depth'],
                                            bootstrap = best_param['bootstrap'])
            
            self.log_object.create_log_file('Starting Training on Best Parameters of Random Forest Model')
            rf_model.fit(x_train,y_train)
            try:
                self.log_object.create_log_file('Training on Best Parameters of Random Forest Model Completed')
                r2=rf_model.score(x_train,y_train)
                self.log_object.create_log_file(f"R2 Score on Training Data: {str(r2)}")
                train_adj_r2=self.obj_score.adj_r2(x_train,y_train,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Training Data: {str(train_adj_r2)}")
                test_adj_r2=self.obj_score.adj_r2(x_test,y_test,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Testing Data: {str(test_adj_r2)}")
                y_pred = rf_model.predict(x_test)
                evaluation_score=self.obj_score.evaluation_r2_score(y_test,y_pred)
                self.log_object.create_log_file(f"Evaluation Score on Testing Data: {str(evaluation_score)}")
                mae=self.obj_score.mae(y_test,y_pred)
                self.log_object.create_log_file(f"Mean Absolute Error(mae) on Testing Data: {str(mae)}")
                rmse=self.obj_score.rmse(y_test,y_pred)
                self.log_object.create_log_file(f"Root Mean Squared Error(rsme) on Testing Data: {str(rmse)}")
                cv_score=self.obj_score.cv_score(rf_model,self.X,self.Y)
                self.log_object.create_log_file(f"Cross Validation Score(CV) on Training Data: {str(cv_score)}")
            except Exception as e:
                self.log_object.create_log_file(f"Error in calculating Score for Training Data: {str(e)}")
            
            self.dict_model['random_forest']=[rf_model,(evaluation_score+rmse)/2]
        
        except Exception as e:
            self.log_object.create_log_file(f"Error on tuning Random Forest model: \n {str(e)}")

    
    def xgb_tuning(self):
        """
        Method: xgb_tuning
        Description: This method is used to tune the parameters for the XGBoost model
        Parameters: None
        Return: Best hyperparameters for the XGBoost model adn tuned model

        Version: 1.0
        """
        try:
            xgb_parameters = self.parameters()[3]
            x_train, x_test, y_train, y_test = self.obj_scale.scaling()

            xgb_reg = XGBRegressor()
            random_xgb = RandomizedSearchCV(estimator=xgb_reg,
                                            param_distributions=xgb_parameters,
                                            cv=5,
                                            scoring='neg_root_mean_squared_error',
                                            n_iter=10,
                                            n_jobs=-1,
                                            verbose=2,
                                            random_state=101
                                        )
            self.log_object.create_log_file('Start Tuning on Randomized Search CV...')
            random_xgb.fit(x_train,y_train)
            self.log_object.create_log_file('Tuning on Randomized Search CV Completed')
            best_param = random_xgb.best_params_

            # adding best hyperparameters obtain from Search
            self.log_object.create_log_file(f"""Best Parameters for XGBoost Model Obtain from Search:- \n
                                                 min_child_weight: {str(best_param['min_child_weight'])} \n
                                                 max_depth: {str(best_param['max_depth'])} \n
                                                 learning_rate: {str(best_param['learning_rate'])} \n
                                                 gamma: {str(best_param['gamma'])} \n
                                                 colsample_bytree: {str(best_param['colsample_bytree'])} \n                                
                                                """)
            xgb_model = XGBRegressor(min_child_weight= best_param['min_child_weight'],
                                    max_depth= best_param['max_depth'],
                                    learning_rate= best_param['learning_rate'],
                                    gamma= best_param['gamma'],
                                    colsample_bytree= best_param['colsample_bytree'])
            self.log_object.create_log_file('Starting Training on Best Parameters of XGBoost Model')
            xgb_model.fit(x_train,y_train)
            self.log_object.create_log_file('Training on Best Parameters of XGBoost Model Completed')
            try:
                r2=xgb_model.score(x_train,y_train)
                self.log_object.create_log_file(f"R2 Score on Training Data: {str(r2)}")
                train_adj_r2 = self.obj_score.adj_r2(x_train,y_train,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Training Data: {str(train_adj_r2)}")
                test_adj_r2 = self.obj_score.adj_r2(x_test,y_test,r2)
                self.log_object.create_log_file(f"Adjusted R2 Score on Testing Data: {str(test_adj_r2)}")
                y_pred = xgb_model.predict(x_test)
                evaluation_score = self.obj_score.evaluation_r2_score(y_test,y_pred)
                self.log_object.create_log_file(f"Evaluation Score on Testing Data: {str(evaluation_score)}")
                mae = self.obj_score.mae(y_test,y_pred)
                self.log_object.create_log_file(f"Mean Absolute Error(mae) on Testing Data: {str(mae)}")
                rmse = self.obj_score.rmse(y_test,y_pred)
                self.log_object.create_log_file(f"Root Mean Squared Error(rsme) on Testing Data: {str(rmse)}")
                cv_score = self.obj_score.cv_score(xgb_model,self.X,self.Y)
                self.log_object.create_log_file(f"Cross Validation Score(CV) on Training Data: {str(cv_score)}")
            except Exception as e:
                self.log_object.create_log_file(f"Error in calculating Score for Training Data: {str(e)}")

            self.dict_model['xgboost_regressor'] = [xgb_model,(evaluation_score+rmse)/2]
        except Exception as e:
            self.log_object.create_log_file(f"Error on tuning XGBoost model: \n {str(e)}")

        
    def algo_run(self):
        """
        Method: algo_run
        Description: This method is used to run multiple algorithms on the dataset
        Parameters: None
        Return: None

        Version: 1.0
        """
        try:
            self.log_object.create_log_file('Starting Linear Regression Tuning')
            self.lr_tuning() # Tuning the Linear Regression model
            self.log_object.create_log_file('Tuning for Linear Regression methode compleated.')
        except Exception as e:
            Self.log_object.create_log_file(f"Error on tuning Linear Regression model: \n {str(e)}")

        try:
            self.log_object.create_log_file('Starting Gradient Boosting Tuning')
            self.gb_tuning() # Tuning the Gradient Boosting model
            self.log_object.create_log_file('Tuning for Gradient Boosting methode completed.')
        except Exception as e:
            Self.log_object.create_log_file(f"Error on tuning Gradient Boosting model: \n {str(e)}")
        
        try:
            self.log_object.create_log_file('Starting Random Forest Tuning')
            self.rf_tuning() # Tuning the Random Forest model
            self.log_object.create_log_file('Tuning for Random Forest methode completed.')
        except Exception as e:
            self.log_object.create_log_file(f"Error on tuning Random Forest model: \n {str(e)}")
        
        try:
            self.log_object.create_log_file('Starting XGBoost Tuning')
            self.xgb_tuning() # Tuning the XGBoost model
            self.log_object.create_log_file('Tuning for XGBoost methode completed.')
        except Exception as e:
            self.log_object.create_log_file(f"Error on tuning XGBoost model: \n {str(e)}")
    
        
    def model_result(self):
        """
        Method: model_result
        Description: This method is used to print the best model and the corresponding score
        Parameters: None
        Return: Store the scores obtain from different algorithms in a dictionary

        Version: 1.0
        """
        try:
            try:
                self.log_object.create_log_file('Start Running algo_run methode')
                self.algo_run() # Running the algo_run methode
            except Exception as e:
                self.log_object.create_log_file(f"Error on running algo_run methode: \n {str(e)}")
            
            d= self.dict_model  
            d = sorted(d.items(), key=lambda a:a[1][1])
            
            best_model_name = d[0][0]
            best_model_object = d[0][1][0]
            best_model_avg_rmse = d[0][1][1]
            self.log_object.create_log_file(f"Best Model: {str(best_model_name)}")
            if not os.path.isdir('./bestmodel/'):
                os.mkdir('./bestmodel/')
            
            with open('./bestmodel/'+best_model_name+'.pkl','wb') as file:
                pickle.dump(best_model_object,file)
            self.log_object.create_log_file(f"Best Model Object Saved in File: {str(best_model_name)}")
                    
            return best_model_name,best_model_avg_rmse
        except Exception as e:
            self.log_object.create_log_file(f"Error on model_result method: \n {str(e)}")




#         self.df_score['linear_regression']=[train_adj_r2,test_adj_r2,evaluation_score,mae,rmse,cv_score]
#         self.df_score=pd.DataFrame(index=['train_adj_r2','test_adj_r2','evaluation_r2_score','mae','rmse','cv_score'])

