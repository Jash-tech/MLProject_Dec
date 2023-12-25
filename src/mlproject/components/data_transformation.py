import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from src.mlproject.utils import read_sql_data,save_object,evaluate_models
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig


@dataclass
class DataTransformationConfig:
    preprocessor_pkl_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ])

            logging.info("Pipelines Created for both Cat and Num Columns")

            preprocessor=ColumnTransformer(
                [
                   ("num_pipeline",num_pipeline,numerical_columns),
                   ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading Train and Test Data")

            logging.info("Obtaining Preprocessor Pkl Object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)

            target_feature_train_df=train_df[target_column_name]
            target_feature_test_df=test_df[target_column_name]


            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_pkl_path,
                obj=preprocessor_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_pkl_path
            )
            



        except Exception as e:
            raise CustomException(e,sys)
        



# if __name__=='__main__':
#     o=DataIngestion()
#     train_data,test_data=o.initiate_data_ingestion()


#     data_transformation=DataTransformation()
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

#     model_trainer=ModelTrainer()
#     print(model_trainer.initiate_model_trainer(train_arr,test_arr))



