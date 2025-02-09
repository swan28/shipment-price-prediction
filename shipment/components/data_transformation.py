import os
import sys
import numpy as np
import pandas as pd

from pandas import DataFrame
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shipment.logger import logging
from shipment.exception import ShippingException
from shipment.entity.config_entity import DataTransformationConfig
from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_transformation_config: DataTransformationConfig,
    ):

        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config

        # Reading train.csv and test.csv from data ingestion artifacts
        self.train_set = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)

   
    def get_data_transformer_object(self) -> object:
        """
        Method Name :   get_data_transformer_object

        Description :   This method gives preprocessor object. 
        
        Output      :   Preprocessor Object. 
        """
        logging.info("Entered get_data_transformer_object method of Data_Ingestion class")

        try:
            # Getting necessary column names from config file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            onehot_columns = self.data_transformation_config.SCHEMA_CONFIG["onehot_columns"]
            binary_columns = self.data_transformation_config.SCHEMA_CONFIG["binary_columns"]

            logging.info("Got numerical cols,one hot cols,binary cols from schema config")

            # Creating transformer objects
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            binary_transformer = BinaryEncoder()

            logging.info("Initialized StandardScaler,OneHotEncoder,BinaryEncoder")

            # Preprocessing pipeline - Using transformer objects in column transformer
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, onehot_columns),
                    ("BinaryEncoder", binary_transformer, binary_columns),
                    ("StandardScaler", numeric_transformer, numerical_columns),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info("Exited get_data_transformer_object method of Data_Ingestion class")
            return preprocessor

        except Exception as e:
            raise ShippingException(e, sys) from e

    
    @staticmethod
    def _outlier_capping(col, df: DataFrame) -> DataFrame:
        """
        Method Name :   _outlier_capping

        Description :   This method performs outlier capping in the dataframe. 
        
        Output      :   DataFrame. 
        """
        logging.info("Entered _outlier_capping method of Data_Transformation class")

        try:
            logging.info("Performing _outlier_capping for columns in the dataframe")

            # Calculating 25 and 75 percentile
            percentile25 = df[col].quantile(0.25)  
            percentile75 = df[col].quantile(0.75)  

            # Calculating upper limit and lower limit
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr

            # Capping the outliers
            df.loc[(df[col] > upper_limit), col] = upper_limit
            df.loc[(df[col] < lower_limit), col] = lower_limit

            logging.info("Performed _outlier_capping method of Data_Transformation class")

            logging.info("Exited _outlier_capping method of Data_Transformation class")
            return df

        except Exception as e:
            raise ShippingException(e, sys) from e

    
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Method Name :   initiate_data_transformation

        Description :   This method initiates data transformation. 
        
        Output      :   Data Transformation Artifacts. 
        """
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")

        try:
            # Creating directory for data transformation artifacts
            os.makedirs(
                self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                exist_ok=True,
            )

            logging.info(
                f"Created artifacts directory for {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}"
            )

            # Getting preprocessor object
            preprocessor = self.get_data_transformer_object()

            logging.info("Got the preprocessor object")

            # Getting target column name from schema file
            target_column_name = self.data_transformation_config.SCHEMA_CONFIG["target_column"]  

            # Getting numerical columns from schema file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"] 

            logging.info("Got target column name and numerical columns from schema config")

            # Outlier capping
            continuous_columns = [
                feature
                for feature in numerical_columns
                if len(self.train_set[feature].unique()) >= 25
            ]

            logging.info("Got a list of continuous_columns")

            [self._outlier_capping(col, self.train_set) for col in continuous_columns]

            logging.info("Outlier capped in train df")

            [self._outlier_capping(col, self.test_set) for col in continuous_columns]

            logging.info("Outlier capped in test df")

            # Getting the input features and target feature of Training dataset
            input_feature_train_df = self.train_set.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = self.train_set[target_column_name]

            logging.info("Got train features and test feature")

            # Getting the input features and target feature of Testing dataset
            input_feature_test_df = self.test_set.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = self.test_set[target_column_name]
            
            logging.info("Got train features and test feature")

            # Applying preprocessing object on training dataframe and testing dataframe
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Used the preprocessor object to transform the test features")

            # Concatinating input feature array and target feature array of Train dataset
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            logging.info("Created train array.")

            # Creating directory for transformed train dataset array 
            os.makedirs(
                self.data_transformation_config.TRANSFORMED_TRAIN_DATA_DIR,
                exist_ok=True,
            )

            # Save the transformed train dataset array
            transformed_train_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                self.data_transformation_config.TRANSFORMED_TRAIN_FILE_PATH, 
                train_arr
            )

            logging.info(
                f"Saved train array to {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}"
            )

            # Concatinating input feature array and target feature array of Test dataset
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Created test array.")

            # Creating directory for transformed test dataset array 
            os.makedirs(
                self.data_transformation_config.TRANSFORMED_TEST_DATA_DIR, exist_ok=True
            )

            # Save the transformed test dataset array
            transformed_test_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                self.data_transformation_config.TRANSFORMED_TEST_FILE_PATH, 
                test_arr
            )

            logging.info(
                f"Saved test array to {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}"
            )

            # Saving the preprocessor object to data transformation artifacts directory
            preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(
                self.data_transformation_config.PREPROCESSOR_FILE_PATH, 
                preprocessor
            )

            logging.info("Saved the preprocessor object in DataTransformation artifacts directory.")

            logging.info("Exited initiate_data_transformation method of Data_Transformation class")

            # Saving data transformation artifacts
            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_object_file_path=preprocessor_obj_file,
                transformed_train_file_path=transformed_train_file,
                transformed_test_file_path=transformed_test_file,
            )

            return data_transformation_artifacts

        except Exception as e:
            raise ShippingException(e, sys) from e