# data_transformation.py

import sys
import os
import re
import pandas as pd
from typing import List
from topicmodeling.exception.exception import TopicModelingException 
from topicmodeling.logging.logger import logging
from topicmodeling.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from topicmodeling.entity.config_entity import DataTransformationConfig
from topicmodeling.utils.main_utils.utils import save_object

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise TopicModelingException(e, sys)

    def read_data(self, file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TopicModelingException(e, sys)

    def text_preprocessing(self, text_series: pd.Series) -> List[List[str]]:
        try:
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            def clean_and_tokenize(text):
                text = text.lower()
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
                return tokens

            tokenized_corpus = text_series.apply(clean_and_tokenize).tolist()
            return tokenized_corpus
        except Exception as e:
            raise TopicModelingException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting Data Transformation Process for Topic Modeling")
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)

            tokenized_corpus = self.text_preprocessing(train_df['text'])
            save_object(self.data_transformation_config.transformed_object_file_path, tokenized_corpus)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=None,
                transformed_test_file_path=None
            )
            logging.info(f"Data Transformation Artifact Created: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise TopicModelingException(e, sys)
