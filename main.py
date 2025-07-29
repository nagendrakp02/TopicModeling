import multiprocessing
import os

if __name__ == '__main__':
    multiprocessing.freeze_support()

    from topicmodeling.components.data_ingestion import DataIngestion
    from topicmodeling.components.data_validation import DataValidation
    from topicmodeling.components.data_transformation import DataTransformation
    from topicmodeling.components.model_trainer import ModelTrainer

    from topicmodeling.entity.config_entity import (
        DataIngestionConfig, DataValidationConfig,
        DataTransformationConfig, ModelTrainerConfig,
        TrainingPipelineConfig
    )

    from topicmodeling.exception.exception import TopicModelingException
    from topicmodeling.logging.logger import logging

    import sys

    def start_training_pipeline():
        try:
            training_pipeline_config = TrainingPipelineConfig()

            # Data Ingestion
            data_ingestion_config = DataIngestionConfig(training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            logging.info("Initiating Data Ingestion")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion Completed")
            print(data_ingestion_artifact)

            # Data Validation
            data_validation_config = DataValidationConfig(training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            logging.info("Initiating Data Validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data Validation Completed")
            print(data_validation_artifact)

            # Data Transformation
            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            logging.info("Initiating Data Transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data Transformation Completed")
            print(data_transformation_artifact)

            # Model Trainer
            model_trainer_config = ModelTrainerConfig(training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            logging.info("Initiating Model Training")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model Training Completed")
            print(model_trainer_artifact)

        except Exception as e:
            raise TopicModelingException(e, sys)

    # Execute the pipeline
    start_training_pipeline()
