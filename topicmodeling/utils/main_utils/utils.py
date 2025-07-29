import yaml
from topicmodeling.exception.exception import TopicModelingException
from topicmodeling.logging.logger import logging
from gensim.models import LdaModel, LsiModel
from gensim.models.coherencemodel import CoherenceModel
import os,sys
import numpy as np
#import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise TopicModelingException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise TopicModelingException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise TopicModelingException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise TopicModelingException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise TopicModelingException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise TopicModelingException(e, sys) from e
    



def evaluate_models(corpus, dictionary, texts, models_config):
    """
    Evaluate Topic Models using Coherence Score.
    :param corpus: Bag-of-words corpus
    :param dictionary: Gensim dictionary
    :param texts: Preprocessed tokenized texts
    :param models_config: dict with model names & corresponding params (num_topics, passes, etc.)
    :return: dict of model names with their coherence scores
    """
    try:
        report = {}

        for model_name, params in models_config.items():
            if model_name == "LDA":
                model = LdaModel(corpus=corpus, id2word=dictionary, **params)
            elif model_name == "LSA":
                model = LsiModel(corpus=corpus, id2word=dictionary, **params)
            else:
                continue  # Skip unknown models

            coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()

            report[model_name] = {
                'model': model,
                'coherence_score': coherence_score
            }

        return report

    except Exception as e:
        raise TopicModelingException(e, sys)
