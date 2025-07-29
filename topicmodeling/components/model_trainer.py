import os
import sys

from topicmodeling.exception.exception import TopicModelingException
from topicmodeling.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from topicmodeling.entity.config_entity import ModelTrainerConfig
from topicmodeling.utils.main_utils.utils import save_object, load_object

from gensim.models.ldamodel import LdaModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import joblib

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise TopicModelingException(e, sys)

    def compute_coherence_score(self, model, texts, dictionary, corpus, coherence='c_v'):
        try:
            coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
            return coherence_model.get_coherence()
        except Exception as e:
            raise TopicModelingException(e, sys)

    def train_topic_models(self, tokenized_texts):
        try:
            dictionary = Dictionary(tokenized_texts)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

            # Sanity Check
            if len(dictionary.token2id) == 0:
                raise Exception("Dictionary is empty after preprocessing.")
            empty_docs = [doc for doc in corpus if len(doc) == 0]
            if len(empty_docs) > 0:
                raise Exception(f"{len(empty_docs)} documents are empty after BoW transformation.")

            # Train LDA Model and pick best coherence
            best_lda_model = None
            best_lda_coherence = -1
            lda_best_num_topics = 0

            for num_topics in [5, 10, 15, 20]:
                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10, eval_every=None)
                coherence = self.compute_coherence_score(lda_model, tokenized_texts, dictionary, corpus)

                if coherence > best_lda_coherence:
                    best_lda_model = lda_model
                    best_lda_coherence = coherence
                    lda_best_num_topics = num_topics

            # Train LSA Model
            lsa_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=10)
            lsa_coherence = self.compute_coherence_score(lsa_model, tokenized_texts, dictionary, corpus)

            # Extract Top 10 Keywords per Topic from Best LDA Model
            topic_keywords = {}
            for topic_id in range(best_lda_model.num_topics):
                words = best_lda_model.show_topic(topic_id, topn=10)
                keywords = [word for word, _ in words]
                topic_keywords[f"Topic_{topic_id}"] = keywords

            # Save Models and Artifacts
            model_dir = self.model_trainer_config.model_trainer_dir
            lda_model_dir = os.path.join(model_dir, "lda_model")
            lsa_model_dir = os.path.join(model_dir, "lsa_model")

            os.makedirs(lda_model_dir, exist_ok=True)
            os.makedirs(lsa_model_dir, exist_ok=True)

            best_lda_model.save(os.path.join(lda_model_dir, "model.lda"))
            lsa_model.save(os.path.join(lsa_model_dir, "model.lsa"))

            dictionary.save(os.path.join(model_dir, "dictionary.dict"))
            joblib.dump(corpus, os.path.join(model_dir, "corpus.pkl"))

            # Save Topic Keywords to File
            keywords_file = os.path.join(model_dir, "lda_topic_keywords.txt")
            with open(keywords_file, "w") as f:
                for topic_id, keywords in topic_keywords.items():
                    f.write(f"Topic {topic_id}: {', '.join(keywords)}\n")

            # Create Artifact to return
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_dir,
                train_metric_artifact=None,
                test_metric_artifact=None
            )

            print(f"\n✅ Model Training Completed Successfully!")
            print(f"LDA Topics: {lda_best_num_topics} | LDA Coherence: {best_lda_coherence:.4f} | LSA Coherence: {lsa_coherence:.4f}")
            print(f"Artifacts saved at: {model_dir}")
            print(f"Topics and Keywords are saved in: {keywords_file}")

            return model_trainer_artifact

        except Exception as e:
            raise TopicModelingException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            tokenized_texts = load_object(self.data_transformation_artifact.transformed_object_file_path)

            if not isinstance(tokenized_texts, list) or not all(isinstance(doc, list) for doc in tokenized_texts):
                raise TopicModelingException("Loaded tokenized_texts is not a list of lists.", sys)

            return self.train_topic_models(tokenized_texts)
        except Exception as e:
            raise TopicModelingException(e, sys)
