from topicmodeling.entity.artifact_entity import CoherenceMetricArtifact
from topicmodeling.exception.exception import TopicModelingException
from gensim.models import CoherenceModel
import sys

def get_coherence_score(model, texts, dictionary, model_name="LDA") -> CoherenceMetricArtifact:
    """
    Compute coherence score for a topic model.
    """
    try:
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        coherence_metric = CoherenceMetricArtifact(
            model_name=model_name,
            coherence_score=coherence_score
        )
        return coherence_metric

    except Exception as e:
        raise TopicModelingException(e, sys)
