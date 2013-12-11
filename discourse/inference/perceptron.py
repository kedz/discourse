import pydecode.model as model
from pystruct.learners import StructuredPerceptron
import discourse.hypergraph as hyper
from discourse.hypergraph import *
import pydecode.constraints as cons

class DiscourseSequenceModel(model.DynamicProgrammingModel):
    def dynamic_program(self, discourse_model, c):
        return hyper.build_hypergraph(discourse_model, c)
    
    def initialize_features(self, discourse_model):
        return None
    
    def constraints(self, discourse_model, hypergraph):
        nsents = len(discourse_model)
        return cons.Constraints(hypergraph, [('sent-{}'.format(i), -1)
                                             for i in range(nsents)]).build(hyper.build_constraints)
    
    def factored_psi(self, discourse_model, transition, data):
        return discourse_model.feature_map(transition)


class PerceptronTrainer:
    def __init__(self, max_iter=25):    
        self.dsm = DiscourseSequenceModel(True)
        self.sp = StructuredPerceptron(self.dsm, verbose=1, max_iter=max_iter)
    
    def fit(self, trainX, trainY):
        import warnings
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.sp.fit(trainX, trainY)


    def predict(self, testX):
        predY = self.sp.predict(testX)
        predicted_sentence_orderings = [recover_order(y) for y in predY]    
        return predicted_sentence_orderings

    def score(self, testX, testY):
        self.sp.score(testX, testY)

    def weights(self):
        return self.dsm._vec.inverse_transform(self.sp.w)

