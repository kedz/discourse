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
    def __init__(self, max_iter=25, verbose=False):    
        self.dsm = DiscourseSequenceModel(True)
        self.sp = StructuredPerceptron(self.dsm, verbose=(1 if verbose else 0), max_iter=max_iter)
    
    def fit(self, trainX, trainY):
        import warnings
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.sp.fit(trainX, trainY)


    def predict(self, testX):
        predY = self.sp.predict(testX)
        return predY        
        #predicted_sentence_orderings = [recover_order(y) for y in predY]    
        #return predicted_sentence_orderings

    def score(self, testX, testY):
        self.sp.score(testX, testY)

    def weights(self):
        return self.dsm._vec.inverse_transform(self.sp.w)
    
    def get_score(self, x, y):
        features = ptron.dsm.psi(x,y) #ptron.dsm._vec.inverse_transform()
        return features * ptron.sp.w.T

def print_gold_feature_diff(ptron, x, gold_y, pred_y):
    weights = ptron.dsm._vec.inverse_transform(ptron.sp.w)[0]
    gold_feat = ptron.dsm._vec.inverse_transform(ptron.dsm.psi(x,gold_y))[0]
    pred_feat = ptron.dsm._vec.inverse_transform(ptron.dsm.psi(x,pred_y))[0]
    print "Gold Features Not In Predicted y:"
    for g in gold_feat:
        if g not in pred_feat:
            print g, '', weights[g] 
    
    print "Predicted Features Not In Gold y:"
    for p in pred_feat:
        if p not in gold_feat:
            print p,'', weights[p]

def num_bigrams_correct(pred_y):
    correct = 0
    o = recover_order(pred_y)
    for i, indx in enumerate(o[:-1]):
        if indx + 1 == o[i+1]:    
            correct += 1
    return correct

def num_trigrams_correct(pred_y):
    correct = 0
    o = recover_order(pred_y)
    for i, indx in enumerate(o[:-2]):
        if indx + 1 == o[i+1] and indx+2 == o[i+2]:    
            correct += 1
    return correct
           
def print_bigram_feature_diff(ptron, x, gold_y, pred_y, f=None):
    pred_gold_pairs = []    
    weights = ptron.dsm._vec.inverse_transform(ptron.sp.w)[0]

    for pt in pred_y:
        for gt in gold_y:
            if pt[1] == gt[1]:
                pred_gold_pairs.append((pt, gt))

    o_pgp = []

    s = 'START'
    while s != 'END':
        for p in pred_gold_pairs:
            if p[0][1] == s:
                s = p[0][0]
                o_pgp.append(p)
                
     
    pred_gold_pairs = o_pgp
    #pred_gold_pairs.sort(key=lambda x: x[0][1])

    for p in pred_gold_pairs:
        pt = p[0]
        gt = p[1]

        #print_gold_feature_diff(ptron, x, set(gt), set(pt))
        gold_feat = x.feature_map(gt)
        pred_feat = x.feature_map(pt)
        if f:
            f.write("Gold Features Not In Predicted {}:\n".format(pt))
        else:
            print "Gold Features Not In Predicted {}:".format(pt)
        for g in gold_feat:
            if g not in pred_feat:
                if f:
                    f.write('{}  {}\n'.format(g, (weights[g] if g in weights else '?')))
                else:
                    print g, '', (weights[g] if g in weights else '?') 
        
        if f:
            f.write("Predicted Features Not In Gold {}:\n".format(gt))
        else:
            print "Predicted Features Not In Gold {}:".format(gt)
        for p in pred_feat:
            if p not in gold_feat:
                if f:
                    f.write('{}  {}\n'.format(p, (weights[p] if p in weights else '?')))
                else :
                    print p,'', (weights[p] if p in weights else '?')
        if f:
            f.write('\n')
        else:
            print 

        if f:
            f.write("Ordered predicted features:\n")
        else:
            print "Ordered predicted features:"
        sf = sorted([feat for feat in pred_feat], key=lambda feat: _score(weights,feat), reverse=True)
        for feat in sf:
            if f:
                f.write('{}  {}\n'.format(feat, _score(weights, feat)))
            else:
                print feat, ' ', _score(weights, feat) 
        if f:
            f.write('\n\n')
            f.flush()
        else:
            print
            print

def _score(weights, feat):
    if feat in weights:
        return weights[feat]
    else:
        return 0
#    o = recover_order(pred_y)
#    bigrams = [(indx, o[i+1]) for i, indx in enumerate(o[:-1])]
#    for t in pred_y:
        #    print t
        #    print x.feature_map(t)       
        #return correct
     


