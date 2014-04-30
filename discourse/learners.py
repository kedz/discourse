import pydecode.model as model
from pystruct.learners import StructuredPerceptron, SubgradientSSVM
import discourse.hypergraph as hyper
from discourse.hypergraph import *
import pydecode.constraints as cons
from itertools import izip
import numpy as np
import pydecode.lp as lp
import pydecode.hyper as ph
import pulp

class DiscourseSequenceModel(model.DynamicProgrammingModel):

    def dynamic_program(self, discourse_model, c):
        return hyper.build_hypergraph(discourse_model, c)

    def initialize_features(self, discourse_model):
        return None

    def constraints(self, discourse_model, hypergraph):
        nsents = len(discourse_model.doc.sents)
        constraints = cons.Constraints(hypergraph,
                                [('sent-{}'.format(i), -1)
                                 for i in range(nsents)])
        constraints.from_vector([hyper.build_constraints(edge.label) 
                                 for edge in hypergraph.edges])
        return constraints

    def factored_joint_feature(self, discourse_model, transition, data):
        return discourse_model.feature_map(transition)
   
    def loss(self, y, y_hat):
        import sys
        sys.stderr.write(u'Warning: this is a dummy loss function.\n')
        sys.stderr.flush()        
        return 0
    
    def zero_one_loss(self, y, y_hat):
        
        y_ord = hyper.recover_order(y)
        y_hat_ord = hyper.recover_order(y_hat)

        total_loss = 0
        for y_i, y_i_hat in izip(y_ord, y_hat_ord):
            print 'GOLD:', y_i.sentences,
            print 'PRED:', y_i_hat.sentences,
            if y_i.sentences != y_i_hat_sentences:
                total_loss = 1
                print 'LOSS TRIGGERED'
            else:
                print 'MATCH'
        print '0-1 LOSS: {}'.format(total_loss)
        #total_loss = 1 if y != y_hat else 0
                   
        return total_loss

    def hamming_loss(self, y, y_hat):
        # Hamming loss:
        total_loss = 0
        
        y_ord = hyper.recover_order(y)
        y_hat_ord = hyper.recover_order(y_hat)
        
        for y_i, y_i_hat in izip(y_ord, y_hat_ord):
            
            l = 1 if y_i.sentences != y_i_hat.sentences else 0
            print 'Gold:', y_i.sentences, 
            print 'Pred:', y_i_hat.sentences, 'LOSS: {}'.format(l)
            total_loss += l
        return total_loss      


    def loss_augmented_inference(self, x, y, w, 
                                 relaxed=False, return_energy=False):
        self.inference_calls += 1
        return self.inference(x, w, relaxed)


    def hamming_loss_aug_inference(self, x, y, w, relaxed=False):
        relaxed = relaxed or self._use_relaxed
        if self._debug:
            a = time.time()
        hypergraph = self._build_hypergraph(x)
        if self._debug:
            print >>sys.stderr, "BUILD HYPERGRAPH:", time.time() - a

        if self._debug:
            a = time.time()
        potentials = self._build_hamming_potentials(hypergraph, x, w)
        
        #for item in potentials.iteritems():
        #    print item
        
        if self._debug:
            print >>sys.stderr, "BUILD POTENTIALS:", time.time() - a
        if not self._constrained:
            if self._debug:
                a = time.time()
            path = ph.best_path(hypergraph, potentials)
            if self._debug:
                print >>sys.stderr, "BEST PATH:", time.time() - a
        else:
            if self._debug:
                a = time.time()
            constraints = self.constraints(x, hypergraph)
            hyperlp = lp.HypergraphLP.make_lp(hypergraph,
                                              potentials,
                                              integral=not relaxed)
            hyperlp.add_constraints(constraints)
            if self._debug:
                print >>sys.stderr, "BUILD LP:", time.time() - a

            if self._debug:
                a = time.time()
            if self._use_gurobi:
                hyperlp.solve(pulp.solvers.GUROBI(mip=1 if not relaxed else 0))
            else:
                hyperlp.solve(pulp.solvers.GLPK(mip=1 if not relaxed else 0))
            if self._debug:
                print >>sys.stderr, "SOLVE LP:", time.time() - a

            if relaxed:
                path = hyperlp.decode_fractional()
            else:
                path = hyperlp.path
        if self._debug:
            print
        y = set([edge.label for edge in path])
        return y

    def _build_hamming_potentials(self, hypergraph, x, w):
        nsents = len(x.doc.sents)
        data = self.initialize_features(x)
        features = [self.factored_joint_feature(x, edge.label, data)
                    for edge in hypergraph.edges]
        f = self._vec.transform(features)
        
        scores = f * w.T
        #print scores
        for i, edge in enumerate(hypergraph.edges):
            idx0 = hyper.s2i(edge.label.sentences[0], nsents),
            idx1 = hyper.s2i(edge.label.sentences[1], nsents)       
            if idx1 + 1 != idx0:
                scores[i] += 1 

        #print "Weights:", self._vec.inverse_transform(w)
        #print
        return ph.Potentials(hypergraph).from_vector(scores)



    def joint_feature(self, x, y):
        joint_feature_map = {}
        for transition in y:
            fmap = self.factored_joint_feature(x, transition, None)
            for feat, val in fmap.iteritems():
                old_val = joint_feature_map.get(feat, 0)
                joint_feature_map[feat] = old_val + val
        f = self._vec.transform(joint_feature_map)

        f = np.array(f.todense()).flatten().transpose()
        return f

class DiscourseSequenceModelZeroOne(model.DynamicProgrammingModel):

    def dynamic_program(self, discourse_model, c):
        return hyper.build_hypergraph(discourse_model, c)

    def initialize_features(self, discourse_model):
        return None

    def constraints(self, discourse_model, hypergraph):
        nsents = len(discourse_model.doc.sents)
        constraints = cons.Constraints(hypergraph,
                                [('sent-{}'.format(i), -1)
                                 for i in range(nsents)])
        constraints.from_vector([hyper.build_constraints(edge.label) 
                                 for edge in hypergraph.edges])
        return constraints

    def factored_joint_feature(self, discourse_model, transition, data):
        return discourse_model.feature_map(transition)
   
    def loss(self, y, y_hat):
        
        y_ord = hyper.recover_order(y)
        y_hat_ord = hyper.recover_order(y_hat)

        total_loss = 0
        for y_i, y_i_hat in izip(y_ord, y_hat_ord):
            print 'GOLD:', y_i.sentences,
            print 'PRED:', y_i_hat.sentences,
            if y_i.sentences != y_i_hat.sentences:
                total_loss = 1
                print 'LOSS TRIGGERED'
            else:
                print 'MATCH'
        print '0-1 LOSS: {}'.format(total_loss)
        #total_loss = 1 if y != y_hat else 0
                   
        return total_loss

    def loss_augmented_inference(self, x, y, w, 
                                 relaxed=False, return_energy=False):
        self.inference_calls += 1
        return self.inference(x, w, relaxed)

    def joint_feature(self, x, y):
        joint_feature_map = {}
        for transition in y:
            fmap = self.factored_joint_feature(x, transition, None)
            for feat, val in fmap.iteritems():
                old_val = joint_feature_map.get(feat, 0)
                joint_feature_map[feat] = old_val + val
        f = self._vec.transform(joint_feature_map)

        f = np.array(f.todense()).flatten().transpose()
        return f




class DiscourseSequenceModelHammingLoss(model.DynamicProgrammingModel):

    def dynamic_program(self, discourse_model, c):
        return hyper.build_hypergraph(discourse_model, c)

    def initialize_features(self, discourse_model):
        return None

    def constraints(self, discourse_model, hypergraph):
        nsents = len(discourse_model.doc.sents)
        constraints = cons.Constraints(hypergraph,
                                [('sent-{}'.format(i), -1)
                                 for i in range(nsents)])
        constraints.from_vector([hyper.build_constraints(edge.label) 
                                 for edge in hypergraph.edges])
        return constraints

    def factored_joint_feature(self, discourse_model, transition, data):
        return discourse_model.feature_map(transition)
   
    def loss(self, y, y_hat):
        # Hamming loss:
        total_loss = 0
        
        y_ord = hyper.recover_order(y)
        y_hat_ord = hyper.recover_order(y_hat)
        
        for y_i, y_i_hat in izip(y_ord, y_hat_ord):
            
            l = 1 if y_i.sentences != y_i_hat.sentences else 0
            print 'Gold:', y_i.sentences, 
            print 'Pred:', y_i_hat.sentences, 'LOSS: {}'.format(l)
            total_loss += l
        return total_loss      

    def loss_augmented_inference(self, x, y, w, relaxed=False):
        relaxed = relaxed or self._use_relaxed
        if self._debug:
            a = time.time()
        hypergraph = self._build_hypergraph(x)
        if self._debug:
            print >>sys.stderr, "BUILD HYPERGRAPH:", time.time() - a

        if self._debug:
            a = time.time()
        potentials = self._build_hamming_potentials(hypergraph, x, w)
        
        
        if self._debug:
            print >>sys.stderr, "BUILD POTENTIALS:", time.time() - a
        if not self._constrained:
            if self._debug:
                a = time.time()
            path = ph.best_path(hypergraph, potentials)
            if self._debug:
                print >>sys.stderr, "BEST PATH:", time.time() - a
        else:
            if self._debug:
                a = time.time()
            constraints = self.constraints(x, hypergraph)
            hyperlp = lp.HypergraphLP.make_lp(hypergraph,
                                              potentials,
                                              integral=not relaxed)
            hyperlp.add_constraints(constraints)
            if self._debug:
                print >>sys.stderr, "BUILD LP:", time.time() - a

            if self._debug:
                a = time.time()
            if self._use_gurobi:
                hyperlp.solve(pulp.solvers.GUROBI(mip=1 if not relaxed else 0))
            else:
                hyperlp.solve(pulp.solvers.GLPK(mip=1 if not relaxed else 0))
            if self._debug:
                print >>sys.stderr, "SOLVE LP:", time.time() - a

            if relaxed:
                path = hyperlp.decode_fractional()
            else:
                path = hyperlp.path
        if self._debug:
            print
        y = set([edge.label for edge in path])
        return y

    def _build_hamming_potentials(self, hypergraph, x, w):
        nsents = len(x.doc.sents)
        data = self.initialize_features(x)
        features = [self.factored_joint_feature(x, edge.label, data)
                    for edge in hypergraph.edges]
        f = self._vec.transform(features)
        
        scores = f * w.T
        #print scores
        for i, edge in enumerate(hypergraph.edges):
            idx0 = hyper.s2i(edge.label.sentences[0], nsents),
            idx1 = hyper.s2i(edge.label.sentences[1], nsents)       
            if idx1 + 1 != idx0:
                scores[i] += 1 

        #print "Weights:", self._vec.inverse_transform(w)
        #print
        return ph.Potentials(hypergraph).from_vector(scores)



    def joint_feature(self, x, y):
        joint_feature_map = {}
        for transition in y:
            fmap = self.factored_joint_feature(x, transition, None)
            for feat, val in fmap.iteritems():
                old_val = joint_feature_map.get(feat, 0)
                joint_feature_map[feat] = old_val + val
        f = self._vec.transform(joint_feature_map)

        f = np.array(f.todense()).flatten().transpose()
        return f


class PerceptronTrainer:
    def __init__(self, max_iter=25, verbose=False):
        self.dsm = DiscourseSequenceModelZeroOne(True)
        self.learner = StructuredPerceptron(self.dsm,
                                            verbose=(1 if verbose else 0),
                                            max_iter=max_iter,
                                            average=True)

    def fit(self, trainX, trainY):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.learner.fit(trainX, trainY)

    def predict(self, testX):
        predY = self.learner.predict(testX)
        return predY

    def score(self, testX, testY):
        return self.sp.score(testX, testY)

    def weights(self):
        return self.dsm._vec.inverse_transform(self.learner.w)

class SubgradientLearner:
    def __init__(self, max_iter=25, verbose=False, loss=None):
        
        if loss == '01':
            self.dsm = DiscourseSequenceModelZeroOne(True)
        elif loss == 'hamming':
            self.dsm = DiscourseSequenceModelHammingLoss(True)

        self.learner = SubgradientSSVM(self.dsm,
                                       verbose=(1 if verbose else 0),
                                       max_iter=max_iter)

#        if loss is None:
#            loss = '01'

#        if loss == '01':
#            self.dsm.loss = self.dsm.zero_one_loss
#            self.dsm.loss_augmented_inference = self.inference
        
#        if loss == 'hamming':
#            self.dsm.loss =  self.dsm.hamming_loss
#            self.dsm.loss_augmented_inference = \
#                self.dsm.hamming_loss_aug_inference


    def fit(self, trainX, trainY):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.learner.fit(trainX, trainY)

    def predict(self, testX):
        predY = self.learner.predict(testX)
        return predY

    def score(self, testX, testY):
        return self.learner.score(testX, testY)

    def weights(self):
        return self.dsm._vec.inverse_transform(self.learner.w)


#    def get_score(self, x, y):
#        features = ptron.dsm.psi(x, y)
#        return features * ptron.sp.w.T
