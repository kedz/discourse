import pydecode.model as model
import pydecode.constraints as cons
import pydecode.lp as lp
import pydecode.hyper as ph
from pystruct.learners import StructuredPerceptron, SubgradientSSVM
from sklearn.feature_extraction import DictVectorizer
from itertools import izip
import numpy as np
import pulp
import discourse.lattice as lattice
from datetime import datetime, timedelta
import time

class DiscourseSequenceModel(model.DynamicProgrammingModel):
    def __init__(self, constrained=True, use_gurobi=False, use_beam=True,
                 use_relaxed=False, verbose=False,
                 loss_function='01'):
        self.inference_calls = 0
        self._vec = DictVectorizer()
        self._loss_function = loss_function
        self._constrained = constrained
        self._use_gurobi = use_gurobi
        self._use_beam = use_beam
        self._use_relaxed = use_relaxed
        self._debug = verbose

        self._total_graph_cons_time = 0
        self._total_potential_cons_time = 0
        self._total_constraint_cons_time = 0
        self._total_inference_time = 0
    
    def dynamic_program(self, discourse_model, c):
        return lattice.build_ngram_lattice(discourse_model, c)

    def initialize_features(self, discourse_model):
        return None

    def constraints(self, discourse_model, hypergraph):
        nsents = len(discourse_model.doc.sents)
        constraints = cons.Constraints(hypergraph,
                                [(i, -1)
                                 for i in range(nsents)])
        g = [lattice.build_constraints(edge.label)
             for edge in hypergraph.edges]
        constraints.from_vector(g)
        return constraints

        return constraints

    def beam_constraints(self, discourse_model, hypergraph):
        """
        For beam search. Transposed constraints.

        """

        gen = []
        for edge in hypergraph.edges:
            b = ph.Bitset()
            b[edge.label.to] = 1
            gen.append(b)

        #gen = [ph.Bitset(edge.label.to )
        #       for edge in hypergraph.edges]
        constraints = ph.BinaryVectorPotentials(hypergraph) \
            .from_vector(gen)
        return constraints

    def build_groups(self, hypergraph):
        return [ i+ 1 for i in [node.label.position for node in hypergraph.nodes] ]

    def factored_joint_feature(self, discourse_model, transition, data):
        return discourse_model.feature_map(transition)

    def loss(self, y, y_hat):
        if self._loss_function == '01':
            return self.zero_one_loss(y, y_hat, verbose=False)
        elif self._loss_function == 'hamming-node':
            return self.hamming_node_loss(y, y_hat)
        elif self._loss_function == 'hamming-edge':
            return self.hamming_edge_loss(y, y_hat)
        elif self._loss_function == 'kendalls-tau':
            print "IMPLEMENT kendalls-tau loss"
            import sys
            sys.exit()
        else:
            print "Bad loss function"
            import sys
            sys.exit()

    def zero_one_loss(self, y, y_hat, verbose=False):
        y_set = set(y)
        total_loss = 0
        for y_i_hat in y_hat:
            if verbose:
                print 'PRED:', y_i_hat,
            if y_i_hat not in y_set:
                total_loss = 1
                if verbose:
                    print 'LOSS TRIGGERED'
            else:
                if verbose:
                    print 'MATCH'
        if verbose:
            print '0-1 LOSS: {}'.format(total_loss)

        return total_loss

    # Overloading inference to try beam search.
    def inference(self, x, w, relaxed=False, verbose=True):
        self.inference_calls += 1

        start_time = int(round(time.time() * 1000)) 
        hypergraph = self._build_hypergraph(x)
        elapsed_time = int(round(time.time() * 1000)) - start_time
        if verbose:
            print 'Graph Construction Time (size: {}):'.format(x.nsents),
            print str(timedelta(milliseconds=elapsed_time))
        self._total_graph_cons_time += elapsed_time
        
        start_time = int(round(time.time() * 1000)) 
        potentials = self._build_potentials(hypergraph, x, w)
        elapsed_time = int(round(time.time() * 1000)) - start_time
        if verbose:
            print 'Edge Potential Construction Time:',
            print str(timedelta(milliseconds=elapsed_time))
        self._total_potential_cons_time += elapsed_time

        start_time = int(round(time.time() * 1000)) 
        if self._use_beam:
            beam_constraints = self.beam_constraints(x, hypergraph)
        else: 
            constraints = self.constraints(x, hypergraph)
        elapsed_time = int(round(time.time() * 1000)) - start_time
        if verbose:
            print 'Constraint Construction Time:',
            print str(timedelta(milliseconds=elapsed_time))
        self._total_constraint_cons_time += elapsed_time

        

        # FOR DEBUGGING

        start_time = int(round(time.time() * 1000))
        if not self._use_beam:
            if verbose:
                print "Running ILP"
            hyperlp = lp.HypergraphLP.make_lp(hypergraph,
                                              potentials,
                                              integral=True)
            hyperlp.add_constraints(constraints)
            if self._use_gurobi:
                hyperlp.solve(pulp.solvers.GUROBI(mip=1 if not relaxed else 0))
            else:
                hyperlp.solve(pulp.solvers.GLPK(mip=1 if not relaxed else 0))

            if verbose:
                print "ILP OBJECTIVE",str(hyperlp.objective)
            path = hyperlp.path

        # BEAM SEARCH
        else:
            if verbose:
                print "Running Beam Search"
            groups = self.build_groups(hypergraph)
            num_groups = max(groups) + 1

            print len(hypergraph.edges), len(hypergraph.nodes)
            in_chart = ph.inside(hypergraph, potentials)
            out = ph.outside(hypergraph, potentials, in_chart)

            beam_chart = ph.beam_search_BinaryVector(hypergraph,
                                                     potentials,
                                                     beam_constraints,
                                                     out,
                                                     -10000000,
                                                     groups,
                                                     [1000] * num_groups,
                                                     num_groups)
            path = beam_chart.path(0)

        if verbose:
            print "OBJECTIVE", potentials.dot(path)

        elapsed_time = int(round(time.time() * 1000)) - start_time
        if verbose:
            print 'Inference Time:',
            print str(timedelta(milliseconds=elapsed_time))
        self._total_inference_time += elapsed_time

        y = set([edge.label for edge in path])
        return y

    def hamming_node_loss(self, y, y_hat, verbose=False):
        s2i = lattice.s2i
        total_loss = 0
        end_pos = len(y) - 1
        for y_i_hat in y_hat:
            if y_i_hat.to != y_i_hat.position:
                l = 1
            else:
                l = 0
            if verbose:
                print 'Gold:', y_i_hat.position,
                print 'Pred:', y_i_hat.to, 'Loss: {}'.format(l)
            total_loss += l

        if verbose:
            print 'Total Loss:', total_loss
        return total_loss

    def hamming_edge_loss(self, y, y_hat, verbose=False):
        total_loss = 0
        y_set = set(y)
        for y_i_hat in y_hat:
            if  y_i_hat not in y_set:
                l = 1
            else:
                l = 0
            if verbose:
                print 'Pred:', y_i_hat, 'Loss: {}'.format(l)
            total_loss += l
        if verbose:
            print 'Total Loss:', total_loss
        return total_loss

    def loss_augmented_inference(self, x, y, w,
                                 relaxed=False, return_energy=False):
        self.inference_calls += 1
        if self._loss_function == '01':
            return self.inference(x, w, relaxed)
        elif self._loss_function == 'hamming-node':
            return self.hamming_loss_augmented_inference(x, y, w, relaxed)
        elif self._loss_function == 'hamming-edge':
            return self.hamming_loss_augmented_inference(x, y, w, relaxed)

    def hamming_loss_augmented_inference(self, x, y, w, relaxed=False):

        relaxed = relaxed or self._use_relaxed
        if self._debug:
            a = time.time()
        hypergraph = self._build_hypergraph(x)
        if self._debug:
            print >>sys.stderr, "BUILD HYPERGRAPH:", time.time() - a

        if self._debug:
            a = time.time()
        potentials = self._build_hamming_potentials(hypergraph, x, y, w)


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

    def _build_hamming_potentials(self, hypergraph, x, y, w):
        nsents = len(x.doc.sents)
        data = self.initialize_features(x)
        features = [self.factored_joint_feature(x, edge.label, data)
                    for edge in hypergraph.edges]
        f = self._vec.transform(features)

        scores = f * w.T

        # Add hamming loss to edge potentials
        if self._loss_function == 'hamming-node':
            lf = self.hamming_node_loss
        elif self._loss_function == 'hamming-edge':
            lf = self.hamming_edge_loss
        else:
            # This should never happen.
            import sys
            print "BADNESS"
            sys.exit()
            lf = self.zero_one_loss

        for i, edge in enumerate(hypergraph.edges):
            l = lf(y, [edge.label])
            scores[i] += l

        return ph.LogViterbiPotentials(hypergraph).from_vector(scores)

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

    def _lattice_constraints(self, x):

        height = x.nsents**x.ngrams
        width = x.nsents
        
        cons = []
        if x.ngrams == 2:
            for i in xrange(height):
                for j in xrange(height):
                    if i != j:
                        cons.append(tuple([i,j]))
        else:
            for i in xrange(1, height + 1):
                for j in xrange(1, height + 1):
                    if i != j and ((i-1) % width) == (i - 1) / width:
                        cons.append(tuple([i,j]))              

class Learner:
    def __init__(self, inference='beam',
                 use_relaxed=False, verbose=False,
                 algorithm='perceptron', loss='01',
                 max_iter=10):

        if inference == 'gurobi':
            use_gurobi = True
        else:
            use_gurobi = False
        if inference == 'beam':
            use_beam = True
        else:
            use_beam = False

        self.dsm = DiscourseSequenceModel(constrained=True, 
                                          use_gurobi=use_gurobi, 
                                          use_beam=use_beam,
                                          use_relaxed=use_relaxed,
                                          verbose=verbose,
                                          loss_function=loss)

        if algorithm == 'perceptron':
            self.learner = StructuredPerceptron(self.dsm,
                                                verbose=(1 if verbose else 0),
                                                max_iter=max_iter,
                                                average=True)

        elif algorithm == 'sg-ssvm':
            self.learner = SubgradientSSVM(self.dsm,
                                           verbose=(1 if verbose else 0),
                                           max_iter=max_iter,
                                           averaging='linear')

    def fit(self, trainX, trainY):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.learner.fit(trainX, trainY)
        
        if trainX[0]._use_cache is True:
            import gc
            for x in trainX:
                x._f_cache = {}
                x._local_edge_feature_cache = {}
            gc.collect()
                

    def predict(self, testX):
        predY = self.learner.predict(testX)

        return predY

    def score(self, testX, testY):
        return self.sp.score(testX, testY)

    def weights(self):
        return self.dsm._vec.inverse_transform(self.learner.w)
