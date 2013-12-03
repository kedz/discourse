import pydecode.hyper as ph
import pydecode.display as display
import pydecode.chart as chart
import pydecode.lp as lp
import pydecode.constraints as cons
import pydecode.display as display
import pulp
from collections import namedtuple
import itertools
import numpy as np
import datetime

class Sentence(namedtuple("Sentence", ["label", "prev_sents", "pos"])):
   
    def __new__(_cls, label, prev_sents, pos):
        non_nulls = [x for x in prev_sents if x != ()]

        return super(_cls, Sentence).__new__(_cls, label, tuple(non_nulls), pos)
   
    def __str__(self):
        return "{}: {}".format(self.pos, self.label)

class Transition(namedtuple("Transition", ['sents'])):
    def __str__(self): 
        label = ''
        if len(self.sents) > 1:
            for sent in self.sents[1:]:
                label = "{} -> ".format(sent) + label
        return label + "{}".format(self.sents[0])



class CoherenceFormat(display.HypergraphPathFormatter):
    def hypernode_attrs(self, node):
        label = self.hypergraph.node_label(node)
        return {"label": label, "shape": ""}
    def hyperedge_node_attrs(self, edge):
        return {"color": "pink", "shape": "point"}

    
def build_edge_scorer(model, weights):    
    def edge_scorer(transition):
        indices = [int(i[5:]) for i in reversed(transition.sents) if 'Sent_' in i]
       
        if len(indices) < 1:
            return 0
        partial_model = model.get_partial_grid(indices)
        
        return np.dot(weights, partial_model.get_trans_cnt_vctr())
        
    return edge_scorer

def cons_name(sent): return "Sent_%s"%(sent)

def build_constraints(transition):
    if transition.sents[0] not in ['START', 'END']:
        return [(transition.sents[0], 1)]
    return []

class LPSolver:
    def __init__(self, weights):
        self.weights = weights
        
        # Solver stats
        self.gbuilder_time = datetime.timedelta(hours=0)
        self.solver_time = datetime.timedelta(hours=0)
        self.potential_time = datetime.timedelta(hours=0)
        self.lpbuilder_time = datetime.timedelta(hours=0)
        self.solved = 0

    def solve(self, model, verbose=False):
        
        # Construct the hypergraph.
        if verbose:
            print 'Constructing hypergraph...\t',
        start = datetime.datetime.utcnow()     
        hypergraph = self._build_hypergraph(model)
        stop = datetime.datetime.utcnow()
        building_delta = stop - start
        self.gbuilder_time += building_delta
        if verbose:
            print 'Elapsed time {}'.format(building_delta)

        # Construct hypergraph edge potentials.
        if verbose:
            print 'Constructing edge potentials...\t',
        start = datetime.datetime.utcnow()     
        potentials = ph.Potentials(hypergraph).build(build_edge_scorer(model, self.weights))       
        stop = datetime.datetime.utcnow()
        potential_delta = stop - start
        self.potential_time += potential_delta
        if verbose:
            print 'Elapsed time {}'.format(potential_delta)
        

        constraints = \
            cons.Constraints(hypergraph, [(cons_name(sent), -1) 
                                              for sent in range(1,len(model.sentences)+1)]).build(
                                             build_constraints)                 
        # Construct LP.
        if verbose:
            print 'Constructing LP...\t',
        start = datetime.datetime.utcnow()     
        hyperlp = lp.HypergraphLP.make_lp(hypergraph, potentials, integral=True) 
        stop = datetime.datetime.utcnow()
        lpbuilder_delta = stop - start
        self.lpbuilder_time += lpbuilder_delta
        if verbose:
            print 'Elapsed time {}'.format(lpbuilder_delta)
        
        # Add constraints.
        hyperlp.add_constraints(constraints)

        # Run the lp solver.
        if verbose:
            print 'Running LP solver...\t',
        start = datetime.datetime.utcnow()     
        hyperlp._status = hyperlp.lp.solve(pulp.solvers.GUROBI(msg=False))
        stop = datetime.datetime.utcnow()
        solver_delta = stop - start
        self.solver_time += solver_delta
        if verbose:
            print 'Elapsed time {}'.format(solver_delta)
            
        lpath = hyperlp.path
        
        self.solved += 1
        
        return (hypergraph, lpath)
        
    def _build_hypergraph(self, model):

        node_labels = ['Sent_'+str(i) for i in model.grids[0].columns]
        history_size = model.history
        n = len(node_labels) + 1

        hist_nodes = [node_labels, ['START',],] + [[()]] * (history_size-1)
        start_sent = Sentence('START', (), 0)

        c = chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing,
                           build_hypergraph = True, debug=False)

        c.init(start_sent)


        for i in range(1,n):

            for labels in itertools.product(*hist_nodes[:-1]):

                c[Sentence(labels[0], (labels[1:]), i)] = \
                        c.sum([c[key] * c.sr(Transition(labels))
                               for label3 in hist_nodes[-1]
                               for key in [Sentence(labels[1], labels[2:]+tuple([label3]), i-1)]
                               if key in c])

            if i <= history_size:

                hist_nodes.pop(-1)
                hist_nodes = [node_labels,] + hist_nodes


        c[Sentence('END', (), n)] = \
                c.sum([c[key] * c.sr(Transition(('END', labels[0])))
                       for labels in itertools.product(*hist_nodes[:-1])
                       for key in [Sentence(labels[0], (labels[1:]), n-1)]
                       if key in c])


        hypergraph = c.finish()
        return hypergraph

    def print_stats(self):
        total_time = self.gbuilder_time + self.potential_time + self.solver_time + self.lpbuilder_time
        avg_mus = total_time.microseconds / float(self.solved)
        avg_total = datetime.timedelta(microseconds=avg_mus)

        avg_mus = self.gbuilder_time.microseconds / float(self.solved)
        avg_build_time = datetime.timedelta(microseconds=avg_mus) 

        avg_mus = self.potential_time.microseconds / float(self.solved)
        avg_potential_time = datetime.timedelta(microseconds=avg_mus) 

        avg_mus = self.solver_time.microseconds / float(self.solved)
        avg_solver_time = datetime.timedelta(microseconds=avg_mus) 

        avg_mus = self.lpbuilder_time.microseconds / float(self.solved)
        avg_lpbuilder_time = datetime.timedelta(microseconds=avg_mus) 


        print "Solved {} graph(s)".format(self.solved)
        print "Graph construction: {} ({})".format(self.gbuilder_time, avg_build_time)
        print "Edge potential time: {} ({})".format(self.potential_time, avg_potential_time)
        print "LP Construction time: {} ({})".format(self.lpbuilder_time, avg_lpbuilder_time)
        print "LP Solver time: {} ({})".format(self.solver_time, avg_solver_time)
        print "Total time: {} ({})".format(total_time, avg_total) 
    

def to_ipython(hypergraph, path):
    disp_format = CoherenceFormat(hypergraph, [path])
    return disp_format.to_ipython()
