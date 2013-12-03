import pydecode.hyper as ph
import pydecode.chart as chart
import itertools
from collections import namedtuple

class Sentence(namedtuple("Sentence", ["label", "prev_sents", "pos"])):

    def __new__(_cls, label, prev_sents, pos):
        non_nulls = [x for x in prev_sents if x != ()]

        return super(_cls, Sentence).__new__(_cls, label, tuple(non_nulls), pos)

    def __str__(self):
        return "{}: {}".format(self.pos, self.label)

class Transition:
    def __init__(self, sents):
        self.sents = sents

    def __str__(self):
        label = ''
        if self.__len__() > 1:
            for sent in self.sents[1:]:
                label = "{} -> ".format(sent) + label
        return label + "{}".format(self.sents[0])
    
    def __iter__(self):
        return iter(self.sents)

    def __getitem__(self,index):
        return self.sents[index]

    def __len__(self):
        return len(self.sents) 

    def __nonzero__(self):
        return True

def build_constraints(transition):
    if transition[0] != 'END':
        return [(transition[0], 1)]
    return []


def build_hypergraph(discourse_model, c):

    node_labels = ['sent-'+str(i) for i,s in enumerate(discourse_model)]
    history_size = discourse_model.history
    n = len(node_labels) + 1

    hist_nodes = [node_labels, ['START',],] + [[()]] * (history_size-1)
    start_sent = Sentence('START', (), 0)
        
    
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

    return c


