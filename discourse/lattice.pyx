from collections import defaultdict
import pydecode.hyper as ph

class SentenceNGram:
    def __init__(self, idxs, position):
        self.idxs = idxs
        self.position = position

    def _attrs(self):
        return (self.idxs, self.position)

    def __eq__(self, other):
        return isinstance(other, SentenceNGram) and \
            self._attrs() == other._attrs()

    def __richcmp__(self, other, int op):
        if op == 2:      
            return isinstance(other, Transition) and \
                self._attrs() == other._attrs()
        elif op == 3:
            return not (isinstance(other, Transition) and \
                self._attrs() == other._attrs())
        else:
            return 0

    def __hash__(self):
        return hash(self._attrs())

    def __str__(self):
        return u'{}: {}'.format(self.position, u'>'.join(str(i) for i in self.idxs[::-1]))

cdef class Transition:

    def __cinit__(self, tuple idxs, int position):
        self.position = position
        self.idxs = idxs
        self.to = self.idxs[0]

    def _attrs(self):
        return (self.idxs, self.position)

#<   0
#==  2
#>   4
#<=  1
#!=  3
#>=  5
    def __richcmp__(self, other, int op):
        if op == 2:      
            return isinstance(other, Transition) and \
                self._attrs() == other._attrs()
        elif op == 3:
            return not (isinstance(other, Transition) and \
                self._attrs() == other._attrs())
        else:
            return 0

#    def __eq__(self, other):
#        return isinstance(other, Transition) and \
#            self._attrs() == other._attrs()

    def __hash__(self):
        return hash(self._attrs())

    def __str__(self):
        return u'{}: {}'.format(self.position, u'>'.join(str(i) for i in self.idxs[::-1]))


cpdef build_ngram_lattice(discourse_model, c):
    """ Builds the graph from the discourse model.

    Parameters
    ----------
    discourse_model : RushModel
        A RushModel discourse coherence model.

    c : ChartBuilder
        A ChartBuilder object for contrsucting the instance graph.

    Returns
    -------
    c : ChartBuilder
        The ChartBuilder object, after having constructed the graph.
    """

    cdef int positions, ngram
    cdef object start_sent
    # The number of sentence positions to predict.
    positions = len(discourse_model.doc.sents)
    
    cdef int i
    # The set of non 'START' or 'END' sentence labels.
    node_labels = [i for i in range(positions)]
    
    # The start node of the graph with label 'START' and position -1.
    # Everyone has to start somewhere :)
    ngram = discourse_model.ngram
    padding = tuple([-1]) * (ngram - 1)
    start_sent = SentenceNGram(tuple(padding), -1)

    # As we build the graph, this will hold the previous position's
    # nodes, that is, the set of nodes it is possible to add edges
    # from.
    nodes = set()

    # Initialize the graph construction.
    c.init(start_sent)
    nodes.add(start_sent)
#    print 'nodes', nodes
    # Create the remaining graph nodes/edges.
    cdef int position
    cdef int node_label1
    cdef object node, next_node, prev_node

    for position in range(positions):
        #print 'Position', position
        # Find all possible edges for each possible new node and add them
        # to n2e.
        n2e = defaultdict(set)
        for node_label1 in node_labels:
            for node in nodes:
                #print 'nodes', node_label1, node
                # This edge does not result in a legal path.
                if node_label1 not in node.idxs:
                    

                    s = SentenceNGram(tuple([node_label1]) + node.idxs[:-1],
                                      position)
                    #print 'new node', s
                    n2e[s].add(node)

        # Create the new nodes and edges.
        for next_node in n2e.keys():
            node_edge = []
            for prev_node in n2e[next_node]:
                transition = Transition(tuple([next_node.idxs[0]]) + \
                                        prev_node.idxs, position)
                node_edge.append((prev_node, transition))
                #print 'making trans', transition

            c[next_node] = c.sum([c[prev_node] * c.sr(transition)
                                  for prev_node, transition in node_edge])
#
#        # Replace nodes with current position nodes.
        nodes = n2e.keys()

    # Create the final END node.
    end_node = SentenceNGram(tuple([positions]), positions)
    c[end_node] = \
        c.sum([c[node] * c.sr(Transition(tuple([positions]) + node.idxs,
                                         positions))
               for node in nodes])

    return c

def build_constraints(transition):
    """ Build a constraint for a transition.

    Parameters
    ----------
    transition : Transition
        A Transition object in the hypergraph.

    Returns
    -------
    cons : list, (string, int)
        A list containing a 2-tuple of a sentence label string and
        the constraint value of 1.
    """

    cons = []
    if transition.labels[0] != u'END':
        cons.append((transition.labels[0], 1))
    return cons

def build_beam_constraints(transition):
    cons = ph.Bitset()
    if transition.labels[0] != u'END':
        cons[s2i(transition.labels[0])] = 1
    return cons


def recover_order(transition_set):
    """Returns an ordered list of transitions where the head
    sentence of the ith transition is the tail sentence of the
    (i+1)th transition.

    Parameters
    ----------
    transition_set : set
        The set of Transitions predicted by an LP solver.
        This must be a valid path in the graph.

    Returns
    -------
    ordered_trans : list
        A list of Transition objects in the order implied by the
        path of the transitions through the graph.
    """
    transitions = list(transition_set)
    ordered_trans = []
    curr_tok = -1

    # If pop_cntr exceeds the number of transitions, the set is not
    # a valid path in the graph, return an empty list.
    pop_cntr = 0
    while len(transitions) > 0:
        t = transitions.pop(0)
        pop_cntr += 1
        if t.idxs[1] == curr_tok:
            ordered_trans.append(t)
            curr_tok = t.to
            pop_cntr = 0
        else:
            transitions.append(t)
        if pop_cntr == len(transitions) + 1:
            print u'Invalid solution'
            for t in transition_set:
                print t
            print
            return []

    return ordered_trans


def s2i(sent_label, end=None):
    """ Convert a sentence label to its index. 'START' label
    returns -1. Optional argument end sets output for the 'END'
    label which defaults to None.

    Parameters
    ----------
    sent_label : string
        A sentence label string, either 'START', 'END', or
        's-i' where i is the sentence index.

    end : Object
        Optional argument specifying the output object for an
        'END' label.

    Returns
    -------
    index : Object
        An integer index, None, or the value
        specified by 'end' argument.
    """
    if sent_label == u'START':
        index = -1
    elif sent_label == u'END':
        index = end
    else:
        index = int(sent_label.split('-')[1])
    return index
