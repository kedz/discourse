from collections import defaultdict

class SentenceNGram:
    def __init__(self, labels, position):
        self.labels = labels
        self.position = position

    def _attrs(self):
        return (self.labels, self.position)
    
    def __eq__(self, other):
        return isinstance(other, SentenceNGram) and \
            self._attrs() == other._attrs()
      
    def __hash__(self):
        return hash(self._attrs()) 

    def __str__(self):
        return u'{}: {}'.format(self.position, u'->'.join(self.labels[::-1]))

class Transition:
    def __init__(self, labels, position):
        self.labels = labels
        self.position = position

    def _attrs(self):
        return (self.labels, self.position)
    
    def __eq__(self, other):
        return isinstance(other, SentenceNGram) and \
            self._attrs() == other._attrs()
      
    def __hash__(self):
        return hash(self._attrs()) 
          
    def __str__(self):
        return u'{}: {}'.format(self.position, u'->'.join(self.labels[::-1]))


def build_ngram_lattice(discourse_model, c):
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

    # The number of sentence positions to predict.
    positions = len(discourse_model.doc.sents)

    # The set of non 'START' or 'END' sentence labels.
    node_labels = [u's-{}'.format(i) for i in range(positions)]

    # The start node of the graph with label 'START' and position -1.
    # Everyone has to start somewhere :)
    ngram = discourse_model.ngram
    padding = tuple([u'START']) * (ngram - 1)
    start_sent = SentenceNGram(tuple(padding), -1)

    # As we build the graph, this will hold the previous position's
    # nodes, that is, the set of nodes it is possible to add edges
    # from.
    nodes = set()

    # Initialize the graph construction.
    c.init(start_sent)
    nodes.add(start_sent)

    # Create the remaining graph nodes/edges.
    for position in range(positions):

        # Find all possible edges for each possible new node and add them
        # to n2e.
        n2e = defaultdict(set)
        for node_label1 in node_labels:
            for node in nodes:

                # This edge does not result in a legal path.
                if node_label1 in node.labels:
                    continue

                s = SentenceNGram(tuple([node_label1]) + node.labels[:-1], 
                                  position)
                n2e[s].add(node)

        # Create the new nodes and edges.
        for next_node in n2e.keys():
            node_edge = []
            for prev_node in n2e[next_node]:
                transition = Transition(tuple([next_node.labels[0]]) + \
                                        prev_node.labels, position)
                node_edge.append((prev_node, transition))

            c[next_node] = c.sum([c[prev_node] * c.sr(transition)
                                  for prev_node, transition in node_edge])

        # Replace nodes with current position nodes.
        nodes = n2e.keys()

    # Create the final END node.
    end_node = SentenceNGram(tuple([u'END']), positions)
    c[end_node] = \
        c.sum([c[node] * c.sr(Transition(tuple([u'END']) + node.labels,
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
    curr_tok = u'START'

    # If pop_cntr exceeds the number of transitions, the set is not
    # a valid path in the graph, return an empty list.
    pop_cntr = 0
    while len(transitions) > 0:
        t = transitions.pop(0)
        pop_cntr += 1
        if t.labels[1] == curr_tok:
            ordered_trans.append(t)
            curr_tok = t.labels[0]
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
