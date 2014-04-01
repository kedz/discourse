import pydecode.chart as chart
import itertools
from collections import defaultdict
from discourse.gazetteers import DiscourseConnectives


def build_hypergraph(discourse_model, c):
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

    # The number of entities to explicitly track in the model.
    # This number should probably not go above 4 or the graph will
    # blow up.
    #nsalient = discourse_model._num_salient_ents

    # The set of non 'START' or 'END' sentence labels.
    node_labels = [u'sent-{}'.format(i) for i in range(positions)]

    # The start node of the graph with label 'START' and position -1. 
    # Everyone has to start somewhere :)
    start_sent = Sentence(u'START', -1, dcon=u'__START__')

    # As we build the graph, this will hold the previous position's
    # nodes, that is, the set of nodes it is possible to add edges
    # from.
    nodes = set()

    # Dict mapping sentence indices to salient entity bit strings.
    s2e = discourse_model.sent2ents

    # Dict mapping sentence indices to explicit discourse connectives
    # or the empty string ''.
    s2dcon = discourse_model.sent2dcon

    # Initialize the graph construction.
    c.init(start_sent)
    nodes.add(start_sent)

    # Create the remaining graph nodes/edges.
    for position in range(positions):

        # Find all possible edges for each possible new node and add them
        # to n2e.
        n2e = defaultdict(set)
        for node_label1 in node_labels:
            next_dcon = s2dcon[s2i(node_label1)]
            for node in nodes:

                # This edge does not result in a legal path.
                if node_label1 == node.label:
                    continue
                
                # Get the next salient entity set.
                ents = s2e[s2i(node_label1)].union(node.entities)
                
                # If the next node's discourse connective is empty,
                # use the previous node discourse connective.
                if next_dcon == u'':
                    next_dcon = node.dcon
                
                s = Sentence(node_label1, position, ents, next_dcon)
                n2e[s].add(node)

        # Create the new nodes and edges.
        for next_node in n2e.keys():
            node_edge = []
            for prev_node in n2e[next_node]:
                transition = Transition((next_node.label, prev_node.label),
                                        position, prev_node.entities,
                                        prev_node.dcon)
                node_edge.append((prev_node, transition))

 
            c[next_node] = c.sum([c[prev_node] * c.sr(transition)
                                  for prev_node, transition in node_edge])
            
        # Replace nodes with current position nodes.
        nodes = n2e.keys()

    # Create the final END node.
    end_ents = reduce(lambda x, y: x.union(y), s2e.values(), frozenset())
    end_node = Sentence(u'END', positions, entities=end_ents, dcon=u'__END__')
    c[end_node] = \
        c.sum([c[node] * c.sr(Transition((u'END', node.label),
                                         positions,
                                         node.entities,
                                         node.dcon))
               for node in nodes if node.entities == end_ents])

    return c


class Sentence:
    """ A Sentence node in the sentence ordering hypergaph.

    Attributes
    ----------
    label : string
        The current sentence label, possibly 'START, 'END',
        or 'sent_i' where i is in {0, ..., n-1} and n is the number 
        of sentences in the problem instance.

    position : int
        The position of this sentence node in the graph.              

    entities : frozenset
        A set of strings representing the salient entities in
        the current sentence.

    dcon : string
        A discourse connective, if any, present in the sentence.
        If there are no discourse connectives, the default 
        value is the empty string.
    """
    def __init__(self, label, position,
                 entities=frozenset(), dcon=u''):
        self.label = label
        self.position = position
        self.entities = entities
        self.dcon = dcon

    def _attrs(self):
        return (self.label, self.position, self.entities, self.dcon)

    def __eq__(self, other):
        return isinstance(other, Sentence) and self._attrs() == other._attrs()

    def __hash__(self):
        return hash(self._attrs())

    def __str__(self):
        return u'{}: {}, {}, {}'.format(self.position, self.label,
                                        self.entities, self.dcon)


class Transition:
    """ A Transition is an edge between sentence nodes in the
    sentence ordering graph.

    Attributes
    ----------
    sentences : tuple, string
        A tuple of sentence labels where the first element is the
        label of the sentence being transitioned to, and subsequent
        sentence labels correspond to sentence taken on the path to
        that sentence.

    position : int
        The position of the sentence node at the head of this edge.

    previous_entities : frozenset, string
        A set of salient entities that have previously occurred
        somewhere on the path up to but not including the sentence
        node at the head of this edge.

    previous_dcon : string
        The last discourse connective that previously occurred
        somewhere on the path up to but not including the sentence
        node at the head of this edge.
    """
    def __init__(self, sentences, position=0,
                 previous_entities=frozenset(), previous_dcon=u''):

        self.sentences = tuple(sentences)
        self.position = position
        self.previous_entities = previous_entities
        self.previous_dcon = previous_dcon

    def _attrs(self):
        return (self.sentences, self.position,
                self.previous_entities, self.previous_dcon)

    def __eq__(self, other):
        return isinstance(other, Transition) \
            and self._attrs() == other._attrs()

    def __hash__(self):
        return hash(self._attrs())

    def __str__(self):
        label = ''
        if self.__len__() > 1:
            for sent in self.sentences[1:]:
                label = u'{} -> '.format(sent) + label
        return label + u'{}, {}, {}, {}'.format(self.sentences[0],
                                                self.previous_entities,
                                                self.previous_dcon,
                                                self.position)
                                                
    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)

    def __nonzero__(self):
        return True


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
    if transition[0] != u'END':
        cons.append((transition[0], 1))
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
        if t[1] == curr_tok:
            ordered_trans.append(t)
            curr_tok = t[0]
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
        'sent_i' where i is the sentence index.

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
        index = int(sent_label[5:])
    return index
