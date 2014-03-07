import pydecode.hyper as ph
import pydecode.chart as chart
import itertools
from collections import namedtuple, defaultdict
from discourse.gazetteers import DiscourseConnectives
from bitstring import Bits, BitArray


class Sentence(namedtuple("Sentence", ["label", "prev_sents",
                                       "pos", 'entities', 'conn'])):
    """
    A Sentence node in the sentence ordering hypergaph.

    label -- The current sentence label, possibly 'START, 'END',
        or 'sent_i' where i is in {0, ..., n-1} and n is the number of
        sentences in the problem instance.

    prev_sent -- A list of the n-1 previous sentences taken to get to
        this node where n is the history size.
    """
    def __new__(_cls, label, prev_sents, pos, entities=Bits(1), conn=''):
        non_nulls = [x for x in prev_sents if x != ()]

        return super(_cls, Sentence).__new__(_cls, label, tuple(non_nulls),
                                             pos, entities, conn)

    def __str__(self):
        return "{}: {}, {}, {}, {}".format(self.pos, self.label,
                                           self.prev_sents, self.entities.bin,
                                           self.conn)


class Transition:
    """
    A Transition is an edge between sentence nodes in the sentence
    ordering hypergraph.
    """
    def __init__(self, sents, pos=0, prev_ents=Bits(1), conn=''):
        """
        sents -- A list of n sentence labels, where n is the history
            size of the models.

        pos -- The position of the head sentence in the predicted
            ordering.

        prev_ents -- A bitstring where each bit represents the presence
            or absence of a salient entity in the path up to but not
            including the head sentence. If the ith bit is 1, it
            indicates that the ith salient entity occurred in the
            path.

        conn -- The most recent explicit discourse connective
            present in the path to the head sentence, including the
            head sentence.
        """
        self.sents = tuple(sents)
        self.pos = pos
        self.prev_ents = prev_ents
        self.conn = conn

    def __str__(self):
        label = ''
        if self.__len__() > 1:
            for sent in self.sents[1:]:
                label = "{} -> ".format(sent) + label
        return label + "{}, {}, {}, {}".format(self.sents[0],
                                               self.prev_ents.bin, self.conn,
                                               self.pos)

    def __eq__(self, other):
        if not isinstance(other, Transition):
            return False
        t1 = (self.sents, self.pos, self.prev_ents, self.conn)
        t2 = (other.sents, other.pos, other.prev_ents, other.conn)
        return t1 == t2

    def __hash__(self):
        return hash((self.sents, self.pos, self.prev_ents, self.conn))

    def __iter__(self):
        return iter(self.sents)

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sents)

    def __nonzero__(self):
        return True


def build_constraints(transition):
    """
    Build a constraint for a transition.
    """
    if transition[0] != 'END':
        return [(transition[0], 1)]
    return []


def build_hypergraph(discourse_model, c):
    """
    Builds the hypergraph for the discourse model.
    """

    # The number of sentence positions to predict - the +1 is for the
    # last sentence to the 'END' label.
    positions = len(discourse_model) + 1

    # The number of salient entities to track in the model.
    # This number should probably not go above 4 or the graph will
    # blow up.
    nsalient = discourse_model._num_salient_ents

    # The set of non 'START' or 'END' sentence labels.
    node_labels = ['sent-'+str(i) for i, s in enumerate(discourse_model)]

    # The start node of the graph. Everyone has to start somewhere :)
    start_sent = Sentence('START', (), 0, entities=Bits(nsalient))

    # As we build the graph, this will hold the previous position's
    # nodes, that is, the set of nodes it is possible to add edges
    # from.
    nodes = set()

    # Dict mapping sentence indices to salient entity bit strings.
    s2b = discourse_model.sent2ent_bstr

    # Dict mapping sentence indices to explicit discourse connectives
    # or the empty string ''.
    s2t = discourse_model.sent2trans

    # Initialize the graph construction.
    c.init(start_sent)

    # Create the first edges from the start node.
    for node_label1 in node_labels:
        # Create new sentence node.
        conn = s2t[s2i(node_label1)]
        ent_bstr = s2b[s2i(node_label1)]
        s = Sentence(node_label1, tuple([start_sent.label]), 1,
                     entities=ent_bstr, conn=conn)

        # Add an edge from start to sentence s in the graph.
        c[s] = c.sum([c[start_sent] * c.sr(Transition(
                      (node_label1, start_sent.label),
                      prev_ents=start_sent.entities))])

        # Add s to the previous position nodes.
        nodes.add(s)

    # Create the remaining graph nodes/edges.
    for position in range(2, positions):
        # Find all possible edges for each possible new node
        n2e = defaultdict(set)
        for node_label1 in node_labels:
            next_conn = s2t[s2i(node_label1)]
            for node in nodes:
                if node_label1 == node.label:
                    continue
                next_bstr = s2b[s2i(node_label1)] | node.entities
                if next_conn == '':
                    next_conn = node.conn
                s = Sentence(node_label1, tuple([node.label, ]), position,
                             entities=next_bstr, conn=next_conn)
                n2e[s].add(node)

        # Create the new nodes and edges.
        for next_node in n2e.keys():
            c[next_node] = c.sum([c[prev_node] * c.sr(Transition(
                                  (next_node.label, prev_node.label),
                                  pos=position, prev_ents=prev_node.entities,
                                  conn=prev_node.conn))
                                  for prev_node in n2e[next_node]])
        # Replace nodes with curr position nodes.
        nodes = n2e.keys()

    # Create the final END node.
    ent_bstrs = set([Bits(nsalient) | item[1]
                     for item in discourse_model.sent2ent_bstr.items()])
    end_bstr = reduce(lambda x, y: x | y, ent_bstrs, Bits(nsalient))
    c[Sentence('END', (), positions, entities=end_bstr)] = \
        c.sum([c[node] * c.sr(Transition(('END', node.label), pos=positions,
                                         prev_ents=node.entities,
                                         conn=node.conn))
               for node in nodes if node.entities == end_bstr])

    return c


def recover_order(transition_set):
    """
    Returns an ordered list of transitions where the head sentence of
    the ith transition is the tail sentence of the (i+1)th transition.

    transition_set -- The set of Transitions predicted by an LP solver.
        This must be a valid path in the graph.
    """

    transitions = list(transition_set)
    ordered_trans = []
    curr_tok = 'START'

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
            print "Invalid solution"
            for t in transition_set:
                print t
            print
            return []

    return ordered_trans


def s2i(sent_label, end=None):
    """
    Convert a sentence label to its index. 'START' label returns -1.
    Optional argument end sets output for the 'END' label which
    defaults to None.

    sent_label -- A sentence label string, either 'START', 'END', or
        'sent_i' where i is the sentence index.

    end -- Optional argument specifying the output for an 'END' label.

    returns an integer index, None, or the value specified by end.
    """
    if sent_label == 'START':
        return -1
    elif sent_label == 'END':
        return end
    else:
        return int(sent_label[5:])
