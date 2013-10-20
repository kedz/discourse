import numpy as np
from collections import defaultdict
from os import listdir
from os.path import join, isfile, splitext
import math


def generate_transitions(role_set, path_size):
    """Create an ordered list of transitions, from a set role labels.
        The number of roles is exponential in the path size."""

    # Each loop iteration i adds all transitions
    # of size (i+1) to the list trans.
    trans = ['']

    for p in range(path_size):
        new_trans = []
        for r in role_set:
            for t in trans:
                new_trans.append(t+r)
        trans = new_trans

    # trans is now a sorted list of all possible role transitions.
    return sorted(trans)


def new_entity_grid(cmat, coref=False, syntax=False, salience=False, length=2):
        """Construct a new EntityGrid from a matrix or chararray of
            entity role transitions."""

        # cmat is a numpy matrix or 2d chararry of representing
        # an entity's role in a sentence.
        # Columns correspond to entities and rows correspond
        # to a document's sentences.
        # e.g. an EntityGrid with 3 entities and 2 sentences:
        #     [['x','-','s'],
        #     ['o','s','x']]
        #
        # When syntax is true, use the role set (s,o,x,-), else use (x,-).
        #
        # When salience is true, split entities into two categories:
        # salient and non salient, by frequency.
        # Transition set then becomes the cartesian product
        # of {salient,nonsalient} x {s,o,x,-}^length.
        #
        # length is the transition length.
        # When length is 3 and syntax=False,
        # the transitions are {---,--x,-x-,x--,-xx,xx-,x-x,xxx}.

        # Initialize the transition set
        if syntax is True:
            trans = generate_transitions(('-', 'x', 's', 'o'), length)
        else:
            trans = generate_transitions(('-', 'x'), length)

        return EntityGrid(cmat, trans, length, salience)


class EntityGrid:

    def __init__(self, trans_mat, trans,
                 length=2, salience=False):
        """An EntityGrid computes the transition count
            vector representation of a document."""

        # Count maps for transistions -- the first index is for non salient
        # entities, and the second for salient ones.
        self.trans_cnts = [defaultdict(int), defaultdict(int)]

        # Store set of possible transitions.
        self.trans = trans

        # Boolean flag to turn salience on or off.
        self.salience = salience

        # Total number of transitions for salient and non salient entities.
        self.num_trans = [0, 0]

        # Transistion matrix representing the document.
        self.trans_mat = trans_mat

        # Count up all the transitions.
        self._count_transitions(trans, length, salience)

        # Transition Probability Vector is None until getter is called.
        self._tpv = None
        
        # Transition Count Vector is None until getter is called.
        self._tcv = None
        

    def _count_transitions(self, transSet, length, salience):
        """Counts the transitions that occur in this document.
            There are possibly two classes of transition: salient
            and nonsalient."""

        # If we are using the salience feature, map columns that have
        # more than 1 non '-' role as salient.
        if salience:
            salmap = {}
            for c in range(self.trans_mat.shape[1]):
                entFreq = 0
                for r in range(self.trans_mat.shape[0]):
                    if not self.trans_mat[r, c] == "-":
                        entFreq += 1
                if entFreq > 1:
                    salmap[c] = 1
                else:
                    salmap[c] = 0

        # For each column and for each row, count the entity transitions
        for c in range(self.trans_mat.shape[1]):
            for r in range(self.trans_mat.shape[0]-1):
                trans = self.trans_mat[r, c]
                i = r+1
                while len(trans) < length and i < self.trans_mat.shape[0]:
                    trans += self.trans_mat[i, c]
                    if trans in transSet:
                        sal = 0
                        if salience:
                            sal = salmap[c]
                        self.trans_cnts[sal][trans] += 1
                        self.num_trans[sal] += 1
                    i += 1

    def _build_vector_rep(self, trans, salience, normalized):
        """Construct the transition count vector representation.
            If normalized, these can be interpreted as a generative model
            for entity transitions."""

        # If salience is used, there are twice as many features.
        if salience:
            v = np.zeros((len(trans)*2, 1))
        else:
            v = np.zeros((len(trans), 1))

        # Get the feature value of each transition and place it in v.
        for i, t in enumerate(trans):

            if trans[i] in self.trans_cnts[0]:
                if self.num_trans[0] > 0:
                    v[i] = self.trans_cnts[0][t]
                    if normalized:
                        v[i] /= float(self.num_trans[0])

            if salience:
                if self.num_trans[1] > 0:
                    v[t+len(trans)] = self.trans_cnts[1][trans[t]]
                if normalized:
                    v[t+len(trans)] /= float(self.num_trans[1])

        # If normalized,
        # cache this vector as a tpv (transition probability vector).
        if normalized:
            self._tpv = v
            return v
        # Otherwise cache this vector as a tcv (transition count vector).
        else:
            self._tcv = v
            return v

    def get_trans_cnt_vctr(self):
        """Get a vector of entity transition counts."""

        if self._tcv is not None:
            return self._tcv
        else:
            return self._build_vector_rep(self.trans, self.salience, False)

    def get_trans_prob_vctr(self):
        """Get a vector of entity transition probabilities."""
        
        if self._tpv is not None:
            return self._tpv
        else:
            return self._build_vector_rep(self.trans, self.salience, True)
