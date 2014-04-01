import corenlp as cnlp
from collections import deque, defaultdict
from discourse.hypergraph import s2i
import discourse.gazetteers as gazetteers
import itertools
from discourse.hypergraph import Transition
import discourse.hypergraph
import nltk
import textwrap

# These are the default features that are active
active_features = {}

# Decorate features if they are active for first and last
# sequence transitions.
active_features['is_first'] = True
active_features['is_last'] = True

# Mark role transitions (e.g. subj --> obj) of matching entities.
active_features['role_match'] = True
active_features['use_det'] = True
active_features['use_sal_ents'] = True

# Mark discourse connective transitions.
active_features['discourse_connectives'] = True
# Mark the first occurence of a salient graph entity.
active_features['discourse_new'] = True

# Mark Tree Syntax sequences of depths 1&2
active_features['syntax_lev1'] = True
active_features['syntax_lev2'] = True

# Mark first word to first word transitions.
active_features['first_word'] = True

# Mark counts of NE types from sentence to sentence.
active_features['ne_types'] = True

# Enable Debug mode -- the correct transitions are the only ones
# that will be score. Do not turn this on unless you are debugging.
active_features['debug'] = False

# Explicit Discourse gazetteers for identifying expl disc tokens in a sentence.
expl_disc = gazetteers.DiscourseConnectives()


class BigramCoherenceInstance:
    """ An bigram discourse coherence sequence mode instance for
    predicting the correct order of sentences in a document.


    Attributes
    ----------
    doc : corenlp.Document
        A corenlp.Doucment object that has pos, ner tags, and
        constituent and dependency parses for the problem
        instance.

    active_feat : dict, string -> bool
        A dict mapping feature names to boolean values, with True
        indicating that a particular feature is being used.

    num_graph_entities : int
        The number of salient entities to track in the model. This
        number should probably not exceed 4 or the graph will blow
        up. A salient entity is any noun that occurs more than once.
        If this value is set to n, the model tracks the top n
        entities by frequency, breaking ties arbitrarily. This
        value is a maximum -- it is possible that a document has
        fewer than n salient entities.

    entity_counts : dict (string -> int)
        A dict mapping document noun strings to occurrence counts.

    _f_cache : dict (Transition -> dict (string -> int))
        This cache stores graph edge (i.e. transitions) feature maps
        so that transition feautres need not be calculated more than
        once.

    graph_ents : frozenset (string)
        The set of salient entities that will be used in graph
        construction.

    sent2ents : dict (int -> frozenset (string))
        A mapping of sentence indices to the set of the graph entities
        that sentence contains if any.

    sent2dcon : dict (int -> string)
        A mapping of sentence indices to an explicit discourse
        connective if present in the corresponding sentence.
        If the current model does not use the
        'discourse_connectives' feature then this dict simply
        maps all sentences to the empty string: ''

    """

    def __init__(self, doc, features=None, num_graph_entities=0):

        self.doc = doc
        self.active_feat = active_features if features is None else features
        self.entity_counts = self._build_entity_counts(doc)
        self._f_cache = {}

        self.graph_ents, self.sent2ents = \
            self._build_graph_ents(doc, num_graph_entities,
                                   self.entity_counts)

        # Update the model with the actual number of salient graph entities
        # found -- this could be less than specified in the num_graph_ents
        # argument.
        self.num_graph_entities = len(self.graph_ents)

        use_expl_disc = self.active_feat.get('discourse_connectives', False)
        self.sent2dcon = self._extract_expl_disc(doc, use_expl_disc)

    def _build_entity_counts(self, doc):
        """ Build a dict of occurrence counts of noun tokens in this
        instances's document.

        Parameters
        ----------
        doc : corenlp.Document
            The problem instance's document containing the sentences
            to be ordered.

        Returns
        -------
        counts : dict (string -> int)
            A dict with occurrence counts for each noun used in the
            document.
        """

        counts = {}
        for s in doc:
            for t in s:
                if t.pos in [u'NN', u'NNS', u'NP', u'NPS']:
                    lem = t.lem.lower()
                    if lem not in counts:
                        counts[lem] = 1
                    else:
                        counts[lem] += 1
        return counts

    def _extract_expl_disc(self, doc, use_expl_disc):
        """ Return a dict mapping sentence indices to the explicit
        discourse connective the respective sentence containts, if
        any. When no connective is present or use_expl_disc is False,
        return the empty string.

        Only checks the first four tokens in the sentence for a
        discourse connective as these are more likely to have
        transition words that hold across sentence, as opposed to
        within a sentence.

        Parameters
        ----------
        doc : corenlp.Document

        use_expl_disc : bool
            A flag indicating whether or not this
            feature is active.

        Returns
        -------
            s2c : dict, corenlp.Sentence -> string
            A dict mapping sentences to the first explicit discourse
            connective, if any, in the first 5 words in the sentence.
        """

        if not use_expl_disc:
            s2c = {}
            for i, s in enumerate(doc):
                s2c[i] = ''
            return s2c

        s2c = {}
        for i, s in enumerate(doc):
            preamble = u' '.join(unicode(token) for token in s[0:5])
            t = expl_disc.contains_connective(preamble)
            if t is not None:
                s2c[i] = t
            else:
                s2c[i] = ''

        return s2c

    def _build_graph_ents(self, doc, num_entities, ent_counts):
        """ Builds the set of graph entites that will be used in
        graph construction as well as a mapping of sentence indices
        to the graph entities that they contain, if any. Entities are
        chosen in order of most frequent within the document. Ties
        are broken arbitrarily.

        Parameters
        ----------
        doc : corenlp.Document
            The problem instance's document containing the sentences
            to be ordered.

        num_entities : int
            The number of salient entities to use in graph
            construction.

        ent_counts : dict (string -> int)
            A dict mapping noun strings to their occurrence counts.

        Returns
        -------
        graph_ents : frozentset (string)
            A set of noun strings corresponding to the salient\
            entities used in graph construction for this problem
            instance.

        sent2ents : dict (int -> frozenset (string))
            A dict mapping sentence indices to the set of graph
            entities in that sentence, if any.
        """

        unsorted_ents = [item for item in ent_counts.items() if item[1] > 1]
        sorted_ents = sorted(unsorted_ents, key=lambda x: x[1], reverse=True)
        graph_ents = frozenset([entity for entity, count
                                in sorted_ents[0:num_entities]])

        sent2ents = {}
        for i, sent in enumerate(doc):
            sent_ents = frozenset(token.lem.lower() for token in sent
                                  if token.lem.lower() in graph_ents)
            sent2ents[i] = sent_ents

        return graph_ents, sent2ents

    def gold_str(self):
        """ Returns the correct sentence ordering as a string.
        E.g.
            (0) This is sentence 0.
            (1) And this is sentence one.
            (2) Notice that our indices are in the correct order.
            (3) This ordering is indicative of the gold ordering.
        """
        strs = [u'({})  {}'.format(i, unicode(s))
                for (i, s) in enumerate(self)]
        return u'\n'.join(strs)

    def feature_map(self, transition):
        """ Build the feature map for this transition (i.e. graph
        edge). If we have already constructed this map, return the
        map in _f_cache.

        Parameters
        ----------

        transition : Transition
            A Transition object corrpesonding to the graph edge at
            hand.

        Returns
        -------
            fmap : dict (string -> int)
            A dict mapping feature names to feature values for
            graph edge that corresponds to transition.
        """

        # Check the cache if we have already created this transition's
        # feature map, otherwise create a new one.
        if transition in self._f_cache:
            fmap = self._f_cache[transition]
            return fmap

        else:

            # Create an empty feature map for this transition.
            fmap = {}

            ### Call each feature function if active. ###
            if self.active_feat.get('role_match', False):
                self._f_role_match(fmap, transition)

            if self.active_feat.get('discourse_new', False):
                self._f_discourse_new(fmap, transition)

            if self.active_feat.get('discourse_connectives', False):
                self._f_discourse_connectives(fmap, transition)
            if self.active_feat.get('first_word', False):
                self._f_first_word(fmap, transition)

            if self.active_feat.get('syntax_lev1', False):
                self._f_syntax_lev(fmap, transition, 1)
            if self.active_feat.get('syntax_lev2', False):
                self._f_syntax_lev(fmap, transition, 2)

            if self.active_feat.get('ne_types', False):
                self._f_ne_types(fmap, transition)

            if self.active_feat.get(u'debug', False):
                self._f_debug(fmap, transition)

            self._f_cache[transition] = fmap
            return fmap

    def _f_debug(self, fmap, transition):
        nsents = len(self.doc.sents)
        head = s2i(transition[0], end=nsents)
        tail = s2i(transition[1], end=nsents)

        if transition.position == head and tail + 1 == head:
            fmap['DEBUG'] = 1

    def _f_first_word(self, fmap, transition):
        """ Marks the sequence first words of the sentences selected
        by the graph edge *transition*.
        E.g. 'a' ---> 'the' .

        Parameters
        ----------
        fmap : dict (string -> int)
            A dict mapping feature names to feature values
            for this transition. This function mutates this dict.

        transition : Transition
            The graph edge, from which this function
            extracts features.
        """

        # Extract first word from tail sentence.
        if transition[1] == u'START':
            word1 = u'START'
            ne1 = u'START'
        else:
            idx = s2i(transition[1])
            sent1 = self.doc[idx]
            token1 = sent1.tokens[0]

            word1 = token1.lem.lower()
            ne1 = token1.ne

        # Extract first word from head sentence.
        if transition[0] == u'END':
            word0 = u'END'
            ne0 = u'END'
        else:
            idx = s2i(transition[0])
            sent0 = self.doc[idx]
            token0 = sent0.tokens[0]
            word0 = token0.lem.lower()
            ne0 = token0.ne

        # Mark the feature
        fstr1 = u'First Word Trans: {} --> {}'.format(word1, word0)
        fmap[fstr1] = 1

        # Mark smoothed versions of this feature.
        fstr2 = u'First Word Trans: __ --> {}'.format(unicode(word0))
        fmap[fstr2] = 1

        fstr3 = u'First Word Trans: {} --> __'.format(unicode(word1))
        fmap[fstr3] = 1

        fstr4 = u'First Word Trans: {} --> {}'.format(ne1, ne0)
        fmap[fstr4] = 1

        fstr5 = u'First Word Trans: {} --> {}'.format(ne1, word0)
        fmap[fstr5] = 1

        fstr6 = u'First Word Trans: {} --> {}'.format(word1, ne0)
        fmap[fstr6] = 1

        fstr7 = u'First Word Trans: {} --> __'.format(ne1)
        fmap[fstr7] = 1

        fstr8 = u'First Word Trans: __ --> {}'.format(ne0)
        fmap[fstr8] = 1

    def _f_syntax_lev(self, fmap, transition, depth):
        """ Marks the non-terminal sequence transition in the
        feature map. E.g. S , NP VP . ---> NP VP .

        Parameters
        ----------
        fmap : dict (string -> int)
            A dict mapping feature names to feature values
            for this transition. This function mutates this dict.

        transition : Transition
            the graph transition, for which this function
            extracts features.

        depth : int
            The depth of the sequence to extract from the parse
            tree.
        """

        # Extract syntax sequence for the tail sentence.
        if transition.sentences[1] == 'START':
            seq1 = 'START'
        else:
            idx = s2i(transition[1])
            seq1_parse = self.doc[idx].parse
            seq1 = syn_sequence(seq1_parse, depth)

        # Extract syntax sequence for the head sentence.
        if transition.sentences[0] == 'END':
            seq0 = 'END'
        else:
            idx = s2i(transition[0])
            seq0_parse = self.doc[idx].parse
            seq0 = syn_sequence(seq0_parse, depth)

        # Assign feature value.
        fmap['{} -sl{}-> {}'.format(seq1, depth, seq0)] = 1

        # Smoothed features.
        fmap['__ -sl{}-> {}'.format(depth, seq0)] = 1
        fmap['{} -sl{}-> __'.format(seq1, depth)] = 1

    def _f_discourse_new(self, fmap, transition):
        """ Marks feature map if the head sentence contains the first
        occurrence of a salient entity, that is, a discourse new
        entity.

        Parameters
        ----------

        fmap : dict (string -> int)
            A dict mapping feature names to feature values
            for this transition. This function mutates this dict.

        transition : Transition
            the graph transition, for which this function
            extracts features.

        """

        if transition.sentences[0] != u'END':
            idx = s2i(transition.sentences[0])

            s2e = self.sent2ents
            num_new = 0
            for ent in s2e[idx]:
                if ent not in transition.previous_entities:
                    num_new += 1

            if num_new > 0:
                fmap[u'Discourse New'] = num_new

    def _entity_roles(self, s):
        """ Returns a set of entity, role tuples for a sentence s.

        Paramters
        ---------
        s : corenlp.Sentence
            A sentence from this problem instance.

        Returns
        -------
            s_ents : set (tuple(Token, string))
            A set of word token, role label tuples.
        """
        s_ents = set()
        dtypes = ['csubj', 'csubjpass', 'dobj', 'iobj', 'nsubj', 'nsubjpass']
        dgraph = s.dep_graph()
        roots = [(rel.dep, rel.type) for rel in dgraph.type['root']]
        roots.extend([(rel.dep, rel.type) for rel in dgraph.type['ccomp']])
        for token, dtype in roots:
            if token.pos in ['NN', 'NNS', 'NP', 'NPS']:
                s_ents.add((token, dtype))
            for rel in dgraph.govs[token]:
                if rel.type in dtypes or 'prep' in rel.type:
                    if rel.dep.pos in ['NN', 'NNS', 'NP', 'NPS']:
                        s_ents.add((rel.dep, rel.type))

        for rel in s.deps:
            if rel.type != 'nn' and rel.dep.pos in ['NN', 'NNS', 'NP', 'NPS']:
                if (rel.dep, rel.type) not in s_ents:
                    s_ents.add((rel.dep, 'other'))
        return s_ents

    def _f_role_match(self, fmap, transition):
        """ This feature counts noun phrase head matches across
        sentences. The feature takes the form of the dependency
        relation for the entity in each sentence, and whether or not
        the entitiy in question is a salient entity.
        E.g. 'nsubj --> dobj SALIENT'. Start and end role transitions
        are similarly captured, e.g. 'START -> other' and
        'iobj -> END'.

        Parameters
        ----------

        fmap : dict (string -> int)
            A dict mapping feature names to feature values
            for this transition. This function mutates this dict.

        transition : Transition
            the graph transition, for which this function
            extracts features.
        """

        # If the tail sentence is START, create a START role for each entity
        # that occurs in the head sentence.
        if transition[1] == u'START':
            idx0 = s2i(transition[0])
            s0_ents = self._entity_roles(self.doc[idx0])
            s1_ents = [(token, u'START') for token, role in s0_ents]

        # If the head sentence is END, create an END role for each entity
        # that occurs in the tail sentence.
        elif transition[0] == u'END':
            idx1 = s2i(transition[1])
            s1_ents = self._entity_roles(self.doc[idx1])
            s0_ents = [(token, u'END') for token, role in s1_ents]

        # Default behavior, extract entity role tuples for each sentence.
        else:
            idx0 = s2i(transition[0])
            idx1 = s2i(transition[1])
            s0_ents = self._entity_roles(self.doc[idx0])
            s1_ents = self._entity_roles(self.doc[idx1])

        # Entity counts
        ecnts = self.entity_counts

        # This set records entities matched in the head sentence.
        # For entites in the head sentence that are NOT matched, this set
        # makes it possible to create a feature of the form "X -> role"
        # where the X indicates that the entitiy did not appear in the tail.
        used_ents = set()

        # This default dict is used to build the feature counts that will be
        # added to fmap.
        role_matches = defaultdict(int)

        # Find matching entities across sentences, and mark them as features.
        for ent1 in s1_ents:
            lem1 = ent1[0].lem.lower()
            is_salient = u'SALIENT' if ecnts[lem1] > 1 else u'not salient'

            #ne1 = ent1[0].ne
            no_match = True
            for ent0 in s0_ents:
                lem0 = ent0[0].lem.lower()
                if lem0 == lem1:
                    no_match = False
                    used_ents.add(lem0)

                    fstr1 = u'Role Trans: {} --> {}'.format(ent1[1], ent0[1])
                    role_matches[fstr1] += 1

                    sfstr1 = fstr1 + u' {}'.format(is_salient)
                    role_matches[sfstr1] += 1

                    # Backoff features with generic __ symbol
                    fstr2 = u'Role Trans: __ --> {}'.format(ent0[1])
                    role_matches[fstr2] += 1

                    sfstr2 = fstr2 + u' {}'.format(is_salient)
                    role_matches[sfstr2] += 1

                    fstr3 = u'Role Trans: {} --> __'.format(ent1[1])
                    role_matches[fstr3] += 1

                    sfstr3 = fstr3 + u' {}'.format(is_salient)
                    role_matches[sfstr3] += 1

            if no_match:
                fstr1 = u'Role Trans: {} --> X'.format(ent1[1])
                role_matches[fstr1] += 1
                sfstr1 = fstr1 + u' {}'.format(is_salient)
                role_matches[sfstr1] += 1

        for ent, role in s0_ents:
            lem = ent.lem.lower()
            if lem not in used_ents:
                is_salient = u'SALIENT' if ecnts[lem] > 1 else u'not salient'
                fstr1 = u'Role Trans: X --> {}'.format(role)
                role_matches[fstr1] += 1
                sfstr1 = fstr1 + u' {}'.format(is_salient)
                role_matches[sfstr1] += 1

        for feature, val in role_matches.items():
            fmap[feature] = val

    def _f_discourse_connectives(self, fmap, transition):
        if transition.sentences[0] == u'END':
            dcon = u'END'
        else:
            idx = s2i(transition.sentences[0])
            dcon = self.sent2dcon[idx]

        prev_dcon = transition.previous_dcon

        fstr1 = u'Discourse Connective {} -> {}'.format(prev_dcon, dcon)
        fmap[fstr1] = 1

        fstr2 = u'Discourse Connective {} -> __'.format(prev_dcon)
        fmap[fstr2] = 1

        fstr3 = u'Discourse Connective __ -> {}'.format(dcon)
        fmap[fstr3] = 1

        is_match = u'MATCH' if dcon == prev_dcon else u'not a match'
        fstr4 = u'Discourse Connective {}'.format(is_match)
        fmap[fstr4] = 1

    def _ne_counts(self, sentence):
        """ Return set of tag, count tuples of  the Named Entity tags
        for a sentence."""

        ne_counts = defaultdict(int)
        for token in sentence:
            if token.ne != u'O':
                ne_counts[token.ne] += 1
        return set(ne_counts.items())

    def _f_ne_types(self, fmap, transition):
        """ Mark NE type transitions and counts.
        E.g. NE Counts ORG_3 --> DATE_1

        Parameters
        ----------

        fmap : dict (string -> int)
            A dict mapping feature names to feature values
            for this transition. This function mutates this dict.

        transition : Transition
            the graph transition, for which this function
            extracts features.
        """

        if transition.sentences[1] == u'START':
            sent1 = set([(u'START', 1)])
            idx = s2i(transition.sentences[0])
            sent0 = self._ne_counts(self.doc[idx])
        elif transition.sentences[0] == u'END':
            sent0 = set([(u'END', 1)])
            idx = s2i(transition.sentences[1])
            sent1 = self._ne_counts(self.doc[idx])
        else:
            idx1 = s2i(transition.sentences[1])
            sent1 = self._ne_counts(self.doc[idx1])
            idx0 = s2i(transition.sentences[0])
            sent0 = self._ne_counts(self.doc[idx0])

        if len(sent1) == 0:
            sent1.add((u'X', 1))

        if len(sent0) == 0:
            sent0.add((u'X', 1))

        for ne1 in sent1:
            for ne0 in sent0:
                fstr1 = u'NE Counts {}_{} --> {}_{}'.format(ne1[0], ne1[1],
                                                            ne0[0], ne0[1])
                fmap[fstr1] = 1

                fstr2 = u'NE Counts {} --> {}'.format(ne1[0], ne0[0])
                fmap[fstr2] = 1

                fstr3 = u'NE Counts __ --> {}'.format(ne0[0])
                fmap[fstr3] = 1

                fstr4 = u'NE Counts {} --> __'.format(ne1[0])
                fmap[fstr4] = 1

    def gold_transitions(self):
        """ Return the gold transitions for a problem instance.
        """
        s2e = self.sent2ents
        s2dcon = self.sent2dcon

        nsents = len(self.doc.sents)
        labels = ['START'] + ['sent-{}'.format(i) for i in range(nsents)] \
            + ['END']
        prevs = labels[0:-1]
        currents = labels[1:]

        dcon = '__START__'

        gold = []

        for i, (prev, current) in enumerate(itertools.izip(prevs, currents)):
            ents = ents.union(s2e[(i-1)]) if i > 0 else frozenset()
            dcon = s2dcon[i-1] if i > 0 and s2dcon[i-1] != '' else dcon
            gold.append(Transition((current, prev), i, ents, dcon))
        return gold

    def indices2str(self, indices):
        """
        Return the document as a string, where the sentences are
        ordered according to the index values in the list *indices*.

        indices -- a list of index values
        """
        txts = []
        wrapper = textwrap.TextWrapper(subsequent_indent=u'        ')
        for i in indices:
            txts.append(u'({})  {}'.format(i+1, unicode(self.doc[i])))
        wrapped_txt = [wrapper.fill(txt) for txt in txts]

        return u'\n'.join(wrapped_txt)

    def trans2str(self, transitions):
        """
        Return the document as a string, where the sentences are
        ordered according to the set of *transitions*.

        transitions -- A set or list of
            discourse.hypergraph.Transition objects.
        """
        ord_trans = discourse.hypergraph.recover_order(transitions)
        indices = [discourse.hypergraph.s2i(t.sentences[0])
                   for t in ord_trans
                   if t.sentences[0] != 'END']
        return self.indices2str(indices)

    def hypergraph(self):
        """
        Build a graph of this problem instance, using PyDecode.
        """
        import pydecode.chart as chart
        import discourse.hypergraph as hyper
        c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing,
                               build_hypergraph=True, strict=False)
        hypergraph = hyper.build_hypergraph(self, c).finish()
        return hypergraph


def syn_sequence(parse_tree, depth):
    """
    Given a parse tree, return the sequence of non-terminals at a
    given depth. See Louis & Nenkova, 2012 for details.
    """
    seqs = [[]]
    Q = []
    d = 0
    curr_d = d
    Q.append((parse_tree[0], d))
    while len(Q) > 0 and curr_d <= depth:
        nt, nt_d = Q.pop(0)
        if nt_d != curr_d:
            curr_d = nt_d
            seqs.append([])
        if isinstance(nt, nltk.Tree):
            seqs[nt_d].append(nt.node)
            for c in nt:
                Q.append((c, nt_d+1))

    if depth < len(seqs):
        return u' '.join(seqs[depth])
    else:
        return u''
