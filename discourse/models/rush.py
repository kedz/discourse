import corenlp as cnlp
from collections import deque, defaultdict
from discourse.hypergraph import s2i
import discourse.gazetteers as gazetteers
import itertools
from bitstring import BitArray, Bits
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
# 'use_det' marks the determiner used for the entity (e.g. a subj --> the obj).
# 'use_sal_ents' marks whether or not the entity in question is salient or not,
# e.g. SALIENT a subj --> the obj.
active_features['role_match'] = True
active_features['use_det'] = True
active_features['use_sal_ents'] = True

# Mark discourse connective transitions.
active_features['discourse_connectives'] = True
# Mark the first occurence of a salient entity.
active_features['discourse_new'] = True

# Mark Tree Syntax sequences of depths 1&2
active_features['syntax_lev1'] = True
active_features['syntax_lev2'] = True

# Mark first word to first word transitions.
active_features['first_word'] = True

# Explicit Discourse gazetteers for identifying expl disc tokens in a sentence.
expl_disc = gazetteers.DiscourseConnectives()


class RushModel:

    def __init__(self, doc, history=2, features=active_features,
                 num_salient_ents=4):
        """
        A discourse model for predicting coherent sentence orderings.

        doc -- A corenlp.Document object.

        history -- This model takes into acount a sentence window of
            size history (currently only 2 is supported).

        features -- a dict mapping feature names to boolean values, with True
            indicating that a particular feature is being used.

        num_salient_entities -- The number of salient entities to track in the
            model. This number should probably not exceed 4 or the graph will
            blow up. A salient entity is any noun phrase head that occurs more
            than once. If this value is set to n, the model tracks the top n
            entities by frequency, breaking ties arbitrarily. This value is
            a maximum -- it is possible that a document has fewer than n
            salient entities.
        """
        self.doc = doc
        self.history = history
        self._active_feat = features

        # Sentence to sentence transition features are cached, since they are
        # need repeatedly in decoding. This dict holds a Transitions feature
        # map.
        self._f_cache = {}

        # If we are not using the salient ents feature, there is no reason to
        # store information about salient entities, so this value is set to 0.
        if not features.get('use_sal_ents', False):
            num_salient_ents = 0

        # Extract two dicts related to salient entities:
        # s2b -- a sentence index to salient entity bitstring map, indicating
        #        the salient entities present in the sentence.
        # i2e -- a bitstring digit to salient entity map, indicating which
        #        bitstring digit corresponds to which salient entity.
        # NOTE: If num_salient_ents is 0 or use_sal_ents is False, these dicts
        # will be empty.
        sal_ents, s2e = self._extract_salient_ents(doc, num_salient_ents)
        self.sent2sal_ents = s2e
        self.sal_ents = sal_ents

        # Update the model with the actual number of salient entities
        # found -- this could be less than specified in the num_salient_ents
        # argument. If there are no salient entities, set the value to 1,
        # indicating that the salient entity bitstring for each sentence, will
        # be a 1 bit string with value 0.
        self._num_salient_ents = len(sal_ents)

        # If we are using discourse connective features, make a dict mapping
        # sentence indices to the explicit discourse connective that occurs in
        # that sentence or the empty string if there is none.
        # When this feature is false, the map always returns the empty string.
        use_expl_disc = features.get('discourse_connectives', False)
        self.sent2trans = self._extract_expl_disc(doc, use_expl_disc)

    def _extract_expl_disc(self, doc, use_expl_disc):
        """
        Return a dict mapping sentence indices to the explicit discourse
        connective the respective sentence containts, if any. When no
        connective is present or use_expl_disc is False, return the empty
        string.

        Only checks the first four tokens in the sentence for a discourse
        connective as these are more likely to have transition words that hold
        across sentence, as opposed to within a sentence.

        doc -- A corenlp.Document object.

        use_expl_disc -- a boolean flag indicating whether or not this
            feature is active.
        """

        if not use_expl_disc:
            s2t = {}
            for i, s in enumerate(doc):
                s2t[i] = ''
            return s2t

        s2t = {}
        for i, s in enumerate(doc):
            preamble = u' '.join(unicode(token) for token in s[0:4])
            t = expl_disc.contains_connective(preamble)
            if t is not None:
                s2t[i] = t
            else:
                s2t[i] = ''

        return s2t

    def _extract_salient_ents(self, doc, nsalient):
        """
        Returns two dicts:
        s2b -  a sentence index to salient entity bitstring map, indicating
               the salient entities present in the sentence.
        i2e -  a bitstring digit to salient entity map, indicating which
               bitstring digit corresponds to which salient entity.
        NOTE: If nsalient == 0, s2b will map any index to Bits(1),
        i.e., a single bit with value 0, and i2e will be an empty dict.

        doc -- a corenlp.Document object.

        nsalient -- The maximum number of salient entities to extract.
        """

        # The number of sentences in the document.
        nsents = len(doc)

        # Map sentences to a set of entities (Nouns) in that sentence.
        s2ents = {}

        # Map entities to the number of occurrences of that entity in the
        # document.
        ent_counts = {}

        # Count noun phrase heads that occur in subject or object
        # dependencies.
        if nsalient > 0:
            for i, sent in enumerate(doc):
                for rel in sent.deps:
                    if 'sub' in rel.type or 'obj' in rel.type:
                        if rel.dep.pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                            ent = rel.dep.lem.lower().strip()
                            if ent not in ent_counts:
                                ent_counts[ent] = 1
                            else:
                                ent_counts[ent] += 1
        

        # Select at most *nsalient* salient entities,
        # starting from most frequent.
        ent_list = sorted(ent_counts.items(), key=lambda x: x[1], reverse=True)
        sal_ent_list = [ent[0] for ent in ent_list if ent[1] > 1][0:nsalient]
        sal_ent_set = set(sal_ent_list)

        for i, sent in enumerate(doc):
            ents = set()
            for rel in sent.deps:
                if 'sub' in rel.type or 'obj' in rel.type:
                    if rel.dep.pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                        ent = rel.dep.lem.lower().strip()
                        if ent in sal_ent_set:
                             ents.add(ent)
            s2ents[i] = frozenset(ents)

        return sal_ent_set, s2ents



    def __len__(self):
        return len(self.doc)

    def __getitem__(self, index):
        return self.doc[index]

    def __iter__(self):
        return iter(self.doc)

    def gold_str(self):
        """
        Returns the correct sentence ordering as a string.
        """
        strs = [u'({})  {}'.format(i, unicode(s))
                for (i, s) in enumerate(self, 1)]
        return u'\n'.join(strs)

    def feature_map(self, transition):
        """
        Return the feature map for a transition. Feature maps are
        cached.

        transition -- the graph transition, for which this function
            extracts features.
        """

        nsents = len(self)

        # Get the sentence indices for this transition.
        idxs = [s2i(s, end=nsents) for s in transition]
        key = tuple(idxs)

        # Check the cache if we have already created this transition's
        # feature map, otherwise create a new one.
        if key not in self._f_cache:

            # Holds feature info for visualition/debug
            self._t = []

            # Create an empty feature map for this transition.
            fmap = {}

            ### Call each feature function if active. ###

            if self._active_feat.get('role_match', False):
                self._f_role_match(idxs, fmap,
                                   transition,
                                   use_det=self._active_feat.get('use_det',
                                                                 False))

            if self._active_feat.get('discourse_new', False):
                self._f_discourse_new(idxs, fmap, transition,
                                      is_first=self._active_feat['is_first'])
            if self._active_feat.get('discourse_connectives', False):
                self._f_discourse_connectives(idxs, fmap, transition)
            if self._active_feat.get('first_word', False):
                self._f_first_word(idxs, fmap, transition)

            if self._active_feat.get('syntax_lev1', False):
                self._f_syntax_lev(idxs, fmap, transition, 1)
            if self._active_feat.get('syntax_lev2', False):
                self._f_syntax_lev(idxs, fmap, transition, 2)

            if self._active_feat.get('is_first', False):
                self._f_is_first(idxs, fmap, transition)
            if self._active_feat.get('is_last', False):
                self._f_is_last(idxs, fmap, transition)

            self._f_cache[key] = fmap
            return fmap

        else:

            return self._f_cache[key]

    def _f_syntax_lev(self, idxs, fmap, transition, depth):
        """
        Marks the *depth* non-terminal sequence transition in the
        feature map. E.g. S , NP VP . ---> NP VP .

        idxs -- the list of sentence indices in this transition in
            order of head sentence to tail sentence.

        fmap -- a dict with feature values for this transition.

        transition -- the graph transition, for which this function
            extracts features.

        depth -- the depth of the sequence to extract from the parse
            tree.
        """

        if transition.sents[1] == 'START':
            seq1 = 'START'
        else:
            seq1 = syn_sequence(self[idxs[1]].parse, depth)
        if transition.sents[0] == 'END':
            seq0 = 'END'
        else:
            seq0 = syn_sequence(self[idxs[0]].parse, depth)
        fmap['{} -sl{}-> {}'.format(seq1, depth, seq0)] = 1
        fmap['__ -sl{}-> {}'.format(depth, seq0)] = 1
        fmap['{} -sl{}-> __'.format(seq1, depth)] = 1


    def _f_discourse_new(self, idxs, fmap, transition,
                         is_first=False):
        """
        Marks feature map if the tail sentence contains the first
        occurrence of a salient entity, that is, a discourse new
        entity.

        idxs -- the list of sentence indices in this transition in
            order of head sentence to tail sentence.

        fmap -- a dict with feature values for this transition.

        transition -- the graph transition, for which this function
            extracts features.

        is_first -- an optional flag indicating whether or not to use
            a distinguished feature for discourse new entities that
            occur in the first position sentence.
        """

        # If we are in the last position, it is impossible to have a discourse
        # new entity, since by definition they occur at least more than once.
        # Simply return.
        if idxs[0] == len(self):
            return

        s2b = self.sent2ent_bstr
        i2e = self.idx2ent

        curr_bstr = s2b[idxs[0]]
        prev_bstr = transition.prev_ents

        disc_new_ents = set()

        # Populate disc_new_ents with discourse new entities, if any.
        for i in range(self._num_salient_ents):
            if prev_bstr[i] == 0 and curr_bstr[i] == 1:
                disc_new_ents.add(i2e[i])

        # Mark features in fmap, and provide token level information
        # for feature visualization/debug.
        for t in self[idxs[0]]:
            if t.lem.lower() in disc_new_ents:
                if transition.pos == 0 and is_first:
                    fstr = 'FIRST SENT - DISC NEW'
                else:
                    fstr = 'DISC NEW'
                # Mark feature in the feature map.
                fmap[fstr] = 1
                # Mark feature token coordinates for visualization.
                self._t.append([fstr, ((idxs[0], t.idx),
                                       (idxs[0], t.idx))])

    def _entity_roles(self, s):
        """
        Returns a set of entity, role tuples for a sentence s.

        s -- A corenlp.Sentence object.
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

    def _f_role_match(self, idxs, fmap,
                      transition, use_det=False):
        """
        Marks feature map if noun phrase heads match across sentence.
        The feature takes the form of the dependency relation for the
        entity in each sentence, and optionally the determiner used
        and whether or not the entitiy in question is a salient
        entity. E.g. 'SALIENT a nsubj --> the dobj'.

        idxs -- the list of sentence indices in this transition in
            order of head sentence to tail sentence.

        fmap -- a dict with feature values for this transition.

        transition -- the graph transition, for which this function
            extracts features.

        use_det -- an optional flag indicating whether or not to mark
            the determiner of the entity.
        """

        # If this is the first transition, or the last transition, there is
        # nothing to mark. Simply return.
        if idxs[0] == len(self):
            return
        if idxs[-1] == -1:
            return

        # Get entity, role tuples for each sentence.
        s0_ents = self._entity_roles(self[idxs[0]])
        s1_ents = self._entity_roles(self[idxs[1]])

        # Find matching entities across sentences, and mark them as features.
        for ent0 in s0_ents:
            lem0 = ent0[0].lem.lower()
            for ent1 in s1_ents:
                lem1 = ent1[0].lem.lower()
                if lem0 == lem1 or lem0 in lem1:

                    if lem0 in self.sal_ents or lem1 in self.sal_ents:
                        is_sal = 'SALIENT'
                    else:
                        is_sal = 'X'

                    det0 = 'X'
                    det1 = 'X'
                    if use_det:
                        for rel in dgraph0.govs[ent0[0]]:
                            if rel.type == 'det':
                                det0 = rel.dep.lem.lower()
                        for rel in dgraph1.govs[ent1[0]]:
                            if rel.type == 'det':
                                det1 = rel.dep.lem.lower()

                    fstr = u'{} {} {} --> {} {}'.format(is_sal,
                                                        det1,
                                                        ent1[1],
                                                        det0,
                                                        ent0[1])

                    # Mark token level information for visualization/debug.
                    self._t.append([fstr,
                                    ((idxs[1], ent1[0].idx),
                                     (idxs[0], ent0[0].idx))])
                    # Mark the feature map.
                    fmap[fstr] = 1

    def _f_discourse_connectives(self, idxs, fmap, transition):
        if idxs[0] == len(self):
            return
        s2t = self.sent2trans
        d0 = s2t[idxs[0]]
        fstr = 'DISC {} --> {}'.format(transition.conn, d0)
        fmap[fstr] = 1
        self._t.append([fstr, ((idxs[0], 0),
                               (idxs[0], 0))])

    def _f_first_word(self, idxs, fmap, transition):
        if transition.sents[1] == 'START':
            word1 = 'START'
        else:
            word1 = self[idxs[1]][0].lem.lower()
 
        if transition.sents[0] == 'END':
            word0 = 'END'
        else:
            word0 = self[idxs[0]][0].lem.lower()
        fstr1 = u'First Word Trans: {} --> {}'.format(unicode(word1), unicode(word0))
        fmap[fstr1] = 1
        fstr2 = u'First Word Trans: __ --> {}'.format(unicode(word0))
        fmap[fstr2] = 1
        fstr3 = u'First Word Trans: {} --> __'.format(unicode(word1))
        fmap[fstr3] = 1

#        if idxs[-1] == -1:
#            fstr = 'First Word: {}'.format(self[idxs[0]][0].lem.lower())
#            fmap[fstr] = 1
#            self._t.append([fstr, ((idxs[0], 0), (idxs[0], 0))])
#        elif idxs[0] < len(self):
#            w1 = self[idxs[0]][0].lem.lower()
#            w2 = self[idxs[1]][0].lem.lower()
#            fstr = '{} --> {}'.format(w2, w1)
#            fmap[fstr] = 1
#            self._t.append([fstr, ((idxs[0], 0), (idxs[1], 0))])
#        else:
#            fstr = 'Last Word: {}'.format(self[idxs[1]][0].lem.lower())
#            fmap[fstr] = 1
#            self._t.append([fstr, ((idxs[1], 0), (idxs[1], 0))])

    def _f_is_first(self, idxs, fmap, trans):
        if trans.pos == 1:
           
            for k in fmap.keys():
                val = fmap[k]
                del fmap[k]
                fmap[k+' <is_first>'] = val
            for feat in self._t:
                feat[0] = '<is_first> ' + feat[0]

    def _f_is_last(self, idxs, fmap, trans):
        if trans.pos == len(self) - 1:
            for k in fmap.keys():
                val = fmap[k]
                del fmap[k]
                fmap[k+' <is_last>'] = val
            for feat in self._t:
                feat[0] = '<is_last> ' + feat[0]

    def gold_transitions(self):
        """
        Return the gold transitions for a problem instance.
        This only works for history size 2.
        """
        s2e = self.sent2sal_ents
        s2t = self.sent2trans

        nsents = len(self)
        labels = ['START'] + ['sent-{}'.format(i) for i in range(nsents)] \
            + ['END']
        prevs = labels[0:-1]
        currents = labels[1:]

        conn = ''

        gold = []

        for i, (prev, current) in enumerate(itertools.izip(prevs, currents)):
            ents = ents.union(s2e[(i-1)]) if i > 0 else frozenset()
            conn = s2t[i-1] if i > 0 and s2t[i-1] != '' else conn
            gold.append(Transition((current, prev), i, ents, conn))
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
            txts.append(u'({})  {}'.format(i+1, unicode(self[i])))
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
        indices = [discourse.hypergraph.s2i(t.sents[0])
                   for t in ord_trans
                   if t.sents[0] != 'END']
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
