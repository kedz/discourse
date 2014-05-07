cimport discourse.lattice as lattice
import discourse.lattice as lattice
import textwrap
import itertools
from collections import defaultdict
import nltk

cdef lattice.Transition Transition

cdef class NGramDiscourseInstance:
    """An ngram discourse coherence sequence model instance for
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

    entity_counts : dict (string -> int)
        A dict mapping document noun strings to occurrence counts.

    _f_cache : dict (Transition -> dict (string -> int))
        This cache stores graph edge (i.e. transitions) feature maps
        so that transition feautres need not be calculated more than
        once.

    """


    def __init__(self, doc, features=None, topic_map=None,
                 ngram=3, use_cache=False):
        self.doc = doc
        self.nsents = len(doc.sents)
        self.active_feat = features
#active_features if features is None else features
        self.entity_counts = self._build_entity_counts(doc)
        self._f_cache = {}
        self._topic_map = topic_map
        self.ngram = ngram
        self._use_cache = use_cache
        self._verb_cache = [None for s in doc.sents]
        self._syntax_cache = [[None, None, None] for s in doc.sents]
        self._fw_cache = [None for s in doc.sents]

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
                if t.pos in [u'NN', u'NNS', u'NNP', u'NNPS']:
                    lem = t.lem.lower()
                    if lem not in counts:
                        counts[lem] = 1
                    else:
                        counts[lem] += 1
        return counts

    def hypergraph(self):
        """
        Build a graph of this problem instance, using PyDecode.
        """
        import pydecode.chart as chart
        c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing,
                               build_hypergraph=True, strict=False)
        hypergraph = lattice.build_ngram_lattice(self, c).finish()
        return hypergraph


    cdef feature_map(self, lattice.Transition transition):
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

        if transition in self._f_cache and self._use_cache is True:
            return self._f_cache[transition]

        else:

            # Create an empty feature map for this transition.
            fmap = {}

                ### Call each feature function if active. ###
            if self.active_feat.get('role_match', False):
                self._f_role_match(fmap, transition)
#
#            if self.active_feat.get('discourse_new', False):
#                self._f_discourse_new(fmap, transition)
#
#            if self.active_feat.get('discourse_connectives', False):
#                self._f_discourse_connectives(fmap, transition)
            if self.active_feat.get('first_word', False):
                self._f_first_word(fmap, transition)
#
            if self.active_feat.get('syntax_lev1', False):
                self._f_syntax_lev(fmap, transition, 1)
            if self.active_feat.get('syntax_lev2', False):
                self._f_syntax_lev(fmap, transition, 2)
#
#            if self.active_feat.get('personal_pronoun_res', False):
#                self._f_personal_pronoun_res(fmap, transition)
#
#            if self.active_feat.get('ne_types', False):
#                self._f_ne_types(fmap, transition)
#
            if self.active_feat.get('verbs', False):
                self._f_verbs(fmap, transition)
            
#
#            
#            if self.active_feat.get('relative_position', False):
#                self._f_relative_position(fmap, transition)
#
#            
#            if self.active_feat.get('sentiment', False):
#                self._f_sentiment(fmap, transition)
#
            if self.active_feat.get('topics_rewrite_20', False):
                self._f_topics_rewrite(fmap, transition, '20')
            if self.active_feat.get('topics_rewrite_40', False):
                self._f_topics_rewrite(fmap, transition, '40')
            if self.active_feat.get('topics_rewrite_60', False):
                self._f_topics_rewrite(fmap, transition, '60')
            if self.active_feat.get('topics_rewrite_80', False):
                self._f_topics_rewrite(fmap, transition, '80')
            if self.active_feat.get('topics_rewrite_100', False):
                self._f_topics_rewrite(fmap, transition, '100')
                
            if self.active_feat.get(u'debug', False):
                self._f_debug(fmap, transition)

            if self.active_feat.get('topics', False):
                self._f_topics(fmap, transition)
            if self.active_feat.get('relative_position_qtr', False):
                self._f_relative_position_qtr(fmap, transition)
            if self._use_cache is True:
                self._f_cache[transition] = fmap
            return fmap

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

        nsents = len(self.doc.sents)
        feature_combs = [self.extract_fws(idx) 
                         for idx in transition.idxs[::-1]]
        null_p = tuple([u'__']) * len(feature_combs)

        for p in itertools.product(*feature_combs):
            p = tuple(p) 
            if p == null_p:
                continue
            
            # Mark the feature
            fstr = u'First Word: {}'.format(u' --> '.join(p))
            fmap[fstr] = 1

    cdef extract_fws(self, int idx):
        if idx == -1:
            return ['START']
        elif idx == self.nsents:
            return ['END']
        else:
            if self._fw_cache[idx] is None:
                t = self.doc.sents[idx].tokens[0]
                if t.ne == 'O':
                    self._fw_cache[idx] = [t.lem.lower(), u'__']
                else:
                    self._fw_cache[idx] = [t.lem.lower(), t.ne, u'__']
            return self._fw_cache[idx] 

    cdef _f_verbs(self, dict fmap, lattice.Transition transition):
        
        nsents = len(self.doc)
                
        feature_combs = [self.extract_verbs(idx) 
                         for idx in transition.idxs[::-1]]
        null_p = tuple([u'__']) * len(feature_combs)

        for p in itertools.product(*feature_combs):
            p = tuple(p) 
            if p == null_p:
                continue
            
            # Mark the feature
            fstr = u'Verbs: {}'.format(u' --> '.join(p))
            fmap[fstr] = 1

    cdef extract_verbs(self, int idx):
 
        if idx == -1:
            return ['START']
        elif idx == self.nsents:
            return ['END']
        else:
            if self._verb_cache[idx] is None:
                s = self.doc.sents[idx]
                verbs = [t.lem.lower() for t in s.tokens if 'VB' in t.pos]
                
                self._verb_cache[idx] = set(verbs).union(set(['__']))
            return self._verb_cache[idx]
   
    cdef _f_relative_position_qtr(self, dict fmap,
                                  lattice.Transition transition):
        nsents = len(self.doc)
        per = transition.position / float(nsents)
        if per <= .25:
            qtr = u'1Q'
        elif per <= .5:
            qtr = u'2Q'
        elif per <= .75:
            qtr = u'3Q'
        else:
            qtr = u'4Q'
        new_feats = []
        for feat, value in fmap.items():
            new_feats.append((u'({} QTR) {}'.format(qtr, feat), value))  
        for feat, value in new_feats:
            fmap[feat] = value        


    def _f_role_match(self, fmap, transition):

        sent_ent_roles = []
        for label in transition.labels[::-1]:
            idx = lattice.s2i(label, end=self.nsents)
            if idx == -1:         
                sent_ent_roles.append(defaultdict(lambda: ['START']))
            elif idx == self.nsents:
                sent_ent_roles.append(defaultdict(lambda: ['END']))
            else:
                s = self.doc.sents[idx]
                sermap = defaultdict(lambda: ['X', '__'])
                for ent_role in self._entity_roles(s):
                    if self.entity_counts[ent_role[0]] > 1:
                        sermap[ent_role[0]] = [ent_role[1], '__'] 
                sent_ent_roles.append(sermap)
                
        null_p = tuple([u'__']) * self.ngram
        for ent, cnt in self.entity_counts.iteritems():
                       
            feature_combs = [sermap[ent] for sermap in sent_ent_roles]

            for p in itertools.product(*feature_combs):
                p = tuple(p) 
                if p == null_p:
                    continue
                
                # Mark the feature
                fstr = u'Role Match: {}'.format(u' --> '.join(p))

                if cnt > 1:
                    fstr = u'Salient {}'.format(fstr)
                score = fmap.get(fstr, 0)                
                fmap[fstr] = score + 1


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
        subj_types = set(['nsubj', 'csubj' 'agent'])
        obj_types = set(['csubjpass', 'dobj', 'nsubjpass'])
        ntags = set(['NN', 'NNS', 'NP', 'NPS'])
        dgraph = s.dep_graph()
        for t in s.tokens:
            if 'VB' in t.pos:
                if t.lem == 'be':
                    s_ents = s_ents.union(self.find_copula_roles(dgraph, t))
                else:
                    for rel in dgraph.govs[t]:
                        if rel.dep.pos in ntags: 
                            role = None
                            if rel.type in subj_types:
                                role = 'SUBJECT'
                            elif rel.type in obj_types:
                                role = 'OBJECT'
                            else:
                                role = 'CLAUSE'    
                            s_ents.add(tuple([rel.dep.lem.lower(), 
                                              role]))
        return s_ents

    def find_copula_roles(self, dgraph, token):
        ntags = set(['NN', 'NNS', 'NP', 'NPS'])
        ent_roles = set()
        cop_obj = None
        for rel in dgraph.deps[token]:
            if rel.type == 'cop' and rel.gov.pos in ntags:
                cop_obj = rel.gov
                ent_roles.add(tuple([cop_obj.lem.lower(), 'OBJECT']))
                break
        if cop_obj is not None:
            for rel in dgraph.govs[cop_obj]:
                if rel.type == 'nsubj' and rel.dep.pos in ntags:
                    t = rel.dep
                    ent_roles.add(tuple([t.lem.lower(), 'SUBJECT']))
        return ent_roles 
#        roots = [(rel.dep, rel.type) for rel in dgraph.type['root']]
#        roots.extend([(rel.dep, rel.type) for rel in dgraph.type['ccomp']])
#        for token, dtype in roots:
#            if token.pos in ['NN', 'NNS', 'NP', 'NPS']:
#                s_ents.add((token, dtype))
#            for rel in dgraph.govs[token]:
#                if rel.type in dtypes or 'prep' in rel.type:
#                    if rel.dep.pos in ['NN', 'NNS', 'NP', 'NPS']:
#                        s_ents.add((rel.dep, rel.type))
#
#        for rel in s.deps:
#            if rel.type != 'nn' and rel.dep.pos in ['NN', 'NNS', 'NP', 'NPS']:
#                if (rel.dep, rel.type) not in s_ents:
#                    s_ents.add((rel.dep, 'other'))
#        return s_ents


    cdef _f_syntax_lev(self, dict fmap, lattice.Transition transition, 
                       int depth):
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
        productions = []
        for idx in transition.idxs[::-1]:
            if idx == -1:         
                productions.append(['START'])
            elif idx == self.nsents:
                productions.append(['END'])
            else:
                if self._syntax_cache[idx][depth] is None:
                    tree = self.doc.sents[idx].parse
                    prod = self.syn_sequence(tree, depth)
                    self._syntax_cache[idx][depth] = [prod, '__']
                    
                productions.append(self._syntax_cache[idx][depth])    

        null_p = tuple([u'__']) * self.ngram

        for p in itertools.product(*productions):
            p = tuple(p) 
            if p == null_p:
                continue
            
            # Mark the feature
            fstr = u'Syntax ({}): {}'.format(depth, u' --> '.join(p))
            fmap[fstr] = 1



    def syn_sequence(self, parse_tree, depth):
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

    cdef _f_topics(self, dict fmap, lattice.Transition transition):

        # Get topic granularities        
        grans = self._topic_map[0].keys()
        null_p = tuple([u'__']) * self.ngram
        
        for gran in grans:
            topics = []    
            for idx in transition.idxs[::-1]:
                if idx == -1:
                    topics.append(['START'])
                elif idx == self.nsents:
                    topics.append(['END'])
                else:
                    topics.append([self._topic_map[idx][gran], '__'])
                       
            for p in itertools.product(*topics):
                p = tuple(p) 
                if p == null_p:
                    continue
                
                # Mark the feature
                fstr = u'Topics ({}): {}'.format(gran, u' --> '.join(p))
                fmap[fstr] = 1

    cdef _f_topics_rewrite(self, dict fmap, lattice.Transition transition,
                           object granularity):

        idx = transition.idxs[0]
        if idx == -1 or idx == self.nsents:
            return
        topic = self._topic_map[idx][granularity]

        for feat, val in fmap.items():
            fstr = u'(TPC (Gran {}) {}) {}'.format(granularity, topic, feat)
            fmap[fstr] = val
 
    def _f_debug(self, fmap, transition):
        s2i = lattice.s2i
        nsents = len(self.doc.sents)
        correct = True
        for i, label in enumerate(transition.labels[:-1]):

            head = s2i(label, end=nsents)
            tail = s2i(transition.labels[i + 1], end=nsents)
            if tail == -1 and head == -1:
                continue
            if tail + 1 != head :
                correct = False
                break
        if transition.position != s2i(transition.labels[0], end=nsents):
            correct = False
        
        if correct:
            fmap['DEBUG'] = 1

    def gold_transitions(self):
        """ Return the gold transitions for a problem instance.
        """
        
        ngrams = self.ngram
        nsents = len(self.doc.sents)
        labels = ['START'] * (ngrams - 1) + ['s-{}'.format(i) 
                                             for i in range(nsents)]

        labels += ['END']
        nlabels = len(labels)
        gold = []

        pos = 0
        
        for i in range(ngrams, nlabels + 1):
            gt = lattice.Transition(tuple(labels[i - ngrams:i])[::-1], 
                                       pos)
            gold.append(gt)
            pos += 1


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
            txts.append(u'({})  {}'.format(i, unicode(self.doc[i])))
        wrapped_txt = [wrapper.fill(txt) for txt in txts]

        return u'\n'.join(wrapped_txt)

    def trans2str(self, transitions):
        """
        Return the document as a string, where the sentences are
        ordered according to the set of *transitions*.

        transitions -- A set or list of
            discourse.hypergraph.Transition objects.
        """
        ord_trans = lattice.recover_order(transitions)
        indices = [lattice.s2i(t.labels[0])
                   for t in ord_trans
                   if t.labels[0] != 'END']
        return self.indices2str(indices)

    def hypergraph(self):
        """
        Build a graph of this problem instance, using PyDecode.
        """
        import pydecode.chart as chart
        c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing,
                               build_hypergraph=True, strict=False)
        hypergraph = lattice.build_ngram_lattice(self, c).finish()
        return hypergraph
