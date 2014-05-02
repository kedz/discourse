import discourse.lattice as lattice
import textwrap
import itertools

class NGramDiscourseInstance:
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

    def __init__(self, doc, features=None, topic_map=None, ngram=3):
        self.doc = doc
        self.active_feat = active_features if features is None else features
        self.entity_counts = self._build_entity_counts(doc)
        self._f_cache = {}
        self._topic_map = topic_map
        self.ngram = ngram

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
        #if transition in self._f_cache:
        #    fmap = self._f_cache[transition]
        #    return fmap

        #else:

            # Create an empty feature map for this transition.
        fmap = {}

            ### Call each feature function if active. ###
#            if self.active_feat.get('role_match', False):
#                self._f_role_match(fmap, transition)
#
#            if self.active_feat.get('discourse_new', False):
#                self._f_discourse_new(fmap, transition)
#
#            if self.active_feat.get('discourse_connectives', False):
#                self._f_discourse_connectives(fmap, transition)
        if self.active_feat.get('first_word', False):
            self._f_first_word(fmap, transition)
#
#            if self.active_feat.get('syntax_lev1', False):
#                self._f_syntax_lev(fmap, transition, 1)
#            if self.active_feat.get('syntax_lev2', False):
#                self._f_syntax_lev(fmap, transition, 2)
#
#            if self.active_feat.get('personal_pronoun_res', False):
#                self._f_personal_pronoun_res(fmap, transition)
#
#            if self.active_feat.get('ne_types', False):
#                self._f_ne_types(fmap, transition)
#
#            if self.active_feat.get('verbs', False):
#                self._f_verbs(fmap, transition)
#            
#            if self.active_feat.get('topics', False):
#                self._f_topics(fmap, transition)
#
#            
#            if self.active_feat.get('relative_position', False):
#                self._f_relative_position(fmap, transition)
#
#            if self.active_feat.get('relative_position_qtr', False):
#                self._f_relative_position_qtr(fmap, transition)
#            
#            if self.active_feat.get('sentiment', False):
#                self._f_sentiment(fmap, transition)
#
#            if self.active_feat.get('topics_rewrite', False):
#                self._f_topics_rewrite(fmap, transition)
            
        if self.active_feat.get(u'debug', False):
            self._f_debug(fmap, transition)

            #self._f_cache[transition] = fmap
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
        def extract_fws(label):
            idx = lattice.s2i(label, end=nsents)
            if idx == -1:
                return ['START']
            elif idx == nsents:
                return ['END']
            else:
                t = self.doc.sents[idx].tokens[0]
                if t.ne == 'O':
                    return [t.lem.lower(), u'__']
                else:
                    return [t.lem.lower(), t.ne, u'__']

        feature_combs = [extract_fws(label) 
                         for label in transition.labels[::-1]]
        null_p = tuple([u'__']) * len(feature_combs)

        for p in itertools.product(*feature_combs):
            p = tuple(p) 
            if p == null_p:
                continue
            
            # Mark the feature
            fstr = u'First Word: {}'.format(u' --> '.join(p))
            fmap[fstr] = 1
                

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
