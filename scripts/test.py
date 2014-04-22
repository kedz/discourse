from bitstring import BitArray, Bits
from collections import defaultdict

# These are the default features that are active
active_features = {}

# Decorate features if they are active for first and last
# sequence transitions.
active_features['is_first'] = True
active_features['is_last'] = True

class RushModel:

    def __init__(self, doc, history=2, features=active_features,
                 num_salient_ents=4):
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

        #s2b, i2e = self._extract_salient_ents(doc, num_salient_ents)
        self.sent2ent_bstr = {}
        self.idx2ent = Bits(1)

        #self.sal_ents = set(i2e.values())

        #self._num_salient_ents = len(i2e) if len(i2e) > 0 else 1

        #use_expl_disc = features.get('discourse_connectives', False)
        #self.sent2trans = self._extract_expl_disc(doc, use_expl_disc)
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
        s2ents = defaultdict(set)

        # Map entities to the number of occurrences of that entity in the
        # document.
        ent_counts = defaultdict(int)

        # Count noun phrase heads that occur in subject or object
        # dependencies.
        if nsalient > 0:
            for i, sent in enumerate(doc):

                # Iterate through all dependencies in a sentence sent, puting
                # salient tokens in a set -- this is done to avoice counting
                # entities twice if they occurred in multiple relevant
                # dependencies.
                deps = set()
                for rel in sent.deps:
                    if 'sub' in rel.type or 'obj' in rel.type:
                        if rel.dep.pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                            ent = rel.dep.lem.lower().strip()
                            deps.add((rel.dep.idx, ent))
                for ent in deps:
                    s2ents[i].add(ent[1])
                    ent_counts[ent[1]] += 1

        # Select at most *nsalient* salient entities,
        # starting from most frequent.
        ent_list = sorted(ent_counts.items(), key=lambda x: x[1], reverse=True)
        sal_ent_list = [ent[0] for ent in ent_list[0:nsalient] if ent[1] > 1]

        # Update nsalient as it might be less than specified if there are
        # fewer salient entities.
        nsalient = len(sal_ent_list)

        # If we found no salient entities, or we set this value to 0
        # initially, return the default dicts.
        if nsalient == 0:
            no_ents = Bits(1)
            s2b = {}
            i2e = {}
            return s2b, i2e

        # Create a unique 'one-hot' bitstring for each salient entity and map
        # the 'hot' digit to the corresponding entity.
        i2e = {}
        sal_ents = {}
        for i, ent in enumerate(sal_ent_list):
            bstr = BitArray(nsalient)
            bstr[i] = 1
            sal_ents[ent] = bstr
            i2e[i] = ent

        # Create a bit string for each sentence where each digit in the
        # bitstring indicates the presence (or absence) of a salient entity
        # in the sentence. In other words, each sentence's bitstring is an
        # ORing of the bitstrings of the salient entities in that sentence.
        # Map sentence indices to the sentence bitstring.
        s2b = {}
        for s in range(nsents):
            bits = Bits(nsalient)
            for e in sal_ents.keys():
                if e in s2ents[s]:
                    bits = bits | sal_ents[e]
            s2b[s] = bits

        return s2b, i2e

