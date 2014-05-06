import codecs
from collections import namedtuple, defaultdict
from math import sqrt
import os

Sentence = namedtuple('Sentence', ['unigrams', 'bigrams', 'trigrams', 'half'])

def load_clusters(fname):
    data = []

    sequences = []
    with codecs.open(fname, 'r', 'utf-8') as f:
        sequence = []
        for line in f:
            line = line.strip()
            if line == u'':
                if len(sequence) > 0:
                    sequences.append(sequence)
                sequence = []
            else:                
                topic, text = line.split(u'\t')
                topic = topic.strip()
                text = text.strip().split(u' ')
                #print topic, text
                sequence.append((topic, text))
    for sequence in sequences:
        slen = len(sequence)
        for i, (y, x) in enumerate(sequence, 1):
            instance = make_instance(x, i, slen)
            data.append((y, instance))
    
    def classify(instance, k=10):
        sims = [(sentence_sim(instance, x), y) for y, x in data]
        sims.sort(key=lambda x: x[0], reverse=True)
        counts = defaultdict(int)
        for score, label in sims[0:k]:
            counts[label] += 1
        votes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return votes[0][0]
    return classify

def load_multigran_clusters(dirname):
    cluster_classifiers = {}
    for fname in os.listdir(dirname):
        fpath = os.path.join(dirname, fname)
        cluster_classifiers[fname] = load_clusters(fpath)
    def classify(instance, k):
        classes = {}
        for gran, g_classify in cluster_classifiers.items():
            classes[gran] = g_classify(instance, k)
        return classes
    return classify

def sentence_sim(sent1, sent2):
    ugrm_sim = cosine_sim(sent1.unigrams, sent2.unigrams)
    bgrm_sim = cosine_sim(sent1.bigrams, sent2.bigrams)
    tgrm_sim = cosine_sim(sent1.trigrams, sent2.trigrams)
    position_sim = 0.0 if sent1.half == sent2.half else -1.0
    return (.2 * ugrm_sim) + (.3 * bgrm_sim) + (.5 * tgrm_sim) + position_sim

def cosine_sim(ngrams1, ngrams2):
    intrsec = len(ngrams1.intersection(ngrams2))
    if intrsec == 0:
        return 0
    return float(intrsec) / (sqrt(len(ngrams1)) * sqrt(len(ngrams2)))

def make_instance(words, position, doc_length):
    return Sentence(frozenset(words), bigrams(words), trigrams(words),
                    1 if position / float(doc_length) <= .5 else 2)

stopwords = set(["the", "a", "an", "'s", "had", "have", "has", "were", 
                 "will", "shall", "be", "have", "for", "from", "of", "to",
                 "in", "it", "not", "other", "at", "off", ",", ".", ";",
                 ":", "!", "?", "*", "+", "/", "(", ")", ",[", "]", "\"",
                 "'", "`", "''", "``"])

def filter_tokens(cnlp_sentence):

    valid_lemmas = []
    for t in cnlp_sentence:
        if t.ne != 'O':
            valid_lemmas.append(t.ne)
        elif unicode(t).lower() not in stopwords:
            valid_lemmas.append(t.lem.lower())
    return valid_lemmas

def unigrams(tokens):
    return frozenset(tokens)
        
def bigrams(tokens):
    bgrm = set()
    #tokens = filter_unigrams(sent) 
    ntokens = len(tokens)
    
    for i in range(ntokens - 1):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        bgrm.add(u'{} {}'.format(t1, t2))
    return frozenset(bgrm)

def trigrams(tokens):
    tgrm = set()
#    tokens = filter_unigrams(sent)
    ntokens = len(tokens)
    
    for i in range(ntokens - 2):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        t3 = tokens[i + 2]
        tgrm.add(u'{} {} {}'.format(t1, t2, t3))
    return frozenset(tgrm)

