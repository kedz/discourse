import argparse
import os
import corenlp
from discourse.models.rush import BigramCoherenceInstance
from math import sqrt
from collections import namedtuple
import sys
import codecs

def main():

    traindir, ofile, cfile, tgt_nclusters, min_size = parse_cmdline()
    clusters = load_clusters(traindir)

    print u'Read {} sentences from {}'.format(len(clusters), traindir)
    print u'Generating at most {} clusters'.format(tgt_nclusters)
    print u'Clusters with less than {}'.format(min_size), 
    print u'elements will be merged into one cluster.'

    
    while tgt_nclusters < len(clusters):
        per = 100.0 * (tgt_nclusters / float(len(clusters)))
        sys.stdout.write(u'Clustering {:2.3f}%\r'.format(per))
        sys.stdout.flush()
        merge_most_similar_clusters(clusters)
    
    output_clusters = threshold_clutsers(clusters, min_size)

    print u'Generated {} clusters.'.format(len(output_clusters))

    write_sequence_output(output_clusters, ofile)
    write_clusters_output(output_clusters, cfile)   
    

def threshold_clutsers(clusters, min_size):
    output_clusters = []
    misc_cluster = []
    while len(clusters) > 0:
        cluster = clusters.pop(0)
        if len(cluster['sentences']) < min_size:
            misc_cluster.extend(cluster['sentences'])
        else:
            output_clusters.append(cluster)
    output_clusters.append({'sentences': misc_cluster}) 
    return output_clusters

def write_clusters_output(clusters, cfile):

    with codecs.open(cfile, 'w', 'utf-8') as f:
        for i, cluster in enumerate(clusters, 1):
            f.write(u'Topic {:3}\n---------\n'.format(i))
            for sentence in cluster['sentences']:
                ug = filter_unigrams(sentence.corenlp_sentence)
                f.write(u' '.join(ug))
                f.write(u'\n')
            f.write(u'\n')
            f.flush()
        

def write_sequence_output(clusters, ofile):

    sent2topic = {}
    sent2tokens = {}
    docs = {}

    for i, cluster in enumerate(clusters, 1):
        if i < len(clusters):
            topic = u'tpc_{}'.format(i)
        else:
            topic = u'tpc_MISC'
        for sentence in cluster['sentences']:
            if sentence.filename not in docs:
                docs[sentence.filename] = []
            snum = sentence.corenlp_sentence.idx
            ug = filter_unigrams(sentence.corenlp_sentence)
            docs[sentence.filename].append((snum, topic, ug))
            
    with codecs.open(ofile, 'w', 'utf-8') as of:
        for filename, sents in docs.iteritems():
            ordered_sents = sorted(sents, key=lambda x: x[0])
            for sent in ordered_sents:
                topic = sent[1]
                tokens = u' '.join(sent[2])
                line = u'{} _@@_ {}\n'.format(topic, tokens)
                of.write(line)
                of.flush()
            of.write(u'\n')
            of.flush()

    

                
def assign_test_sentences(clusters, testdir):

    tfiles = [os.path.join(testdir, fname) for fname in os.listdir(testdir)]
    tdocs = [corenlp.Document(f) for f in tfiles]

    test_sents = []    
    for tdoc in tdocs:
        nsents = len(tdoc)
        for i, sent in enumerate(tdoc):
            inst = make_instance(sent, i, nsents)
            max_sim = 0
            max_cluster = None    
            for cluster in clusters:
                sim = min_cluster_sim([inst], cluster['sentences'])
                if sim > max_sim:
                    max_sim = sim
                    max_cluster = cluster
            if max_sim > 0:
                if 'test' not in max_cluster:
                    max_cluster['test'] = []
                max_cluster['test'].append(inst)             
 

def load_clusters(indir):
    files = [os.path.join(indir, fname) for fname in os.listdir(indir)]

    clusters = []
    for f in files:
        doc = corenlp.Document(f)
        nsents = len(doc)
        for i, sent in enumerate(doc):
            inst = make_instance(sent, i, nsents, f)
            clusters.append({'sentences': frozenset([inst]), 'cache': {} })
    return clusters    
 

def sort_sentences(sentences):

    sent_sims = []
    nsims = float(len(sentences) - 1)
    if nsims <= 0:
        return [(sentence, 0.0) for sentence in sentences]
    else:                      
        for sent1 in sentences:
            tot_sim = 0.0
            for sent2 in sentences:
                if sent1 != sent2:
                    tot_sim += sentence_sim(sent1, sent2)
            avg_sim = tot_sim / nsims
            sent_sims.append((sent1, avg_sim))
    sent_sims.sort(key=lambda x: x[1], reverse=True)
    return sent_sims

def merge_most_similar_clusters(clusters):

    nclusters = len(clusters)
    
    best_i = None
    best_j = None
    max_min_sim = -999.0


    for i in range(nclusters):
        for j in range(i + 1, nclusters):
            cache_sim = clusters[i]['cache'].get(clusters[j]['sentences'],
                                                 None)
            if cache_sim is not None:
                sim = cache_sim
            else:
                sim = min_cluster_sim(clusters[i]['sentences'],
                                      clusters[j]['sentences'])
                clusters[i]['cache'][clusters[j]['sentences']] = sim
                    
            if sim > max_min_sim:
                best_i = i
                best_j = j
                max_min_sim = sim

    sents_j = clusters.pop(best_j)['sentences']
    sents_i = clusters[best_i]['sentences']
    clusters[best_i]['sentences'] = sents_i.union(sents_j)

def min_cluster_sim(cluster1, cluster2):
    min_sim = 999
    for sent1 in cluster1:
        for sent2 in cluster2:
            sim = sentence_sim(sent1, sent2)
            if sim < min_sim:
                min_sim = sim
    return min_sim


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

Sentence = namedtuple('Sentence', ['string', 'unigrams', 'bigrams', 'trigrams', 'half', 'corenlp_sentence', 'filename'])

def make_instance(sent, position, doc_length, filename):
    return Sentence(unicode(sent),
                    unigrams(sent), bigrams(sent), trigrams(sent),
                    1 if position / float(doc_length) <= .5 else 2,
                    sent,
                    filename)

stopwords = set(["the", "a", "an", "'s", "had", "have", "has", "were", 
                 "will", "shall", "be", "have", "for", "from", "of", "to",
                 "in", "it", "not", "other", "at", "off", ",", ".", ";",
                 ":", "!", "?", "*", "+", "/", "(", ")", ",[", "]", "\"",
                 "'", "`", "''", "``"])

def filter_unigrams(sent):

    valid_lemmas = []
    for t in sent:
        if t.ne != 'O':
            valid_lemmas.append(t.ne)
        elif unicode(t).lower() not in stopwords:
            valid_lemmas.append(t.lem.lower())
    return valid_lemmas

def unigrams(sent):
    ugrm = set()
    for word in filter_unigrams(sent):
        ugrm.add(word)
    return frozenset(ugrm)
        
def bigrams(sent):
    bgrm = set()
    tokens = filter_unigrams(sent) 
    ntokens = len(tokens)
    
    for i in range(ntokens - 1):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        bgrm.add(u'{} {}'.format(t1, t2))
    return frozenset(bgrm)

def trigrams(sent):
    tgrm = set()
    tokens = filter_unigrams(sent)
    ntokens = len(tokens)
    
    for i in range(ntokens - 2):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        t3 = tokens[i + 2]
        tgrm.add(u'{} {} {}'.format(t1, t2, t3))
    return frozenset(tgrm)

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-train', u'--train-dir',
                        help=u'Training data directory.',
                        type=unicode, required=True)
        
    parser.add_argument(u'-of', u'--output-file', 
                        help=u'Location to write clusters.',
                        type=unicode, required=True)

    parser.add_argument(u'-cf', u'--cluster-file',
                        help=u'Outputed list of cluster assignments for debug',
                        type=unicode, required=True)

    parser.add_argument(u'-n', u'--n-clusters', 
                        help=u'Target number of clusters.',
                        type=int, default=40, required=False)

    parser.add_argument(u'-m', u'--min-size', 
                        help=u'Minimum cluster size. Output clusters '\
                             +u'below this are merged into one.',
                        type=int, default=10, required=False)
    

    args = parser.parse_args()
    traindir = args.train_dir
    ofile = args.output_file
    cfile = args.cluster_file
    tgt_nclusts = args.n_clusters
    min_size = args.min_size

    if tgt_nclusts < 2:
        import sys
        sys.stderr.write(u'Target cluster size must be at least 2.\n')
        sys.stderr.flush()
        sys.exit()    

    if not os.path.exists(traindir) or not os.path.isdir(traindir):
        import sys
        sys.stderr.write((u'{} either does not exits ' +
                          u'or is not a directory.\n').format(traindir))
        sys.stderr.flush()
        sys.exit()

    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    
    cdir = os.path.dirname(cfile)
    if cdir != '' and not os.path.exists(cdir):
        os.makedirs(cdir)

    return traindir, ofile, cfile, tgt_nclusts, min_size

if __name__ == '__main__':
    main()
