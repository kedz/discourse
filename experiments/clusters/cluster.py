import argparse
import os
import corenlp
import discourse.topics as topics
from collections import namedtuple
import sys
import codecs


def main():
    """Cluster sentences in training documents. Generates a cluster
    assignment file and a cluster-sentence sequence file."""

    traindir, sfile, cfile, tgt_nclusters, min_size = parse_cmdline()
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

    write_sequence_output(output_clusters, sfile)
    write_clusters_output(output_clusters, cfile)   
    

def threshold_clutsers(clusters, min_size):
    """Combine all clusters with less than min_size items into one
    cluster."""

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
    """Write cluster sentences for each cluster -- this is mainly
    for debugging."""

    with codecs.open(cfile, 'w', 'utf-8') as f:
        for i, cluster in enumerate(clusters, 1):
            f.write(u'Topic {:3}\n---------\n'.format(i))
            for sentence in cluster['sentences']:
                ug = topics.filter_tokens(sentence.corenlp_sentence)
                f.write(u' '.join(ug))
                f.write(u'\n')
            f.write(u'\n')
            f.flush()

def write_sequence_output(clusters, ofile):
    """Write training documents with cluster labels."""

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
            ug = topics.filter_tokens(sentence.corenlp_sentence)
            docs[sentence.filename].append((snum, topic, ug))
            
    with codecs.open(ofile, 'w', 'utf-8') as of:
        for filename, sents in docs.iteritems():
            ordered_sents = sorted(sents, key=lambda x: x[0])
            for sent in ordered_sents:
                topic = sent[1]
                if len(sent[2]) == 0:
                    continue
                tokens = u' '.join(sent[2])
                line = u'{}\t{}\n'.format(topic, tokens)
                of.write(line)
                of.flush()
            of.write(u'\n')
            of.flush()
                
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
 

def merge_most_similar_clusters(clusters):
    """Merge most similar pair of clusters using complete-link
    clustering."""

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
            sim = topics.sentence_sim(sent1, sent2)
            if sim < min_sim:
                min_sim = sim
    return min_sim


Sentence = namedtuple('Sentence', ['string', 'unigrams', 'bigrams', 'trigrams', 'half', 'corenlp_sentence', 'filename'])

def make_instance(sent, position, doc_length, filename):
    tokens = topics.filter_tokens(sent)
    return Sentence(unicode(sent),
                    topics.unigrams(tokens),
                    topics.bigrams(tokens),
                    topics.trigrams(tokens),
                    1 if position / float(doc_length) <= .5 else 2,
                    sent,
                    filename)


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
