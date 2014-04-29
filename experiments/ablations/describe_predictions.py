import argparse
import os
import corenlp
from discourse.models.rush import BigramCoherenceInstance
import discourse.inference.perceptron as perceptron
import pickle
import discourse.evaluation as evaluation
import pandas as pd
from itertools import izip
import discourse.hypergraph as hypergraph
import textwrap
import codecs
import discourse.data as data
import discourse.topics as topics

def make_data(feats, docs, num_graph_ents, tmapper):
    X = [BigramCoherenceInstance(doc, feats, num_graph_ents, tmapper(doc))
         for doc in docs]
    Y = [x.gold_transitions() for x in X]
    return X, Y

def load_docs(path):
    files = [os.path.join(path, fname) for fname in os.listdir(path)]
    docs = [corenlp.Document(f) for f in files]
    return docs    

def build_topic_mapper(topic_classifier, k):
    def tm(doc):
        mapping = {}
        nsents = len(doc)
        for i, s in enumerate(doc):
            words = topics.filter_tokens(s)
            instance = topics.make_instance(words, i, nsents)
            mapping[i] = topic_classifier(instance, k)
        return mapping
    return tm    

def main():

    pfile, ofile = _parse_cmdline()

    with open(pfile, 'rb') as f:

        d = pickle.load(f)

    model = d['model']
    feats = d['features']
    traindir = d['traindir']
    testdir = d['testdir']
    num_graph_ents = d['num_graph_ents']
    tpc_file = d['topic_file']

    if tpc_file is not None: 
        topic_classifier = topics.load_topics(tpc_file)
        train_topic_mapper = build_topic_mapper(topic_classifier, 1)
        test_topic_mapper = build_topic_mapper(topic_classifier, 1)
    else:
        train_topic_mapper = lambda x: None
        test_topic_mapper = lambda x: None

    traindocs = load_docs(traindir)
    testdocs = load_docs(testdir)
    
    
    trainX, gtrainY = make_data(feats, traindocs, num_graph_ents, 
                                train_topic_mapper)
    ptrainY = model.predict(trainX)

    testX, gtestY = make_data(feats, testdocs, num_graph_ents,
                              test_topic_mapper)
    ptestY = model.predict(testX)


#    trainX = d['trainX']
#    gtrainY = d['gtrainY']
#    ptrainY = d['ptrainY']

#    testX = d['testX']
#    gtestY = d['gtestY']
#    ptestY = d['ptestY']


    with codecs.open(ofile, 'w', 'utf-8') as out:
        out.write(unicode(model_metrics(ptrainY, ptestY)))
        out.write(u'\n')
        out.flush()

        for i, datum in enumerate(izip(trainX, gtrainY, ptrainY), 1):
            trainx, gtrainy, ptrainy = datum
            kt, pval = evaluation.kendalls_tau(ptrainy)


            out.write(u'TRAIN NO: {:3}\n============\n'.format(i))
            out.write(u'K\'s Tau: {:2.3f} (pval {:2.3f})\n\n'.format(kt, pval))
            
            out.write(u'PREDICTED ORDERING\n==================\n')
            out.write(trainx.trans2str(ptrainy))
            out.write(u'\n\n')
            

            for t in hypergraph.recover_order(ptrainy):

                out.write(u'TRANSITION: {}\n'.format(unicode(t)))
                out.write(u'=' * 79)
                out.write(u'\n\n')

                idx1 = hypergraph.s2i(t.sentences[1])
                sent1 = trainx.doc[idx1] if idx1 > -1 else u'START'
                idx2 = hypergraph.s2i(t.sentences[0])
                sent2 = trainx.doc[idx2] if idx2 is not None else u'END'

                out.write(textwrap.fill(u'({:3}) {}\n'.format(idx1,
                                                              unicode(sent1))))
                out.write(u'\n |\n V\n')
                out.write(textwrap.fill(u'({:3}) {}\n'.format(idx2,
                                                              unicode(sent2))))
                out.write(u'\n\n')
                out.write(unicode(evaluation.explain_transition(t, model, 
                                                                trainx)))
                out.write(u'\n\n')

        out.write(u'\n\n')
        out.flush()

        for i, datum in enumerate(izip(testX, gtestY, ptestY), 1):
            testx, gtesty, ptesty = datum
            kt, pval = evaluation.kendalls_tau(ptesty)

            out.write(u'TEST NO: {:3}\n============\n'.format(i))
            out.write(u'Kendall\'s Tau: {} (pval: {})\n\n'.format(kt, pval))
            out.write(u'PREDICTED ORDERING\n==================\n')
            out.write(testx.trans2str(ptesty))
            out.write(u'\n\n')

            for t in hypergraph.recover_order(ptesty):

                out.write(u'TRANSITION: {}\n'.format(unicode(t)))
                out.write(u'=' * 79)
                out.write(u'\n\n')

                idx1 = hypergraph.s2i(t.sentences[1])
                sent1 = testx.doc[idx1] if idx1 > -1 else u'START'
                idx2 = hypergraph.s2i(t.sentences[0])
                sent2 = testx.doc[idx2] if idx2 is not None else u'END'

                out.write(textwrap.fill(u'({:3}) {}\n'.format(idx1,
                                                              unicode(sent1))))
                out.write(u'\n |\n V\n')
                out.write(textwrap.fill(u'({:3}) {}\n'.format(idx2,
                                                              unicode(sent2))))
                
                out.write(u'\n\n')
                out.write(unicode(evaluation.explain_transition(t, model, testx)))
                out.write(u'\n\n')

        out.write(u'\n\n')
        out.flush()



def model_metrics(ptrainY, ptestY):
    train_kt, train_pval = evaluation.avg_kendalls_tau(ptrainY)
    test_kt, test_pval = evaluation.avg_kendalls_tau(ptestY)

    train_bg = evaluation.mac_avg_bigram_acc(ptrainY)
    test_bg = evaluation.mac_avg_bigram_acc(ptestY)

    train_oso_acc = evaluation.avg_oso_acc(ptrainY)
    test_oso_acc = evaluation.avg_oso_acc(ptestY)
    
    data = [[train_kt, train_pval, train_bg, train_oso_acc],
            [test_kt, test_pval, test_bg, test_oso_acc]]


    return pd.DataFrame(data,
                        index=(u'train', u'test'),
                        columns=(u'Kendalls Tau', u'KT pvalue',
                                 u'Bigram Acc.', u'OSO Acc.'))


def _parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-p', u'--pickle-file',
                        help=u'The pickled model and data file.',
                        type=unicode, required=True)
    parser.add_argument(u'-of', u'--output-file',
                        help=u'File to write descriptions to.',
                        type=unicode, required=True)

    args = parser.parse_args()
    pfile = args.pickle_file
    ofile = args.output_file

    if not os.path.exists(pfile) or os.path.isdir(pfile):
        import sys
        sys.stderr.write((u'{} either does not exits ' +
                          u'or is a directory.\n').format(pfile))
        sys.stderr.flush()
        sys.exit()


    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)

    return pfile, ofile

if __name__ == '__main__':
    main()
