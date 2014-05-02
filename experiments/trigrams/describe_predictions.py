import argparse
import os
import corenlp
from discourse.models.ngram import NGramDiscourseInstance
import discourse.evaluation as evaluation
import discourse.lattice as lattice
import discourse.topics as topics
import pickle
import pandas as pd
from itertools import izip
import textwrap
import codecs

def make_data(feats, docs, tmapper, ngram):
    X = [NGramDiscourseInstance(doc, feats, tmapper(doc), ngram)
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

    if os.path.exists(ofile):
        print u'{} already exists, stopping...'.format(ofile)
        import sys
        sys.exit()

    with open(pfile, 'rb') as f:

        d = pickle.load(f)

    learner = d['learner']
    feats = d['features']
    traindir = d['traindir']
    devdir = d['devdir']
    testdir = d['testdir']
    tpc_file = d['topic_files']
    ngram = d['ngram']

    print learner.learner.w
    if tpc_file is not None: 
        topic_classifier = topics.load_topics(tpc_file)
        train_topic_mapper = build_topic_mapper(topic_classifier, 1)
        test_topic_mapper = build_topic_mapper(topic_classifier, 1)
    else:
        train_topic_mapper = lambda x: None
        test_topic_mapper = lambda x: None

    traindocs = load_docs(traindir)
    testdocs = load_docs(testdir)
    devdocs = load_docs(devdir)
    
    trainX, gtrainY = make_data(feats, traindocs,
                                train_topic_mapper, ngram)
    ptrainY = learner.predict(trainX)

    devX, gdevY = make_data(feats, devdocs,
                                train_topic_mapper, ngram)
    pdevY = learner.predict(devX)


    testX, gtestY = make_data(feats, testdocs,
                              test_topic_mapper, ngram)
    ptestY = learner.predict(testX)


    with codecs.open(ofile, 'w', 'utf-8') as out:

        out.write(unicode(model_metrics(ptrainY, pdevY, ptestY)))
        out.write(u'\n')
        out.flush()

        write_explanation(trainX, gtrainY, ptrainY, out, 'TRAIN', learner)
        write_explanation(devX, gdevY, pdevY, out, 'DEV', learner)

        write_explanation(testX, gtestY, ptestY, out, 'TEST', learner)


def write_explanation(dataX, gdataY, pdataY, out, name, learner):

    for i, datum in enumerate(izip(dataX, gdataY, pdataY), 1):
        datax, gdatay, pdatay = datum
        kt, pval = evaluation.kendalls_tau(pdatay)

        out.write(u'{} NO: {:3}\n============\n'.format(name, i))
        out.write(u'K\'s Tau: {:2.3f} (pval {:2.3f})\n\n'.format(kt, pval))
        
        out.write(u'PREDICTED ORDERING\n==================\n')
        out.write(datax.trans2str(pdatay))
        out.write(u'\n\n')
        

        for t in lattice.recover_order(pdatay):

            out.write(u'TRANSITION: {}\n'.format(unicode(t)))
            out.write(u'=' * 79)
            out.write(u'\n\n')

            texts = []
            for label in t.labels[::-1]:
                idx = lattice.s2i(label, end=u'END')
                if idx == -1:
                    sent = u'START'
                elif idx == u'END':
                    sent = u'END'
                else:
                    sent = unicode(datax.doc.sents[idx])
                texts.append(textwrap.fill(u'({:3}) {}\n'.format(idx,
                                                                 sent)))
            out.write(u'\n |\n V\n'.join(texts))
            out.write(u'\n') 
            out.write(u'\n\n')
            out.write(unicode(evaluation.explain_transition(t, learner, 
                                                            datax)))
            out.write(u'\n\n')

    out.write(u'\n\n')
    out.flush()

def model_metrics(ptrainY, pdevY, ptestY):
    train_kt, train_pval = evaluation.avg_kendalls_tau(ptrainY)
    dev_kt, dev_pval = evaluation.avg_kendalls_tau(pdevY)
    test_kt, test_pval = evaluation.avg_kendalls_tau(ptestY)

    train_bg = evaluation.mac_avg_bigram_acc(ptrainY)
    dev_bg = evaluation.mac_avg_bigram_acc(pdevY)
    test_bg = evaluation.mac_avg_bigram_acc(ptestY)

    train_oso_acc = evaluation.avg_oso_acc(ptrainY)
    dev_oso_acc = evaluation.avg_oso_acc(pdevY)
    test_oso_acc = evaluation.avg_oso_acc(ptestY)
    
    data = [[train_kt, train_pval, train_bg, train_oso_acc],
            [dev_kt, dev_pval, dev_bg, dev_oso_acc],
            [test_kt, test_pval, test_bg, test_oso_acc]]


    return pd.DataFrame(data,
                        index=(u'train', u'dev', u'test'),
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