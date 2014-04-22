import discourse.data as data
import corenlp as cnlp
import discourse.evaluation as evaluation
from itertools import izip
import discourse.hypergraph as hypergraph
import textwrap

def print_model_features(feats):
    print u'Model Features:'
    print u'=' * 79
    for item in feats.items():
        print u'{:35}  |  {}'.format(item[0], item[1])
    print u'\n'


def print_model_metrics(trainY, testY):

    train_kt, train_pval = evaluation.avg_kendalls_tau(trainY)
    test_kt, test_pval = evaluation.avg_kendalls_tau(testY)
    
    train_bg = evaluation.mac_avg_bigram_acc(trainY)
    test_bg = evaluation.mac_avg_bigram_acc(testY)

    print u'                | TRAIN DATA         | TEST DATA'
    print u'{:15} | {:.3f} (pval {:.3f}) | {:.3f} (pval {:.3f})'.format(
        'Kendall\'s Tau',
        train_kt, train_pval,
        test_kt, test_pval)
    print u'{:15} | {:.3f}              | {:.3f}'.format(
        'Bigram Accuracy',
        train_bg, test_bg)
    print u'\n'


print 'MODEL NAME: '
print

test_docs = [cnlp.Document(xml)
             for xml in data.corenlp_apws_test()]

baseline_dict = data.get_apws_model('fword_synseq12.p')
model = baseline_dict['model']
feats = baseline_dict['features']
doc_cutoff = baseline_dict['doc_cutoff']

trainX = baseline_dict['trainX']
gtrainY = baseline_dict['gtrainY']
ptrainY = baseline_dict['ptrainY']

testX = baseline_dict['testX']
gtestY = baseline_dict['gtestY']
ptestY = baseline_dict['ptestY']

print_model_features(feats)
print

print u'ALL DATA PERFORMANCE'
print u'=' * 79
print_model_metrics(ptrainY, ptestY)


if doc_cutoff is not None:
    
    cutoff_trainY = [y for y in ptrainY if len(y) <= doc_cutoff]
    cutoff_testY = [y for y in ptestY if len(y) <= doc_cutoff]
    print u'CUTOFF DATA PERFORMANCE (Length <= {})'.format(doc_cutoff)
    print u'=' * 79
    print_model_metrics(cutoff_trainY, cutoff_testY)
    

for i, datum in enumerate(izip(testX, gtestY[0:20], ptestY), 1):
    print u'TEST NO: {:3}\n============\n'.format(i)
    testx, gtesty, ptesty = datum
    kt, pval = evaluation.kendalls_tau(ptesty)
    print u'Kendall\'s Tau : {:.3f} (pval {:.3f})'.format(kt, pval)
    print u'Bigram Acc.    : {:.3f}'.format(evaluation.bigram_acc(ptesty))
    print

    print u'GOLD ORDERING\n==================\n'
    print unicode(testx.trans2str(gtesty))
    print 

    for t in hypergraph.recover_order(gtesty):
        
        print u'TRANSITION: {}'.format(unicode(t))
        print u'=' * 79

        idx1 = hypergraph.s2i(t.sents[1])
        sent1 = testx[idx1] if idx1 > -1 else 'START'
        idx2 = hypergraph.s2i(t.sents[0])
        sent2 = testx[idx2] if idx2 is not None else 'END'

        print textwrap.fill(u'({:3}) {}'.format(idx1, unicode(sent1)))
        print u' |\n V'
        print textwrap.fill(u'({:3}) {}\n'.format(idx2, unicode(sent2)))
        evaluation.explain_transition(t, model, testx)
        print 

    print u'PREDICTED ORDERING\n==================\n'
    print unicode(testx.trans2str(ptesty))
    print
    
    for t in hypergraph.recover_order(ptesty):
        
        print u'TRANSITION: {}'.format(unicode(t))
        print u'=' * 79

        idx1 = hypergraph.s2i(t.sents[1])
        sent1 = testx[idx1] if idx1 > -1 else 'START'
        idx2 = hypergraph.s2i(t.sents[0])
        sent2 = testx[idx2] if idx2 is not None else 'END'

        print textwrap.fill(u'({:3}) {}'.format(idx1, unicode(sent1)))
        print u' |\n V'
        print textwrap.fill(u'({:3}) {}\n'.format(idx2, unicode(sent2)))
        evaluation.explain_transition(t, model, testx)
        print 

    print
    print
#evaluation.explain_predicted(test_docs, ptest, model, feats)
