from itertools import izip
import os
import discourse.data as data
import cPickle
import corenlp as cnlp
import discourse.inference.perceptron as perceptron
from discourse.models.rush import RushModel


feats1 = {'first_word': True}

feats2 = {'first_word': True,
          'is_first':   True,
          'is_last':    True}

feats3 = {'syntax_lev1': True,
          'syntax_lev2': True}

feats4 = {'syntax_lev1': True,
          'syntax_lev2': True,
          'is_first':    True,
          'is_last':     True}

feats5 = {'first_word':  True,
          'syntax_lev1': True,
          'syntax_lev2': True,
          'is_first':    True,
          'is_last':     True}

feats6 = {'first_word':  True,
          'syntax_lev1': True,
          'syntax_lev2': True}
 


apwsmodels = [(feats1, 'apws_fword.p'),
              (feats2, 'apws_fword_fl.p'),
              (feats3, 'apws_synseq12.p'),
              (feats4, 'apws_synseq12_fl.p'),
              (feats5, 'apws_fword_synseq12_fl.p'),
              (feats6, 'apws_fword_synseq12.p')]

apwsmodels = [(feats6, 'apws_fword_synseq12.p')]
#feats3 = {'first_word': True,
#          'is_first': True,
#          'is_last': True,
#          'role_match': True}

#feats4 = {'first_word': True,
#          'is_first': True,
#          'is_last': True,
#          'role_match': True,
#          'discourse_new': True,
#          'use_sal_ents': True}

#feats5 = defaultdict(lambda: False)
#feats5['first_word'] = True
#feats5['is_first'] = True
#feats5['is_last'] = True
#feats5['discourse_connectives'] = True

#feats6 = defaultdict(lambda: False)
#feats6['first_word'] = True
#feats6['is_first'] = True
#feats6['is_last'] = True
#feats6['syntax_lev1'] = True
#feats6['syntax_lev2'] = True


def fit_model(dataX, dataY, history=2, features=None, doc_cutoff=15):
    ptron = perceptron.PerceptronTrainer(max_iter=10, verbose=True)
    trainX = []
    trainY = []
    for x, y in izip(dataX, dataY):
        if len(x) > doc_cutoff: continue
        trainX.append(x)
        trainY.append(y)

    ptron.dsm._use_relaxed = False
    ptron.dsm._use_gurobi = True
    ptron.dsm._debug = False
    ptron.fit(trainX, trainY)
    
    return ptron


for apwsmodel in apwsmodels:
    feats, modelname = apwsmodel
    print 'Running model {}'.format(modelname)
    print 'With features\n=============\n'
    print feats.keys()
    print
    
    doc_cutoff = 15

    train_apws = [cnlp.Document(xml) for xml in data.corenlp_apws_train()]
    test_apws = [cnlp.Document(xml) for xml in data.corenlp_apws_test()]

    trainX = [RushModel(doc, history=2, features=feats) 
              for doc in train_apws]
    gtrainY = [x.gold_transitions() for x in trainX]
    
    testX = [RushModel(doc, history=2, features=feats)
             for doc in test_apws]
    gtestY = [x.gold_transitions() for x in testX]
    
    model = fit_model(trainX, gtrainY, history=2,
                      features=feats, doc_cutoff=doc_cutoff)
    ptrainY = model.predict(trainX)
    ptestY = model.predict(testX)


    modelpath = os.path.join(data._DATA_DIR, 'models', modelname)
    with open(modelpath, 'wb') as f:    
        cPickle.dump({'model': model,
                      'trainX': trainX,
                      'gtrainY': gtrainY,
                      'ptrainY': ptrainY,
                      'testX': testX,
                      'gtestY': gtestY,
                      'ptestY': ptestY,
                      'features': feats,
                      'doc_cutoff': doc_cutoff}, 
                  f)
