import argparse
import os
import corenlp
from discourse.models.rush import BigramCoherenceInstance
import discourse.inference.perceptron as perceptron
import pickle
import discourse.data as data
import discourse.topics as topics


### NON TOPIC FEATURES ###

vbz_feats = {'verbs': True}
fw_feats = {'first_word': True}
rm_feats = {'role_match': True}
sx1_feats = {'syntax_lev1': True}
sx2_feats = {'syntax_lev2': True}
sx12_feats = {'syntax_lev1': True,
              'syntax_lev2': True}
s_feats = {'sentiment': True}

fw_vbz_feats = {'verbs': True,
                'first_word': True}

fw_rm_feats = {'first_word': True,
               'role_match': True}

fw_vbz_sx1_feats = {'first_word': True,
                    'verbs': True,
                    'syntax_lev1': True}
              
fw_vbz_s_feats = {'first_word': True,
                  'verbs': True,
                  'sentiment': True}

vbz_posq_feats = {'verbs': True,
                  'relative_position_qtr': True}

fw_posq_feats = {'first_word': True,
                 'relative_position_qtr': True}
                
rm_posq_feats = {'role_match': True,
                 'relative_position_qtr': True}

sx1_posq_feats = {'syntax_lev1': True,
                  'relative_position_qtr': True}
sx2_posq_feats = {'syntax_lev2': True,
                  'relative_position_qtr': True}

sx12_posq_feats = {'syntax_lev1': True,
                   'syntax_lev2': True,
                   'relative_position_qtr': True}

s_posq_feats = {'sentiment': True,
                'relative_position_qtr': True}
                
fw_vbz_posq_feats = {'verbs': True,
                     'first_word': True,
                     'relative_position_qtr': True}

fw_rm_posq_feats = {'first_word': True,
                    'role_match': True,
                    'relative_position_qtr': True}

fw_vbz_sx1_posq_feats = {'first_word': True,
                         'verbs': True,
                         'syntax_lev1': True,
                         'relative_position_qtr': True}
              
fw_vbz_s_posq_feats = {'first_word': True,
                       'verbs': True,
                       'sentiment': True,
                       'relative_position_qtr': True}


ftr_settings = [(vbz_feats, 0, 'vbz_feats.p'),
                (vbz_posq_feats, 0, 'vbz_posq_feats.p'),
                (fw_feats, 0, 'fw_feats.p'),
                (fw_posq_feats, 0, 'fw_posq_feats.p'),
                (rm_feats, 0, 'rm_feats.p'),
                (rm_posq_feats, 0, 'rm_posq_feats.p'),
                (sx1_feats, 0, 'sx1_feats.p'),
                (sx1_posq_feats, 0, 'sx1_posq_feats.p'),
                (sx2_feats, 0, 'sx2_feats.p'),
                (sx2_posq_feats, 0, 'sx2_posq_feats.p'),
                (sx12_feats, 0, 'sx12_feats.p'),
                (sx12_posq_feats, 0, 'sx12_posq_feats.p'),
                (s_feats, 0, 's_feats.p'),
                (s_posq_feats, 0, 's_posq_feats.p'),
                (fw_vbz_feats, 0, 'fw_vbz_feats.p'),
                (fw_vbz_posq_feats, 0, 'fw_vbz_posq_feats.p'),
                (fw_rm_feats, 0, 'fw_rm_feats.p'),
                (fw_rm_posq_feats, 0, 'fw_rm_posq_feats.p'),
                (fw_vbz_sx1_feats, 0, 'fw_vbz_sx1_feats.p'),
                (fw_vbz_sx1_posq_feats, 0, 'fw_vbz_sx1_posq_feats.p'),
                (fw_vbz_s_feats, 0, 'fw_vbz_s_feats.p'),
                (fw_vbz_s_posq_feats, 0, 'fw_vbz_s_posq_feats.p')
                ]

### TOPIC FEATURES ###

tpc_f = {'topics': True,
         'topics_rewrite': True}

tpc_settings = [(dict(vbz_feats.items()+tpc_f.items()), 0,
                 'vbz_tpc_feats.p'),
                (dict(vbz_posq_feats.items() + tpc_f.items()), 0,
                 'vbz_posq_tpc_feats.p'),
                (dict(fw_feats.items() + tpc_f.items()), 0,
                 'fw_tpc_tpc_feats.p'),
                (dict(fw_posq_feats.items() + tpc_f.items()), 0,
                 'fw_posq_tpc_feats.p'),
                (dict(rm_feats.items() + tpc_f.items()), 0,
                 'rm_tpc_feats.p'),
                (dict(rm_posq_feats.items() + tpc_f.items()), 0, 
                 'rm_posq_tpc_feats.p'),
                (dict(sx1_feats.items() + tpc_f.items()), 0,
                 'sx1_tpc_feats.p'),
                (dict(sx1_posq_feats.items() + tpc_f.items()), 0,
                 'sx1_posq_tpc_feats.p'),
                (dict(sx2_feats.items() + tpc_f.items()), 0,
                 'sx2_tpc_feats.p'),
                (dict(sx2_posq_feats.items() + tpc_f.items()), 0,
                 'sx2_posq_tpc_feats.p'),
                (dict(sx12_feats.items() + tpc_f.items()), 0,
                 'sx12_tpc_feats.p'),
                (dict(sx12_posq_feats.items() + tpc_f.items()), 0,
                 'sx12_posq_tpc_feats.p'),
                (dict(s_feats.items() + tpc_f.items()), 0,
                 's_tpc_feats.p'),
                (dict(s_posq_feats.items() + tpc_f.items()), 0,
                 's_posq_tpc_feats.p'),
                (dict(fw_vbz_feats.items() + tpc_f.items()), 0,
                 'fw_vbz_tpc_feats.p'),
                (dict(fw_vbz_posq_feats.items() + tpc_f.items()), 0,
                 'fw_vbz_posq_tpc_feats.p'),
                (dict(fw_rm_feats.items() + tpc_f.items()), 0,
                 'fw_rm_tpc_feats.p'),
                (dict(fw_rm_posq_feats.items() + tpc_f.items()), 0,
                 'fw_rm_posq_tpc_feats.p'),
                (dict(fw_vbz_sx1_feats.items() + tpc_f.items()), 0,
                 'fw_vbz_sx1_tpc_feats.p'),
                (dict(fw_vbz_sx1_posq_feats.items() + tpc_f.items()), 0,
                 'fw_vbz_sx1_posq_tpc_feats.p'),
                (dict(fw_vbz_s_feats.items() + tpc_f.items()), 0,
                 'fw_vbz_s_tpc_feats.p'),
                (dict(fw_vbz_s_posq_feats.items() + tpc_f.items()), 0,
                 'fw_vbz_s_posq_tpc_feats.p')
               ]

def main():

    traindir, testdir, outdir, tpc_file = _parse_cmdline()

    if tpc_file is not None:

        topic_classifier = topics.load_topics(tpc_file)

        train_topic_mapper = build_topic_mapper(topic_classifier, 1)
        test_topic_mapper = build_topic_mapper(topic_classifier, 1)

        settings = tpc_settings
    
    else:
        train_topic_mapper = build_topic_mapper(None, 1)
        test_topic_mapper = build_topic_mapper(None, 1)

        settings = ftr_settings
       

    traindocs = load_docs(traindir)
    testdocs = load_docs(testdir)

    for feats, num_graph_ents, pfile in settings:
        if tpc_file is not None:
            pfile = u'{}_{}'.format(tpc_file.replace(u'.txt', u''), pfile)
        print u'Training {}...'.format(pfile)
        
        trainX, gold_trainY = make_data(feats, traindocs, num_graph_ents,
                                        train_topic_mapper)
        testX, gold_testY = make_data(feats, testdocs, num_graph_ents,
                                      test_topic_mapper)
        model = fit_model(trainX, gold_trainY)

        print u'Pickling model and data...'
        outfile = os.path.join(outdir, pfile)
        with open(outfile, 'wb') as f:
            pickle.dump({'model': model,
                         'features': feats,
                         'traindir': traindir,
                         'testdir': testdir,
                         'num_graph_ents': num_graph_ents,
                         'topic_file': tpc_file
                         },
                        f)

def build_topic_mapper(topic_classifier, k):

    def tm(doc):
        mapping = {}
        nsents = len(doc)
        for i, s in enumerate(doc):
            words = topics.filter_unigrams(s)
            instance = topics.make_instance(words, i, nsents)
            mapping[i] = topic_classifier(instance, k)
        return mapping

    def no_class(doc):
        return None
    if topic_classifier is not None:
        return tm    
    else:
        return no_class                
        
def fit_model(dataX, dataY):
    ptron = perceptron.PerceptronTrainer(max_iter=10, verbose=True)

    ptron.dsm._use_relaxed = False
    ptron.dsm._use_gurobi = True
    ptron.dsm._debug = False
    ptron.fit(dataX, dataY)
    
    return ptron


def make_data(feats, docs, num_graph_ents,
              tmapper):
    X = [BigramCoherenceInstance(doc, feats, num_graph_ents, tmapper(doc))
         for doc in docs]
    Y = [x.gold_transitions() for x in X]
    return X, Y

def load_docs(path):
    files = [os.path.join(path, fname) for fname in os.listdir(path)]
    docs = [corenlp.Document(f) for f in files]
    return docs    
    

def _parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-train', u'--train-dir',
                        help=u'Training data directory.',
                        type=unicode, required=True)
        
    parser.add_argument(u'-test', u'--test-dir',
                        help=u'Test data directory.',
                        type=unicode, required=True)

    parser.add_argument(u'-od', u'--output-dir', 
                        help=u'Location to write pickled models.',
                        type=unicode, required=True)

    parser.add_argument(u'-t', u'--topic_file', 
                        help=u'Location of topic file if any.',
                        type=unicode, required=False,
                        default=None)

    args = parser.parse_args()
    traindir = args.train_dir
    testdir = args.test_dir
    outdir = args.output_dir
    tpc_file = args.topic_file

    if not os.path.exists(traindir) or not os.path.isdir(traindir):
        import sys
        sys.stderr.write((u'{} either does not exist ' +
                          u'or is not a directory.\n').format(traindir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(testdir) or not os.path.isdir(testdir):
        import sys
        sys.stderr.write((u'{} either does not exist ' +
                          u'or is not a directory.\n').format(testdir))
        sys.stderr.flush()
        sys.exit()


#    if tpc_file is not None and not os.path.exists(tpc_file):
#        import sys
#        sys.stderr.write(u'{} does not exist.\n'.format(tpc_file))
#        sys.stderr.flush()
#        sys.exit()


    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    return traindir, testdir, outdir, tpc_file

if __name__ == '__main__':
    main()
