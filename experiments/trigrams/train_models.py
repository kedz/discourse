import argparse
import os
import corenlp
from discourse.models.ngram import NGramDiscourseInstance
import discourse.learners as learners
import discourse.topics as topics
import pickle


### NON TOPIC FEATURES ###

dbg_feats = {'debug': True}

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


ftr_settings = [
                (fw_feats, 'fw_feats.p'),
#                (vbz_feats, 0, 'vbz_feats.p'),
#                (vbz_posq_feats, 0, 'vbz_posq_feats.p'),
#                (fw_posq_feats, 0, 'fw_posq_feats.p'),
#                (rm_feats, 0, 'rm_feats.p'),
#                (rm_posq_feats, 0, 'rm_posq_feats.p'),
#                (sx1_feats, 0, 'sx1_feats.p'),
#                (sx1_posq_feats, 0, 'sx1_posq_feats.p'),
#                (sx2_feats, 0, 'sx2_feats.p'),
#                (sx2_posq_feats, 0, 'sx2_posq_feats.p'),
#                (sx12_feats, 0, 'sx12_feats.p'),
#                (sx12_posq_feats, 0, 'sx12_posq_feats.p'),
#                (s_feats, 0, 's_feats.p'),
#                (s_posq_feats, 0, 's_posq_feats.p'),
#                (fw_vbz_feats, 0, 'fw_vbz_feats.p'),
#                (fw_vbz_posq_feats, 0, 'fw_vbz_posq_feats.p'),
#                (fw_rm_feats, 0, 'fw_rm_feats.p'),
#                (fw_rm_posq_feats, 0, 'fw_rm_posq_feats.p'),
#                (fw_vbz_sx1_feats, 0, 'fw_vbz_sx1_feats.p'),
#                (fw_vbz_sx1_posq_feats, 0, 'fw_vbz_sx1_posq_feats.p'),
#                (fw_vbz_s_feats, 0, 'fw_vbz_s_feats.p'),
#                (fw_vbz_s_posq_feats, 0, 'fw_vbz_s_posq_feats.p')
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

dbg_settings = [(dbg_feats, 'dbg_feats.p')]


def main():

    traindir, testdir, devdir, outdir, tpc_file, opts = _parse_cmdline()

    if opts['debug']:
        print u'Mode: ###DEBUG MODE###'
        settings = dbg_settings
        opts['iters'] = 1
        train_topic_mapper = build_topic_mapper(None, 1)

    elif tpc_file is not None:

        topic_classifier = topics.load_topics(tpc_file)

        train_topic_mapper = build_topic_mapper(topic_classifier, 1)

        settings = tpc_settings
        print u'Mode: topic features'

    else:

        train_topic_mapper = build_topic_mapper(None, 1)

        settings = ftr_settings
        print u'Mode: non-topic features'

    print u'Learner:', opts['learner']
    print u'Loss-function:', opts['loss-func']
    print u'Inference:', opts['inference']
    print u'Max Iterations:', opts['iters']
    print u'Train:', traindir
    print u'Dev:', devdir
    print u'Test:', testdir
    print u'Topic files:', tpc_file

    traindocs = load_docs(traindir)

    for feats, pfile in settings:
        if tpc_file is not None:
            tpc_basename = os.path.basename(tpc_file.replace(u'.txt', u''))
            pfile = u'{}_{}'.format(tpc_basename, pfile)
        pfile = u'{}gram_{}'.format(opts['ngram'], pfile)
        pfile = u'{}_{}_{}'.format(opts['learner'], 
                                   opts['loss-func'], pfile)
         
        outfile = os.path.join(outdir, pfile)
        if os.path.exists(outfile):
            print u'{} already exists, moving on...'.format(outfile)
            continue
        
        print u'Training {}...'.format(pfile)
        trainX, gold_trainY = make_data(feats, traindocs,
                                        train_topic_mapper, opts['ngram'])
        learner = build_learner(opts)
        learner.fit(trainX, gold_trainY)

        print u'Pickling model and data...'
        with open(outfile, 'wb') as f:
            pickle.dump({'learner': learner,
                         'learner_name': opts['learner'],
                         'loss-function': opts['loss-func'],
                         'ngram': opts['ngram'],
                         'features': feats,
                         'traindir': traindir,
                         'devdir': devdir,
                         'testdir': testdir,
                         'topic_files': tpc_file,
                         },
                        f)

        print pfile, learner.learner.w
def build_learner(opts):
    use_gurobi = True if opts['inference'] == 'gurobi' else False
    return learners.Learner(use_gurobi, False, False, opts['learner'],
                            opts['loss-func'], opts['iters'])

def build_topic_mapper(topic_classifier, k):

    def tm(doc):
        mapping = {}
        nsents = len(doc)
        for i, s in enumerate(doc):
            words = topics.filter_tokens(s)
            instance = topics.make_instance(words, i, nsents)
            mapping[i] = topic_classifier(instance, k)
        return mapping

    def no_class(doc):
        return None
    if topic_classifier is not None:
        return tm    
    else:
        return no_class                
        

def make_data(feats, docs, tmapper, ngram):
    X = [NGramDiscourseInstance(doc, feats, tmapper(doc), ngram)
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

    parser.add_argument(u'-dev', u'--dev-dir',
                        help=u'Test data directory.',
                        type=unicode, required=False)
        
    parser.add_argument(u'-test', u'--test-dir',
                        help=u'Test data directory.',
                        type=unicode, required=True)

    parser.add_argument(u'-od', u'--output-dir', 
                        help=u'Location to write pickled models.',
                        type=unicode, required=True)

    parser.add_argument(u'-t', u'--topic-file', 
                        help=u'Location of topic file if any.',
                        type=unicode, required=False,
                        default=None)

    parser.add_argument(u'-l', u'--learner', 
                        help=u'Learning algorithm.',
                        type=unicode, required=False,
                        default='perceptron')

    parser.add_argument(u'-lf', u'--loss-function', 
                        help=u'Loss function.',
                        type=unicode, required=False,
                        default='01')

    parser.add_argument(u'-inf', u'--inference', 
                        help=u'LP solver.',
                        type=unicode, required=False,
                        default='gurobi')

    parser.add_argument(u'-n', u'--ngram', 
                        help=u'Ngram size of sequence model (2 or 3)',
                        type=int, required=False,
                        default=2)

    parser.add_argument(u'-dbg', u'--debug-mode', 
                        help=u'For debuging only',
                        action='store_true')

    args = parser.parse_args()
    traindir = args.train_dir
    devdir = args.dev_dir
    testdir = args.test_dir
    outdir = args.output_dir
    tpc_file = args.topic_file

    opts = {}

    opts['ngram'] = args.ngram
    if opts['ngram'] <= 1:
        import sys
        sys.stderr.write('Must have ngram size of at least 2.\n')
        sys.stderr.flush()
        sys.exit()

    opts['learner'] = args.learner
    if args.learner not in set(['perceptron', 'sg-ssvm']):
        import sys
        sys.stderr.write('{} is not a valid learner\n'.format(args.learner))
        sys.stderr.flush()
        sys.exit()
    opts['iters'] = 10 if opts['learner'] == 'perceptron' else 25

    opts['loss-func'] = args.loss_function
    valid_loss = set(['01', 'hamming-node', 'hamming-edge', 'kendalls-tau'])
    if opts['loss-func'] not in valid_loss:
        import sys
        sys.stderr.write('{} is not a valid loss function\n'.format(
            opts['loss-func']))
        sys.stderr.flush()
        sys.exit()

    opts['inference'] = args.inference
    if args.inference not in set(['gurobi', 'glpk', 'beam']):
        import sys
        sys.stderr.write('{} is not a valid inference arg\n'.format(
            args.inference))
        sys.stderr.flush()
        sys.exit()
    

    opts['debug'] = args.debug_mode


    if not os.path.exists(traindir) or not os.path.isdir(traindir):
        import sys
        sys.stderr.write((u'{} either does not exist ' +
                          u'or is not a directory.\n').format(traindir))
        sys.stderr.flush()
        sys.exit()

    if devdir is not None:
        if not os.path.exists(devdir) or not os.path.isdir(devdir):
            import sys
            sys.stderr.write((u'{} either does not exist ' +
                              u'or is not a directory.\n').format(testdir))
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
    
    return traindir, testdir, devdir, outdir, tpc_file, opts

if __name__ == '__main__':
    main()
