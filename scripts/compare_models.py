import discourse.data as data
import corenlp as cnlp

test_docs = [cnlp.Document(xml) 
             for xml in data.corenlp_apws_test()]

baseline_dict = data.get_apws_model('test1.p')
base_model = baseline_dict['model']
base_ptest = baseline_dict['ptest']
base_ptrain = baseline_dict['ptrain']
base_feats = baseline_dict['features']

new_dict = data.get_apws_model('test2.p')
new_model = new_dict['model']
new_ptest = new_dict['ptest']
new_ptrain = new_dict['ptrain']
new_feats = new_dict['features']

import discourse
import discourse.evaluation as evaluation
evaluation.eval_against_baseline(test_docs,
                                 base_ptest,
                                 new_ptest,
                                 base_model,
                                 new_model,
                                 base_feats,
                                 new_feats,
                                 base_ptrain,
                                 new_ptrain
                                 )
