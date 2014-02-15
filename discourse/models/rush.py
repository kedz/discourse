import corenlp_xml as cnlp
from collections import deque, defaultdict
from discourse.hypergraph import s2i
import discourse.gazetteers as gazetteers
import itertools
from scipy.spatial.distance import cosine
#import gensim

active_features = defaultdict(lambda: False)
#active_features['is_first'] = True
#active_features['is_last'] = True

active_features['role_match'] = True
active_features['use_det'] = True
active_features['use_subs'] = True

active_features['pron_res'] = True
#active_features['ne_tags'] = True
#active_features['num_caps'] = True
#active_features['quotes'] = True
active_features['first_word'] = True
active_features['root_pos'] = True
#active_features['person'] = True
active_features['root_verb'] = True
active_features['sent_length'] = True
active_features['lsa'] = False

female_fnames = gazetteers.FemaleNames()
male_fnames = gazetteers.MaleNames()

class RushModel:

    def __init__(self, doc, history=2, features=active_features):
        self.doc = doc
        self.history = history
        self._f_cache = {}
        self._active_feat = features

    def __len__(self):
        return len(self.doc)
    
    def __getitem__(self, index):
        return self.doc[index]

    def __iter__(self):
        return iter(self.doc)

    def set_lsi(self, w2i, lsi):
        self._w2i = w2i
        self._lsi = lsi

    def gold_str(self):
        strs = ['({})  {}'.format(i, s) for (i, s) in enumerate(self, 1)]
        return '\n'.join(strs)
    
    def feature_map(self, transition):  
        
        nsents = len(self)
        idxs = [s2i(s, end=nsents) for s in transition if s != ()] 
        key = tuple(idxs)
        if key not in self._f_cache:
            self._t = []
        
            fmap = {}

            if self._active_feat['person']:
                self._f_person(idxs, fmap)
            if self._active_feat['role_match']:
                self._f_mark_role_matches(idxs, fmap, use_det=self._active_feat['use_det'], use_subs=self._active_feat['use_subs'])
            if self._active_feat['pron_res']:
                self._f_pron_res(idxs,fmap) 
            if self._active_feat['ne_tags']:
                self._f_ne_tags(idxs,fmap)
            if self._active_feat['num_caps']:
                self._f_num_caps(idxs,fmap)
            if self._active_feat['quotes']:
                self._f_quotes(idxs,fmap)
            if self._active_feat['first_word']:
                self._f_first_word(idxs,fmap)
            if self._active_feat['root_pos']:
                self._f_root_pos(idxs,fmap)
            if self._active_feat['root_verb']:
                self._f_root_verb(idxs,fmap)
            if self._active_feat['sent_length']:
                self._f_sent_length(idxs,fmap)
            if self._active_feat['lsa']:
                self._f_lsa(idxs, fmap)
            if self._active_feat['is_first']:
                self._f_is_first(idxs, fmap)
            if self._active_feat['is_last']:
                self._f_is_last(idxs, fmap)
 
            self._f_cache[key] = fmap
            return fmap
        
        else:
            
            return self._f_cache[key]
        #indices = [int(i[5:]) - 1 for i in reversed(transition.sents) if 'Sent_' in i]
        
        #return {'first_ne:{}'.format(self._f_first_ne_tag(indices)): 1}

    #def _det_role(self, idxs, fmap):


    def _f_lsa(self, idxs, fmap):

        fstr = ''
        if idxs[0] < len(self) and idxs[-1] > -1:
            bows = []
            for idx in idxs:
                counts = defaultdict(float)
                for t in self[idx]:
                    if t.pos not in ['.', ',', '\'', ';', ':', '`', '"', '\'\'', '``', '-LRB-', '-RRB-', 'POS', 'CD', 'CC']:
                        lem = t.lem.lower()
                        if lem in self._w2i:
                            counts[self._w2i[lem]] += 1 
                bows.append(counts.items())
            for i, bow in enumerate(bows[:-1]):
                vec1 = gensim.matutils.corpus2csc([lsi[bow]]).transpose().todense()
                vec2 = gensim.matutils.corpus2csc([lsi[bows[i+1]]]).transpose().todense()
                fmap['s_{}_cos_s_{}'.format(i, (i+1))] = cosine(vec1, vec2)[0][0]


    def _f_person(self, idxs, fmap):
        if idxs[0] < len(self) and idxs[1] > -1:
            shehe = False
            person = False
            for t in self[idxs[0]]:
                if t.ne == 'PERSON':
                    person = True
            for t in self[idxs[1]]:
                if t.lem.lower() in ['he', 'she']:
                    shehe = True
            fmap['person in 0:{} and he/she:{}'.format(person,shehe)] = 1
                                                        
 
    def _f_root_pos(self, idxs, fmap):
       
        fstr = ''
        root_positions = []


        if idxs[0] < len(self) and idxs[-1] > -1:
            for i, idx in enumerate(idxs):
                #print idx
                dgraph = self[idx].get_dependency_graph()
                #for rel in dgraph:
                #    print rel
                if dgraph.root:
                    rel = dgraph.root
                    root_positions.append((idx, rel.dep_idx))
                    fstr += '-s{}_root_pos_{}'.format(i, rel.dep.pos)
        if fstr != '':
            fmap[fstr] = 1
            self._t.append([fstr, root_positions])
    
    def _f_root_verb(self, idxs, fmap):
       
        fstr = ''
        root_positions = []


        if idxs[0] < len(self) and idxs[-1] > -1:
            for i, idx in enumerate(idxs):
                #print idx
                dgraph = self[idx].get_dependency_graph()
                #for rel in dgraph:
                #    print rel
                if dgraph.root:
                    rel = dgraph.root
                    root_positions.append((idx, rel.dep_idx))
                    fstr += '-s{}_{}'.format(i, rel.dep.lem.lower())
        if fstr != '':
            fmap[fstr] = 1
            self._t.append([fstr, root_positions])
          
          
    def _f_sent_length(self, idxs, fmap):
        fstr = ''
        #root_positions = []

        if idxs[0] < len(self) and idxs[-1] > -1:
            for i, idx in enumerate(idxs):
                fstr = 's{}_{}-'.format(i, (len(self[idx].tokens))/5) + fstr
                
        if fstr != '':
            fmap[fstr] = 1
            #self._t.append([fstr, root_positions])
          
                     
    def _f_first_word(self, idxs, fmap):
        fstr = ''
        fw_positions = []
        if idxs[-1] > -1 and idxs[0] < len(self):
            for i, idx in enumerate(idxs):
                first = self[idx][0]
                fw_positions.append((idx, 0)) 
                #if first.ne != 'O':
                fstr = '{}.{}-'.format(i, first.lem.lower(), first.ne[0:3]) + fstr
        if fstr != '':
            self._t.append([fstr, fw_positions])
            fmap[fstr] = 1

    def _f_quotes(self, idxs, fmap):
        fstr = ''
        for i, idx in enumerate(idxs):
            if idx > -1 and idx < len(self):
                has_q = False
                for t in self[idx]:
                    if t.word in ['\'\'','"','``']:
                        has_q = True
                        break
                fstr += '-{}_{}'.format(i, has_q)
        if fstr.count('False') < 2:
            fmap[fstr] = 1

    def _f_num_caps(self, idxs, fmap):
        fstr = ''
        for i, idx in enumerate(idxs):
            if idx > -1 and idx < len(self):
                cnt = 0
                for t in self[idx]:
                    if t.word.isupper():
                        cnt += 1
                if cnt > 5:
                    cnt = 'lots'
                fstr += '-s'+str(i)+'_caps_c_'+str(cnt)        
        if fstr.count('_c_0') < 2:
            fmap[fstr] = 1

    def _f_ne_tags(self, idxs, fmap):
        flists = []
        
        for i, idx in enumerate(idxs):
            if idx > -1 and idx < len(self):
                feats = []
                orgcnt = 0
                percnt = 0
                loccnt = 0

                for t in self[idx]:
                    if t.ne == 'PERSON':
                        percnt += 1
                    if t.ne == 'ORGANIZATION':
                        orgcnt += 1
                    if t.ne == 'LOCATION':
                        loccnt += 1
                if loccnt > 5:
                    loccnt = 'lots'
                feats.append('-s'+str(i)+"_LOCS_c"+str(loccnt))
                #locf += '-s'+str(i)+"_LOCS_"+str(loccnt)
                if orgcnt > 5:
                    orgcnt = 'lots'
                feats.append('-s'+str(i)+"_ORGS_c"+str(orgcnt))
                #orgf += '-s'+str(i)+"_ORGS_"+str(orgcnt)
                if percnt > 5:
                    percnt = 'lots'
                feats.append('-s'+str(i)+"_PERS_c"+str(percnt))
                #perf += '-s'+str(i)+"_PERS_"+str(percnt)
                flists.append(feats)
        for f in itertools.product(*flists):
            fstr = ''.join(f)
            if fstr.count('_c0') < 2:
                fmap[fstr] = 1

        
        #fmap[perf] = 1
        #fmap[orgf] = 1

    def _f_pron_res(self, idxs, fmap):
        
        global female_fnames 
        global male_fnames


        plural_pronouns = ['we', 'us', 'our', 'ours', 'your', 'yours',
                           'they', 'them', 'their', 'theirs', 
                           'ourselves', 'yourselves']


        if idxs[0] < len(self) and idxs[1] > -1:
            prns = [] 
            pprns = []
            
                              
            for idx, t in enumerate(self[idxs[0]]):
                if 'PRP' == t.pos:
                    if t.lem.lower() in plural_pronouns:
                        pprns.append((t.lem.lower(), (idxs[0], idx))) 
                    else:
                        prns.append((t.lem.lower(), (idxs[0], idx))) 
                     
            #prns = set(prns)
            fpeople = []
            mpeople = []
            unknownpeople = []
            orgs = []
            for rel in self[idxs[1]].get_dependency_graph():
                if rel.dep.ne == 'PERSON' and rel.type not in ['nn', 'conj_and', 'conj_or']:
                    #if rel.dep.lem.lower() in female_fnames:
                    #    fpeople.append((rel.dep, (idxs[1], rel.dep_idx)))
                    #elif rel.dep.lem.lower() in male_fnames:
                    #    mpeople.append((rel.dep, (idxs[1], rel.dep_idx)))
                    #else:
                    unknownpeople.append((rel, (idxs[1], rel.dep_idx)))
                if rel.dep.ne == 'ORGANIZATION' and rel.type not in ['nn', 'conj_and', 'conj_or']:
                    orgs.append((rel, (idxs[1], rel.dep_idx)))

            #fpeople = [(t.lem.lower(), (idxs[1], idx)) for idx, t in enumerate(self[idxs[1]]) if t.ne == 'PERSON' and t.lem.lower() in female_fnames]
            #mpeople = [(t.lem.lower(), (idxs[1], idx)) for idx, t in enumerate(self[idxs[1]]) if t.ne == 'PERSON' and t.lem.lower() in male_fnames]
            #unknownpeople = [(t.lem.lower(), (idxs[1], idx)) for idx, t in enumerate(self[idxs[1]]) 
            #                 if t.ne == 'PERSON'
            #                 and t.lem.lower() not in male_fnames
            #                 and t.lem.lower() not in female_fnames]
            #orgs = False
            #for t in self[idxs[1]]:
            #    if t.ne == 'ORGANIZATION':
            #        orgs = True
            #        break
                            
            for t in prns:
                token, loc = t
                if token not in ['it', 'its']:
                #if token in ['she'] and len(fpeople) > 0:
                #    for f in fpeople:
                #        ftoken, floc = f    
                #        self._t.append(["pron_female_resolution", [loc, floc]])
                #        fmap["pron_female_resolution"] = 1    
                #elif token in ['he'] and len(mpeople) > 0:
                #    for m in mpeople:
                #        mtoken, mloc = m
                #        self._t.append(["pron_male_resolution", [loc, mloc]])
                #        fmap["pron_male_resolution"] = 1
                #else:
                    for p in unknownpeople:
                        if p[0].dep.pos in ['NNP', 'NN']:
                            prel, ploc = p
                            self._t.append(["{}X{}_resolution".format(token, prel.type), [loc, ploc]])
                            fmap["{}X{}_resolution".format(token, prel.type)] = 1

                else:
                    if len(orgs) > 0:
                        for p in orgs:
                            #if p[0].dep.pos in ['NNP', 'NN']:
                            prel, ploc = p
                            self._t.append(["{}X{}_resolution".format(token, prel.type), [loc, ploc]])
                            fmap["{}X{}_resolution".format(token, prel.type)] = 1
                    else:
                        for rel in self[idxs[1]].get_dependency_graph():
                            if rel.dep.ne == 'O' and rel.type not in ['nn', 'conj_and', 'conj_or'] and rel.dep.pos in ['NN']:
                                

                                self._t.append(["{}X{}_resolution".format(token, rel.type), [loc, (idxs[1], rel.dep_idx)]])
                                fmap["{}X{}_resolution".format(token, rel.type)] = 1



            for t in pprns:
                token, loc = t
                for p in unknownpeople: 
                    if p[0].dep.pos in ['NNPS', 'NNS']:
                        prel, ploc = p
                        self._t.append(["{}X{}_resolution".format(token, prel.type), [loc, ploc]])
                        fmap["{}X{}_resolution".format(token, prel.type)] = 1



                #if orgs:
                #    fmap[t+"_org"] = 1
                
    def _f_is_first(self, idxs, fmap):
        if idxs[-1] == -1:
            for k in fmap.keys():
                val = fmap[k]
                del fmap[k]
                fmap[k+'is_first'] = val
            #fmap['is_first'] = 1
            for feat in self._t:
                feat[0] = 'is_first ' + feat[0]

    def _f_is_last(self, idxs, fmap):
        if idxs[0] == len(self):
            for k in fmap.keys():
                val = fmap[k]
                del fmap[k]
                fmap[k+'is_last'] = val
            for feat in self._t:
                feat[0] = 'is_last ' + feat[0]

            #fmap['is_last'] = 1

    def _f_mark_role_matches(self, idxs, fmap, use_det=True, use_subs=True):
        start = 0
        end = len(idxs) - 1
        if idxs[0] == len(self):    
            return 
        if idxs[-1] == -1:
            return 
       
       
        dgraph = self[idxs[-1]].get_dependency_graph()
        is_noun = lambda rel: rel.dep.pos in ['NN', 'NNS', 'NNP', 'NNPS'] and ('subj' in rel.type or 'obj' in rel.type or 'prep' in rel.type or 'agent' in rel.type)
               
        for rel in dgraph.filter_iterator(is_noun):
            #print rel.dep, rel.type
            num_mods = len(rel.dep.deps)
            if use_det: 
                det = ''
                for rel2 in rel.dep.deps:
                    if rel2.type == 'det':
                        det = rel2.dep.lem    
            else:
                det = 'x'
            features = [('role0:{} det:{}'.format(rel.type, det), [(idxs[-1], rel.dep_idx)])]
            #feature_index 
            matches = lambda other_rel: other_rel.dep.lem.lower() in rel.dep.lem.lower() and other_rel.dep.pos in ['NN', 'NNS', 'NNP', 'NNPS'] and ('subj' in other_rel.type or 'obj' in other_rel.type or 'prep' in other_rel.type or 'agent' in other_rel.type)
                
            for i, idx in enumerate(reversed(idxs[:-1]), 1):
                #print self[idx]
                next_features = []
                for next_rel in self[idx].get_dependency_graph().filter_iterator(matches):
                    num_nmods = len(next_rel.dep.deps)
                    if use_det: 
                        next_det = ''
                        for rel2 in next_rel.dep.deps:
                            if rel2.type == 'det':
                                next_det = rel2.dep.lem    
                    else:
                        next_det = 'x'
                    for feature in features: 
                        subsumed = (num_nmods < num_mods) if use_subs else 'x'
                        next_indices = []
                        next_indices.extend(feature[1])
                        next_indices.append((idx, next_rel.dep_idx)) 
                        new_f = ['{} _ role{}:{} det:{}_subsumed:{}'.format(feature[0], i,next_rel.type, next_det, subsumed), next_indices]
                        next_features.append(new_f)
                features = next_features             

            self._t.extend(features)
            for feature in features:
                #print feature
                fmap[feature[0]] = 1
               
         
    def _f_first_ne_tag(self, indices):
        f = [self.document.sentences[i].tokens[0].ne 
             for i in indices] 
        return '_'.join(f)
        
    def gold_transitions(self):
        num_prefix = self.history - 2
        slabels = deque([()]*num_prefix)
        slabels.appendleft('START')
        
        nsents = len(self)
        
        slabels.extendleft(['sent-{}'.format(i) 
                            for i in range(nsents)])
        slabels = list(slabels)
        
        from discourse.hypergraph import Transition  
        
        trans = []
        lsize = len(slabels) - self.history + 1
        trans.insert(0, Transition(['END',slabels[0]]))
        for i in range(lsize):
            idx = i+self.history
            trans.insert(0, Transition(slabels[i:idx]))
        
        return trans 

    def ordering2str(self, indices):
        strs = []
        for i in indices:
            strs.append('({})  {}'.format(i+1, self[i].space_sep_str()))
        return '\n'.join(strs)

    def hypergraph(self):
        import pydecode.chart as chart
        import discourse.hypergraph as hyper  
        c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing,
                               build_hypergraph=True, strict=False)
        hypergraph = hyper.build_hypergraph(self, c).finish()
        return hypergraph 

def make_from_corenlp_xml(xml_file, history_size=2):
    doc = cnlp.Document(xml_file)
    return RushModel(doc, history_size)     



