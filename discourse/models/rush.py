import corenlp_xml as cnlp
from collections import deque
from discourse.hypergraph import s2i
import discourse.gazetteers as gazetteers
import itertools

active_features = set(['is_first',
                       'is_last',
                       'role_match',
                       'pron_res',
                       'ne_tags',
                       'num_caps',
                       'quotes'])
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

    def gold_str(self):
        strs = [str(s) for s in self]
        return '\n'.join(strs)
    
    def feature_map(self, transition):  
        
        nsents = len(self)
        idxs = [s2i(s, end=nsents) for s in transition if s != ()] 
        key = tuple(idxs)
        if key not in self._f_cache:
        
            fmap = {}

            if 'role_match' in self._active_feat:
                self._f_mark_role_matches(idxs, fmap)
            if 'is_first' in self._active_feat:
                self._f_is_first(idxs, fmap)
            if 'is_last' in self._active_feat:
                self._f_is_last(idxs, fmap)
            if 'pron_res' in self._active_feat:
                self._f_pron_res(idxs,fmap) 
            if 'ne_tags' in self._active_feat:
                self._f_ne_tags(idxs,fmap)
            if 'num_caps' in self._active_feat:
                self._f_num_caps(idxs,fmap)
            if 'quotes' in self._active_feat:
                self._f_quotes(idxs,fmap)

            self._f_cache[key] = fmap
            return fmap
        
        else:
            
            return self._f_cache[key]
        #indices = [int(i[5:]) - 1 for i in reversed(transition.sents) if 'Sent_' in i]
        
        #return {'first_ne:{}'.format(self._f_first_ne_tag(indices)): 1}
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
        
        if idxs[0] < len(self) and idxs[1] > -1:
            prns = set([t.lem.lower() for t in self[idxs[0]] if 'PRP' in t.pos]) 
            fpeople = [t.lem.lower() for t in self[idxs[1]] if t.ne == 'PERSON' and t.lem.lower() in female_fnames]
            mpeople = [t.lem.lower() for t in self[idxs[1]] if t.ne == 'PERSON' and t.lem.lower() in male_fnames]
            unknownpeople = [t.lem.lower() for t in self[idxs[1]] if t.ne == 'PERSON' and t.lem.lower() not in male_fnames and t.lem.lower() not in female_fnames]
            orgs = False
            for t in self[idxs[1]]:
                if t.ne == 'ORGANIZATION':
                    orgs = True
                    break
                            
            for t in prns:
                if len(fpeople) > 0:
                    fmap[t+"_female"] = 1    
                if len(mpeople) > 0:
                    fmap[t+"_male"] = 1
                if len(unknownpeople) > 0:
                    fmap[t+"_person"] = 1
                if orgs:
                    fmap[t+"_org"] = 1
                
    def _f_is_first(self, idxs, fmap):
        if idxs[1] == -1:
            fmap['is_first'] = 1

    def _f_is_last(self, idxs, fmap):
        if idxs[0] == len(self):
            fmap['is_last'] = 1

    def _f_mark_role_matches(self, idxs, fmap):
        start = 0
        end = len(idxs) - 1
        if idxs[0] == len(self):    
            return 
        if idxs[-1] == -1:
            return 
        
        dgraph = self[start].get_dependency_graph()
        is_noun = lambda rel: rel.dep.pos in ['NN', 'NNS', 'NNP', 'NNPS']
        
        for rel in dgraph.filter_iterator(is_noun):
            features = ['role0:{}'.format(rel.type)]
            matches = lambda other_rel: other_rel.dep.lem == rel.dep.lem
                
            for i, idx in enumerate(idxs[start+1:], 1):
                next_features = []
                for next_rel in self[idx].get_dependency_graph().filter_iterator(matches):
                    
                    for feature in features:    
                        next_features.append('role{}:{}:{}'.format(i,next_rel.type, feature))
                features = next_features             

            for feature in features:
                #print feature
                fmap[feature] = 1
         
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
            strs.append(self[i].space_sep_str())
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



