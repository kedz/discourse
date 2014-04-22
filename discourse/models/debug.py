from collections import deque
from discourse.hypergraph import s2i

class GoldModel:
    def __init__(self, doc, history=2):
        self.doc = doc
        self.history = history    
            
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
        idxs = [s2i(s, end=nsents) for s in transition if s2i(s,end=nsents) != -1000] 
        
        for i, idx in enumerate(idxs):
            if i+1 < len(idxs):
                if idx - 1 != idxs[i+1]:
                    return {'GOLD':0}

        return {'GOLD': 1}

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
            strs.append(str(self[i]))
        return '\n'.join(strs)

    def hypergraph(self):
        import pydecode.chart as chart
        import discourse.hypergraph as hyper  
        c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing,
                               build_hypergraph=True, strict=False)
        hypergraph = hyper.build_hypergraph(self, c).finish()
        return hypergraph 
