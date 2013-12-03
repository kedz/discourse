from collections import deque

class GoldModel:
    def __init__(self, doc, history=2):
        self.doc = doc
        self.history = history    
            
    def __len__(self):
        return len(self.doc)
    
    def __iter__(self):
        return iter(self.doc)

    def feature_map(self, transition):
        idxs = [_s2i(self,s) for s in transition if _s2i(self,s) != -1000] 
        
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

        

    def hypergraph(self):
        import pydecode.chart as chart
        import discourse.hypergraph as hyper  
        c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing,
                               build_hypergraph=True, strict=False)
        hypergraph = hyper.build_hypergraph(self, c).finish()
        return hypergraph 

def _s2i(model, sent_label):
    if sent_label == 'START':
        return -1
    elif sent_label == 'END':
        return len(model)     
    elif sent_label == ():
        return -1000
    else:
        return int(sent_label[5:])
