import corenlp_xml as cnlp


class RushModel:

    def __init__(self, document, history_size=2):
        self.document = document
        self.num_sents = len(document.sentences)
        self.history_size = history_size

    def get_feature_map(self, transition):
  
        indices = [int(i[5:]) - 1 for i in reversed(transition.sents) if 'Sent_' in i]
        return {'first_ne:{}'.format(self._f_first_ne_tag(indices)): 1}

    def _f_first_ne_tag(self, indices):
        f = [self.document.sentences[i].tokens[0].ne 
             for i in indices] 
        return '_'.join(f)
        

def make_from_corenlp_xml(xml_file, history_size=2):
    doc = cnlp.Document(xml_file)
    return RushModel(doc, history_size)     



