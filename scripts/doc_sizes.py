import discourse.data as data
import corenlp as cnlp
from collections import defaultdict

def make_counts(corpus, counts):
    tots = 0
    for doc in corpus:
        counts[len(doc)] += 1
        if len(doc) < 17:
            tots +=1 
    return tots


train_apws = [cnlp.Document(xml) for xml in data.corenlp_apws_train()]
train_ntsb = [cnlp.Document(xml) for xml in data.corenlp_ntsb_train()]
test_apws = [cnlp.Document(xml) for xml in data.corenlp_apws_test()]
test_ntsb = [cnlp.Document(xml) for xml in data.corenlp_ntsb_test()]

train_apws_counts = defaultdict(int)
test_apws_counts = defaultdict(int)
train_ntsb_counts = defaultdict(int)
test_ntsb_counts = defaultdict(int)

tr_a_tots = make_counts(train_apws, train_apws_counts)
te_a_tots = make_counts(test_apws, test_apws_counts)
tr_n_tots = make_counts(train_ntsb, train_ntsb_counts)
te_n_tots = make_counts(test_ntsb, test_ntsb_counts)

print '{} :   \t\t{} :   \t\t{} :   \t\t{} :  \n'.format('train_apws', 'test_apws', 'train_ntsb', 'test_ntsb')
for l in range(40):

    print '{} : {}\t\t{} : {}\t\t{} : {}\t\t{} :{}'.format(l, train_apws_counts[l], l, test_apws_counts[l], l, train_ntsb_counts[l], l, test_ntsb_counts[l])  

l = '<17'
print
print '{} : {}\t\t{} : {}\t\t{} : {}\t\t{} :{}'.format(l, tr_a_tots, l, te_a_tots, l, tr_n_tots, l, te_n_tots)  

