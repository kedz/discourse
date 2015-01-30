import re
from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join


txt_pat = re.compile( "<TEXT>(.*?)<\/TEXT>", re.DOTALL | re.MULTILINE ) 
sent_pat = re.compile( "<s[^>]*?>(.*?)<\/s>", re.DOTALL | re.MULTILINE ) 

class duc02Doc():
    def __init__( self, sents, tkn_sents ):
        self.sents = sents
        self.tkn_sents = tkn_sents

def read_DUC_02_doc( fileid ): 

    f = open( fileid, 'r' )
    filetxt = ''
    for line in f:
        filetxt = filetxt + line

    m = txt_pat.search( filetxt )
    
    articletxt = ''

    if m:
        articletxt = m.groups()[0] 
        #print articletxt
    
    sentences = []
    tokenized_sentences = []    
    for sent in sent_pat.findall( articletxt ):
        sentences.append( sent )
        tokenized_sentences.append( [ token.lower() for token in word_tokenize( sent ) ] )
        
    return duc02Doc( sentences, tokenized_sentences )

def read_DUC_02_doc_set( dirid ):
    return [ read_DUC_02_doc( join( dirid, f ) ) for f in listdir( dirid ) if isfile( join( dirid, f ) ) ]

#testfile = "/home/chris/Desktop/DUC/duc02/data/test/docs.with.sentence.breaks/d061j"

#docs =  read_DUC_02_doc_set( testfile )

#for d in range( len(docs) ):
#    print "Doc "+str(d)
#    for s in docs[d].tkn_sents:
#        print "\t",
#        print s



