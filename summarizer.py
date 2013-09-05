from os.path import exists, basename, join
import sys, getopt
from duc02 import read_DUC_02_doc_set
from nltk import Text, TextCollection
from scipy.spatial import distance
import numpy as np
import datetime

class Summarizer:
    """Summarizer for documents in the DUC 2002 format."""
    
    def __init__( self, docset_dir, debug=False ):
        
        self.docset_dir = docset_dir
        self.dbg = debug
        self.docset = read_DUC_02_doc_set( docset_dir ) # read in all documents in a docset (these are related docs for summarization)
              
        self.texts = [] # list of documents as Text object representation 
        # populate self.texts -- Texts represent the document as a whole and not discrete sentences
        # tfidf score are determined by document, not by sentence 
        for d in self.docset:
            doc_txt = []
            for s in d.tkn_sents:
                doc_txt.extend( s )
            self.texts.append( Text( doc_txt ) )        
    
        self.collection = TextCollection( self.texts ) # TextCollection of texts
        self.num_docs = len( self.texts ) # total number of docs in docset
        self.num_terms = len( self.collection ) # total number of tokens in docset
        self.unique_terms = list( set( self.collection ) ) # vocabulary set
        self.num_vocab = len( self.unique_terms ) # vocab size
    
        # num_docs x num_vocab matrix holding tf idf scores for each doc/word term vector 
        self.tfidf_table =  np.zeros( [ self.num_docs, self.num_vocab ] )

        for d in range( self.num_docs ):
            doc = self.texts[ d ]
            for w in range( self.num_vocab ):
                word = self.unique_terms[ w ]           
                self.tfidf_table[ d, w ] = self.collection.tf_idf( word, doc )
            
        # document x sentence 2d list holding a reference to a 1d array of tfidf weighted term vector
        self.d_s_termvector = []
        for d in range ( self.num_docs ):
            doc = self.docset[ d ]
            termvectors = []
            for s in range( len( doc.tkn_sents ) ): 
                wordset = set( doc.tkn_sents[ s ] )
                term_vector = np.zeros( [ self.num_vocab ] )
                for w in range( self.num_vocab ):
                    word = self.unique_terms[ w ]                 
                    
                    if word in wordset:
                        term_vector[ w ] = self.tfidf_table[ d, w ]
                    else:
                        term_vector[ w ] = 0.0
                
                termvectors.append( term_vector )

            self.d_s_termvector.append( termvectors )

    def greedy_sum( self, max_length=250 ):
        """Greedy summarization algorithm from McDonald (2007). Default summary length constraint is 250 words""" 
        summary_units = [] # text units selected for summary
        txt_units = [] # text units remaining
        
        # create text units for each sentence in each document where a text unit is a tuple( d, s, r, l ) where
        #   d = doc index
        #   s = sentence index   
        #   r = relevance score
        #   l = number of words in text unit
        for d in range( self.num_docs ):
            tvectors = self.d_s_termvector[ d ]
            for s in range( len( tvectors ) ):
                tvector = tvectors[ s ]
                pos = s + 1
                sim = self._sim( tvector ) # get average cosine similarity of this vector to the rest of the text units             
                rel = pos**-1 + sim # compute relevance score
                
                txt_units.append( ( d, s, rel, len( self.docset[ d ].tkn_sents[ s ] ) ) )
        
        
        # sort text units by relevance(desc) and select the most relevant sentence as the first sentence in the summary
        txt_units = sorted( txt_units, key=lambda txt_unit: txt_unit[ 2 ], reverse=True )  
       
         
        if self.dbg: 
            print 
            print "{} : Greedy Summarization for docset: {} ".format( datetime.datetime.now(), self.docset_dir )
            print   "TEXT UNITS\n----------\n\n"
            for t in txt_units:
                print 'Doc: {} Sent: {} Rel: {} Len: {}\n\tTokens: {}'.format( t[0], t[1], t[2], t[3], self.docset[ t[0] ].sents[ t[1] ] )
            
        
        summary_units.append( txt_units.pop( 0 ) )
        
        
        length = summary_units[ 0 ][ 3 ] # summary length -- stop when we can no longer add sentences that violate the max length constraint

        cur_rel = summary_units[ 0 ][ 2 ] # current total relevance score for this summary
        cur_red = 0 # current redundancy for this summary
        

        loop_iters = 1
        while length < max_length - 1 and len( txt_units ) > 0:

            max_index = 0
            max_score = 0
            max_red = 0

            tus_scores = [] # for debugging, holds the text units, and gets sorted by score after each round
    
            # for each text unit (aka sentence), calculate its MMR (relevance - redundancy)
            for i in range( len( txt_units ) ):

                
                tu = txt_units[ i ]
                rel = cur_rel + tu[ 2 ] # add candidate text unit's relevance score to the current summary score
                red = cur_red + self._calc_red( summary_units, tu ) # calc candidate text unit's redundancy compared to current summary sentences and add to red score
                score = rel - red # calculate MMR
                l = length + tu[3]
                tus_scores.append( ( tu, score, rel, red, l ) ) # for debugging -- can print out text units by MMR for each iteration 
                
                # if this score beats the current max_score, mark this text unit as the current candidate to add to the summary
                if score > max_score:
                    max_index = i
                    max_score = score
                    max_red = red
            

            # remove the highest scoring(by MMR) text unit, from candidates list
            # if adding the candidate text unit does not violate the summary length constraint, add it to the summary
           
            max_tu = None
            tus_scores = sorted( tus_scores, key=lambda x: x[1], reverse=True ) 
            while len(tus_scores) > 0 and tus_scores[0][4]  >= max_length:
                tus_scores.pop(0)        
            
            if len( tus_scores ) > 0:
                max_tu_score = tus_scores.pop(0)
            
            txt_units = map( lambda t: t[0], tus_scores ) 
            
            #max_tu = txt_units.pop( max_index )
            
            if not max_tu_score == None: 
                summary_units.append( max_tu_score[0] )
                cur_rel = max_tu_score[ 2 ]
                cur_red = max_tu_score[ 3 ]
                length = max_tu_score[ 4 ]    

            # display current summary sentences and top candidates
            if self.dbg:
                print "{} :\tSUMMARIZATION ITERATION {}".format( datetime.datetime.now(), loop_iters )
                print "{} :\t{}(MMR) = {}(REL) - {}(RED)\t | Current summary length: {}".format( datetime.datetime.now(), cur_rel - cur_red, cur_rel, cur_red, length )
                print "{} :\tCurrent Summary Listing:".format( datetime.datetime.now() )
            
                for t in summary_units:
                    print ' Doc: {} Sent: {} Rel: {} Len: {} \n   Sent: {}'.format( t[0], t[1], t[2], t[3], self.docset[ t[0] ].sents[ t[1] ] )
                
                print "----------------------------------------------------------------------------------------------\n"
                print "{} :\tTop Candidate Text Units Listing:".format( datetime.datetime.now() )
                i = 0
                while i < 15 and i < len( tus_scores ):
                    tu_score = tus_scores[i]
                    t = tu_score[0]
                    score = tu_score[1]
                    rel = tu_score[2]
                    red = tu_score[3]
                    print ' MMR: {} Rel: {} Red: {} Doc: {} Sent: {} Rel: {} Len: {} \n   Sent: {}'.format( score, rel, red, t[0], t[1], t[2], t[3], self.docset[ t[0] ].sents[ t[1] ] )
                    i += 1

            loop_iters += 1
        return summary_units




    def _sim( self, tv1, tv2=None ):
        """Compute cosine similarity between term vectors. If tv2 == None, the avg cosine sim between tv1 and the entire docset is calculated."""          
        sim = 0.0

        # compute avg cosine similarity of tv to all term vectors
        if tv2 == None:
            tot_tvs = 0.0
            for d in range( self.num_docs ):
                tvectors = self.d_s_termvector[ d ]
                for s in range( len( tvectors ) ):
                    if not tv1 is tvectors[ s ]:
                        sim += distance.cosine( tv1, tvectors[ s ] )
                        tot_tvs += 1.0
            sim = sim / tot_tvs
        # compute cosine sim between tv1 and tv2
        else:
            sim += distance.cosine( tv1, tv2 )
        
        return sim


    def _calc_red( self, stus, tu ):
        """Compute redundancy score if text unit tu is added to summary text units (stus)."""          
        tv = self.d_s_termvector[ tu[ 0 ] ][ tu[ 1 ] ]
        
        red = 0.0
        for stu in stus:
            stv = self.d_s_termvector[ stu[ 0 ] ][ stu[ 1 ] ]
            red += self._sim( tv, stv )


        return red 
        


def main( docset_dir, summary_length, output_dir, debug=False ):
    """Sumarize all DUC 2002 docsets in docset_dir subject to the summary_length constraint, and write them out in a ROUGE compatible format to output_dir."""

    print datetime.datetime.now(),
    print " :\t### SUMMARIZER ###"
    print datetime.datetime.now(),
    print " :\t"
    print datetime.datetime.now(),
    print " :\tReading document set in: " + docset_dir

    # initialize summarizer 
    sumpy = Summarizer( docset_dir, debug )

    print datetime.datetime.now(),
    print " :\tDocument Set Size: " + str( sumpy.num_docs )
    print datetime.datetime.now(),
    print " :\tCreated a collection of "+ str( sumpy.num_terms ) + " terms."
    print datetime.datetime.now(),
    print " :\tUnique terms found: " +  str( sumpy.num_vocab )

    
    print datetime.datetime.now(),
    print " :\tGenerating summary...",
    sys.stdout.flush()
    # generate summary subject to summary length
    import time
    start = time.time()
    sum_units = sumpy.greedy_sum( summary_length )
    elapsed = time.time() - start
    print "\tcomplete!"
    print datetime.datetime.now(),
    print " :\tTotal time elapsed: " + str( datetime.timedelta( seconds=elapsed ) )
    actual_length = sum( map( lambda s: s[ 3 ], sum_units ) )
    print datetime.datetime.now(),
    print " :\tDesired length: {}".format( summary_length - 1 )
    print datetime.datetime.now(),
    print " :\tActual length: {}".format( actual_length )  

    # write summary to file 
    system_title = 'greedy.L{}.{}'.format( summary_length, basename( docset_dir ) )
    sum_file = join( output_dir, system_title+'.html' )
    print datetime.datetime.now(),
    print " :\tWriting summary to file: {}".format( sum_file )    
    
    f = open( sum_file, 'w' )    
    f.write( '<html>\n' )
    f.write( '<head>\n' )
    f.write( '<title>{}</title>\n'.format( system_title ) )
    f.write( '</head>\n' )
    f.write( '<body bgcolor="white">\n' )  
    for i in range( len( sum_units ) ):
        snum = i+1
        txt_unit = sum_units[ i ]
        d = txt_unit[ 0 ]
        s = txt_unit[ 1 ]
        rel = txt_unit[ 2 ]
        f.write( '<a name="{}">[{}]</a> <a href="#{}" id={}>{}</a>\n'.format( snum, snum, snum, snum, sumpy.docset[ d ].sents[ s ].lower() ) )
    f.write( '</body>\n' )
    f.write( '</html>\n' )      
    f.close() 

    print datetime.datetime.now(),
    print " :\tSummarizer complete."

if __name__ == '__main__':


    # Get options/handle option errors
    script = basename( __file__ )
    argnames = '-d <docsetdir> -l <summarylength> -o <outputdir>'
    help_msg = '{} {}'.format( script, argnames )
    debug = False

    docset_dir = ''
    output_dir = ''
    summary_length = 250
    try:
        opts, args = getopt.getopt( sys.argv[1:], "hd:l:o:", ["docset-directory=", "summary-length=", "output-directory=", "debug"] )
    except getopt.GetoptError:
        print help_msg
        sys.exit(-1)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit(1)
        elif opt in ("-d", "--docset-directory"):
            docset_dir = arg
        elif opt in ("-l", "--summary-length"):
            summary_length = int(arg)
        elif opt in ("-o", "--output-directory"):
            output_dir = arg
        elif opt in ("--debug"):
            debug = True                                                  

    if docset_dir == '' or not exists( docset_dir ):
        print help_msg
        print 'Invalid DUC docset directory. Please check the DUC docset directory path.'
        sys.exit(-1)
    if output_dir == '' or not exists( output_dir ):
        print help_msg
        print 'Invalid output directory. Please check the output directory path.'
        sys.exit(-1)   
    if not isinstance( summary_length, int ):
        print help_msg
        print 'Invalid argument for option -l. Argument must be a positive integer.'
        sys.exit(-1) 

    main( docset_dir, summary_length, output_dir, debug )
