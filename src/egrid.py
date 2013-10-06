import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
from os import listdir
from os.path import join, isfile, splitext
import math


def generate_transitions(role_set, path_size):
    """Create an ordered list of transitions, from a set role labels. The number of roles is exponential in the path size.""" 
   
    # Each loop iteration i adds all transitions of size (i+1) to the list trans. 
    trans = ['']   

    for p in range(path_size):
        new_trans = []
        for r in role_set:
            for t in trans:
                new_trans.append(t+r)
        trans = new_trans
    
    # trans is now a sorted list of all possible role transitions.
    return sorted(trans)  


class EntityGridFactory:
    
    def __init__(self, coref=False, syntax=False, salience=False, tsize=3, normalized=True):
        """Factory class for constructing EntityGrids."""
        
        # Use coreference flag - currently this does nothing.
        self.coref = coref
        
        # Use syntax - when true use the role set (s,o,x,-), else use (x,-).
        self.syntax = syntax
        
        # Use salience - split entities into two categories salient and non salient, by frequency. 
        # Transition set then becomes the cartesian product of {salient,nonsalient} x {s,o,x,-}^tsize. 
        self.salience= salience
        
        # The size of the transitions - when tsize is 3 and syntax=False, the transitions are {---,--x,-x-,x--,-xx,xx-,x-x,xxx}. 
        self.tsize = tsize

        # When true, the feature vector representation is normalized by the total number of transitions in the document. 
        self.normalized = normalized

        # Initialize the transition set 
        if self.syntax == True:
            self.trans = generate_transitions(('-','x','s','o'), self.tsize)
        else:
            self.trans = generate_transitions(('-','x'), self.tsize)   
        
    
    def buildEntityGrid(self, cmat):
        """Turn a numpy matrix or chararray of entity role transitions into an EntityGrid."""     

        # cmat is a numpy matrix or 2d chararry of representing an entity's role in a sentence.
        # Columns correspond to entities and rows correspond to a document's sentences.
        # e.g. an EntityGrid with 3 entities and 2 sentences:  
        #     [['x','-','s'],
        #     ['o','s','x']]
          
        return EntityGrid(cmat, self.trans, self.tsize, self.salience, self.normalized)



class EntityGrid:

    def __init__(self, trans_mat, trans, maxTrans=2, salience=False, normalized=True):
        """An EntityGrid computes the transition count vector representation of a document."""
        
        # Count maps for transistions -- the first index is for non salient entities, and the second for salient ones.
        self.trans_cnts = [defaultdict(int), defaultdict(int)]
        
        # Total number of transitions for salient and non salient entities.
        self.num_trans = [0,0]

        # Transistion matrix representing the document.   
        self.trans_mat = trans_mat

        # Count up all the transitions. 
        self._count_transitions(trans, maxTrans, salience)
        
        # Create the feature vector
        self._build_vector_rep(trans, salience, normalized)
        
    def _count_transitions(self, transSet, maxTrans, salience):        
        """Counts the transitions that occur in this document. There are possibly two classes of transition: salient and nonsalient."""    
        
        # If we are using the salience feature, map columns that have more than 1 non '-' role as salient. 
        if salience:
            salmap = {}
            for c in range(self.trans_mat.shape[1]):
                entFreq = 0
                for r in range(self.trans_mat.shape[0]):
                    if not self.trans_mat[r,c] == "-": 
                        entFreq += 1
                if entFreq > 1:
                    salmap[c] = 1
                else: 
                    salmap[c] = 0


        # For each column and for each row, count the entity transitions
        for c in range(self.trans_mat.shape[1]):
            for r in range(self.trans_mat.shape[0]-1):
                trans = self.trans_mat[r,c]
                i = r+1
                while len(trans) < maxTrans and i < self.trans_mat.shape[0]:
                    trans += self.trans_mat[i,c]
                    if trans in transSet:           
                        sal = 0
                        if salience:
                            sal = salmap[c]
                        self.trans_cnts[sal][trans] += 1
                        self.num_trans[sal] += 1
                    i+=1

    def _build_vector_rep(self, trans, salience, normalized): 
        """Construct the transition count vector representation. If normalized, these can be interpreted as a generative model for entity transitions."""    
        
        # If salience is used, there are twice as many features.
        if salience:
            self.tpv = np.zeros((len(trans)*2,1))
        else:
            self.tpv = np.zeros((len(trans),1))    
               
        # Get the feature value of each transition and place it in self.tpv       
        for t in range(len(trans)):           
            
            if trans[t] in self.trans_cnts[0]:
                if self.num_trans[0] > 0:
                    self.tpv[t] = self.trans_cnts[0][trans[t]] 
                    if normalized:
                        self.tpv[t] = self.tpv[t] / float(self.num_trans[0])
                            
            if salience:
                if self.num_trans[1] > 0:
                    self.tpv[t+len(trans)] = self.trans_cnts[1][trans[t]]
                if normalized:
                    self.tpv[t+len(trans)] = self.tpv[t+len(trans)] / float(self.num_trans[1])



#int2state = {}
#int2state[0] = '-'
#int2state[1] = 'x'
#int2state[2] = 's'
#int2state[3] = 'o'

#def loadPermutations(permDir, fileid):
#    
#    filePermTuples = []
#    perms = []
#    for f in listdir(permDir):
#        if f.startswith(fileid):
#            perms.append(join(permDir,f))
#    
#    for p in perms:
#        mat = []
#        pfile = open(p,"r")
#        for line in pfile:
#            mat.append( [ int(x) for x in line.strip().split(" ") ] )
#        
#        mat = np.matrix(mat)                            
#        filePermTuples.append((p,mat))
#    return filePermTuples
#

#def genPerms(imat):
#    
#    perms = []
#    eye = np.eye(imat.shape[0],imat.shape[0],0,dtype=np.int8)
#    signatures = set()
#    signatures.add(np.ravel(eye).tostring())
#    maxPerms = min(math.factorial(imat.shape[0]), 20)
#    
#        
#    while(len(signatures) + 1< maxPerms+1):
#        rperm = np.random.permutation(eye)  
#        flat = np.ravel(rperm).tostring()
#        if flat not in signatures:
#            signatures.add(flat)
#            perms.append(rperm) 
#
#    return perms
#
#def loadProcessedDocument(fileid, coref=False, syntax=False): 
#    tree = ET.parse(fileid)
#    root = tree.getroot()    
#   
#    numSents = 0
#    salience = defaultdict(int)
#    allEntities = set() 
#    sentenceEntitySets = []
#
#    if coref:
#        numSents = len(root[0][0])
#        ents = [ set() for i in range(numSents) ]
#
#        for corefs in root[0][1].findall('coreference'):
#            repmen = corefs
#            
#            for mention in corefs.findall('mention'):
#                snum = int(mention.find('sentence').text)
#                start = int(mention.find('start').text)
#                end = int(mention.find('end').text)
#                head = int(mention.find('head').text)
#                if 'representative' in mention.attrib:
#                    repmen = ''
#                    for i in range(start-1,end-1):
#                        repmen += root[0][0][snum-1][0][i].find('word').text +" "    
#                    
#                    
#                
#                deplabel = 'x'
#                if syntax:
#                    for dep in root[0][0][snum-1][4].findall('dep'):
#                        if int(dep.find('dependent').attrib['idx']) == head:
#                            if dep.attrib['type'] == 'nsubj':
#                                deplabel = 's'
#                            elif dep.attrib['type'] == 'dobj':
#                                deplabel = 'o'
#                            else:
#                                deplabel = 'x'
#
#                ents[snum-1].add((repmen,deplabel))
#                allEntities.add(repmen)
#        
#        tmat = np.zeros((numSents,len(allEntities)), dtype=np.int8)
#        col = 0
#        for ent in allEntities:
#            for r in range(numSents):
#                if (ent,'x') in ents[r]:
#                    tmat[r,col] = 1
#                elif (ent,'s') in ents[r]:
#                    tmat[r,col] = 2
#                elif (ent,'o') in ents[r]:
#                    tmat[r,col] = 3
#                else:
#                    tmat[r,col] = 0
#
#            col += 1
#        return tmat
#
#    if not coref:
#    
#        for sentence in root[0][0].findall('sentence'):
#            numSents += 1
#            entitySet = set()
#            for token in sentence[0].findall('token'):
#                pos = token.find('POS').text
#                if pos in ('NN','NNP','NNS','NNPS'):
#                    word = token.find('word').text.lower() 
#                    entitySet.add(word)
#                    salience[word] += 1
#                    allEntities.add(word)
#            sentenceEntitySets.append(entitySet)    
#            
#                
#            tmat = np.zeros((numSents,len(allEntities)), dtype=np.int8)
#            col = 0
#            for entity in allEntities:
#                for s in range(0,numSents):
#                    if entity in sentenceEntitySets[s]:
#                        if syntax:
#                            label = set()
#                            for dep in root[0][0][s][4].findall('dep'):
#                                if dep.get('type') == "nsubj" and dep[1].text.lower() == entity:
#                                    label.add('s')
#                                elif dep.get('type') == "dobj" and dep[1].text.lower() == entity:
#                                    label.add('o')
#                                elif dep[1].text.lower() == entity:
#                                    label.add('x')        
#                            if 's' in label:
#                                tmat[s,col] = 2
#                            elif 'o' in label:
#                                tmat[s,col] = 3
#                            else:
#                                tmat[s,col] = 1    
#                                                 
#                        else:
#                            tmat[s,col] = 1        
#                    else:
#                        tmat[s,col] = 0
#                col += 1            
#        return tmat
#
#                 
#def intMat2CharMat(imat): 
#
#
#    charmat = np.chararray((imat.shape[0],imat.shape[1]))
#    for r in range(imat.shape[0]):
#        for c in range(imat.shape[1]):
#            charmat[r,c] = int2state[imat[r,c]]
#    return charmat


