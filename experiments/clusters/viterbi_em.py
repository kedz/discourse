import argparse
import os
import codecs
from collections import defaultdict
from math import log
import shutil

def main():
    trainfile, ofile_template = parse_cmdline()
    
    state_smoothing_parameter = 0.00001
    lm_smoothing_parameter = 0.00001

    iteration = 1
    delta = float('+inf')
    last_avg_ll = float('+inf')
    while delta > 0.00001:
    
        ofile = ofile_template.repace('.txt', '_{}.txt'.format(iteration))

        print u'Reading state transition counts from {}'.format(trainfile)
        states, Q = make_transition_table(trainfile, state_smoothing_parameter)
        print u'Reading state language model counts from {}'.format(trainfile)
        E = make_state_language_models(trainfile, states,
                                       lm_smoothing_parameter)


        print u'Streaming input from {}'.format(trainfile)

        sum_score = 0
        ninstances = 0
        with cr(trainfile) as rf, cw(ofile) as wf:
            sequence = next_instance(rf)
            while sequence is not None:
                ninstances += 1
                labeled_sequence, score = decode(sequence, states, Q, E)
                sum_score += score         
                write_sequence(labeled_sequence, wf)

                sequence = next_instance(rf)
        avg_ll = sum_score / float(ninstances)
        delta = abs(last_avg_ll - avg_ll)
        last_avg_ll = avg_ll
        trainfile = ofile
        print "Average log likelihood: {}".format(avg_ll)
        print 'Delta: {}'.format(delta)
        iteration += 1 

    completed_sequence = ofile_template.replace('.txt', '_final.txt')
    shutil.move(ofile, completed_sequence)

def write_sequence(labeled_sequence, f):
    for y, x in labeled_sequence:
        f.write(u'{}\t{}\n'.format(y, u' '.join(x)))
    f.write(u'\n')
    f.flush()

def cr(fname):
    return codecs.open(fname, 'r', 'utf-8')
def cw(fname):
    return codecs.open(fname, 'w', 'utf-8')

def decode(sequence, states, Q, E):
#    for i, s in enumerate(sequence, 1):
#        print i, s

    pi = {}
    bp = {}
    
    #print sequence[0]
    for state in states:
        
        pi[(state, 1)] = log(Q('__START__', state)) + E(state, sequence[0])
        bp[(state, 1)] = u'__START__'

    for i, component in enumerate(sequence[1:], 2):
        for state2 in states:
            max_score = None
            max_state = None
            for state1 in states:
                
                score = pi[(state1, i - 1)] \
                        + log(Q(state1, state2)) + E(state2, component)
                if max_state is None or score > max_score:
                    max_score = score
                    max_state = state1
            pi[(state2, i)] = max_score
            bp[(state2, i)] = max_state

    ncoms = len(sequence)
    end = u'__STOP__'
    max_score = 0
    max_state = None
    for state in states:
        #pi[(end, ncoms + 1)] 
        score = pi[(state, ncoms)] + log(Q(state, end))
        if max_state is None or score > max_score:
            max_score = score
            max_state = state

    optimal_states = []
    pointer = max_state
    position = ncoms
    while position > 0:
        optimal_states.append(pointer)
        #if position == 0:
        #    break
        pointer = bp[(pointer, position)]
        position -= 1
#    print optimal_states
        #bp[(end, 1)] = u'__START__'
    labeled_sequence = []
    optimal_states.reverse()
    for i, state in enumerate(optimal_states):
        labeled_sequence.append((state, sequence[i]))    

    return tuple(labeled_sequence), max_score

def next_instance(f):
    components = []
    line = f.readline().strip()
    while line != u'':
        y, x = line.split('\t')
        components.append(tuple(x.strip().split(u' ')))
        line = f.readline().strip()
    if len(components) > 0:
        return components
    else:
        return None        

def make_state_language_models(lmfile, states, smoothing_parameter):

    state_unigram_count = {}
    state_bigram_count = {}
    vocab = set()

    with codecs.open(lmfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            if line == u'':
                continue
            #print line
            state, token_str = line.split(u'\t')
            tokens = token_str.strip().split(u' ')
            ntokens = len(tokens)
            for i in range(ntokens):
                word1 = tokens[i]
                if state not in state_unigram_count:
                    state_unigram_count[state] = {}
                if word1 not in state_unigram_count[state]:
                    state_unigram_count[state][word1] = 0
                
                state_unigram_count[state][word1] += 1
                vocab.add(word1)      

                if i + 1 >= ntokens:
                    continue

                word2 = tokens[i + 1] 
                if state not in state_bigram_count:
                    state_bigram_count[state] = {}
                if word1 not in state_bigram_count[state]:
                    state_bigram_count[state][word1] = {}
                if word2 not in state_bigram_count[state][word1]:
                    state_bigram_count[state][word1][word2] = 0
                state_bigram_count[state][word1][word2] += 1

    nwords = len(vocab)

    def probability(state, word1, word2):
        count1 = state_unigram_count[state].get(word1, 0)
        if count1 == 0 or state_bigram_count[state].get(word1, None) is None:
            return  (smoothing_parameter) / float(nwords * smoothing_parameter)
        
        count2 = state_bigram_count[state][word1].get(word2, 0)
        bgfreq = (count2 + smoothing_parameter)
        return bgfreq / float(count1 + (nwords * smoothing_parameter))


    misc_numerators = {}
    misc_denoms = defaultdict(float)
    def misc_probability(word1, word2):
        if word1 not in misc_denoms:            
            for u in vocab:
                max_prob = None
                for state in states:
                    if state == 'tpc_MISC':
                        continue
                    prob = probability(state, word1, u)
                    if prob > 1:
                        print '???', state, word1, u, prob
                    if max_prob is None or prob > max_prob:
                        max_prob = prob
                
                misc_denoms[word1] += (1 - max_prob)
                misc_numerators[u] = (1 - max_prob)
                if max_prob > 1:
                    print 'misc_denoms ', word1, u,  (1 - max_prob)
        #print "MISC",  misc_numerators[word2] / misc_denoms[word1]       
        return misc_numerators[word2] / misc_denoms[word1]                

    def emission(state, tokens):
        tot_emission = 0
        for t, token1 in enumerate(tokens[:-1]):
            token2 = tokens[t + 1]
            if state != 'tpc_MISC':
                tot_emission += log(probability(state, token1, token2))
            else:
                tot_emission += log(misc_probability(token1, token2))
        #print tokens, tot_emission
        return tot_emission
    
#    for state in state_unigram_count.keys():
#        for token in state_unigram_count[state].keys():
#            for token2 in state_bigram_count[state][token].keys():
#                print state, token, state_unigram_count[state][token],
#                print token2, state_bigram_count[state][token][token2]            
            
    return emission


def make_transition_table(trainfile, state_smoothing_parameter):
    state_counts = {}
    state_state_counts = {}
    last_state = '__START__'
    states = set()

    def count(state1, state2):
        if state1 not in state_counts:
            state_counts[state1] = 0
        state_counts[state1] += 1

        if state1 not in state_state_counts:
            state_state_counts[state1] = {}
        
        if state2 not in state_state_counts[state1]:
            state_state_counts[state1][state2] = 0
        state_state_counts[state1][state2] += 1

    with codecs.open(trainfile, 'r', 'utf-8') as f:

        for line in f:
            line = line.strip()
            if line == u'':
                count(last_state, u'__STOP__')
                last_state = u'__START__'

            else:
                state = line.split('\t')[0].strip()
                count(last_state, state)
                states.add(state)
                last_state = state

    nstates = len(state_counts.keys())
    
    def Q(state1, state2):
        numerator = state_state_counts[state1].get(state2, 0) \
                    + state_smoothing_parameter                    
        denominator = state_counts[state1] \
                      + (state_smoothing_parameter * nstates)
        return numerator / float(denominator)

    states = states - frozenset([u'__START__', u'__STOP__'])

#    for state1 in states.union(set(['__START__'])):
#        for state2 in states.union(set(['__STOP__'])):
#            print state1, state2, Q(state1, state2)
    return states, Q

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-tf', u'--train-file',
                        help=u'Training data file',
                        type=unicode, required=True)

    parser.add_argument(u'-of', u'--output-file', 
                        help=u'Location to write clusters.',
                        type=unicode, required=True)

    args = parser.parse_args()
    trainfile = args.train_file
    ofile = args.output_file

    if not os.path.exists(trainfile):
        import sys
        sys.stderr.write(u'{} does not exist!\n'.format(trainfile))
        sys.stderr.flush()
        sys.exit()

    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    
    return trainfile, ofile

if __name__ == '__main__':
    main()
