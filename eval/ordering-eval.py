import sys
import argparse
from scipy.stats import kendalltau

def readNextGoldOrdering(gold_in):
    """Reads the next gold sentence ordering and returns ordered list of tuples
        where the first element is the sentence number,
        and the second is the sentence string."""

    # Get the ordered list of gold sentence strings.
    ordering = readNextOrdering(gold_in)

    # If there was another ordering, return a list of tuples
    # with sentence indices and strings.
    # Otherwise return None
    if ordering is not None:
        return [(i, ordering[i - 1]) for i in range(1, len(ordering) + 1)]
    else:
        return None


def readNextOrdering(file_in):
    """Read the next ordering of sentences as a list of strings.
        The start of a new documents is indicate by an empty line."""

    # When preorder is true, we are in a blank line before the document start.
    # We need to read lines until we find one with text on.
    preorder = True

    # When readingsents is true, we should add another line to sentence list.
    readingsents = False

    # List of sentence strings as they appeared in the odering file.
    sentences = []

    # While preorder is True, read blank lines until EOF
    # or we encounter a string.
    line = ''
    while preorder is True:

        line = next(file_in, '')
        if not line == '\n':
            preorder = False
            readingsents = True

    # While readingsents is True, add lines to the list of sentences.
    while readingsents is True:
        if not line == '\n' and not line == '':
            sentences.append(line.strip())
            line = next(file_in, '')
        else:
            readingsents = False

    # If sentences is not empty, return it, otherwise return None
    if len(sentences) > 0:
        return sentences
    else:
        return None


def countInversions(order):
    """Count the inversions in a list of numbers. The indices (i,j) are an
        inversion if i < j and A[i] > A[j]."""

    # Get counts using a modified merge sort.
    result = mergeSortCount(order)
    return result[0]


def mergeSortCount(order):
    """An implementation of merge sort that counts
        the number of inversions in the original unsorted list.
        This function returns a tuple where the first item is number of
        inversions in the original unsorted list
        and the second is the sorted list"""

    # If order is length 1, just return it.
    size = len(order)
    if size == 1:
        return (0, order)

    # Split order in two. Recursively sort and count each subsequence.
    p = size / 2

    leftOrd = order[0:p]
    rightOrd = order[p:size]

    leftResult = mergeSortCount(leftOrd)
    rightResult = mergeSortCount(rightOrd)

    # Merge results and sum all inversion counts.
    thisResult = mergeAndCount(leftResult[1], rightResult[1])
    totalInversions = leftResult[0] + rightResult[0] + thisResult[0]

    # Return the total counts with the ordered list.
    return (totalInversions, thisResult[1])


def mergeAndCount(a, b):
    """Merge two lists and count the number of inversions.
        a and b must be sorted lists of numbers."""

    inversions = 0
    sortedList = []

    # Starting from the ends of the lists, insert the greatest element
    # of a[-1] and b[-1]. If a[-1] is greater, then it is greter than all of
    # the elements currently in b. Add these inversion counts. Pop the greater
    # element and insert it into sortedList and repeat.
    while len(a) > 0 and len(b) > 0:
        if a[-1] > b[-1]:
            inversions += len(b)
            sortedList.insert(0, a.pop())
        else:
            sortedList.insert(0, b.pop())
    while len(a) > 0:
        sortedList.insert(0, a.pop())
    while len(b) > 0:
        sortedList.insert(0, b.pop())

    return (inversions, sortedList)


def main():
    """Run evaluation of sentence ordering task. This script reads a file of
        gold sentence orderings and compares them to a file of predicted
        orderings. Each file contains the sentence orderings for all documents
        in the test set. Each document is separated by
        at least 1 blank line."""

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(prog='ordering-eval', add_help=True)
    parser.add_argument('-m', action='store_true',
                        help='Print in machine readable format: '
                        + 'tot_docs num_correct accuracy avg_k_tau')
    parser.add_argument('--gold', nargs=1, type=argparse.FileType('r'),
                        required=True,
                        help='File with gold sentence orderings, '
                        + 'documents separated by a blank line.')
    parser.add_argument('--predicted', nargs=1, type=argparse.FileType('r'),
                        required=True,
                        help='File with predicted sentence orderings, '
                        + 'documents separated by a blanks line.')
    parser.add_argument('--noheadline', action='store_true',
                        help='Evaluate without the first sentence '
                        + 'which is often a headline or noise.')
    
    parser._optionals.title = "arguments"
    args = parser.parse_args()

    # Read in the first gold ordering and predicted ordering.
    gold_in = args.gold[0]
    pred_in = args.predicted[0]

    goldorder = readNextGoldOrdering(gold_in)
    predictedorder = readNextOrdering(pred_in)

    # Number of correctly ordered documents.
    numcorrect = 0

    # Total number of documents.
    total = 0

    # Total number of sentence inversions for all predicted documents.
    total_kendalls_tau = 0

    # sum of pvals for avg
    total_pvals = 0 

    #total pairwise correct
    pairwise_correct = 0

    # While there are more documents to check, evaluate prediction.
    while goldorder is not None and predictedorder is not None:

        total += 1

        # If the number of sentences is different something is seriously wrong.
        if len(goldorder) is not len(predictedorder):

            print "Sentence pair {} is misaligned.".format(total)
            sys.exit()

        # Evaluate without first sentence which is often a headline/noise.
        if args.noheadline:
            predictedorder.remove(goldorder[0][1])
            goldorder.pop(0)
        
        
        # Iterate through each gold sentence and predicted
        # sentence in gold order. If they are ever different,
        # mark this prediction as incorrect.
        correct = True
        for i in range(0, len(goldorder)):
            if not goldorder[i][1] == predictedorder[i]:
                correct = False
            

        # Increment correct count if ordering is correct.
        if correct is True:
            numcorrect += 1
            total_kendalls_tau += 1
            pairwise_correct += 1
        # Otherwise, count the number of inverted sentence
        # orderings in the bad prediction.
        else:
            # Map sentence to correct sentence index
            smap = {}
            for g in goldorder:
                smap[g[1]] = g[0]

            # Get the predicted index ordering.
            p_indices = [smap[predictedorder[i]]
                         for i in range(len(predictedorder))]

            if len(p_indices) > 1:
                score = 0
                for i, idx in enumerate(p_indices[:-1]):
                    if idx+1 == p_indices[i+1]:
                        score += 1
                pairwise_correct = score / (len(p_indices) - 1.0)
            else:
                import sys
                print 'Can\'t order passage of size 1'
                sys.exit()


            scipytau, pval = kendalltau(p_indices, [i for i, _ in enumerate(p_indices, 1)])
            #ktau, pval = kendalltau(x,y)
            #        print ktau
            #            avg_ktau += ktau
            # Calculate Kendall's Tau.
            normalization = len(goldorder) * (len(goldorder) - 1) / float(2)
            tau = 1 - (2 * countInversions(p_indices)) / float(normalization)
            total_kendalls_tau += scipytau
            if abs(tau-scipytau) > .01:
                print "My tau {} scipy tau {}".format(tau, scipytau)
            total_pvals += pval

        # Grab the next gold and predicted ordering.
        goldorder = readNextGoldOrdering(gold_in)
        predictedorder = readNextOrdering(pred_in)

    # Display the results.
    accuracy = 1

    if total > 0:
        avg_kendalls_tau = total_kendalls_tau / float(total)
        accuracy = numcorrect / float(total)
        avg_pval = total_pvals / float(total)
        avg_pw_corr = pairwise_correct / float(total)
    else:
        import sys
        print 'Cannot evaluate empty dataset.'
        sys.exit()

    if args.m:

        print "{} {} {} {} {} {}".format(total, numcorrect,
                                         accuracy, avg_kendalls_tau,
                                         avg_pval, avg_pw_corr)

    else:
        print "Total documents: {}".format(total)
        print "Total correct: {}".format(numcorrect)
        print "Accuracy: {}".format(accuracy)
        print "Avg. Kendall's Tau {}".format(avg_kendalls_tau)
        print "pval: {}".format(avg_pval)
        print "pairwise correct: {}".format(avg_pw_corr)

if __name__ == '__main__':
    main()
