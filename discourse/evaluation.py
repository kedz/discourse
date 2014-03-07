from itertools import izip
from discourse.hypergraph import s2i, recover_order
import discourse
from collections import OrderedDict
import textwrap
from discourse.models.rush import RushModel
import scipy as sp


def eval_against_baseline(testX, baselineY, newY, baseline_model, new_model,
                          base_feats, new_feats):
    """
    Evaluate differences in two models. Prints out per instance
    analysis of transitions predicted by baseline and new models.

    testX -- A list of corenlp.Document objects to evaluate on.

    baselineY -- A list of lists of discourse.hypergraph.Transition
        objects predicted by the baseline model for the documents
        in testX.

    newY -- A list of lists of discourse.hypergraph.Transition
        objects predicted by the new model for the documents
        in testX.

    baseline_model -- A discourse.perceptron.Perceptron object trained
        on the features in base_feats.

    new_model -- A discourse.perceptron.Perceptron object trained
        on the features in new_feats.

    base_feats -- A dict of feature names to boolean values,
        indicating the features active in the baseline model.

    new_feats -- A dict of feature names to boolean values,
        indicating the features active in the new model.
    """

    # Limit text output to 80 chars and wrap nicely.
    wrapper = textwrap.TextWrapper(subsequent_indent='\t')

    print u'OVERALL STATS FOR TEST DOCUMENTS'

    # Print macro averaged Kendall's Tau and pvalues for baseline
    # and new model.
    bl_avg_kt, bl_avg_pval = avg_kendalls_tau(baselineY)
    new_avg_kt, new_avg_pval = avg_kendalls_tau(newY)
    print u'\t     | BASELINE      | NEW'
    print u'{:14} {:.3f} ({:.3f}) | {:.3f} ({:.3f})\n'.format(u'Kendalls Tau',
                                                              bl_avg_kt,
                                                              bl_avg_pval,
                                                              new_avg_kt,
                                                              new_avg_pval)

    # Print bigram gold sequence overlap (accuracy) for baseline and
    # new model.
    bl_bg_acc = mac_avg_bigram_acc(baselineY)
    new_bg_acc = mac_avg_bigram_acc(newY)
    print u'\t     | BASELINE      | NEW'
    print u'{:12} | {:.3f}         | {:.3f} \n'.format(u'bigram acc',
                                                       bl_bg_acc,
                                                       new_bg_acc)

    # Print stats for individual test instances.
    for test_idx, datum in enumerate(izip(testX, baselineY, newY), 1):
        testx, baseliney, newy = datum
        print u'TEST NO. {:4}\n=============\n'.format(test_idx)

        # Print Kendalls Tau and pvalue for baseline and new model
        # for this test instance.
        bl_kt, bl_pval = kendalls_tau(baseliney)
        new_kt, new_pval = kendalls_tau(newy)
        print u'\t     | BASELINE      | NEW'
        print u'{:14} {:.3f} ({:.3f}) | {:.3f} ({:.3f})\n'.format(u'K. Tau',
                                                                  bl_kt,
                                                                  bl_pval,
                                                                  new_kt,
                                                                  new_pval)

        # Print bigram gold sequence overlap (accuracy) for baseline
        # and new model.
        bl_acc = bigram_acc(baseliney)
        new_acc = bigram_acc(newy)
        print u'\t     | BASELINE      | NEW'
        print u'{:12} | {:.3f}         | {:.3f} \n'.format(u'bigram acc',
                                                           bl_acc,
                                                           new_acc)

        # Print document sentences in correct order.
        print u'GOLD TEXT\n=========\n'
        for i, s in enumerate(testx):
            print wrapper.fill(u'({:3}) {}'.format(i, unicode(s)))
        print u'\n\n'

        # Print document sentences in baseline order.
        print u'BASELINE TEXT\n=========\n'
        indices = [s2i(t.sents[0]) for t in recover_order(baseliney)[:-1]]
        for i in indices:
            print wrapper.fill(u'({}) {}'.format(i, unicode(testx[i])))
        print u'\n\n'

        # Print document sentences in new model order.
        print u'NEW MODEL TEXT\n=========\n'
        indices = [s2i(t.sents[0]) for t in recover_order(newy)[:-1]]
        for i in indices:
            print wrapper.fill(u'({}) {}'.format(i, unicode(testx[i])))
        print u'\n\n'

        # Get predicted transitions in order for both models.
        # NOTE: The predict function of the Perceptron object returns
        # the predicted transitions in no particular order.
        # When in doubt, use recover_order on any predicted output
        # if you want to iterate over it as if you were traversing the
        # graph of sentence transitions.
        baseline_trans = discourse.hypergraph.recover_order(baseliney)
        new_trans = discourse.hypergraph.recover_order(newy)

        # Map tail sentence of a transition to the transition.
        p2t_baseline = _position2transition_map(baseline_trans)
        p2t_new = _position2transition_map(new_trans)

        # For each transition leaving the same sentence, if the models
        # disagree on what the next sentence is, print analysis of
        # the model features.
        for pos, t_bl in p2t_baseline.items():
            if p2t_new[pos].sents[0] != t_bl.sents[0]:
                t_new = p2t_new[pos]

                # Print tail sentence.
                if pos > -1:
                    pos_str = unicode(testx[pos])
                else:
                    pos_str = u'START'
                print u'=' * 80
                print wrapper.fill(u'({:3}) {}'.format(pos, pos_str))
                print (u'-' * 80)
                print u'  |\n  V'

                # Print baseline head sentence
                if s2i(t_bl.sents[0]) is not None:
                    bl_str = unicode(testx[s2i(t_bl.sents[0])])
                else:
                    bl_str = u'END'
                print wrapper.fill(u'(OLD) {}\n'.format(bl_str)) + u'\n'

                # Print baseline model features for the predicted
                # baseline transition.
                explain(t_bl, baseline_model, new_model, testx,
                        base_feats, new_feats)

                # Print new model head sentence.
                if s2i(t_new.sents[0]) is not None:
                    new_str = unicode(testx[s2i(t_new.sents[0])])
                else:
                    new_str = 'END'
                print wrapper.fill(u'(NEW) {}\n'.format(new_str)) + u'\n'

                # Print new model features for the predicted new
                # model transition.
                explain(t_new, baseline_model, new_model, testx,
                        base_feats, new_feats)

                # Print gold head sentence, that is, the sentence the
                # models should have selected.
                if pos + 1 < len(testx):
                    gstr = u'(GLD) {}\n'.format(unicode(testx[pos + 1]))
                    print wrapper.fill(gstr) + u'\n'

                if pos + 1 == s2i(t_bl.sents[0], end=len(testx)):
                    print 'OLD MODEL IS CORRECT\n'
                if pos + 1 == s2i(t_new.sents[0], end=len(testx)):
                    print 'NEW MODEL IS CORRECT\n'
                print


def explain(t, baseline_model, new_model, testdoc, base_feats, new_feats):
    """
    Prints the features and feature scores for a transition t under a
    baseline and new model.

    t -- A discourse.hypergraph.Transition object to explain.

    baseline_model -- A discourse.perceptron.Perceptron object trained
        on the features in base_feats.

    new_model -- A discourse.perceptron.Perceptron object trained
        on the features in new_feats.

    testdoc -- A corenlp.Document object corresponding to the test
        instance that the transition t is from.

    base_feats -- A dict of feature names to boolean values,
        indicating the features active in the baseline model.

    new_feats -- A dict of feature names to boolean values,
        indicating the features active in the new model.
    """

    # Create RushModels for each model feature set.
    base_rmodel = RushModel(testdoc, history=2, features=base_feats)
    new_rmodel = RushModel(testdoc, history=2, features=new_feats)

    # Get each model's weight vector.
    bl_weights = baseline_model.dsm._vec.inverse_transform(
        baseline_model.sp.w)[0]
    new_weights = new_model.dsm._vec.inverse_transform(
        new_model.sp.w)[0]

    # Get baseline and new model features for this transition t.
    bl_feat = set(baseline_model.dsm._vec.inverse_transform(
        baseline_model.dsm.psi(base_rmodel, [t]))[0].keys())
    new_feat = set(new_model.dsm._vec.inverse_transform(
        new_model.dsm.psi(new_rmodel, [t]))[0].keys())

    # Print baseline model features for this transition and their score
    # under the baseline and new model.
    print '\tBASELINE FEATURES'
    print 'FEATURE                      |BASELINE     | NEW'
    for feat in bl_feat:
        bl_w = bl_weights[feat] if feat in bl_weights else '?'
        new_w = new_weights[feat] if feat in new_weights else '?'
        print u'{:28} | {:11} | {}'.format(feat, bl_w, new_w)
    print

    # Print new model features for this transition and their score
    # under the baseline and new model.
    print '\tNEW MODEL FEATURES'
    print 'FEATURE                      |BASELINE     | NEW'
    for feat in new_feat:
        bl_w = bl_weights[feat] if feat in bl_weights else '?'
        new_w = new_weights[feat] if feat in new_weights else '?'
        print u'{:28} | {:11} | {}'.format(feat, bl_w, new_w)
    print
    print


def avg_kendalls_tau(dataY):
    """
    Returns the macro averaged Kendall's tau and pvalues for list
    of lists of predicted transitions.

    dataY -- A list of lists of discourse.hypergaph.Transition objects
        predicted by a discourse model.

    returns (avg_kt, avg_pval)
    """
    # Sum all kendalls tau and pvalues over predicted instances.
    kt_sum = 0
    pval_sum = 0
    for transitions in dataY:
        kt, pval = kendalls_tau(transitions)
        kt_sum += kt
        pval_sum += pval

    # Return (None, None) if dataY is empty, else return macro
    # averaged Kendall's tau and pvals.
    if len(dataY) > 0:
        avg_kt = float(kt_sum) / len(dataY)
        avg_pval = float(pval_sum) / len(dataY)
    else:
        avg_kt = None
        avg_pval = None
    return (avg_kt, avg_pval)


def kendalls_tau(transitions):
    """
    Compute Kendall's tau and pvalue for a list of
    discourse.hypergraph.Transition objects.

    transitions -- A list of discourse.hypergaph.Transition objects.

    returns (kt, pval)
    """
    # Get list sentence indices implied by the transition set.
    indices = [s2i(t.sents[0]) for t in recover_order(transitions)[:-1]]
    # Get gold indices.
    gold = [i for i in range(len(indices))]
    # Compute Kendall's tau for these two sequences.
    kt, pval = sp.stats.kendalltau(indices, gold)
    return kt, pval


def mac_avg_bigram_acc(dataY):
    """
    Computes the macro average bigram overlap (accuracy) for list of
    lists of predicted Transitions.

    dataY -- A list of lists of discourse.hypergaph.Transition objects
        predicted by a discourse model.

    returns avg_acc
    """

    ndata = len(dataY)

    # If dataY is empty, return None, else return avg acc.
    if ndata == 0:
        return None
    sum_acc = 0
    for y in dataY:
        acc = bigram_acc(y)
        sum_acc += acc
    avg_acc = sum_acc / float(ndata)
    return avg_acc


def bigram_acc(transitions):
    """
    Compute the bigram overlap (accuracy) for a list of predicted
    Transitions.

    transitions -- A list of discourse.hypergaph.Transition objects.

    returns bigram overlap (accuracy)
    """
    ntrans = len(transitions)
    # Get predicted bigrams.
    pred_bg = set([(s2i(t.sents[1]), s2i(t.sents[0], end='end'))
                   for t in recover_order(transitions)])

    # Create gold bigrams.
    gold = set([(i, i+1) for i in range(-1, ntrans - 2)])
    gold.add((ntrans - 2, 'end'))

    # If either sets are empty return None.
    if len(pred_bg) == 0 or len(gold) == 0:
        return None

    nbigrams = len(gold)
    acc = len(pred_bg & gold) / float(nbigrams)
    return acc


def _position2transition_map(transitions):
    """
    Return a dict mapping transition tail sentence indices to
    transitions.

    transitions -- A list of discourse.hypergaph.Transition objects.
    """
    m = OrderedDict()
    for t in transitions:
        m[s2i(t.sents[1])] = t
    return m
