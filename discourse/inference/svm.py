import random
import math
import numpy as np
import pandas as pd
import discourse.models.entity_grid as eg
import svmlight
from tempfile import NamedTemporaryFile
from os import remove


class RankSVMTrainer:
    """Trains an entity grid coherence model using a ranking svm."""

    def __init__(self, random_perms=20, normalized=True):

        # Number of random permutations of train/tests document to generate
        self._num_perms = random_perms

        # Training instances in format
        # (<label>, [(<feature>, <value>), ...], <queryid>)
        self._training_data = []

        # Flag for using probabilties vs counts
        self._normalized = normalized

        # Each model/permutation pair gets a unique query id
        # in the ranking algorithm
        self._queryid = 0

        # Model weights - when learned will be numpy array
        self.weights = None

        # Model weights as data frame with feature labels for pretty printing
        self.weights_df = None

        # Feature labels i.e. transition features
        self._feature_labels = None

    def add_model(self, model, permutations=None):
        """Add an entity grid model to the training data.
            If permutations is none, generate random
            permutations of this model."""

        # Get model feature lables
        if self._feature_labels is None:
            self._feature_labels = model.labels

        if permutations is None:
            permutations = permute_model(model, self._num_perms)

        # For each random permutation pmodel, make a training instance pair
        # where original model is ranked higher than the perm
        for pmodel in permutations:
            self._queryid += 1
            if self._normalized:

                model_features = [(f, x) for f, x
                                  in enumerate(model.get_trans_prob_vctr(),
                                               start=1)]
                model_inst = (2, model_features, self._queryid)
                self._training_data.append(model_inst)

                pmodel_features = [(f, x) for f, x
                                   in enumerate(pmodel.get_trans_prob_vctr(),
                                                start=1)]
                pmodel_inst = (1, pmodel_features, self._queryid)
                self._training_data.append(pmodel_inst)
            else:

                model_features = [(f, x) for f, x
                                  in enumerate(model.get_trans_cnt_vctr(),
                                               start=1)]
                model_inst = (2, model_features, self._queryid)
                self._training_data.append(model_inst)

                pmodel_features = [(f, x) for f, x
                                   in enumerate(pmodel.get_trans_cnt_vctr(),
                                                start=1)]
                pmodel_inst = (1, pmodel_features, self._queryid)
                self._training_data.append(pmodel_inst)

    def train(self):
        """Learn model weights from training instances."""

        # Train using svmlight
        self._svmmodel = svmlight.learn(self._training_data, type='ranking')

        # Write svmlight output to a temp file and recover weights
        modelout = NamedTemporaryFile(delete=False)
        svmlight.write_model(self._svmmodel, modelout.name)
        modelout.close()
        self._recover_weights(modelout.name)
        remove(modelout.name)

    def _recover_weights(self, modelfile):
        """Recover feature weights from svmlight model output."""

        # Parse svm modelfile
        lines = open(modelfile, "r").readlines()
        num_feats = int(lines[7].split(' ')[0])
        weights = [0]*num_feats

        for i in range(11, len(lines)):
            alpha = float(lines[i].split(' ')[0])
            for pair in lines[i].split(' ', 1)[1].split('#')[0].split(' '):
                if ':' in pair:
                    f = int(pair.split(':')[0]) - 1
                    x = float(pair.split(':')[1])
                    weights[f] += alpha*x

        # Store weights as numpy array and dataframe for pretty printing.
        self.weights = np.asarray(weights)
        self.weights_df = pd.DataFrame(np.asmatrix(weights),
                                       columns=self._feature_labels,
                                       index=['w'])


def permute_model(model, num_perms):
    """Generate random permutations of a model."""
    # Create the original ordered sequence: 0 1 2 ... n-1.
    ordered_sent = [i for i in range(0, len(model.sentences))]

    # Set the number of permutations to make.
    # If the total possible document permutatiosn is smaller than num_perms
    # set that as the max, since we can't possibly make any more.
    maxperms = min(math.factorial(len(ordered_sent))-1, num_perms)

    # Set up our set of permutations. Initially this holds the original
    # sequence so that we don't add the correct sequence even if it is
    # randomly generated.
    perms = set()
    perms.add(tuple(ordered_sent))

    # Make new permutations until we hit our limit.
    rsent = list(ordered_sent)
    while len(perms) < maxperms + 1:
        random.shuffle(rsent)
        perms.add(tuple(rsent))
    perms.remove(tuple(ordered_sent))

    # For each permutation, create a permuted entity grid model.
    pmodels = []
    for perm in perms:

        psentences = [model.sentences[i] for i in perm]
        grids = []

        for grid in model.grids:
            orig = np.asarray(grid)
            pmat = [[orig[i][j] for j in perm]
                    for i in range(orig.shape[0])]
            grids.append(pd.DataFrame(pmat, index=grid.index,
                                      columns=grid.columns))

        perm_model = eg.EntityGrid(grids,
                                   model.trans,
                                   psentences,
                                   model.history)
        pmodels.append(perm_model)

    # Return the set of permuted entity grid models.
    return pmodels
