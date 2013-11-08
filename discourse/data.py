from os import getenv, remove
from os.path import join
from collections import namedtuple
from tempfile import NamedTemporaryFile
from subprocess import check_output
import pandas as pd

# Location of training and test for apws and ntsb corpora.
_DATA_DIR = getenv("DISCOURSEDATA", ".")

# Location of Brown Coherence Toolkit bin directory.
_COHERENCE_BIN = getenv("BROWNCOHERENCEPATH", ".")

# Command used to generate entity grid.
_TEST_GRID_CMD = join(_COHERENCE_BIN, "TestGrid")


class CoherenceInstance(namedtuple("CoherenceInstance",
                        ["parses", "dataframe"])):
    """Each train/test CoherenceInstance is a list of
        parsed sentences in the correct order, and a
        pandas DataFrame for storing the entity grid."""

    def __str__(self):
        return "CoherenceInstance:\n{}\n{}".format(self.parses, self.dataframe)


def _parse_grid_string(grid_str):
    """Parses stdout of TestGrid from the Brown Coherence Toolkit into
        a pandas DataFrame."""

    lines = grid_str.strip().split("\n")

    # A list of lists holding entity transitions.
    # This will become the data in a pandas
    # DataFrame representing the entity grid.
    egrid = []

    # A list of entities in this document.
    # This will become the index of a pandas
    # DataFrame representing the entity grid.
    entities = []

    for line in lines:

        if (line.strip() is not ""):

            # Each entry is separated by a space.
            roles = line.strip().split(" ")

            # The first entry is the entity name. Add this to the index.
            entities.append(roles[0].decode("ascii", "ignore"))

            # Add the transitions (these start at index 1)
            # for this entity to egrid.
            roles = [roles[t].lower() for t in range(1, len(roles))]
            egrid.append(roles)

    egrid_df = pd.DataFrame(egrid, index=entities)

    return egrid_df


def ntsb_train():
    """Return the next instance in the NTSB training data."""

    return _instance_generator(join(_DATA_DIR, "ntsb_train.txt"))


def ntsb_test():
    """Return the next instance in the NTSB testing data."""

    return _instance_generator(join(_DATA_DIR, "ntsb_test.txt"))


def apws_train():
    """Return the next instance in the APWS training data."""

    return _instance_generator(join(_DATA_DIR, "apws_train.txt"))


def apws_test():
    """Return the next instance in the APWS testing data."""

    return _instance_generator(join(_DATA_DIR, "apws_test.txt"))


def _instance_generator(instances_file):
    """Yields the next CoherenceInstance from instances_file."""

    # List of parses
    parse_list = []

    # Temp file to put parses to be read by TestGrid.
    tmpFile = NamedTemporaryFile(delete=False)

    # Read in lines until finding a blank line.
    # Then build a CoherenceInstance from them.
    for line in open(instances_file, "r"):

        # Line is empty -- build an instance.
        if (line.strip() is ""):

            # Close our temp file with parses and run TestGrid on this file.
            # The output is a string representation of the entity grid
            # for this document.
            tmpFile.close()
            grid_str = check_output([_TEST_GRID_CMD, tmpFile.name])

            # Parse the grid_str into a DataFrame
            # and yield the next CoherenceInstance.
            yield CoherenceInstance(parse_list, _parse_grid_string(grid_str))

            # Reset parse list and temp file for next instance.
            remove(tmpFile.name)
            tmpFile = NamedTemporaryFile(delete=False)
            parse_list = []

        # Write this parse to a temp file for processing while also adding
        # this parse to a list of parses.
        else:
            tmpFile.write(line)
            parse_list.append(line.strip())
