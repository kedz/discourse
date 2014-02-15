from os import getenv, remove, listdir, makedirs
from os.path import join, isdir
from collections import namedtuple, defaultdict
from tempfile import NamedTemporaryFile
from subprocess import check_output
import pandas as pd
import re
#from gensim.matutils import Sparse2Corpus
#from gensim.models.lsimodel import LsiModel
from scipy.sparse import coo_matrix
import corenlp_xml as cnlp

# Location of training and test for apws and ntsb corpora.
_DATA_DIR = getenv("DISCOURSEDATA", ".")

# Location of Brown Coherence Toolkit bin directory.
_COHERENCE_BIN = getenv("BROWNCOHERENCEPATH", ".")

# Command used to generate entity grid.
_TEST_GRID_CMD = join(_COHERENCE_BIN, "TestGrid")

# Directories of xml output of stanford corenlp for test and training data.
_XML_NTSB_TRAIN = join(_DATA_DIR, 'xml', 'ntsb',  'train')
_XML_NTSB_TEST = join(_DATA_DIR, 'xml', 'ntsb', 'test')
_XML_APWS_TRAIN = join(_DATA_DIR, 'xml', 'apws', 'train')
_XML_APWS_TEST = join(_DATA_DIR, 'xml', 'apws', 'test')

# Gold pretty sentence ordering files.
_GOLD_NTSB_ORDERINGS = join(_DATA_DIR, "ntsb.gold.txt")
_GOLD_APWS_ORDERINGS = join(_DATA_DIR, "apws.gold.txt")

# Directories with Regina's original grids.
_BARZILAY_GRIDS_APWS_TRAIN = join(_DATA_DIR,
                                  "barzilay", "grids", "train", "apws")
_BARZILAY_GRIDS_APWS_TEST = join(_DATA_DIR,
                                 "barzilay", "grids", "test", "apws")
_BARZILAY_GRIDS_NTSB_TRAIN = join(_DATA_DIR,
                                  "barzilay", "grids", "train", "ntsb")
_BARZILAY_GRIDS_NTSB_TEST = join(_DATA_DIR,
                                 "barzilay", "grids", "test", "ntsb")

_RAW_TXT_APWS_TRAIN = join(_DATA_DIR,
                           'raw',
                           'apws',
                           'train')

_RAW_TXT_APWS_TEST = join(_DATA_DIR,
                           'raw',
                           'apws',
                           'test')

_RAW_TXT_NTSB_TRAIN = join(_DATA_DIR,
                           'raw',
                           'ntsb',
                           'train')

_RAW_TXT_NTSB_TEST = join(_DATA_DIR,
                           'raw',
                           'ntsb',
                           'test')



def apws_train_xml_dir():
    if not isdir(_XML_APWS_TRAIN):
        makedirs(_XML_APWS_TRAIN)    
    return _XML_APWS_TRAIN

def apws_test_xml_dir():
    if not isdir(_XML_APWS_TEST):
        makedirs(_XML_APWS_TEST)    
    return _XML_APWS_TEST

def ntsb_train_xml_dir():
    if not isdir(_XML_NTSB_TRAIN):
        makedirs(_XML_NTSB_TRAIN)    
    return _XML_NTSB_TRAIN

def ntsb_test_xml_dir():
    if not isdir(_XML_NTSB_TEST):
        makedirs(_XML_NTSB_TEST)    
    return _XML_NTSB_TEST



def raw_apws_train_files():
    for f in listdir(_RAW_TXT_APWS_TRAIN):
        yield join(_RAW_TXT_APWS_TRAIN, f)

def raw_apws_test_files():
    for f in listdir(_RAW_TXT_APWS_TEST):
        yield join(_RAW_TXT_APWS_TEST, f)

def raw_ntsb_train_files():
    for f in listdir(_RAW_TXT_NTSB_TRAIN):
        yield join(_RAW_TXT_NTSB_TRAIN, f)

def raw_ntsb_test_files():
    for f in listdir(_RAW_TXT_NTSB_TEST):
        yield join(_RAW_TXT_NTSB_TEST, f)






def gold_ntsb_file():
    return _GOLD_NTSB_ORDERINGS


def gold_apws_file():
    return _GOLD_APWS_ORDERINGS


def corenlp_ntsb_train():
    return _xml_file_generator(_XML_NTSB_TRAIN)


def corenlp_ntsb_test():
    return _xml_file_generator(_XML_NTSB_TEST)


def corenlp_apws_train():
    return _xml_file_generator(_XML_APWS_TRAIN)


def corenlp_apws_test():
    return _xml_file_generator(_XML_APWS_TEST)


def _xml_file_generator(xml_dir):

    for f in sorted(listdir(xml_dir)):
        yield join(xml_dir, f)


def apws_barzilay_grids_train():
    return _grid_dataframe_generator(_BARZILAY_GRIDS_APWS_TRAIN)


def apws_barzilay_grids_test():
    return _grid_dataframe_generator(_BARZILAY_GRIDS_APWS_TEST)


def ntsb_barzilay_grids_train():
    return _grid_dataframe_generator(_BARZILAY_GRIDS_NTSB_TRAIN)


def ntsb_barzilay_grids_test():
    return _grid_dataframe_generator(_BARZILAY_GRIDS_NTSB_TEST)


def _grid_dataframe_generator(data_dir):
    """Yield tuples of
        (grid_file, list_of_permuted_grid_files)."""
    inst_map = defaultdict(dict)

    pattern = re.compile('(.*)perm-(\d+)-')

    for f in listdir(data_dir):
        filename = join(data_dir, f)
        m = pattern.search(filename)
        if m:
            inst_map[m.groups()[0]][m.groups()[1]] = filename
    keys = [k for k in inst_map]
    keys = sorted(keys)

    for key in keys:
        original_file = inst_map[key]['1']

        del inst_map[key]['1']
        yield (original_file, inst_map[key].values())


class CoherenceInstance(namedtuple("CoherenceInstance",
                        ["parses", "dataframe"])):
    """Each train/test CoherenceInstance is a list of
        parsed sentences in the correct order, and a
        pandas DataFrame for storing the entity grid."""

    def __str__(self):
        return "CoherenceInstance:\n{}\n{}".format(self.parses, self.dataframe)


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

def get_ntsb_gensim(k=50):
    return get_gensim([xml for xml in data.corenlp_ntsb_train()], k=k)

def get_apws_gensim():
    return get_gensim([xml for xml in data.corenlp_apws_train()], k=k)


def get_gensim(xmls, k=50):
    
    w2i = {}
    windex = 0
    d = []
    rows = []
    cols = []
    row = 0

    for xml in xmls:
        for s in cnlp.Document(xml):
            counts = defaultdict(float)
            
            for t in s:
                if t.pos not in ['.', ',', '\'', ';', ':', '`', '"', '\'\'', '``', '-LRB-', '-RRB-', 'POS', 'CD']:
                    lem = t.lem.lower()
                    #print lem+'_{}'.format(t.pos), 
                    
                    if lem not in w2i:
                        w2i[lem] = windex
                        windex += 1
                    
                    counts[w2i[lem]] += 1 
            
            for item in counts.items():
        
                rows.append(row)
                cols.append(item[0])
                d.append(item[1])
                
        row += 1

    D = coo_matrix((d, (rows, cols)))
    C = Sparse2Corpus(D, documents_columns=False)
    lsi = LsiModel(corpus=C, num_topics=k)
    return (w2i, lsi)
