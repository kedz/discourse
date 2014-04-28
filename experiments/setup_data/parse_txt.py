import argparse
import os
import codecs
import re
import sys
from corenlp.pipeline import dir2dir

def _parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-id', u'--input-dir',
                        help=u'Text input directory.',
                        type=unicode, required=True)

    parser.add_argument(u'-od', u'--output-dir',
                        help=u'XML output directory',
                        type=unicode, required=True)

    args = parser.parse_args()
    indir = args.input_dir
    outdir = args.output_dir

    if not os.path.exists(indir) or not os.path.isdir(indir):
        import sys
        sys.stderr.write((u'{} either does not exits ' +
                          u'or is not a directory.\n').format(indir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return indir, outdir

def main():
    """Parse all txt files in input directory using the
    Stanford Corenlp library."""

    indir, outdir = _parse_cmdline()
    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner',
                  'parse', 'dcoref', 'sentiment']
    dir2dir(indir, outdir, annotators=annotators, mem=u'6G')

if __name__ == '__main__':
    main()
