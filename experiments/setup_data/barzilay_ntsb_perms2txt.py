import argparse
import os
import codecs
import re
import sys

def parse_cmdline():

    parser = argparse.ArgumentParser()
    parser.add_argument(u'-perm', u'--ntsb-perm',
                        help=u'Location of the ntsb ' +
                             u'permutation directory.',
                        type=unicode, required=True)
        
    parser.add_argument(u'-text', u'--ntsb-text', 
                        help=u'Location of ntsb raw text directory.',
                        type=unicode, required=True)

    parser.add_argument(u'-f', u'--filter',
                        help=u'Filter out headers and footer text.',
                        action=u'store_true', default=False)

    args = parser.parse_args()
    permdir = args.ntsb_perm
    textdir = args.ntsb_text
    use_filter = args.filter

    train_permdir = os.path.join(permdir, u'train')
    test_permdir = os.path.join(permdir, u'test')

    if not os.path.exists(train_permdir) or not os.path.isdir(train_permdir):
        import sys
        sys.stderr.write((u'{} either does not exits ' +
                          u'or is not a directory.\n').format(train_permdir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(test_permdir) or not os.path.isdir(test_permdir):
        import sys
        sys.stderr.write((u'{} either does not exits ' +
                          u'or is not a directory.\n').format(test_permdir))
        sys.stderr.flush()
        sys.exit()

    train_textdir = os.path.join(textdir, u'train')
    if not os.path.exists(train_textdir):
        os.makedirs(train_textdir)
    test_textdir = os.path.join(textdir, u'test')
    if not os.path.exists(test_textdir):
        os.makedirs(test_textdir)

    o = (train_permdir, test_permdir, train_textdir, test_textdir, use_filter)
    return o

def process(src, tgt):

    print u'Extracting text from {} --> {}'.format(src, tgt)
    nfiles = len(os.listdir(src))
    for i, fname in enumerate(os.listdir(src), 1):

        per = 100 * float(i) / nfiles
        sys.stdout.write(u'Percent Complete: {:2.3f}%\r'.format(per))
        if i == nfiles:
            sys.stdout.write(u'\n')
        sys.stdout.flush()        
            
        if u'perm' not in fname or fname.endswith(u'perm-1'):
            path = os.path.join(src, fname)
            textfname = os.path.split(fname)[1] + u'.txt'
            textfname = textfname.replace(u'.perm-1', '')
            textpath = os.path.join(tgt, textfname)
                        
            with codecs.open(path, u'r', u'utf-8') as f:
                with codecs.open(textpath, u'w', u'utf-8') as out:
                    for line in f:
                        text = line.split(u' ', 1)[-1]
                        out.write(text)
                        out.flush()

def process_filtered(src, tgt):

    bad1 = r'This is preliminary information, subject to change, and may ' \
           r'contain errors\.|' \
           r'Any errors in this report will be corrected when the final ' \
           r'report has been completed\.'
    bad2 = r'investigation is under|' \
           r'Any further information|' \
           r'This report is filed for|' \
           r'For further information, contact' 

    print u'Extracting filtered text from {} --> {}'.format(src, tgt)
    nfiles = len(os.listdir(src))
    for i, fname in enumerate(os.listdir(src), 1):
        per = 100 * float(i) / nfiles
        sys.stdout.write(u'Percent Complete: {:2.3f}%\r'.format(per))
        if i == nfiles:
            sys.stdout.write(u'\n')
        sys.stdout.flush()        
            
        if u'perm' not in fname or fname.endswith(u'perm-1'):
            path = os.path.join(src, fname)
            textfname = os.path.split(fname)[1] + u'.txt'
            textfname = textfname.replace(u'.perm-1', '')
            textpath = os.path.join(tgt, textfname)
                        
            with codecs.open(path, u'r', u'utf-8') as f:
                with codecs.open(textpath, u'w', u'utf-8') as out:
                    for line in f:
                        if re.search(bad2, line):
                            break
                        if not re.search(bad1, line):
                            text = line.split(u' ', 1)[-1]
                            out.write(text)
                            out.flush()

def main():
    """Create raw text for training/test data from the
    permutation files. Optionally filter header/footer text."""

    args = parse_cmdline()
    train_permdir = args[0]
    test_permdir = args[1]
    train_textdir = args[2]
    test_textdir = args[3]
    use_filter = args[4]

    if use_filter:
        process_filtered(train_permdir, train_textdir)
        process_filtered(test_permdir, test_textdir)
    else:
        process(train_permdir, train_textdir)
        process(test_permdir, test_textdir)

if __name__ == '__main__':
    main()
