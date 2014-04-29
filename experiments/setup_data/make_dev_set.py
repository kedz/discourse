import argparse
import os
import shutil
import random

def main():
    """Split 20 random docs from the training set into a
    a development set."""

    tdir, ddir, seed = parse_cmdline()
    random.seed(seed)

    files = [os.path.join(tdir, fname) for fname in os.listdir(tdir)]
    random.shuffle(files)

    dev_files = files[:20]
    train_files = files[20:]

    for src in dev_files:
        bname = os.path.basename(src)
        tgt = os.path.join(ddir, bname)
        shutil.move(src, tgt)

    if len(train_files) != len(os.listdir(tdir)):
        import sys
        sys.stderr.write(u'Training directory files and the number of training')
        sys.stderr.write(u' files differs.\n')
        sys.stderr.flush()
    if len(dev_files) != len(os.listdir(ddir)):
        import sys
        sys.stderr.write(u'Development directory files and the number of ')
        sys.stderr.write(u'development files differs.\n')
        sys.stderr.flush()

    print u'Training set: {}'.format(tdir)
    print u'Training set has {} files.'.format(len(os.listdir(tdir)))
    print u'Created development set: {}'.format(ddir)
    print u'Development set has {} files.'.format(len(os.listdir(ddir)))

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-td', u'--train-dir',
                        help=u'Directory of training data.',
                        type=unicode, required=True)

    parser.add_argument(u'-dd', u'--dev-dir',
                        help=u'Location of develop directory data',
                        type=unicode, required=True)

    parser.add_argument(u'-s', u'--seed',
                        help=u'Seed for random number gen.',
                        type=int, required=True)


    args = parser.parse_args()
    tdir = args.train_dir
    ddir = args.dev_dir
    seed = args.seed

    if not os.path.exists(tdir) or not os.path.isdir(tdir):
        import sys
        sys.stderr.write((u'{} either does not exits ' +
                          u'or is not a directory.\n').format(tdir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(ddir):
        os.makedirs(ddir)

    return tdir, ddir, seed

if __name__ == u'__main__':
    main()
