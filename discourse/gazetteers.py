from os import getenv
from os.path import join

male_fnames = None
female_fnames = None

class MaleNames:
    def __init__(self, lc=False):
        global male_fnames
        if male_fnames == None:
            fname = join(getenv('DISCOURSEDATA','.'), 'gazetteers', 'male.txt')
            f = open(fname, 'r')
            male_fnames = set()
            for line in f:
                if lc:
                    male_fnames.add(line.strip().lower())
                else:
                    male_fnames.add(line.strip())
            f.close()
        self._names = male_fnames

    def __iter__(self):
        return iter(self._names)

class FemaleNames:
    def __init__(self, lc=False):
        global female_fnames
        if female_fnames == None:
            fname = join(getenv('DISCOURSEDATA','.'), 'gazetteers', 'female.txt')
            f = open(fname, 'r')
            female_fnames = set()
            for line in f:
                if lc:
                    female_fnames.add(line.strip().lower())
                else:
                    female_fnames.add(line.strip())
            f.close()
        self._names = female_fnames

    def __iter__(self):
        return iter(self._names)

