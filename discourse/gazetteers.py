from os import getenv
from os.path import join
import re

male_fnames = None
female_fnames = None


class DiscourseConnectives:
    def __init__(self):
        ellipses = []
        no_ellipses = []

        fname = join(getenv('DISCOURSEDATA', '.'),
                     'gazetteers', 'explicit_disc.txt')
        f = open(fname, 'r')
        for line in f:
            if '...' not in line:
                no_ellipses.append(line.strip())
            else:
                splits = line.strip().split('...')
                restr = '.*?'.join(['({})'.format(s.strip()) for s in splits])
                ellipses.append(restr)
        f.close()
        self.p2 = re.compile(r'\b({})\b'.format('|'.join(no_ellipses)), re.I)

        self.p1 = [re.compile(restr, re.I) for restr in ellipses]

    def contains_connective(self, a_string):
        for patt in self.p1:
            m = patt.search(a_string)
            if m:
                return (m.group(1).lower(), m.group(2).lower())
        m = self.p2.search(a_string)
        if m:
            return m.group(0).lower()
        else:
            return None


class MaleNames:
    def __init__(self, lc=False):
        global male_fnames
        if male_fnames is None:
            fname = join(getenv('DISCOURSEDATA', '.'),
                         'gazetteers', 'male.txt')
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
        if female_fnames is None:
            fname = join(getenv('DISCOURSEDATA', '.'),
                         'gazetteers', 'female.txt')
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
