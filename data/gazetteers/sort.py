
names = set()
for line in open('female.txt'):
    names.add(line.strip())

nlist = list(names)
nlist.sort()
for n in nlist:
    print n

