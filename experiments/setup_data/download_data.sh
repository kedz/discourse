${DISCOURSE_DATA:?"Set DISCOURSE_DATA environment variable."}

OLDLOC=`pwd`

if [ ! -d "$DISCOURSE_DATA" ]; then
    mkdir -p $DISCOURSE_DATA
fi

cd $DISCOURSE_DATA
echo "Downloading Barzilay ntsb permutations..."
wget http://people.csail.mit.edu/regina/coherence/data2-train-perm.tar.Z
wget http://people.csail.mit.edu/regina/coherence/CLsubmission/ntsb-test.tgz
tar zxf data2-train-perm.tar.Z
tar zxf ntsb-test.tgz
mkdir -p ntsb/perm
mv training ntsb/perm/train
mv ntsb-test ntsb/perm/test

echo "Downloading Barzilay apws permutations..."
wget http://people.csail.mit.edu/regina/coherence/data1-train-perm.tar
wget http://people.csail.mit.edu/regina/coherence/CLsubmission/data1-test.tgz
tar xf data1-train-perm.tar
tar zxf data1-test.tgz  
mkdir -p apws/perm
mv training apws/perm/train
mv data1-test apws/perm/test

rm data1-train-perm.tar ntsb-test.tgz data2-train-perm.tar.Z data1-test.tgz

cd $OLDLOC

echo "Permutations downloaded."
