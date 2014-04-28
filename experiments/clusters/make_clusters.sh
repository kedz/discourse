APWSF_TRAIN=$DISCOURSE_DATA/apws/filtered_corenlp/train
APWSF_TEST=$DISCOURSE_DATA/apws/filtered_corenlp/test

APWS_TRAIN=$DISCOURSE_DATA/apws/corenlp/train
APWS_TEST=$DISCOURSE_DATA/apws/corenlp/test

NTSBF_TRAIN=$DISCOURSE_DATA/ntsb/filtered_corenlp/train
NTSBF_TEST=$DISCOURSE_DATA/ntsb/filtered_corenlp/test

NTSB_TRAIN=$DISCOURSE_DATA/ntsb/corenlp/train
NTSB_TEST=$DISCOURSE_DATA/ntsb/corenlp/test

CS_DIR=$DISCOURSE_DATA/cluster_sequences
C_DIR=$DISCOURSE_DATA/clusters


for size in `seq 20 20 100`;
do
    echo "APWS: Initial clustering, $size target topics and min size threshold 4" 
    python cluster.py -train $APWS_TRAIN -cf $C_DIR/apws_clusters_n${size}m4.txt -of $CS_DIR/apws_c_seq_n${size}m4.txt -n $size -m 4
done    

echo "Done!"


for size in `seq 20 20 100`;
do
    echo "APWS-f: Initial clustering, $size target topics and min size threshold 4" 
    python cluster.py -train $APWSF_TRAIN -cf $C_DIR/apwsf_clusters_n${size}m4.txt -of $CS_DIR/apwsf_c_seq_n${size}m4.txt -n $size -m 4
done    

echo "Done!"

for size in `seq 20 20 100`;
do
    echo "NTSB: Initial clustering, $size target topics and min size threshold 4" 
    python cluster.py -train $NTSB_TRAIN -cf $C_DIR/ntsb_clusters_n${size}m4.txt -of $CS_DIR/ntsb_c_seq_n${size}m4.txt -n $size -m 4

done    

echo "Done!"


for size in `seq 20 20 100`;
do
    echo "NTSB-f: Initial clustering, $size target topics and min size threshold 4" 
    python cluster.py -train $NTSBF_TRAIN -cf $C_DIR/ntsbf_clusters_n${size}m4.txt -of $CS_DIR/ntsbf_c_seq_n${size}m4.txt -n $size -m 4

done    

echo "Done!"
