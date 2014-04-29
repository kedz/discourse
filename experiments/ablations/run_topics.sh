APWSF_TRAIN=$DISCOURSE_DATA/apws/filtered_corenlp/train
APWSF_DEV=$DISCOURSE_DATA/apws/filtered_corenlp/dev

APWS_TRAIN=$DISCOURSE_DATA/apws/corenlp/train
APWS_DEV=$DISCOURSE_DATA/apws/corenlp/dev

NTSBF_TRAIN=$DISCOURSE_DATA/ntsb/filtered_corenlp/train
NTSBF_DEV=$DISCOURSE_DATA/ntsb/filtered_corenlp/dev

NTSB_TRAIN=$DISCOURSE_DATA/ntsb/corenlp/train
NTSB_DEV=$DISCOURSE_DATA/ntsb/corenlp/dev

CS_DIR=$DISCOURSE_DATA/cluster_sequences

for ntpcs in `seq 20 20 60`;
do
    echo $ntpcs
    python train_models.py -train $NTSB_TRAIN -test $NTSB_DEV -od models/topics/ntsb_models -t $CS_DIR/ntsb_cvem_seq_n${ntpcs}m4_final.txt
done

for ntpcs in `seq 20 20 60`;
do
    python train_models.py -train $APWS_TRAIN -test $APWS_DEV -od models/topics/apws_models -t $CS_DIR/apws_cvem_seq_n${ntpcs}m4_final.txt
done

for ntpcs in `seq 20 20 100`;
do
    python train_models.py -train $NTSBF_TRAIN -test $NTSBF_DEV -od models/topics/ntsbf_models -t $CS_DIR/ntsbf_cvem_seq_n${ntpcs}m4_final.txt
done


for ntpcs in `seq 20 20 100`;
do
    python train_models.py -train $APWSF_TRAIN -test $APWSF_DEV -od models/topics/apwsf_models -t $CS_DIR/apwsf_cvem_seq_n${ntpcs}m4_final.txt
done


#python train_models.py -train $APWS_TRAIN -test $APWS_DEV -od models/apws_models
#python train_models.py -train $NTSBF_TRAIN -test $NTSBF_DEV -od models/ntsbf_models
#python train_models.py -train $NTSB_TRAIN -test $NTSB_DEV -od models/ntsb_models
#python train_models.py -train $APWSF_TRAIN -test $APWSF_DEV -od models/apwsf_models
