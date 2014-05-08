APWSF_TRAIN=$DISCOURSE_DATA/apws/filtered_corenlp/train
APWSF_DEV=$DISCOURSE_DATA/apws/filtered_corenlp/dev
APWSF_TEST=$DISCOURSE_DATA/apws/filtered_corenlp/test

APWS_TRAIN=$DISCOURSE_DATA/apws/corenlp/train
APWS_DEV=$DISCOURSE_DATA/apws/corenlp/dev
APWS_TEST=$DISCOURSE_DATA/apws/corenlp/test

NTSBF_TRAIN=$DISCOURSE_DATA/ntsb/filtered_corenlp/train
NTSBF_DEV=$DISCOURSE_DATA/ntsb/filtered_corenlp/dev
NTSBF_TEST=$DISCOURSE_DATA/ntsb/filtered_corenlp/test

NTSB_TRAIN=$DISCOURSE_DATA/ntsb/corenlp/train
NTSB_DEV=$DISCOURSE_DATA/ntsb/corenlp/dev
NTSB_TEST=$DISCOURSE_DATA/ntsb/corenlp/test

CS_DIR=$DISCOURSE_DATA/cluster_sequences

python -u train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/bigram/apws_models -l perceptron -lf 01 -inf beam -n 2 -dbg
python -u train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/bigram/apws_models -l perceptron -lf 01 -inf beam -n 2 
python -u train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/bigram/apws_models -l perceptron -lf 01 -inf beam -n 2 -t $CS_DIR/apws

python -u train_models.py -train $NTSB_TRAIN -dev $NTSB_DEV -test $NTSB_TEST -od models/bigram/ntsb_models -l perceptron -lf 01 -inf beam -n 2 -dbg
python -u train_models.py -train $NTSB_TRAIN -dev $NTSB_DEV -test $NTSB_TEST -od models/bigram/ntsb_models -l perceptron -lf 01 -inf beam -n 2
python -u train_models.py -train $NTSB_TRAIN -dev $NTSB_DEV -test $NTSB_TEST -od models/bigram/ntsb_models -l perceptron -lf 01 -inf beam -n 2 -t $CS_DIR/ntsb
