
#${DISCOURSE_DATA:?"Set DISCOURSE_DATA environment variable."}

SEED=1986

APWS_TRAIN=$DISCOURSE_DATA/apws/corenlp/train
APWS_DEV=$DISCOURSE_DATA/apws/corenlp/dev

python make_dev_set.py -td $APWS_TRAIN -dd $APWS_DEV -s $SEED

APWSF_TRAIN=$DISCOURSE_DATA/apws/filtered_corenlp/train
APWSF_DEV=$DISCOURSE_DATA/apws/filtered_corenlp/dev

python make_dev_set.py -td $APWSF_TRAIN -dd $APWSF_DEV -s $SEED

NTSB_TRAIN=$DISCOURSE_DATA/ntsb/corenlp/train
NTSB_DEV=$DISCOURSE_DATA/ntsb/corenlp/dev

python make_dev_set.py -td $NTSB_TRAIN -dd $NTSB_DEV -s $SEED

NTSBF_TRAIN=$DISCOURSE_DATA/ntsb/filtered_corenlp/train
NTSBF_DEV=$DISCOURSE_DATA/ntsb/filtered_corenlp/dev

python make_dev_set.py -td $NTSBF_TRAIN -dd $NTSBF_DEV -s $SEED
