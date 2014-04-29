APWS_TEXT=${DISCOURSE_DATA:?"Set DISCOURSE_DATA environment variable."}/apws/text
APWS_CNLP=$DISCOURSE_DATA/apws/corenlp

#python parse_txt.py -id $APWS_TEXT/train -od $APWS_CNLP/train
#python parse_txt.py -id $APWS_TEXT/test -od $APWS_CNLP/test

APWS_TEXT_FLTR=$DISCOURSE_DATA/apws/filtered_text
APWS_CNLP_FLTR=$DISCOURSE_DATA/apws/filtered_corenlp

python parse_txt.py -id $APWS_TEXT_FLTR/train -od ~/sentiment/train   #-od $APWS_CNLP_FLTR/train
python parse_txt.py -id $APWS_TEXT_FLTR/test -od ~/sentiment/test    #-od $APWS_CNLP_FLTR/test

NTSB_TEXT=${DISCOURSE_DATA}/ntsb/text
NTSB_CNLP=$DISCOURSE_DATA/ntsb/corenlp

#python parse_txt.py -id $NTSB_TEXT/train -od $NTSB_CNLP/train
#python parse_txt.py -id $NTSB_TEXT/test -od $NTSB_CNLP/test

NTSB_TEXT_FLTR=$DISCOURSE_DATA/ntsb/filtered_text
NTSB_CNLP_FLTR=$DISCOURSE_DATA/ntsb/filtered_corenlp

#python parse_txt.py -id $NTSB_TEXT_FLTR/train -od $NTSB_CNLP_FLTR/train
#python parse_txt.py -id $NTSB_TEXT_FLTR/test -od $NTSB_CNLP_FLTR/test
