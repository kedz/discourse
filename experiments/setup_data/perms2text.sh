APWS_PERM=$DISCOURSE_DATA/apws/perm/
NTSB_PERM=$DISCOURSE_DATA/ntsb/perm/

APWS_TEXT=$DISCOURSE_DATA/apws/text/
NTSB_TEXT=$DISCOURSE_DATA/ntsb/text/

python barzilay_apws_perms2txt.py -perm $APWS_PERM -text $APWS_TEXT
python barzilay_ntsb_perms2txt.py -perm $NTSB_PERM -text $NTSB_TEXT

APWS_TEXT_FLTR=$DISCOURSE_DATA/apws/filtered_text/
NTSB_TEXT_FLTR=$DISCOURSE_DATA/ntsb/filtered_text/

python barzilay_apws_perms2txt.py -perm $APWS_PERM -text $APWS_TEXT_FLTR -f
python barzilay_ntsb_perms2txt.py -perm $NTSB_PERM -text $NTSB_TEXT_FLTR -f
