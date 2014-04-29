# This script downloads the discourse ordering data used in the Barzilay
# entity grid experiments and extracts text of the gold orderings and
# processes it using the stanford corenlp java library.
# If you do not have the DISCOURSE_DATA environment variable set,
# uncomment the following line and set the location where you would like
# to keep the data:
#DISCOURSE_DATA=path/to/data

if [ ${DISCOURSE_DATA:+x} ]
    then echo "Data will be install under the directory: $DISCOURSE_DATA"
        # Download Barzilay's permutation data
        bash download_data.sh

        # Extract gold ordering text
        bash perms2text.sh

        # Process text using the stanford corenlp library.
        bash parse_texts.sh 

        # Split off 20 random docs from training sets to create a development
        # set.
        bash make_dev_sets.sh

    else echo "DISCOURSE_DATA is not set or is empty"
fi



