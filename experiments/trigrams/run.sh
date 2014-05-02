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

#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l perceptron -lf 01 -inf gurobi -n 2 -dbg
#
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l perceptron -lf 01 -inf gurobi -n 3 -dbg
#
#
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l sg-ssvm -lf 01 -inf gurobi -n 2 -dbg
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l sg-ssvm -lf hamming-node -inf gurobi -n 2 -dbg
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l sg-ssvm -lf hamming-edge -inf gurobi -n 2 -dbg
#
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l sg-ssvm -lf 01 -inf gurobi -n 3 -dbg
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l sg-ssvm -lf hamming-node -inf gurobi -n 3 -dbg
#python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_DEV -od models/apws_models -l sg-ssvm -lf hamming-edge -inf gurobi -n 3 -dbg


python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models2 -l perceptron -lf 01 -inf glpk -n 2

# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l perceptron -lf 01 -inf gurobi -n 3


# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l sg-ssvm -lf 01 -inf gurobi -n 2
# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l sg-ssvm -lf hamming-node -inf gurobi -n 2
# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l sg-ssvm -lf hamming-edge -inf gurobi -n 2

# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l sg-ssvm -lf 01 -inf gurobi -n 3
# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l sg-ssvm -lf hamming-node -inf gurobi -n 3
# python train_models.py -train $APWS_TRAIN -dev $APWS_DEV -test $APWS_TEST -od models/apws_models -l sg-ssvm -lf hamming-edge -inf gurobi -n 3





#for ntpcs in `seq 20 20 60`3
#do
#    python train_models.py -train $APWS_TRAIN -test $APWS_DEV -od models/topics/apws_models -t $CS_DIR/apws_cvem_seq_n${ntpcs}m4_final.txt
#done

#for ntpcs in `seq 20 20 60`;
#do
#    echo $ntpcs
#    python train_models.py -train $NTSB_TRAIN -test $NTSB_DEV -od models/topics/ntsb_models -t $CS_DIR/ntsb_cvem_seq_n${ntpcs}m4_final.txt
#done
#
#
#for ntpcs in `seq 20 20 100`;
#do
#    python train_models.py -train $NTSBF_TRAIN -test $NTSBF_DEV -od models/topics/ntsbf_models -t $CS_DIR/ntsbf_cvem_seq_n${ntpcs}m4_final.txt
#done
#
#
#for ntpcs in `seq 20 20 100`;
#do
#    python train_models.py -train $APWSF_TRAIN -test $APWSF_DEV -od models/topics/apwsf_models -t $CS_DIR/apwsf_cvem_seq_n${ntpcs}m4_final.txt
#done


#python train_models.py -train $APWS_TRAIN -test $APWS_DEV -od models/apws_models
#python train_models.py -train $NTSBF_TRAIN -test $NTSBF_DEV -od models/ntsbf_models
#python train_models.py -train $NTSB_TRAIN -test $NTSB_DEV -od models/ntsb_models
#python train_models.py -train $APWSF_TRAIN -test $APWSF_DEV -od models/apwsf_models
