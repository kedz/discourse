#for p in `ls models/topics/ntsb_models`;
#do
#    RFILE=`echo "$p" | sed -r 's/\.p/.txt/g'`
#    echo $RFILE
#    python describe_predictions.py -p models/topics/ntsb_models/$p -of results/topics/ntsb/$RFILE
#
#done

for p in `ls models/topics/apws_models`;
do
    RFILE=`echo "$p" | sed -r 's/\.p/.txt/g'`
    echo $RFILE
    python describe_predictions.py -p models/topics/apws_models/$p -of results/topics/apws/$RFILE

done

#for p in `ls models/topics/apwsf_models`;
#do
#    RFILE=`echo "$p" | sed -r 's/\.p/.txt/g'`
#    echo $RFILE
#    python describe_predictions.py -p models/topics/apwsf_models/$p -of results/topics/apwsf/$RFILE
#
#done
#
#for p in `ls models/topics/ntsbf_models`;
#do
#    RFILE=`echo "$p" | sed -r 's/\.p/.txt/g'`
#    echo $RFILE
#    python describe_predictions.py -p models/topics/ntsbf_models/$p -of results/topics/ntsbf/$RFILE
#
#done


#for p in `ls models/ntsbf_models`;
#do
#    RFILE=`echo "$p" | sed -r 's/\.p/.txt/g'`
#    echo $RFILE
#    python describe_predictions.py -p models/ntsbf_models/$p -of results/ntsbf/$RFILE

#done


#for p in `ls models/apwsf_models`;
#do
#    RFILE=`echo "$p" | sed -r 's/\.p/.txt/g'`
#    echo $RFILE
#    python describe_predictions.py -p models/apwsf_models/$p -of results/apwsf/$RFILE

#done


#python describe_predictions.py -p apwsf_models/vbz_feats.p -of results/apwsf/vbz_feats.txt
#python describe_predictions.py -p apwsf_models/apws_topics_vbz_tpc_trw_feats.p -of results/apwsf/apws_topics_vbz_tpc_trw_feats.txt

#apws_topics_vbz_tpc_trw_feats.p  vbz_feats.p
#python describe_predictions.py -p apws_models/fw_ne_prn_sx1_posq_tpc_trw_feats.p -of results/fw_ne_prn_sx1_posq_tpc_trw_feats.txt


#python describe_predictions.py -p apws_models/fw_rm_sx1_posq_tpc_trw_feats.p -of results/fw_rm_sx1_posq_tpc_trw_feats.txt

#python describe_predictions.py -p apws_models/fw_rm_posq_tpc_trw_feats.p -of results/fw_rm_posq_tpc_trw_feats.txt
#python describe_predictions.py -p apws_models/fw_sx12_posq_tpc_trw_feats.p -of results/fw_sx12_posq_tpc_trw_feats.txt
#python describe_predictions.py -p apws_models/fw_sx1_posq_tpc_trw_feats.p -of results/fw_sx1_posq_tpc_trw_feats.txt
#python describe_predictions.py -p apws_models/fw_sx2_posq_tpc_trw_feats.p -of results/fw_sx2_posq_tpc_trw_feats.txt


#python describe_predictions.py -p apws_models/fw_posq_tpc_trw_feats.p -of results/fw_posq_tpc_trw_feats.txt
#python describe_predictions.py -p apws_models/fw_posq_trw_feats.p -of results/fw_posq_trw_feats.txt

#python describe_predictions.py -p apws_models/fw_s_feats.p -of results/fw_s_feats.txt
#python describe_predictions.py -p apws_models/fw_posq_tpc68r_feats.p -of results/fw_posq_tpc68r_feats.txt

#python describe_predictions.py -p apws_models/fw_posq_tpc68_feats.p -of results/fw_posq_tpc68_feats.txt
#python describe_predictions.py -p apws_models/fw_pos_tpc68_feats.p -of results/fw_pos_tpc68_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_prn_pos_feats.p -of results/fw_rm_sx_ne_prn_pos_feats.txt
#python describe_predictions.py -p apws_models/rm_tpc68_feats.p -of results/rm_tpc68_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_tpc_feats.p -of results/fw_rm_sx_tpc_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_prn_tpc68_feats.p -of results/fw_rm_sx_ne_prn_tpc68_feats.txt
#python describe_predictions.py -p apws_models/fw_tpc68_feats.p -of results/fw_tpc68_feats.txt


#python describe_predictions.py -p apws_models/fw_feats.p -of results/fw_feats.txt

#python describe_predictions.py -p apws_models/fw_posq_tpc_feats.p -of results/fw_posq_tpc_feats.txt
#python describe_predictions.py -p apws_models/fw_pos_tpc_feats.p -of results/fw_pos_tpc_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_prn_pos_feats.p -of results/fw_rm_sx_ne_prn_pos_feats.txt
#python describe_predictions.py -p apws_models/rm_tpc_feats.p -of results/rm_tpc_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_tpc_feats.p -of results/fw_rm_sx_tpc_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_prn_tpc_feats.p -of results/fw_rm_sx_ne_prn_tpc_feats.txt
#python describe_predictions.py -p apws_models/fw_feats.p -of results/fw_feats.txt
#python describe_predictions.py -p apws_models/fw_tpc_feats.p -of results/fw_tpc_feats.txt

#python describe_predictions.py -p apws_models/rm_prn_feats.p -of results/rm_prn_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_prn_feats.p -of results/fw_rm_sx_ne_prn_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_dc_prn_feats.p -of results/fw_rm_sx_ne_dc_prn_feats.txt

#python describe_predictions.py -p apws_models/fw_rm_sx_ne_dc_feats.p -of results/fw_rm_sx_ne_dc_feats.txt
#python describe_predictions.py -p apws_models/all_feats.p -of results/all_feats.txt
#python describe_predictions.py -p apws_models/fw_feats.p -of results/fw_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_feats.p -of results/fw_rm_sx_feats.txt
#python describe_predictions.py -p apws_models/fw_rm_sx_ne_feats.p -of results/fw_rm_sx_ne_feats.txt
#python describe_predictions.py -p apws_models/rm_feats.p -of results/rm_feats.txt
#python describe_predictions.py -p apws_models/sx_feats.p -of results/sx_feats.txt


