# awk '/^>/ {sub(/ .*/, "", $0)} {print}' ../data/test.fasta > ../data/test_clean.fasta

cd ../LucaOneApp/LucaOneApp-master/algorithms
python inference_embedding_lucaone.py \
    --llm_dir /data3/zd/LucaOneApp/models \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../test/PV241319.1_nt.fasta \
    --save_path ../test/embedding \
    --embedding_type vector \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 1000 \
    --gpu_id 0
