export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

for attr in OT; do
  for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
    for method in mean knn xgboost iim front; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/ETT-small/ \
        --data_path ETTh1_missing_${attr}_${missing_rate}_imputed_by_${method}.csv \
        --model_id ETTh1_rate_missing_${attr}_${missing_rate}_imputed_by_${method} \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 32 \
        --des 'Exp' \
        --itr 1 \
        --top_k 5
    done
  done
done