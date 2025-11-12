export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1
done
