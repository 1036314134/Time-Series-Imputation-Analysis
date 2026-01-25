export CUDA_VISIBLE_DEVICES=0


model_name=TimesNet

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/illness/ \
      --data_path national_illness_${metric}_${strength}.csv \
      --model_id national_illness_${metric}_${strength} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len 24 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 768 \
      --d_ff 768 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1
  done
done


model_name=Nonstationary_Transformer

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/illness/ \
      --data_path national_illness_${metric}_${strength}.csv \
      --model_id national_illness_${metric}_${strength} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len 24 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --p_hidden_dims 32 32 \
      --p_hidden_layers 2
  done
done


model_name=PatchTST

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/illness/ \
      --data_path national_illness_${metric}_${strength}.csv \
      --model_id national_illness_${metric}_${strength} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len 24 \
      --e_layers 4 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --n_heads 4 \
      --d_model 1024\
      --itr 1
  done
done


model_name=Autoformer

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/illness/ \
      --data_path national_illness_${metric}_${strength}.csv \
      --model_id national_illness_${metric}_${strength} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len 24 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1
  done
done