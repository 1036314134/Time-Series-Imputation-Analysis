export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.15 0.25 0.35 0.45; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1_${metric}_${strength}.csv \
      --model_id ETTh1_${metric}_${strength} \
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

model_name=Nonstationary_Transformer

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1_${metric}_${strength}.csv \
      --model_id ETTh1_${metric}_${strength} \
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
      --des 'Exp' \
      --itr 1 \
      --p_hidden_dims 256 256 \
      --p_hidden_layers 2 \
      --d_model 128
  done
done


model_name=PatchTST

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.15 0.25 0.35 0.45; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1_${metric}_${strength}.csv \
      --model_id ETTh1_${metric}_${strength} \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --n_heads 2 \
      --itr 1
  done
done

model_name=Autoformer

for metric in trend seasonal cycle ; do
  for strength in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1_${metric}_${strength}.csv \
      --model_id ETTh1_${metric}_${strength} \
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
      --des 'Exp' \
      --itr 1
  done
done

#
#
#model_name=SegRNN
#
#for metric in trend seasonal cycle ; do
#  for strength in 0.1 0.2 0.3 0.4 0.5; do
#    python -u run.py \
#      --task_name long_term_forecast \
#      --is_training 1 \
#      --root_path ../dataset/ETT-small/ \
#      --data_path ETTh1_${metric}_${strength}.csv \
#      --model_id ETTh1_${metric}_${strength} \
#      --model_id ETTh1 \
#      --model $model_name \
#      --data ETTh1 \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --seg_len 24 \
#      --enc_in 7 \
#      --d_model 512 \
#      --dropout 0.5 \
#      --learning_rate 0.0001 \
#      --des 'Exp' \
#      --itr 1
#  done
#done

#
#model_name=TimeMixer
#
#for metric in trend seasonal cycle ; do
#  for strength in 0.1 0.2 0.3 0.4 0.5; do
#    python -u run.py \
#      --task_name long_term_forecast \
#      --is_training 1 \
#      --root_path ../dataset/ETT-small/ \
#      --data_path ETTh1_${metric}_${strength}.csv \
#      --model_id ETTh1_${metric}_${strength} \
#      --model $model_name \
#      --data ETTh1 \
#      --features M \
#      --seq_len 96 \
#      --label_len 0 \
#      --pred_len 96 \
#      --e_layers 2 \
#      --enc_in 7 \
#      --c_out 7 \
#      --des 'Exp' \
#      --itr 1 \
#      --d_model 16 \
#      --d_ff 32 \
#      --learning_rate 0.01 \
#      --train_epochs 10 \
#      --patience 10 \
#      --batch_size 128 \
#      --down_sampling_layers 3 \
#      --down_sampling_method avg \
#      --down_sampling_window 2
#  done
#done