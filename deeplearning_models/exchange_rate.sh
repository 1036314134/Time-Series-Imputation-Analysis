export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

for attr in OT; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id exchange_rate \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 1

  for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
    for method in mean knn xgboost iim front; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/exchange_rate/ \
        --data_path exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method}.csv \
        --model_id exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --d_model 32 \
        --d_ff 32 \
        --top_k 5 \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 1
    done
  done
done


model_name=Mamba

for attr in OT; do
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 8 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 8 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1

  for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
    for method in mean knn xgboost iim front; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/exchange_rate/ \
        --data_path exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method}.csv \
        --model_id exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in 8 \
        --expand 2 \
        --d_ff 16 \
        --d_conv 4 \
        --c_out 8 \
        --d_model 128 \
        --des 'Exp' \
        --itr 1

    done
  done
done


model_name=Nonstationary_Transformer

for attr in OT; do
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2

  for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
    for method in mean knn xgboost iim front; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/exchange_rate/ \
        --data_path exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method}.csv \
        --model_id exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --des 'Exp' \
        --itr 1 \
        --p_hidden_dims 256 256 \
        --p_hidden_layers 2

    done
  done
done


model_name=PatchTST

for attr in OT; do
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

  for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
    for method in mean knn xgboost iim front; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/exchange_rate/ \
        --data_path exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method}.csv \
        --model_id exchange_rate_missing_${attr}_${missing_rate}_imputed_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --des 'Exp' \
        --itr 1

    done
  done
done