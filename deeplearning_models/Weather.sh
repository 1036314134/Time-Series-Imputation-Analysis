export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

for attr in OT; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --model_id weather \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1

  for method in mean front knn xgboost; do
    for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/weather/ \
        --data_path weather_${attr}_${missing_rate}_${method}.csv \
        --model_id weather_${attr}_${missing_rate}_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --d_model 32 \
        --d_ff 32 \
        --top_k 5 \
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
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --model_id weather \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 3 \
    --p_hidden_dims 256 256 \
    --p_hidden_layers 2

  for method in mean front knn xgboost; do
    for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/weather/ \
        --data_path weather_${attr}_${missing_rate}_${method}.csv \
        --model_id weather_${attr}_${missing_rate}_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 3 \
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
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --model_id weather \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 4 \
    --train_epochs 3

  for method in mean front knn xgboost; do
    for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/weather/ \
        --data_path weather_${attr}_${missing_rate}_${method}.csv \
        --model_id weather_${attr}_${missing_rate}_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --n_heads 4 \
        --train_epochs 3
    done
  done
done


model_name=SegRNN

for attr in OT; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --model_id weather \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --seg_len 48 \
    --enc_in 21 \
    --d_model 512 \
    --dropout 0.5 \
    --learning_rate 0.0001 \
    --des 'Exp' \
    --itr 1

  for method in mean front knn xgboost; do
    for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/weather/ \
        --data_path weather_${attr}_${missing_rate}_${method}.csv \
        --model_id weather_${attr}_${missing_rate}_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --seg_len 48 \
        --enc_in 21 \
        --d_model 512 \
        --dropout 0.5 \
        --learning_rate 0.0001 \
        --des 'Exp' \
        --itr 1
    done
  done
done


model_name=TimeMixer

for attr in OT; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --model_id weather \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --d_model 16 \
    --d_ff 32 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --train_epochs 20 \
    --patience 10 \
    --down_sampling_layers 3 \
    --down_sampling_method avg \
    --down_sampling_window 2

  for method in mean front knn xgboost; do
    for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/weather/ \
        --data_path weather_${attr}_${missing_rate}_${method}.csv \
        --model_id weather_${attr}_${missing_rate}_by_${method} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 0 \
        --pred_len 96 \
        --e_layers 3 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --d_model 16 \
        --d_ff 32 \
        --batch_size 128 \
        --learning_rate 0.01 \
        --train_epochs 20 \
        --patience 10 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2
    done
  done
done