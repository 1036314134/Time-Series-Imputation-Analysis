from results.show_metirc_change import show_metirc_change

if __name__ == '__main__':
    # ======== ETTh1 ========
    # ======== TimesNet ========
    TimesNet_pre_mse = {
        "Modify Trend": [1.796163321	1.804508448	1.866905928	1.83067584	1.847233772	1.879895926	1.88581419	1.918817759	1.956220746	1.978688478	2.078234196],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(TimesNet_pre_mse, "Modification intensity", "predict_MSE",
                       "TimesNet MSE under Different Modification")

    TimesNet_pre_mae = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(TimesNet_pre_mae, "Modification intensity", "predict_MAE",
                       "TimesNet MAE under Different Modification")

    # ======== Non_Transformer ========
    Non_Trans_pre_mse = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(Non_Trans_pre_mse, "Modification intensity", "predict_MSE",
                       "Non_Transformer MSE under Different Modification")

    Non_Trans_pre_mae = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(Non_Trans_pre_mae, "Modification intensity", "predict_MAE",
                       "Non_Transformer  MAE under Different Modification")

    # ======== PatchTST ========
    PatchTST_pre_mse = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(PatchTST_pre_mse, "Modification intensity", "predict_MSE",
                       "PatchTST MSE under Different Modification")

    PatchTST_pre_mae = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(PatchTST_pre_mae, "Modification intensity", "predict_MAE",
                       "PatchTST MAE under Different Modification")

    # ======== Autoformer ========
    Autoformer_pre_mse = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(Autoformer_pre_mse, "Modification intensity", "predict_MSE",
                       "Autoformer MSE under Different Modification")

    Autoformer_pre_mae = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(Autoformer_pre_mae, "Modification intensity", "predict_MAE",
                       "Autoformer MAE under Different Modification")

    # ======== 指标变化 ========
    trend_strength = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(trend_strength, "Modification intensity", "trend strength",
                       "trend strength under Different Modification")

    seasonal_strength = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(seasonal_strength, "Modification intensity", "seasonal strength",
                       "seasonal strength under Different Modification")

    cycle_strength = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(cycle_strength, "Modification intensity", "cycle strength",
                       "cycle strength under Different Modification")

    acf1 = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(acf1, "Modification intensity", "acf@1",
                       "acf@1 under Different Modification")

    Ljung_Box_stat = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(Ljung_Box_stat, "Modification intensity", "Ljung Box stat",
                       "Ljung Box stat under Different Modification")

    spectral_entropy = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(spectral_entropy, "Modification intensity", "spectral entropy",
                       "spectral entropy under Different Modification")

    MSE = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(MSE, "Modification intensity", "MSE",
                       "MSE under Different Modification")

    MAE = {
        "Modify Trend": [],
        "Modify Seasonal": [],
        "Modify Cycle": [],
    }
    show_metirc_change(MAE, "Modification intensity", "MAE",
                       "MAE under Different Modification")