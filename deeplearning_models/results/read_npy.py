import numpy as np

#导入npy文件路径位置
root_path = './long_term_forecast_Exchange_96_96_TimesNet_custom_ftM_sl96_ll48_pl96_dm64_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0'
data_path = root_path + '/pred.npy'
test = np.load(data_path)

print(test)