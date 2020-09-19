import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as stats
import pandas as pd
from scipy.optimize import curve_fit
import sympy as sym
import seaborn as sns
from IPython import embed as shell

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/')
test_df = pd.read_csv('test_pre_post.csv') 
train_df = pd.read_csv('training.csv') 

# Reformat test
test_df = test_df[['High_1','High_2','Low_1','Low_2','No1_1','No1_2','No2_1','No2_2']]
train_df = train_df[['High_2','High_3','High_4','Low_2','Low_3','Low_4','No1_2','No1_3','No1_4','No2_2','No2_3','No2_4']]

corrs = []

for i in train_df:
	for j in test_df:
		corrs.append(stats.pearsonr(train_df[i], test_df[j]))

# Training (across all trials) with Test RT 
corrs = np.array(corrs)
corrs_values = corrs[:,0].reshape(12,8)
corrs_sigs = corrs[:,1].reshape(12,8)

corrs_df = pd.DataFrame(corrs_sigs, index =['High_2','High_3','High_4','Low_2','Low_3','Low_4','No1_2','No1_3','No1_4','No2_2','No2_3','No2_4'])
corrs_df.index.name = 'Train (RT)'

corrs_df.columns =   ['High_1','High_2','Low_1','Low_2','No1_1','No1_2','No2_1','No2_2']
corrs_df.columns.name = 'Train (SRT)'

fig, ax = plt.subplots(figsize=(4.5, 7))
cax = ax.imshow(corrs_df, interpolation='nearest', cmap=cm.magma)
cbar = fig.colorbar(cax)
cbar.set_label("p-values")
plt.xticks(np.arange(0,8),fontsize=12, rotation='vertical')
plt.yticks(np.arange(0,12),fontsize=12)
x_labels =  ['High_1','High_2','Low_1','Low_2','No1_1','No1_2','No2_1','No2_2']
y_labels = ['High_2','High_3','High_4','Low_2','Low_3','Low_4','No1_2','No1_3','No1_4','No2_2','No2_3','No2_4']
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)
plt.xlabel('Test locations (RT)', fontsize=14)
plt.ylabel('Training locations (SRT)', fontsize=14)
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs')
#plt.savefig('Correlation_Training_v_Test_allTrials')
plt.show()
plt.close()


# Correlations on gains
gain_train_high = train_df['High_2']-train_df['High_4']
gain_train_low = train_df['High_2']-train_df['Low_4']
gain_train_no1 = train_df['No1_2']-train_df['No1_4']
gain_train_no2 = train_df['No2_2']-train_df['No2_4']
gain_test_high = test_df['High_1']-test_df['High_2']
gain_test_low = test_df['Low_1']-test_df['Low_2']
gain_test_no1 = test_df['No1_1']-test_df['No1_2']
gain_test_no2 = test_df['No2_1']-test_df['No2_2']
stats.pearsonr(gain_train_high,gain_test_high)
stats.pearsonr(gain_train_low,gain_test_low)
stats.pearsonr(gain_train_no1,gain_test_no1)
stats.pearsonr(gain_train_no2,gain_test_no2)

#Direct correlation comparisons of SRT and RT at each location
stats.pearsonr(train_df['High_4'],test_df['High_2'])
stats.pearsonr(train_df['High_3'],test_df['High_2'])
stats.pearsonr(train_df['High_2'],test_df['High_2'])
stats.pearsonr(train_df['Low_4'],test_df['Low_2'])
stats.pearsonr(train_df['Low_3'],test_df['Low_2'])
stats.pearsonr(train_df['Low_2'],test_df['Low_2'])
stats.pearsonr(train_df['No1_4'],test_df['No1_2'])
stats.pearsonr(train_df['No1_3'],test_df['No1_2'])
stats.pearsonr(train_df['No1_2'],test_df['No1_2'])
stats.pearsonr(train_df['No2_4'],test_df['No2_2'])
stats.pearsonr(train_df['No2_3'],test_df['No2_2'])
stats.pearsonr(train_df['No2_2'],test_df['No2_2'])

shell()

