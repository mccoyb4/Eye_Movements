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

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/results/')
test_df = pd.read_csv('test_pre_post.csv') 
train_df = pd.read_csv('training.csv') 

# Reformat test
test_df = test_df[['High_1','High_2','Low_1','Low_2','No1_1','No1_2','No2_1','No2_2']]

#Train
gain_train_high = train_df['High_2']-train_df['High_4']
gain_train_low = train_df['High_2']-train_df['Low_4']
gain_train_no1 = train_df['No1_2']-train_df['No1_4']
gain_train_no2 = train_df['No2_2']-train_df['No2_4']

gain_train_high = train_df['baseline_mean']-train_df['High_4']
gain_train_low = train_df['baseline_mean']-train_df['Low_4']
gain_train_no1 = train_df['baseline_mean']-train_df['No1_4']
gain_train_no2 = train_df['baseline_mean']-train_df['No2_4']

#Test
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

# Correlation in difference in high v nocomb across training and test phases?
gain_train_nocomb = (gain_train_no1 + gain_train_no2)/2
train_diff = gain_train_high - gain_train_nocomb
gain_test_nocomb = (gain_test_no1 + gain_test_no2)/2
test_diff = gain_test_high - gain_test_nocomb
stats.pearsonr(train_diff, test_diff)

shell()


