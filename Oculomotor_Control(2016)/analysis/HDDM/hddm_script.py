# Based on Wiecki et al (2013). Webpage Info : http://ski.clps.brown.edu/hddm_docs/index.html

os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/results/hddm')

#In IPython shell, use : %pylab

data = hddm.load_csv('hddm.csv')

data = hddm.utils.flip_errors(data) #changes response times of incorrect trials to negative

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for i, subj_data in data.groupby('subj_idx'):
	subj_data.rt.hist(bins=20, histtype='step', ax=ax)
	
plt.savefig('hddm_demo_fig_00.pdf')

## Drift rate model  - allow change in angle and reward#
m = hddm.HDDM(data, bias = True, depends_on = {'v': ['angle', 'reward']})
m.find_starting_values()
m.sample(10000, burn = 1000, thin = 5, dbname = 'full_model.db',db = 'pickle')
m.save('full_model')

# Can look at intertrial variability at the group level as follows : but this still takes a very very long time.
# model = hddm.HDDM(data, bias = True, include=('sv','st','sz'), group_only_nodes=['sv', 'st', 'sz'], depends_on = {'v': ['angle', 'reward']})

# Include everything, even intertrial variability at the single subject level. This takes a VERY VERY long time to sample.
# model = hddm.HDDM(data, bias = True, include = 'all', depends_on = {'v': ['angle', 'reward']})

stats = m.gen_stats()
stats[stats.index.isin(['a', 'a_std', 'a_subj.0', 'a_subj.1'])] #etc
m.plot_posteriors(['a','t','v','a_std'])

# If Outliers are a problem:
m_outlier = hddm.HDDM(data, p_outlier=.05)
m_outlier.sample(10000, burn=1000)
m_outlier.plot_posterior_predictive(figsize=(14, 10))

m.plot_posteriors_conditions()

v_large_high, v_large_low, v_large_no, v_med_high, v_med_low, v_med_no, v_small_high, v_small_low, v_small_no = m.nodes_db.node[['v(l.h)', 'v(l.l)', 'v(l.n)', 'v(m.h)','v(m.l)','v(m.n)','v(s.h)','v(s.l)','v(s.n)']]
hddm.analyze.plot_posterior_nodes([v_large_high, v_large_low, v_large_no])
hddm.analyze.plot_posterior_nodes([v_med_high, v_med_low, v_med_no])
hddm.analyze.plot_posterior_nodes([v_small_high, v_small_low, v_small_no])

# Angle probabilities
print "P(large_h < small_h) = ", (v_large_high.trace() < v_small_high.trace()).mean() 
print "P(large_h < med_h) = ", (v_large_high.trace() < v_med_high.trace()).mean() 
print "P(med_h < small_h) = ", (v_med_high.trace() < v_small_high.trace()).mean() 

print "P(large_l < small_l) = ", (v_large_low.trace() < v_small_low.trace()).mean() 
print "P(large_l < med_l) = ", (v_large_low.trace() < v_med_low.trace()).mean() 
print "P(med_l < small_l) = ", (v_med_low.trace() < v_small_low.trace()).mean() 

print "P(large_n < small_n) = ", (v_large_no.trace() < v_small_no.trace()).mean() 
print "P(large_n < med_n) = ", (v_large_no.trace() < v_med_no.trace()).mean() 
print "P(med_n < small_n) = ", (v_med_no.trace() < v_small_no.trace()).mean() 

# Reward probabilities
print "P(small_h < small_n) = ", (v_small_high.trace() < v_small_no.trace()).mean() 
print "P(small_h < small_l) = ", (v_small_high.trace() < v_small_low.trace()).mean() 
print "P(small_l < small_n) = ", (v_small_low.trace() < v_small_no.trace()).mean() 

print "P(med_h < med_n) = ", (v_med_high.trace() <  v_med_no.trace()).mean() 
print "P(med_h < med_l) = ", (v_med_high.trace() < v_med_low.trace()).mean() 
print "P(med_l < med_n) = ", (v_med_low.trace() < v_med_no.trace()).mean() 

print "P(large_h < large_n) = ", (v_large_high.trace() < v_large_no.trace()).mean() 
print "P(large_h < large_l) = ", (v_large_high.trace() < v_large_low.trace()).mean() 
print "P(large_l < large_n) = ", (v_large_low.trace() < v_large_no.trace()).mean() 



# Angle only
m_angle = hddm.HDDM(data, bias = True, depends_on = {'v': ['angle']})
m_angle.find_starting_values()
m_angle.sample(10000, burn = 1000, thin = 5, dbname = 'angle_model.db',db = 'pickle')
m_angle.save('angle_model')

# Reward only
m_reward = hddm.HDDM(data, bias = True, depends_on = {'v': ['reward']})
m_reward.find_starting_values()
m_reward.sample(10000, burn = 1000, thin = 5, dbname = 'reward_model.db',db = 'pickle')
m_reward.save('reward_model')


## Combined Boundary and Drift rate model - allow varying angle and reward boundary separations and drift rates ##
model_av = hddm.HDDM(data, depends_on={'a': ['angle', 'reward'], 'v': ['angle', 'reward']})
m_av.find_starting_values()
m_av.sample(10000, burn = 1000, thin = 5, dbname = 'av_model.db',db = 'pickle')
m_av.save('av_model')

# Within-subject effects : taking no reward condition as "baseline"
dmatrix("C(reward, Treatment('n'))", data.head(10)) #needs patsy.dmatrix
m_within_subj = hddm.HDDMRegressor(data, "v ~ C(reward, Treatment('n'))")
v_no, v_high, v_low = m_within_subj.nodes_db.ix[["v_Intercept", "v_C(reward, Treatment('n'))[T.h]", "v_C(reward, Treatment('n'))[T.l]"], 'node']
hddm.analyze.plot_posterior_nodes([v_no, v_high, v_low])


### Hemifield ###
hemi_data = hddm.load_csv('hddm_hemi.csv')
hemi_data = hddm.utils.flip_errors(hemi_data)

m_hemi = hddm.HDDM(hemi_data, bias = True, depends_on = {'v': ['hemifield', 'reward']})
v_same_high, v_same_low, v_same_no, v_opp_high, v_opp_low, v_opp_no = m_hemi.nodes_db.node[['v(0.h)','v(0.l)','v(0.n)','v(1.h)','v(1.l)','v(1.n)']]

# hemifield
print "P(opp_high < same_high) = ", (v_opp_high.trace() < v_same_high.trace()).mean() 
print "P(opp_l < same_l) = ", (v_opp_low.trace() < v_same_low.trace()).mean() 
print "P(opp_n < same_n) = ", (v_opp_no.trace() < v_same_no.trace()).mean() 

# reward
print "P(same_h < same_n) = ", (v_same_high.trace() < v_same_no.trace()).mean() 
print "P(same_h < same_l) = ", (v_same_high.trace() < v_same_low.trace()).mean() 
print "P(same_l < same_n) = ", (v_same_low.trace() < v_same_no.trace()).mean() 

print "P(opp_h < opp_n) = ", (v_opp_high.trace() < v_opp_no.trace()).mean() 
print "P(opp_h < opp_l) = ", (v_opp_high.trace() < v_opp_low.trace()).mean() 
print "P(opp_l < opp_n) = ", (v_opp_low.trace() < v_opp_no.trace()).mean() 


## Model comparison ##
# DIC : Deviance Information Criterion (lower is better (more negative))
print "Lumped model DIC: %f" %m.dic
print "Angle model DIC: %f" %m_angle.dic
print "Reward model DIC: %f" %m_reward.dic

## Assessing Model Convergence ##

# Geweke statistic #
# Can't seem to get this working properly - not really sure how to interpret the output
from kabuki.analyze import check_geweke
print check_geweke(m)

# Gelman-Rubin #
# All r-hat values less than 1.1 indicates successful convergence
from kabuki.analyze import gelman_rubin

models = []
for i in range(5):
	m_gr = hddm.HDDM(m.data)
	m_gr.find_starting_values()
	m_gr.sample(10000, burn=1000, thin = 5)
	models.append(m_gr)
 gelman_rubin(models)

## Model Fit - Post Posterior Checks ##
# Simulate new data from the posterior of the fitted model #
# Then apply the the summary statistic to each of the simulated data sets from the posterior and see if the model does a good job of reproducing this pattern by comparing the summary statistics from the simulations to the summary statistic caluclated over the model.

ppc_data = hddm.utils.post_pred_gen(m) #this simulates 500 datasets by default
ppc_compare = hddm.utils.post_pred_stats(m.data, ppc_data)
print ppc_compare # compares 'observed' (your actual data) to 'mean' (simulations). The last column 'credible' tells you True if the comparison is within the 95% credible interval.


### Roger Ratcliff method ###
# Quantile Optimization. If you have lots of data (>100 trials per condition) and don't care about posterior estimates #

# First load model m
m_quantile = hddm.HDDM(data, depends_on={'v' : ['angle','reward']})
params_quantile = m_quantile.optimize('chisquare') # can also use 'gsquare' or 'ML' here
print params_quantile

# Running different models for each individual subject
subj_params = []
for subj_idx, subj_data in data.groupby('subj_idx'):
	m_subj = hddm.HDDM(subj_data, depends_on={'v' : ['angle','reward']})
	subj_params.append(m_subj.optimize('chisquare'))


### LOADING MODELS & PLOTTING ###

# Make sure in results directory where the model is located

# Full model
m_full = hddm.load('full_model')
v_large_high, v_large_low, v_large_no, v_med_high, v_med_low, v_med_no, v_small_high, v_small_low, v_small_no = m_full.nodes_db.node[['v(l.h)', 'v(l.l)', 'v(l.n)', 'v(m.h)','v(m.l)','v(m.n)','v(s.h)','v(s.l)','v(s.n)']]

# Hemi model
m_hemi = hddm.load('hemi_model')
v_same_high, v_same_low, v_same_no, v_opp_high, v_opp_low, v_opp_no = m_hemi.nodes_db.node[['v(0.h)','v(0.l)','v(0.n)','v(1.h)','v(1.l)','v(1.n)']]

# Creating figures
#%pylab
from matplotlib.pylab import figure

bins = 10
lb = 4
ub = 10

# FULL MODEL
nodes = [v_small_high, v_small_low, v_small_no, v_med_high, v_med_low, v_med_no,v_large_high, v_large_low, v_large_no]

figure(figsize=(3,3)) # inches, for J Neurophysiology measurements (this is the size only of the axes themselves, it doesn't include the border -> leaving an extra 0.5 inches)
plt.figure(facecolor="white")
if lb is None:
    lb = min([min(node.trace()[:]) for node in nodes])
if ub is None:
    ub = max([max(node.trace()[:]) for node in nodes])

x_data = np.linspace(lb, ub, 300)
color_list = ['b','g','r','b','g','r','b','g','r'] # high = blue, low = green, no = red
linestyles = [':', ':', ':', '--', '--', '--', '-', '-', '-'] # 30 deg = ':', 120 deg = '--', 180 deg = '-'

for i,node in enumerate(nodes):
	trace = node.trace()[:]
	hist = hddm.analyze.utils.interpolate_trace(x_data, trace, range=(lb, ub), bins=bins)
	lines = plt.plot(x_data, hist, label=node.__name__, lw=2.)
	plt.setp(lines, color=color_list[i], linestyle = linestyles[i], linewidth=2.0)

leg = plt.legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.xlabel('Drift rate, v', fontsize = '14')
plt.ylabel('Posterior Probability', fontsize = '14')
plt.ylim((0,1.0))
plt.legend(loc = 1, labels = ('30-high', '30-low', '30-no','120-high', '120-low', '120-no','180-high', '180-low', '180-no'), fontsize = '10', title="Condition")
plt.savefig('drift_rates_full_model.png')
plt.savefig('drift_rates_full_model.eps',format='eps')
plt.close()

# HEMI MODEL
nodes = [v_same_high, v_same_low, v_same_no, v_opp_high, v_opp_low, v_opp_no]

figure(figsize=(3,3)) # for J Neurophysiology measurements
plt.figure(facecolor="white")
if lb is None:
    lb = min([min(node.trace()[:]) for node in nodes])
if ub is None:
    ub = max([max(node.trace()[:]) for node in nodes])

x_data = np.linspace(lb, ub, 300)
color_list = ['b','g','r','b','g','r'] # high = blue, low = green, no = red
linestyles = ['--', '--', '--', '-', '-', '-']

for i,node in enumerate(nodes):
	trace = node.trace()[:]
	hist = hddm.analyze.utils.interpolate_trace(x_data, trace, range=(lb, ub), bins=bins)
	lines = plt.plot(x_data, hist, label=node.__name__, lw=2.)
	plt.setp(lines, color=color_list[i], linestyle = linestyles[i], linewidth=2.0)

leg = plt.legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.xlabel('Drift rate, v', fontsize = '14')
plt.ylabel('Posterior Probability', fontsize = '14')
plt.ylim((0,1.0))
plt.legend(loc = 1, labels = ('ipsi-high', 'ipsi-low', 'ipsi-no','contra-high', 'contra-low', 'contra-no'), fontsize = '10',title = 'Condition')
plt.savefig('drift_rates_hemi_model.png')
plt.savefig('drift_rates_hemi_model.eps',format='eps')
plt.close()

