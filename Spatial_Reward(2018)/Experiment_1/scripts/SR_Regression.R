# Using blme package - Linear mixed effects modelling

install.packages('lme4')
library(lme4)

datadir='~/disks/Aeneas_Home/projects/Spatial_Reward_Toolbox/experiment_1/results/'
setwd(datadir)

options(max.print=3000)

# TEST PHASE #
# All pairs logistic regression
highlow_regression_baseline = read.csv('highlow_regression_file_baseline.csv')
highlow_regression_test = read.csv('highlow_regression_file_test.csv')

high_baseline = read.csv('high_regression_file_baseline.csv')
high_test = read.csv('high_regression_file_test.csv')

low_baseline = read.csv('low_regression_file_baseline.csv')
low_test = read.csv('low_regression_file_test.csv')

# (1|Subject) always included in mixed-effects models (random within-subject effects)

#### RT as dependent measure (distributed as inverse gaussian) ####

# BASELINE
# Remember: using an inverse gaussian link function here, so if estimates are negative this relates to a HIGER RT, not a lower one.
latency_baseline = glmer(RT ~ condition + template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=highlow_regression_baseline)
summary(latency_baseline)
# Main effect of condition only (higher reward -> lower RT)

#mean-centred template duration (so mean-centred fixed effects)
highlow_regression_baseline['template_duration'] = scale(highlow_regression_baseline['template_duration'])
latency_baseline = glmer(RT ~ condition + template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=highlow_regression_baseline)
summary(latency_baseline)

# TEST (mean-centred)
highlow_regression_test['template_duration'] = scale(highlow_regression_test['template_duration'])
latency_test = glmer(RT ~ condition + template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=highlow_regression_test)
summary(latency_test)
# Main effect of condition (higher reward = higher RT) and main effect of template duration (higher duration = higher RT). Including 
# an interaction term is not significant and gives higher AIC, BIC so the above model is the best.

# MERGE baseline and test, with new fixed effect 'phase', to see how it changes across test phases
highlow_regression_baseline['phase']=0
highlow_regression_test['phase']=1

phases = rbind(highlow_regression_baseline,highlow_regression_test)
#phases['template_duration'] = scale(phases['template_duration'])
latency_merged_0 = glmer(RT ~ condition + template_duration + phase + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=phases)
latency_merged_1 = glmer(RT ~ condition + template_duration + phase + phase*condition + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=phases)
latency_merged_2 = glmer(RT ~ condition + template_duration + phase + template_duration*condition + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=phases)
summary(latency_merged_1)

# Check whether adding interactions improves the model fit
anova(latency_merged_0,latency_merged_1) # yes : p(>chisq) = .001
anova(latency_merged_0,latency_merged_2) # no : p(>chisq) = .712

# Hemifield analysis
hemi_90 = c(5,6,7,8,13,14,15,16,21,22,23,24)
hemi_180 = c(1,2,3,4,9,10,11,12,17,18,19,20)
phases_90 = phases[phases$Subject %in% hemi_90,]
phases_180 = phases[phases$Subject %in% hemi_180,]

latency_merged_90 = glmer(RT ~ condition + template_duration + phase + phase*condition + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=phases_90)
summary(latency_merged_90)
# No condition*phase interaction, trend for template duration (p=.058)

latency_merged_180 = glmer(RT ~ condition + template_duration + phase + phase*condition + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=phases_180)
summary(latency_merged_180)
# Main effect of condition (p=.005), template duration (p=.01), and a strong condition*phase interaction (p=10^-5)

# Separate high or low files
# high
high_baseline_glm = glmer(RT ~ template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=high_baseline)
summary(high_baseline_glm)
high_test_glm = glmer(RT ~ template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=high_test)
summary(high_test_glm)
# main effect of template duration in test phase only

#low
low_baseline_glm = glmer(RT ~ template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=low_baseline)
summary(low_baseline_glm)
low_test_glm = glmer(RT ~ template_duration + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=low_test)
summary(low_test_glm)
# main effect of template duration in test phase only

# Overall, both high and low reward locations separately experience an increase in RT for longer template durations, so longer template
# duration does not explain the difference found between these two locations.

# LRT (Likelihood Ratio Test), using http://www.ssc.wisc.edu/sscc/pubs/MM/MM_TestEffects.html.
# Check Suzanne's paper too (Jongman, 2017)


# TRAINING PHASE #
hl_regression = read.csv('hl_regression_file.csv')

hl_regression['template_duration'] = scale(hl_regression['template_duration'])
latency_train = glmer(srt ~ condition + template_duration + block + condition*block + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=hl_regression)
summary(latency_train)

hl_regression_90 = hl_regression[hl_regression$Subject %in% hemi_90,]
hl_regression_180 = hl_regression[hl_regression$Subject %in% hemi_180,]

latency_train_90 = glmer(srt ~ condition + template_duration + block + condition*block + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=hl_regression_90)
summary(latency_train_90)

latency_train_180 = glmer(srt ~ condition + template_duration + block + condition*block + (1|Subject), family = inverse.gaussian(link = "1/mu^2"), data=hl_regression_180)
summary(latency_train_180)
