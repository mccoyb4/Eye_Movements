
library(BayesFactor)

setwd('~/Documents/Spatial_Reward_Toolbox/experiment_2/results/')
test = read.csv('test_pre_post.csv')

# Difference in RT between high and low reward locations across baseline and test phases
diff_scores = (test$High_1-test$High_2)-(test$Low_1 - test$Low_2)
bf = ttestBF(x = diff_scores)
bf_null = 1/bf

# Looking at the mu posterior and chain convergence (10000 iterations, twice)
chains = posterior(bf, iterations = 10000)
chains2 = recompute(chains, iterations = 10000)
plot(chains2[,1:2])

# Can also compare evidence for negative difference to positive difference
# i.e. slowing at the high reward location compared to low versus speeding up of RT at high compared to low.
bfInterval = ttestBF(x = diff_scores,nullInterval=c(-Inf,0))
print(bfInterval[1] / bfInterval[2])
allbf = c(bf, bfInterval)
plot(allbf)