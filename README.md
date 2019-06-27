# HDP
Python implementation of Gibbs sampling Hierarchical Dirichlet Process.

# Reference
[Sharing Clusters among Related Groups: Hierarchical Dirichlet Processes](http://papers.nips.cc/paper/2698-sharing-clusters-among-related-groups-hierarchical-dirichlet-processes), Teh et al., NIPS, 2004.

# Building evironment
python 3.6.0, tensorflow 1.14, tensorflow probability 0.7.0, SciPy 1.0.0

# Author
Lihui Lin, School of Data and Computer Science, Sun Yat-sen University.

# Results
## HDP (this implementation)
Topics:

('loss', '0.02226'), ('convex', '0.01590'), ('learning', '0.01590'), ('mechanism', '0.01590'), ('unhinged', '0.01113'), ('stability', '0.01113'), ('SLN-robust', '0.01033'), ('show', '0.01033'), ('Monte', '0.00954'), ('potential', '0.00954')

('conditional', '0.01522'), ('data', '0.01522'), ('low-complexity', '0.01304'), ('distribution', '0.01087'), ('propose', '0.01087'), ('algorithmic', '0.01087'), ('convex', '0.01087'), ('learning', '0.01087'), ('mechanism', '0.01087'), ('datapredictive', '0.00870')

('result', '0.02703'), ('convex', '0.02252'), ('algorithmic', '0.01802'), ('stability', '0.01802'), ('central', '0.01351'), ('low-complexity', '0.01351'), ('hypothesis', '0.01351'), ('generalization', '0.01351'), ('approach', '0.01351'), ('one', '0.01351')

('stability', '0.04372'), ('algorithmic', '0.03279'), ('size', '0.02732'), ('relationship', '0.02186'), ('hypothesis', '0.02186'), ('generalization', '0.02186'), ('large-sample', '0.01639'), ('likelihood', '0.01639'), ('improve', '0.01639'), ('parametric', '0.01639')

('mechanism', '0.07812'), ('unique', '0.04688'), ('crowdsourcing', '0.04688'), ('rates', '0.03125'), ('spammers', '0.03125'), ('mild', '0.03125'), ('possible', '0.03125'), ('one', '0.03125'), ('payment', '0.03125'), ('show', '0.03125')

Perplexity:

Epoch 5 | Perplexity per word: 67.146

## Variational Inference LDA (gensim.models.ldamodel.LdaModel)
Topics:

(0, '0.030*"loss" + 0.025*"convex" + 0.025*"sln" + 0.015*"unhinged" + 0.015*"robust" + 0.015*"potential" + 0.015*"classification" + 0.011*"result" + 0.011*"servedio" + 0.011*"equivalent"')

(1, '0.039*"stability" + 0.033*"algorithmic" + 0.020*"size" + 0.020*"space" + 0.020*"hypothesis" + 0.020*"generalization" + 0.014*"learning" + 0.014*"result" + 0.014*"relationship" + 0.014*"processing"')

(2, '0.034*"mechanism" + 0.016*"payment" + 0.016*"crowdsourcing" + 0.013*"possible" + 0.012*"may" + 0.011*"workers" + 0.011*"show" + 0.010*"free" + 0.010*"unique" + 0.010*"lunch"')

(3, '0.018*"mechanism" + 0.016*"data" + 0.014*"large" + 0.012*"low" + 0.012*"parameter" + 0.011*"show" + 0.010*"distribution" + 0.010*"dirichlet" + 0.010*"likelihood" + 0.010*"complexity"')

(4, '0.003*"loss" + 0.003*"can" + 0.003*"learning" + 0.003*"propose" + 0.003*"large" + 0.003*"approach" + 0.003*"sln" + 0.003*"algorithmic" + 0.003*"low" + 0.003*"noise"')

Perplexity:

Per word perplexity: 72.549
