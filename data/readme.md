# Data description
The data contains abstracts of accepted papers in NIPS 2017. The result is ran on toy.txt, which only contains 5 samples of the corpus.

# Result
## HDP (this implementation)
Topics:

('classes', '0.067'), ('expression', '0.067'), ('prior', '0.067'), ('artificial', '0.067'), ('limiting', '0.067'), ('benefit', '0.067'), ('address', '0.067'), ('To', '0.067'), ('popularity', '0.067'), ('immense', '0.067')

('loss', '0.028'), ('stability', '0.024'), ('training', '0.013'), ('potential', '0.013'), ('algorithmic', '0.013'), ('convex', '0.013'), ('mechanism', '0.013'), ('Monte', '0.011'), ('generalization', '0.011'), ('experiments', '0.011')

('mechanism', '0.018'), ('data', '0.017'), ('loss', '0.017'), ('unhinged', '0.012'), ('learning', '0.012'), ('machine', '0.011'), ('propose', '0.011'), ('space', '0.009'), ('result', '0.009'), ('algorithmic', '0.009')

('mechanism', '0.017'), ('learning', '0.015'), ('stability', '0.014'), ('algorithmic', '0.013'), ('convex', '0.013'), ('distribution', '0.011'), ('methods', '0.011'), ('show', '0.011'), ('loss', '0.011'), ('process', '0.010')

Perplexity:

Epoch 5 | Perplexity per word: [66.904]

## Variational Inference LDA (gensim.LdaModel.ldamodel)
Topics:

(0, '0.028*"loss" + 0.024*"sln" + 0.024*"convex" + 0.021*"mechanism" + 0.015*"classification" + 0.015*"robust" + 0.015*"potential" + 0.015*"unhinged" + 0.011*"show" + 0.010*"result"')

(1, '0.022*"stability" + 0.018*"algorithmic" + 0.014*"data" + 0.011*"hypothesis" + 0.011*"size" + 0.011*"process" + 0.011*"complexity" + 0.011*"generalization" + 0.011*"space" + 0.010*"mechanism"')

(2, '0.014*"used" + 0.014*"machine" + 0.014*"distribution" + 0.014*"variance" + 0.014*"stochastic" + 0.014*"learning" + 0.014*"monte" + 0.014*"carlo" + 0.014*"covariance" + 0.014*"posterior"')

(3, '0.003*"loss" + 0.003*"convex" + 0.003*"data" + 0.003*"learning" + 0.003*"show" + 0.003*"sln" + 0.003*"stability" + 0.003*"large" + 0.003*"algorithmic" + 0.003*"can"')

Perplexity:

Per word perplexity: 67.218
