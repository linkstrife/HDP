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

(0, '0.003*"loss" + 0.003*"large" + 0.003*"mechanism" + 0.003*"convex" + 0.003*"learning" + 0.003*"stability" + 0.003*"show" + 0.003*"can" + 0.003*"algorithmic" + 0.003*"process"')

(1, '0.014*"distribution" + 0.014*"large" + 0.014*"parameter" + 0.010*"learning" + 0.010*"data" + 0.010*"likelihood" + 0.010*"carlo" + 0.010*"posterior" + 0.010*"approach" + 0.010*"machine"')

(2, '0.047*"mechanism" + 0.021*"crowdsourcing" + 0.021*"payment" + 0.021*"possible" + 0.014*"data" + 0.014*"lunch" + 0.014*"may" + 0.014*"workers" + 0.014*"show" + 0.014*"free"')

(3, '0.029*"loss" + 0.025*"stability" + 0.021*"algorithmic" + 0.021*"sln" + 0.021*"convex" + 0.017*"result" + 0.013*"potential" + 0.013*"unhinged" + 0.013*"robust" + 0.013*"can"')

(4, '0.003*"mechanism" + 0.003*"loss" + 0.003*"data" + 0.003*"convex" + 0.003*"learning" + 0.003*"large" + 0.003*"propose" + 0.003*"sln" + 0.003*"parameter" + 0.003*"stability"')

Perplexity:

Per word perplexity: 64.203
