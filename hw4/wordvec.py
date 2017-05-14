import word2vec
import numpy as np
import nltk

# DEFINE your parameters for training
MIN_COUNT = 5
WORDVEC_DIM = 1500
WINDOW = 6
NEGATIVE_SAMPLES = 5
ITERATIONS = 150000
MODEL = 1
LEARNING_RATE = 0.0025

# train model
# cat all .txt files to a single train.txt
word2vec.word2vec(
    train='train.txt',
    output='model.bin',
    cbow=MODEL,
    size=WORDVEC_DIM,
    min_count=MIN_COUNT,
    window=WINDOW,
    negative=NEGATIVE_SAMPLES,
    iter_=ITERATIONS,
    alpha=LEARNING_RATE,
    verbose=True)

# load model for plotting
model = word2vec.load('model.bin')

vocabs = []
vecs = []
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:500]
vocabs = vocabs[:500]

'''
Dimensionality Reduction
'''
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)


'''
Plotting
'''
import matplotlib.pyplot as plt
from adjustText import adjust_text

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]


plt.figure(figsize=(12,8))
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y, c = np.random.rand(15,3))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

# plt.savefig('hp.png', dpi=600)
plt.show()
