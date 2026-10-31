import time, logging
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
t=time.time()
m=Word2Vec(Text8Corpus('dl/text8'), vector_size=100, window=5, min_count=5, sg=1, negative=5, sample=1e-3, epochs=5, workers=2, seed=1, compute_loss=True)
m.save('w2v_text8.model'); print('done', time.time()-t, len(m.wv))
