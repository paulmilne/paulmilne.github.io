# Embeddings series

Code behind `ai/embeddings-part-1-words-to-vectors` and `ai/embeddings-part-2-search-and-retrieval`.
Python 3 with numpy, scipy, scikit-learn, gensim, safetensors and tokenizers. Model weights are not in git:
put them in `_models/` at the root of the blog (ignored by git and by Quarto).

| File | Purpose |
|---|---|
| `bpe.py` | Byte-pair encoding learned on text8 word counts (writes `bpe_text8.json`; the page keeps the first 8,000 merges). |
| `w2v_train.py` | word2vec skip-gram with negative sampling on text8 (gensim). |
| `pmi_check.py` | Compares trained dot products with shifted PMI counted with gensim's window weighting and subsampling. |
| `glove_an.py` | GloVe analogy accuracy with and without excluding the input words; neighbours. |
| `export_glove.py` | The 20,000-word, 8-bit GloVe table used by the page. |
| `gpt2_stats.py` | GPT-2 embedding table: norms, anisotropy, tokens nearest the mean direction, neighbours, position-embedding PCA. |
| `minilm_np.py` | all-MiniLM-L6-v2 forward pass in NumPy (BERT encoder, mean pooling, L2 normalisation). |
| `bm25.py` | Okapi BM25. |
| `chunk_blog.py` | Splits the blog's posts into passages. |
| `queries.json` | The 30 test questions and their relevant post indices. |
| `build_part2.py` | Embeds the passages, evaluates BM25 / embedding / hybrid, compression, geometry, t-SNE; writes Part 2's data. |

Data sources: text8 and `glove-wiki-gigaword-100` from the gensim-data GitHub releases;
`openai-community/gpt2` and `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face.

`gpt2_stats.py` reads only the embedding tables (`wte`, `wpe`), extracted from GPT-2's `model.safetensors` into
`_models/gpt2-embeddings.safetensors`, and a JSON list of token strings by id (from the `gpt-tokenizer` npm package's
`r50k_base` encoding, which is GPT-2's).

Part 2's passages exclude Part 2 itself, because it contains the test questions:

```bash
python3 chunk_blog.py ../.. blog_chunks.json
python3 build_part2.py ../../_models blog_chunks.json queries.json \
  ../../ai/embeddings-part-2-search-and-retrieval/data/blog_index.json
```
