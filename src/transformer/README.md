# Transformer series

Code behind `ai/transformer-part-1-operation-by-operation`, `-part-2-inside-gpt-2` and `-part-3-advances`.
Part 1 is computed entirely in the page.

| File | Purpose |
|---|---|
| `gpt2_np.py` | GPT-2 small forward pass in NumPy (row-vector convention), returning attention, residual stream and sublayer outputs; logit lens. |
| `build_gpt2_viz.py` | Runs it on the example inputs and the repeated random sequence; writes Part 2's `data/gpt2_viz.json` (attention at 8 bits, logit lens, norms, head scores). |
| `enc.js` | Tokenises strings with GPT-2's `r50k_base` encoding (`npm install gpt-tokenizer`). |
| `timeline.py` | The 80 dated entries behind Part 3's timeline (`data/timeline.json`). |

Weights: `openai-community/gpt2` from Hugging Face, kept in `_models/` (not in git), split for transfer into
`gpt2-embeddings.safetensors` (wte, wpe), `gpt2-blocks-0-5.safetensors` and `gpt2-blocks-6-11.safetensors` (with ln_f).

```bash
cd src/transformer
ln -s ../../_models models
python3 build_gpt2_viz.py ../../ai/transformer-part-2-inside-gpt-2/data/gpt2_viz.json
```
