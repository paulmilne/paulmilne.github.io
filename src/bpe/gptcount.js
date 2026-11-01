const fs = require('fs');
const text = fs.readFileSync('/home/claude/emb/dl/text8', 'latin1').slice(95000000);
const out = {};
for (const enc of ['r50k_base', 'cl100k_base', 'o200k_base']) {
  const m = require('/tmp/cdn/node_modules/gpt-tokenizer/cjs/encoding/' + enc + '.js');
  let n = 0; const CH = 200000;
  for (let i = 0; i < text.length; i += CH) { let j = Math.min(text.length, i + CH); while (j < text.length && text[j] !== ' ') j++; n += m.encode(text.slice(i, j)).length; i = j - CH; }
  out[enc] = {tokens: n, chars: text.length, cpt: text.length / n}; console.log(enc, out[enc]);
}
fs.writeFileSync('/home/claude/bpe/gpt_counts.json', JSON.stringify(out));
