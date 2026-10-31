const r=require('gpt-tokenizer/cjs/encoding/r50k_base');
const S=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(JSON.stringify(S.map(s=>r.encode(s))));
