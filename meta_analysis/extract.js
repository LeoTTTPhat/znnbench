// Rubric-oriented text extraction for a paper opened in the built-in browser.
// Paste the whole file as the `text` of mcp__Claude_Browser__javascript_tool (always pass your own tabId).
// Optional first line: `var EXTRA = /regex/i;` (extra search terms), `var MAXC = 20000;` (output cap),
// `var FROM = 0.0;` (fraction of the document where the experiment search starts; default = first
// simulation/experiment heading).
// It (1) turns rendered formulas back into text (MathJax 3 lazy items, MathJax 2 scripts, MathML),
// (2) splits the page text into lines and cuts at the reference list, and returns (3) the section headings,
// (4) "anywhere" hits (discretization analysis, step size, code/data availability, hardware) and
// (5) "experiment" hits: lines of the simulation/experiment part that mention solvers, steps, gains,
// baselines, noise, dimensions or timing.  Lines are capped at 420 characters; output at MAXC characters.
var EXTRA = typeof EXTRA === 'undefined' ? null : EXTRA, MAXC = typeof MAXC === 'undefined' ? 14000 : MAXC,
    FROM = typeof FROM === 'undefined' ? null : FROM;
await new Promise(r => setTimeout(r, 3000));
let nmath = 0;
try { if (window.MathJax && MathJax.startup && MathJax.startup.document) {
  for (const it of [...MathJax.startup.document.math]) {
    const r = it.typesetRoot || (it.start && it.start.node);
    if (r && r.parentNode) { r.replaceWith(document.createTextNode(' ' + String(it.math || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() + ' ')); nmath++; }
  } } } catch (e) {}
document.querySelectorAll('script[type^="math/"]').forEach(s => { s.replaceWith(document.createTextNode(' $' + s.textContent.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() + '$ ')); nmath++; });
document.querySelectorAll('.MathJax_Preview, .MathJax, .MathJax_Display, .MJX_Assistive_MathML, mjx-assistive-mml').forEach(e => e.remove());
document.querySelectorAll('math').forEach(m => { m.replaceWith(document.createTextNode(' ' + (m.getAttribute('alttext') || m.textContent).replace(/\s+/g, ' ') + ' ')); nmath++; });
const body = document.body ? document.body.innerText : '';
const lines = body.split(/\n+/).map(s => s.trim()).filter(Boolean);
let end = lines.length;
for (let i = lines.length - 1; i > lines.length * 0.45; i--) { if (/^(\d+\.?\s*|[IVX]+\.\s*)?(references|bibliography|literature cited)$/i.test(lines[i])) { end = i; break; } }
const isHead = s => s.length < 110 && /^(\d+(\.\d+)*\.?|[IVX]+\.|[A-H]\.)\s+[A-Z]/.test(s);
const heads = []; lines.slice(0, end).forEach((s, i) => { if (isHead(s)) heads.push(i + ': ' + s); });
let sim = Math.floor(end * 0.3);
if (FROM !== null) sim = Math.floor(end * FROM);
else { const h = lines.slice(0, end).findIndex((s, i) => i > end * 0.25 && s.length < 110 && /simulat|experiment|numerical|illustrative|example|verification|case stud|application|comparison|results/i.test(s)); if (h > 0) sim = h; }
const ANY = /step[- ]?size|sampling (gap|period|interval|time|rate|frequency)|truncation error|O\(\s*(τ|\\tau|h)|discrete[- ]time (model|znn)|zero[- ]stab|github|gitee|zenodo|code ocean|source code|code (is|are) (available|provided)|data availab|supplementary (material|code)|on request|fpga|circuit implement|hardware|physical (experiment|robot|platform)|real robot|kinova|ur5|franka|puma/i;
const EXP = /ode ?45|ode15s|ode23|solver|reltol|abstol|tolerance|simulink|matlab|python|euler|runge|kutta|zead|taylor|n-instant|\bτ\b|\btau\b|\bh\s*=|step|sampling|interval|\bγ\b|gamma|\bλ\b|lambda|\bζ\b|\bμ\b|\bη\b|design parameter|scaling factor|gain|convergence time|settling time|upper bound|predefined|fixed[- ]time|finite[- ]time|noise|dimension|\bn\s*=|×|compared|comparison|existing|baseline|\bgnn\b|model[s]? \(|initial (state|value)|random/i;
const any = [], exp = [];
lines.slice(0, end).forEach((s, i) => {
  if (s.length < 20 || isHead(s)) return;
  if (ANY.test(s) || (EXTRA && EXTRA.test(s))) any.push(i + ': ' + s.slice(0, 420));
  else if (i >= sim && EXP.test(s)) exp.push(i + ': ' + s.slice(0, 420));
});
const low = body.toLowerCase();
const info = {title: document.title.slice(0, 140), url: location.href, chars: body.length, nlines: lines.length, refs_at: end, exp_from: sim, nmath,
  full_text: body.length > 15000 && /conclusion/.test(low) && /(references|bibliography)/.test(low) && !/just a moment|verify you are human|access denied/.test(low),
  heads: heads.slice(0, 45), n_any: any.length, any: any.slice(0, 25), n_exp: exp.length, exp};
JSON.stringify(info).slice(0, MAXC);
