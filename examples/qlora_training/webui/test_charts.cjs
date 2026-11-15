const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {
    innerHTML: '', textContent: '', style: {setProperty() {}},
    setAttribute() {}, focus() {}, setPointerCapture() {}, releasePointerCapture() {},
    querySelector() { return {remove: () => { this.innerHTML = this.innerHTML.replace(/<g class="chart-analysis">[\s\S]*?<\/g>/, ''); }}; },
    insertAdjacentHTML(position, html) { this.innerHTML += html; },
    getScreenCTM() { return {inverse() { return {}; }}; },
    createSVGPoint() { return {x: 0, matrixTransform() { return {x: this.x}; }}; },
  });
  return elements.get(id);
};
const source = fs.readFileSync(`${__dirname}/app.js`, 'utf8');
const context = vm.createContext({$, value: (id, v) => $(id).textContent = v});
vm.runInContext(source.slice(source.indexOf('function chartTrendLine'), source.indexOf('function render(data)')), context);
context.drawChart('chart', 'range', [{step: 0, loss: 1}, {step: 1, loss: 2}, {step: 10, loss: 3}], 'loss', 'red');
assert.match($('chart').innerHTML, /L60\.0,/);
context.drawComparisonChart('combined', 'range', [], [{step: 5, loss: 2}], 'loss', 'red', 'blue');
assert.match($('combined').innerHTML, /validation-point/);
context.drawComparisonChart('combined', 'range', [{step: 0, loss: 1}, {step: 10, loss: 2}], [{step: 5, loss: 2}], 'loss', 'red', 'blue');
assert.match($('combined').innerHTML, /cx="300\.0"/);
assert.doesNotMatch($('combined').innerHTML, /NaN|Infinity/);
context.drawChart('chart', 'range', [{step: 0, loss: 1}, {step: 1, loss: 2}, {step: 10, loss: 999}], 'loss', 'red', {start: 0, end: 2});
assert.match($('chart').innerHTML, /L300\.0,10\.0/);
assert.doesNotMatch($('range').innerHTML, /999/);
context.drawComparisonChart('combined', 'range', [{step: 0, loss: 1}, {step: 1, loss: 2}], [{step: 1, loss: 2}, {step: 10, loss: 999}], 'loss', 'red', 'blue', {start: 0, end: 2});
assert.match($('combined').innerHTML, /cx="300\.0" cy="10\.0"/);
assert.doesNotMatch($('range').innerHTML, /999/);
context.drawChart('chart', 'range', [{step: 0, loss: 1}, {step: 0.25, loss: 2}], 'loss', 'red', {start: 0, end: 0.5});
assert.match($('chart').innerHTML, /L300\.0,10\.0/);
context.drawChart('single', 'range', [{step: 0, loss: 2}], 'loss', 'red', {start: 0, end: 5});
assert.doesNotMatch($('single').innerHTML, /class="area"/);
assert.doesNotMatch($('single').innerHTML, /class="line"/);
assert.match($('single').innerHTML, /class="point"/);
const samples = Array.from({length: 30}, (_, i) => ({step: i, loss: i + 1}));
context.drawChart('analysis', 'analysis-range', samples, 'loss', 'red', {start: 0, end: 30});
const chart = $('analysis');
const pointer = x => ({button: 0, isPrimary: true, pointerId: 1, clientX: x, clientY: 50, preventDefault() {}});
chart.onpointerdown(pointer(600));
chart.onpointermove(pointer(200));
chart.onpointerup(pointer(200));
assert.match(chart.innerHTML, /lower-trendline/);
assert.match(chart.innerHTML, /upper-trendline/);
assert.match(chart.innerHTML, /moving-average/);
assert.match($('analysis-range').innerHTML, /Lower 11\.0000 -> 31\.0000 .*기울기 1\.00\/step.*Upper 11\.0000 -> 31\.0000/);
context.drawChart('analysis', 'analysis-range', samples, 'loss', 'red', {start: 0, end: 30});
assert.equal((chart.innerHTML.match(/class="chart-analysis"/g) || []).length, 1);
assert.match($('analysis-range').innerHTML, /SMA20 20\.5000/);
chart.onkeydown({key: 'Escape'});
assert.doesNotMatch(chart.innerHTML, /lower-trendline/);
chart.onpointerdown(pointer(300));
chart.onpointerup(pointer(300));
assert.doesNotMatch(chart.innerHTML, /lower-trendline/);
chart.onpointerdown(pointer(0));
chart.onpointermove(pointer(1000));
chart.onpointerup(pointer(1000));
assert.match($('analysis-range').innerHTML, /SMA20 20\.5000/);
chart.onpointercancel();
chart.onpointermove(pointer(100));
assert.equal(chart.analysisPointer, null);
context.drawComparisonChart('comparison', 'comparison-range', samples, [{step: 15, loss: 100}], 'loss', 'red', 'blue', {start: 0, end: 30});
$('comparison').onpointerdown(pointer(0));
$('comparison').onpointerup(pointer(600));
assert.match($('comparison-range').innerHTML, /Train: Lower 1\.0000 -> 31\.0000/);
assert.match($('comparison-range').innerHTML, /Val: 추세선에 최소 2개 step 필요/);
assert.doesNotMatch($('comparison').innerHTML, /NaN|Infinity/);
console.log('Chart coordinates, selection, upper/lower trendlines, SMA20 and refresh passed');

for (const values of [[1,4,3,6,5,8],[9,7,8,4,5,2],[2,2,2,2]]) {
  const points = values.map((loss,i) => ({step: i*i+1,loss}));
  for (const side of [1,-1]) {
    const line = context.chartTrendLine(points,'loss',side);
    const touches = points.filter(p => Math.abs(p.loss-line.at(p.step)) < 1e-9);
    assert.ok(touches.length >= 2, 'trend must touch at least two points');
    assert.ok(points.every(p => side*(p.loss-line.at(p.step)) >= -1e-9), 'upper/lower trendlines must bound the selected samples');
    if(values[0] === 2) assert.equal(line.slope,0);
    else assert.ok(values[0] === 1 ? line.slope > 0 : line.slope < 0, 'rising/falling trend must have the correct slope');
  }
}
const lines = [...chart.innerHTML.matchAll(/<line class="(?:lower|upper)-trendline"[^>]*y1="([^"]+)"[^>]*y2="([^"]+)"/g)];
assert.equal(lines.length,2);
assert.ok(lines.every(m => Number(m[1]) !== Number(m[2])), 'rendered trend lines must slope');
assert.equal(context.chartTrendLine([{step:1,loss:1},{step:1,loss:3}],'loss',1),null);
console.log('Sloped upper/lower trendlines: two contacts, envelope bounds, irregular steps, rising/falling and flat data passed');
