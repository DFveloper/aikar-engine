const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require('../../../tools/ui/node_modules/playwright');
(async () => {
 const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
 const errors = [];
 const base = await (await fetch('http://127.0.0.1:8787/api/status')).json();
 for (const fixture of ['live', 'epoch2']) {
  const page = await browser.newPage({viewport:{width:1280,height:1000}});
  page.on('pageerror', e=>errors.push(e.message));
  if (fixture === 'epoch2') {
   const sample = {...base, epoch:2, chart_start:0, chart_current:100.5, current:100, total:200};
   sample.history = Array.from({length:240}, (_, i)=>({step:100.5*i/239,loss:2+Math.sin(i/20)*0.3,accuracy:50+i/24}));
   sample.nll_history = sample.history.map(p=>({step:p.step,nll:p.loss}));
   sample.validation_history = Array.from({length:16},(_, i)=>({step:i*6,loss:2.1+i/100,accuracy:51+i/3}));
   const body = process.env.WEBUI_EPOCH2_FIXTURE
     ? fs.readFileSync(process.env.WEBUI_EPOCH2_FIXTURE,'utf8') : JSON.stringify(sample);
   await page.route('**/api/status*', route=>route.fulfill({contentType:'application/json',body}));
  }
  await page.goto('http://127.0.0.1:8787');
  await page.waitForFunction(()=>document.getElementById('loss').textContent!=='--');
  for (const width of [1280, 390]) {
   await page.setViewportSize({width,height:1000});
   for (const combined of [false,true]) {
    await page.locator('#combine-charts').setChecked(combined);
    const result = await page.evaluate(()=>({
     nan: document.body.innerText.includes('NaN'),
     overflow: document.body.scrollWidth > innerWidth,
     train: document.querySelector('.shell').classList.contains('combined')
       ? document.querySelector('#combined-weighted-loss-chart .line')?.getAttribute('d')
       : document.querySelector('#loss-chart .line')?.getAttribute('d'),
     invalidPaths: [...document.querySelectorAll('svg path')].some(p=>/NaN|Infinity/.test(p.getAttribute('d'))),
     valXs: [...document.querySelectorAll('#combined-accuracy-chart .validation-point')].map(e=>Number(e.getAttribute('cx'))),
     singleValArea: !!document.querySelector('#validation-loss-chart .area')
    }));
    assert.equal(result.nan,false);
    assert.equal(result.overflow,false);
    assert.equal(result.invalidPaths,false);
    assert.ok(result.train.split(' L').length > 100, 'microbatch curve was discarded');
    assert.ok(Number(result.train.match(/^M([\d.]+)/)[1]) < 5, 'first training point was shifted');
    if (fixture === 'epoch2') assert.ok(new Set(result.valXs).size > 10, 'Val markers collapse');
    else assert.equal(result.singleValArea,false, 'one Val point must not produce an area');
    await page.screenshot({path:`/tmp/pulsar-recompute/ui-fixed-${fixture}-${width}-${combined?'combined':'separate'}.png`,fullPage:true});
   }
  }
  await page.close();
 }
 assert.deepEqual(errors,[]);
 await browser.close();
 console.log('Browser checks passed: live/epoch2, desktop/mobile, separate/combined, first point and Val positions');
})().catch(error=>{console.error(error);process.exit(1)});
