const httpServer = require('http-server');
const puppeteer = require('puppeteer');

let server;
let browser;
let page;

beforeAll(async () => {
  server = httpServer.createServer({ root: 'frontend' });
  await new Promise(resolve => server.listen(3000, resolve));
  browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  page = await browser.newPage();
});

afterAll(async () => {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
});

test('Load time under 2.2s', async () => {
  const start = Date.now();
  await page.goto('http://localhost:3000/consumer.html');
  const loadTime = Date.now() - start;
  expect(loadTime).toBeLessThan(2200);
});

test('CTA button click triggers onboarding', async () => {
  await page.goto('http://localhost:3000/consumer.html');
  await page.click('#register-btn');
  await page.waitForSelector('#onboarding', { visible: true });
  const display = await page.$eval('#onboarding', el => el.style.display);
  expect(display).toBe('block');
});

test('Scroll triggers demo button in enterprise view', async () => {
  await page.goto('http://localhost:3000/enterprise.html');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight / 2));
  await page.waitForSelector('#demo-btn', { visible: true });
  const display = await page.$eval('#demo-btn', el => el.style.display);
  expect(display).toBe('block');
});

test('Fallback visual loads', async () => {
  await page.goto('http://localhost:3000/consumer.html');
  const visual = await page.$('.spectral-visual');
  expect(visual).not.toBeNull();
});

test('UVP above fold', async () => {
  await page.goto('http://localhost:3000/consumer.html');
  const headerBounds = await page.$eval('header', el => el.getBoundingClientRect());
  expect(headerBounds.top).toBeLessThanOrEqual(0);
  expect(headerBounds.height).toBeGreaterThan(500);
});
