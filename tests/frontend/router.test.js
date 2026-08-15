const test = require('node:test');
const assert = require('node:assert');

const { ROUTES, parseRoute, hashFor, panelVisibility } = require('../../frontend/router.js');

test('ルートは静止画抽出 / 動画 / カタログの3つ', () => {
  assert.deepStrictEqual(ROUTES, ['extract', 'video', 'catalog']);
});

test('ハッシュ無しは抽出画面', () => {
  assert.strictEqual(parseRoute(''), 'extract');
  assert.strictEqual(parseRoute('#'), 'extract');
  assert.strictEqual(parseRoute('#/'), 'extract');
});

test('既知のハッシュをルートへ変換する', () => {
  assert.strictEqual(parseRoute('#/catalog'), 'catalog');
  assert.strictEqual(parseRoute('#/video'), 'video');
  assert.strictEqual(parseRoute('#/extract'), 'extract');
});

test('未知のハッシュは抽出画面へフォールバックする', () => {
  assert.strictEqual(parseRoute('#/nope'), 'extract');
  assert.strictEqual(parseRoute('#catalog/../admin'), 'extract');
});

test('ルートからハッシュを作る', () => {
  assert.strictEqual(hashFor('catalog'), '#/catalog');
  assert.strictEqual(hashFor('extract'), '#/extract');
});

test('抽出画面ではカタログと動画UIを隠す', () => {
  assert.deepStrictEqual(panelVisibility('extract'), {
    controls: true,
    status: true,
    stage: true,
    catalogPanel: false,
    videoMode: false,
  });
});

test('カタログ画面ではカタログだけ出す', () => {
  assert.deepStrictEqual(panelVisibility('catalog'), {
    controls: false,
    status: false,
    stage: false,
    catalogPanel: true,
    videoMode: false,
  });
});

test('動画画面では動画UIだけ出す', () => {
  assert.deepStrictEqual(panelVisibility('video'), {
    controls: false,
    status: false,
    stage: false,
    catalogPanel: false,
    videoMode: true,
  });
});
