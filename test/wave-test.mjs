import assert from 'assert';

import Data from '../src/data.mjs';

describe('waves', () => {
  const d = new Data();

  function run(f) {
    const out = [];
    for (let i = 0; i < 16; i++) {
      out.push(f(0.125, i).toFixed(2));
    }
    return out;
  }

  it('should generate sine', () => {
    assert.deepStrictEqual(run((a, b) => d.sin(a, b)), [
      '0.00', '0.71', '1.00', '0.71', '0.00', '-0.71', '-1.00', '-0.71',
      '-0.00', '0.71', '1.00', '0.71', '0.00', '-0.71', '-1.00', '-0.71',
    ]);
  });

  it('should generate square', () => {
    assert.deepStrictEqual(run((a, b) => d.square(a, b)), [
      '1.00', '1.00', '1.00', '1.00', '1.00', '-1.00', '-1.00', '-1.00',
      '-1.00', '1.00', '1.00', '1.00', '1.00', '-1.00', '-1.00', '-1.00',
    ]);
  });

  it('should generate saw', () => {
    assert.deepStrictEqual(run((a, b) => d.saw(a, b)), [
      '-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75',
      '-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75',
    ]);
  });
});
