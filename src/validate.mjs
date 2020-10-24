import path from 'path';
import tf from '@tensorflow/tfjs-node';

import Data from './data.mjs';

function stats(l) {
  let mean = 0;
  let stddev = 0;

  for (const e of l) {
    mean += e;
    stddev += e ** 2;
  }

  mean /= l.length;
  stddev = Math.sqrt(stddev / l.length - mean ** 2);

  return { mean, stddev };
}

async function main(filename) {
  const data = new Data();

  const model = await tf.loadLayersModel(`file://${path.resolve(filename)}`);

  const notes = [];

  for (let i = 0; i < 55; i++) {
    const freq = 55 * Math.pow(2, i / 12);

    notes.push(freq);
  }

  const xs = [];
  const ys = [];
  for (const freq of notes) {
    for (let i = 0; i < 64; i++) {
      xs.push(data.sample(freq).fft);
      ys.push(freq);
    }
  }

  console.time('predict');
  let prediction = await model.predict(tf.tensor(xs));
  console.timeEnd('predict');
  prediction = await prediction.squeeze(-1)
    .div(100).mul(Math.log(2))
    .exp().mul(440).array();

  const errors = new Map();
  for (const [ i, freq ] of ys.entries()) {
    const predicted = prediction[i];
    const error = Math.abs(Math.log(predicted / freq) / Math.log(2) * 100);

    if (errors.has(freq)) {
      errors.get(freq).push(error);
    } else {
      errors.set(freq, [ error ]);
    }
  }

  const list = [];
  for (const [ key, value ] of errors) {
    list.push({ freq: key, ...stats(value) });
  }

  list.sort((a, b) => {
    return b.mean - a.mean;
  });

  const bestMean = list[list.length - 1];
  console.log('best mean', bestMean);

  const worstMean = list[0];
  console.log('worst mean', worstMean);

  list.sort((a, b) => {
    return b.stddev - a.stddev;
  });

  const worstStddev = list[0];
  console.log(worstMean);
}

main(process.argv[2]).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
