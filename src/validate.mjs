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
    for (let i = 0; i < 128; i++) {
      xs.push(data.note(freq).fft);
      ys.push(freq);
    }
  }

  console.time('predict');
  let [ predFreq, predPres ] =
    await model.predict(tf.tensor(xs)).split([ 1, 1 ], -1);
  console.timeEnd('predict');
  predFreq = await predFreq.squeeze(-1)
    .div(100).mul(Math.log(2))
    .exp().mul(440).array();

  predPres = await predPres.squeeze(-1).array();

  const map = new Map();
  for (const [ i, freq ] of ys.entries()) {
    const predicted = predFreq[i];
    const error = Math.abs(Math.log(predicted / freq) / Math.log(2) * 100);
    const present = predPres[i] > 0 ? 1 : 0;

    if (map.has(freq)) {
      const entry = map.get(freq);
      entry.errors.push(error);
      entry.present.push(present);
    } else {
      map.set(freq, { errors: [ error ], present: [ present ] });
    }
  }

  const list = [];
  for (const [ key, value ] of map) {
    list.push({
      freq: key,
      errors: stats(value.errors),
      present: stats(value.present),
    });
  }

  list.sort((a, b) => {
    return b.errors.mean - a.errors.mean;
  });

  const bestMean = list[list.length - 1];
  console.log('least mean error', bestMean);

  const worstMean = list[0];
  console.log('most mean error', worstMean);

  list.sort((a, b) => {
    return b.present.mean - a.present.mean;
  });

  const worstPresent = list[0];
  console.log('worst mean present', worstMean);
}

main(process.argv[2]).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
