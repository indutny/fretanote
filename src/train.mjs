import path from 'path';

import tf from '@tensorflow/tfjs-node';

import Data from './data.mjs';

const LR = 0.001;

const data = new Data();

function generateData(size) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < size; i++) {
    const s = data.sample();
    xs.push(s.fft);

    // Cents-scale
    ys.push([
      100 * Math.log(s.freq / 440) / Math.log(2),
      s.presence ? 1 : -1,
    ]);
  }

  return {
    xs: tf.tensor(xs),
    ys: tf.tensor(ys),
  };
}

async function main() {
  const model = tf.sequential();

  const name = '2048-32x4-1';

  model.add(tf.layers.dense({
    inputShape: [ data.fftSize ],
    units: 32,
    activation: 'relu',
  }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 2 }));

  model.compile({
    optimizer: tf.train.adam(LR),
    loss: 'meanSquaredError',
    metrics: [ tf.metrics.meanAbsoluteError ],
  });

  const datasets = new Map();
  console.log('Training: ' + name);

  for (let epoch = 0; epoch < 100000; epoch++) {
    const ds = tf.tidy(() => generateData(512));

    await model.fit(ds.xs, ds.ys, {
      initialEpoch: epoch,
      epochs: epoch + 1,
      verbose: 0,
      callbacks: [ tf.node.tensorBoard('./logs/' + name) ],
    });
    tf.dispose(ds);

    if (epoch % 100 === 0) {
      await model.save('file://' + path.resolve('./saves/' + name));
    }
  }
}


main().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
