import path from 'path';

import tf from '@tensorflow/tfjs-node';

import Data from './data.mjs';

const LR = 0.001;

const LOG_DIR = path.join('.', 'logs');
const SAVE_DIR = path.join('.', 'saves');

const data = new Data();

function generateFreqData(size) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < size; i++) {
    const s = data.note();
    xs.push(s.fft);

    // Cents-scale
    ys.push(100 * Math.log(s.freq / 440) / Math.log(2));
  }

  return {
    xs: tf.tensor(xs),
    ys: tf.tensor(ys),
  };
}

function generateNoteData(size) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < size; i++) {
    if (Math.random() > 0.5) {
      xs.push(data.note().fft);
      ys.push([ 0, 1 ]);
    } else {
      xs.push(data.noise().fft);
      ys.push([ 1, 0 ]);
    }
  }

  return {
    xs: tf.tensor(xs),
    ys: tf.tensor(ys),
  };
}

async function main() {
  const freq = tf.sequential({
    name: 'freq-32x4-1',
  });

  freq.add(tf.layers.dense({
    inputShape: [ data.fftSize >>> 1 ],
    units: 32,
    activation: 'relu',
  }));
  freq.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  freq.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  freq.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  freq.add(tf.layers.dense({ units: 1 }));

  const detect = tf.sequential({
    name: 'detect-32x4-1',
  });

  detect.add(tf.layers.dense({
    inputShape: [ data.fftSize >>> 1 ],
    units: 4,
    activation: 'relu',
  }));
  detect.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

  freq.compile({
    optimizer: tf.train.adam(LR),
    loss: 'meanSquaredError',
    metrics: [ tf.metrics.meanAbsoluteError ],
  });

  detect.compile({
    optimizer: tf.train.adam(LR),
    loss: 'meanSquaredError',
    metrics: [ tf.metrics.binaryAccuracy ],
  });

  const models = [ {
    model: freq,
    callbacks: [ tf.node.tensorBoard(path.join(LOG_DIR, freq.name)) ],
    genData: () => generateFreqData(512),
  }, {
    model: detect,
    callbacks: [ tf.node.tensorBoard(path.join(LOG_DIR, detect.name)) ],
    genData: () => generateNoteData(512),
  }].filter((m) => {
    return m.model.name.includes(process.argv[2]);
  });
  console.log(`Training: ${models.map((m) => m.model.name).join(', ')}`);

  for (let epoch = 0; epoch < 100000; epoch++) {
    for (const { model, callbacks, genData } of models) {
      const ds = tf.tidy(() => genData());

      await model.fit(ds.xs, ds.ys, {
        initialEpoch: epoch,
        epochs: epoch + 1,
        verbose: 0,
        callbacks,
      });
      tf.dispose(ds);

      if (epoch % 100 === 0) {
        await model.save('file://' +
          path.resolve(path.join(SAVE_DIR, model.name)));
      }
    }
  }
}


main().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
