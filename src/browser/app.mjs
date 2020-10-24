import * as tf from '@tensorflow/tfjs';
import Data from '../data.mjs';

const out = document.getElementById('freq');

async function main() {
  const detect = await tf.loadLayersModel('tf/detect/model.json');
  const freq = await tf.loadLayersModel('tf/freq/model.json');

  const fftSize = detect.inputs[0].shape[1] * 2;

  const data = new Data({ fftSize });

  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  analyser.fftSize = fftSize;
  const buffer = new Float32Array(fftSize);

  const source = ctx.createMediaStreamSource(stream);
  source.connect(analyser);

  for (;;) {
    analyser.getFloatTimeDomainData(buffer);

    const fft = data.fromBuffer(buffer);

    let prediction = await
      detect.predict(tf.tensor([ fft ])).squeeze().array();

    if (prediction[0] > prediction[1]) {
      out.textContent = '';
    } else {
      prediction = await freq.predict(tf.tensor([ fft ]))
        .squeeze()
        .div(100).mul(Math.log(2))
        .exp().mul(440).array();

      out.textContent = prediction.toFixed(2);
    }

    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  }
}

main();
