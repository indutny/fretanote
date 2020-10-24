import * as tf from '@tensorflow/tfjs';
import FFT from 'fft.js';

const out = document.getElementById('freq');

function normalize(l, len) {
  let max = 1e-23;
  for (let i = 0; i < buffer.length; i++) {
    max = Math.max(max, Math.abs(buffer[i]));
  }
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] /= max;
  }
}

async function main() {
  const detect = await tf.loadLayersModel('tf/detect/model.json');
  const freq = await tf.loadLayersModel('tf/freq/model.json');

  const fft = new FFT(model.inputs[0].shape[1]);
  const fftOut = fft.createComplexArray();

  let lastFreq = 0;

  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();

  const stream = await navigator.mediaDevices.getUserMedia ({ audio: true });

  analyser.fftSize = fft.size;
  const buffer = new Float32Array(fft.size);

  const source = ctx.createMediaStreamSource(stream);
  source.connect(analyser);

  for (;;) {
    analyser.getFloatTimeDomainData(buffer);

    // Normalize
    normalize(buffer);

    // Compute FFT
    fft.realTransform(fftOut, buffer);
    for (let i = 0; i < fft.size; i += 2) {
      fftOut[i >>> 1] = Math.sqrt(fftOut[i] ** 2 + fftOut[i + 1] ** 2);
    }

    let prediction = await model.predict(
      tf.tensor([ fftOut.slice(0, fft.size) ]));
    prediction = await prediction.squeeze()
      .div(100).mul(Math.log(2))
      .exp().mul(440).array();

    lastFreq = lastFreq * 0.99 + prediction * 0.01;

    out.textContent = lastFreq.toFixed(2);

    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  }
}

main();
