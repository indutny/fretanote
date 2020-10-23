'use strict';

const TRIALS = 1000;
const BINS = 4;
const SAMPLE_RATE = 44100;
const LR = 0.1;

function waveform(t, freq = 440) {
  return Math.sin(2 * Math.PI * ((freq / SAMPLE_RATE) * t ));
}

const logFreq = { mean: Math.log(220), stddev: 4 };
const logAmp = { mean: Math.log(0.5), stddev: 1 };
const phase = { mean: 0, stddev: 0.25 };

const newLogFreq = { mean: 0, stddev: 0 };
const newLogAmp = { mean: 0, stddev: 0 };
const newPhase = { mean: 0, stddev: 0 };

let totalLFD = 0;
let totalLAD = 0;
let totalPD = 0;

function summarize(p, n) {
  p.mean /= n;
  p.stddev = Math.sqrt(Math.max(0, p.stddev / n - p.mean ** 2));
}

function interpolate(p1, p2, lr) {
  p1.mean = p1.mean + (p2.mean - p1.mean) * lr;
  p1.stddev = p1.stddev  + (p2.stddev - p1.stddev) * lr;
}

function normalDist(val, config) {
  return 1 / Math.sqrt(2 * Math.PI) * Math.exp(-Math.pow(val, 2) / 2);
}

function update(t, actual) {
  const binnedActual = Math.round((actual + 1) * BINS);

  for (let i = 0; i < TRIALS; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const u3 = Math.random();
    const u4 = Math.random();

    const n1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const n2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
    const n3 = Math.sqrt(-2 * Math.log(u3)) * Math.sin(2 * Math.PI * u4);

    const randomLogFreq = n1 * logFreq.stddev + logFreq.mean;
    const randomLogAmp = n2 * logAmp.stddev + logAmp.mean;
    const randomPhase = n3 * phase.stddev + phase.mean;

    const randomFreq = Math.exp(randomLogFreq);
    const randomAmp = Math.exp(randomLogAmp);

    const prediction = randomAmp *
      Math.sin(2 * Math.PI * (randomFreq / SAMPLE_RATE * t));
    const binnedPrediction = Math.round((prediction + 1) * BINS);

    if (binnedPrediction === binnedActual) {
      const lfd = normalDist(n1);
      const lad = normalDist(n2);
      const pd = normalDist(n3);

      newLogFreq.mean += randomLogFreq * lfd;
      newLogFreq.stddev += randomLogFreq ** 2 * lfd;
      newLogAmp.mean += randomLogAmp * lad;
      newLogAmp.stddev += randomLogAmp ** 2 * lad;
      newPhase.mean += randomPhase * pd;
      newPhase.stddev += randomPhase ** 2 * pd;

      totalLFD += lfd;
      totalLAD += lad;
      totalPD += pd;
    }
  }

  if (totalLFD >= 100) {
    summarize(newLogFreq, totalLFD);
    interpolate(logFreq, newLogFreq, LR);
    newLogFreq.mean = 0;
    newLogFreq.stddev = 0;
    totalLFD = 0;
  }
  if (totalLAD >= 100) {
    summarize(newLogAmp, totalLAD);
    interpolate(logAmp, newLogAmp, LR);
    newLogAmp.mean = 0;
    newLogAmp.stddev = 0;
    totalLAD = 0;
  }
  if (totalPD >= 100) {
    summarize(newPhase, totalPD);
    interpolate(phase, newPhase, LR);
    newPhase.mean = 0;
    newPhase.stddev = 0;
    totalPD = 0;
  }

  return 0;
}

let t = 0;
while (true) {
  t++;
  const hits = update(t, waveform(t, 440));

  if (t % 100 !== 0) {
    continue;
  }
  console.log(`t=${t} hits=${hits} f=${Math.exp(logFreq.mean).toFixed(2)} ` +
    `lfd=${logFreq.stddev.toFixed(2)} ` +
    `a=${Math.exp(logAmp.mean).toFixed(2)} ` +
    `lad=${logAmp.stddev.toFixed(2)} ` +
    `phase=${phase.mean.toFixed(2)} ` +
    `pd=${phase.stddev.toFixed(2)}`);
}
