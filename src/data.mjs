import FFT from 'fft.js';

export default class Data {
  constructor({ sampleRate = 44100, fftSize = 2048 } = {}) {
    this.sampleRate = sampleRate;
    this.fftSize = fftSize;

    this.fft = new FFT(fftSize);
    this.fftIn = new Array(fftSize);
    this.fftOut = this.fft.createComplexArray();

    this.waves = [
      (a, b) => this.sin(a, b),
      (a, b) => this.square(a, b),
      (a, b) => this.saw(a, b),
    ];
  }

  normalize(list) {
    let max = 0;
    for (let i = 0; i < list.length; i++) {
      max = Math.max(max, Math.abs(list[i]));
    }

    for (let i = 0; i < list.length; i++) {
      list[i] /= max;
    }
  }

  sin(freq, t) {
    return Math.sin(2 * Math.PI * freq * t);
  }

  square(freq, t) {
    return this.sin(freq, t) >= 0 ? 1 : -1;
  }

  saw(freq, t) {
    return 2 * (t % (1 / freq)) * freq - 1;
  }

  sample(freq) {
    // E6=1318
    // A1=55
    if (!freq) {
      freq = Math.pow(2, Math.random() * 4.7) * 55;
    }
    const f = freq / this.sampleRate;
    const phase = Math.random();
    const harmonicFade = 0.25 + 0.25 * Math.random();

    const presence = Math.random() > 0.5;

    const wave = this.waves[Math.random(this.waves.length) | 0];

    this.fftIn.fill(0);
    if (presence) {
      for (let t = 0; t < this.fftIn.length; t++) {
        let signal = 0;

        for (let h = 0; h < 3; h++) {
          const harmonicFreq = Math.pow(2, h) * f;
          signal += wave(harmonicFreq, t + phase / harmonicFreq) *
            Math.pow(harmonicFade, h);
        }

        this.fftIn[t] = signal;
      }
      this.normalize(this.fftIn);
    }

    const noise = 0.2 + Math.random() * 0.3;
    for (let t = 0; t < this.fftIn.length; t++) {
      this.fftIn[t] = this.fftIn[t] * (1 - noise) + Math.random() * noise;
    }
    this.normalize(this.fftIn);

    this.fft.realTransform(this.fftOut, this.fftIn);

    return {
      freq,
      presence,
      fft: this.fftOut.slice(0, this.fftSize),
    };
  }
}
