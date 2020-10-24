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

  normalize(list, len = list.length) {
    let max = 1e-23;
    for (let i = 0; i < len; i++) {
      max = Math.max(max, Math.abs(list[i]));
    }

    for (let i = 0; i < len; i++) {
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

  downsampleFFT() {
    for (let i = 0; i < this.fftSize; i += 2) {
      this.fftOut[i >>> 1] = Math.sqrt(
        this.fftOut[i] ** 2 + this.fftOut[i + 1] ** 2);
    }
  }

  note(freq) {
    // E6=1318
    // A1=55
    if (!freq) {
      freq = Math.pow(2, Math.random() * 4.7) * 55;
    }
    const f = freq / this.sampleRate;
    const phase = Math.random();
    const harmonicFade = 0.25 + 0.25 * Math.random();

    const harmonics = [];
    for (let i = 0; i < 4; i++) {
      harmonics.push({
        freq: Math.pow(2, i) * f,
        wave: this.waves[Math.random(this.waves.length) | 0],
        amp: Math.pow(harmonicFade, i),
      });
    }

    this.fftIn.fill(0);
    for (const { freq, wave, amp } of harmonics) {
      for (let t = 0; t < this.fftIn.length; t++) {
        this.fftIn[t] = amp * wave(freq, t + phase / freq);
      }
    }
    this.normalize(this.fftIn);

    const noise = Math.random() * 0.75;
    for (let t = 0; t < this.fftIn.length; t++) {
      this.fftIn[t] = this.fftIn[t] * (1 - noise) + Math.random() * noise;
    }
    this.normalize(this.fftIn);

    this.fft.realTransform(this.fftOut, this.fftIn);
    this.downsampleFFT();

    const fftNoise = Math.random() * 0.5;
    for (let i = 0; i < this.fftSize >>> 1; i += 2) {
      this.fftOut[i] = this.fftOut[i] * (1 - fftNoise) +
        Math.random() * fftNoise;
    }

    this.normalize(this.fftOut, this.fftSize >>> 1);

    return {
      freq,
      fft: this.fftOut.slice(0, this.fftSize >>> 1),
    };
  }

  noise() {
    this.fftIn.fill(0);

    for (let t = 0; t < this.fftIn.length; t++) {
      this.fftIn[t] = Math.random();
    }
    this.normalize(this.fftIn);

    this.fft.realTransform(this.fftOut, this.fftIn);
    this.downsampleFFT();

    const fftNoise = Math.random() * 0.5;
    for (let i = 0; i < this.fftSize >>> 1; i += 2) {
      this.fftOut[i] = this.fftOut[i] * (1 - fftNoise) +
        Math.random() * fftNoise;
    }

    this.normalize(this.fftOut, this.fftSize >>> 1);

    return {
      fft: this.fftOut.slice(0, this.fftSize >>> 1),
    };
  }

  fromBuffer(buf) {
    this.fft.realTransform(this.fftOut, buf);
    this.downsampleFFT();
    this.normalize(this.fftOut, this.fftSize >>> 1);

    return this.fftOut.slice(0, this.fftSize >>> 1);
  }
}
