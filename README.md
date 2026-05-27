# wav2ays

Convert a WAV (or any libsndfile-readable audio file) into an `.ays` digital
sample stream for **single-channel AY-3-8910 / YM2149 playback** on the ZX
Spectrum and compatibles.

The AY's volume DAC has only 16 logarithmically-spaced levels, so a naive linear
quantization sounds terrible. `wav2ays` decodes the source, conditions it
(DC-block, optional pre-emphasis, steep anti-alias low-pass), resamples to the
exact player loop rate, normalizes onto the DAC span, and quantizes against a
**measured logarithmic codebook** with TPDF dither and sigma-delta noise shaping.
Output bytes are AY volume codes `0..15`, ready to write to register R8.

## Install

Python 3.9+ and three packages:

```bash
pip install -r requirements.txt   # numpy, scipy, soundfile
```

## Usage

```bash
./wav2ays.py in.wav                  # -> in.ays (defaults)
./wav2ays.py in.wav --preview-wav    # also in.preview.wav to audition on a PC
```

The playback **rate must match your player loop** exactly
(`cpu_clock / cycles_per_iteration`) or the pitch drifts — set it with `-r/--rate`
(default 15556 Hz).

### Key options

| Flag | Purpose |
|---|---|
| `-r/--rate Hz` | Playback rate; must equal `cpu_clock / loop_cycles`. |
| `-l/--level 0..15` | Highest AY volume code the converter may emit. |
| `--chip ay\|ym` | DAC curve to quantize/preview against. |
| `--headroom-db D` | Attenuate below the selected level. |
| `--map bipolar\|antidc` | How the waveform lands on the unipolar DAC (see below). |
| `--norm-percentile P` | Map the P-th percentile of the swing to full scale (parks the loud body in the dense mid-codes; rare peaks soft-clip). `~97-98` for percussion, `~85` for steady tones, `100` = absolute peak. |
| `--pre-emphasis-db D` | High-shelf HF boost (off by default); permanently brightens the output (no playback de-emphasis). |
| `--lpf-margin F` / `--lpf-order N` | Explicit steep anti-alias low-pass at `F * rate` (default 0.45, order 8). `0` disables. |
| `--dither` / `--noise-shaping` | Both on by default → sigma-delta quantizer. |
| `--pack byte\|nibble` | One sample per byte, or two 4-bit samples per byte (2× density). |
| `--preview-wav [PATH]` | Also write a float WAV of the AY-quantized result. |

Run `./wav2ays.py --help` for the full list.

### `--map`: bipolar vs antidc

- **`bipolar`** (default) maps the static range to `[0, full]`, so silence sits
  at mid-DAC. Best fidelity on **sustained / looping** material, but quiet tails
  ride a loud pedestal that never decays.
- **`antidc`** lifts the waveform's time-varying lower envelope to ~0 so the
  trough rides at code 0 and **quiet tails decay to true silence**, preserving
  the full waveform shape. Best for **decaying one-shots and material with quiet
  gaps** (snare, toms, plucks, phrased vocals). Adds some sub-bass; useless for
  constant-envelope tones.

## `.ays` format

Little-endian 16-byte header followed by the sample data:

```text
0   4   magic 'AYS1'
4   1   version = 1
5   1   channels = 1
6   1   flags (bit0: 1 = nibble-packed)
7   1   reserved = 0
8   2   sample rate, Hz
10  4   logical sample count (per channel, pre-packing)
14  2   reserved = 0
16  ..  data (AY volume codes 0..15)
```

## Credits & license

MIT licensed (see [LICENSE](LICENSE)).

The AY/YM DAC amplitude tables are reproduced as measured DAC data from the
[ayumi](https://github.com/true-grue/ayumi) library by Peter Sovietov.
