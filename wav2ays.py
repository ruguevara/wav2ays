#!/usr/bin/env python3
"""
wav2ays - prepare a digital sound sample for AY-3-8910 playback (single channel).

Copyright (c) 2026 Ru Grantez. Released under the MIT License.

The AY/YM DAC amplitude tables (AY_DAC_TABLE / YM_DAC_TABLE) are reproduced as
measured DAC data from the ayumi library by Peter Sovietov
(https://github.com/true-grue/ayumi).

Pipeline
--------
1. Decode source to mono float.
2. High-pass (~20 Hz) to strip DC. A DC offset wastes codebook range and clicks
   on start/stop.
3. (Optional) Pre-emphasis: a high-shelf HF boost (--pre-emphasis-db, off by
   default) applied before resampling, to perceptually offset the AY's coarse
   top codes / noise shaper masking quiet treble. There is no playback-side
   de-emphasis, so it permanently brightens the output.
4. Explicit steep anti-alias low-pass (high-order Butterworth, --lpf-margin,
   on by default) just below the playback Nyquist. Sharper than the resampler
   FIR's transition band, and a guard against energy pre-emphasis pushes up.
5. Resample to the exact playback rate. The rate is dictated by the player loop
   (cpu_clock / cycles_per_iteration), so it must be hit precisely or the pitch
   drifts. resample_poly applies a polyphase anti-alias FIR; without it every
   component above Nyquist folds back as audible aliasing.
6. Map and normalize onto the unipolar DAC span [0, full], where `full` is the
   requested AY volume level (0..15, less optional headroom). The level selects
   the highest AY register value the converter may emit. Two mappings (--map):
   - bipolar (default): scale the static min..max range to [0, full], so silence
     sits at mid-DAC. Best on sustained/looping material, but quiet tails ride a
     loud pedestal that never decays.
   - antidc: first lift the waveform's time-varying lower envelope to ~0 (see
     antidc_bend), so the trough rides at code 0 and quiet tails decay to true
     silence while the full waveform shape is preserved. Best for decaying
     one-shots and material with quiet gaps; adds some sub-bass.
   --norm-percentile maps a high percentile of the swing to full scale instead of
   the absolute peak, parking the loud body in the dense mid-codes.
7. Quantize to the AY's 16 volume levels. The AY DAC is logarithmic: steps are
   roughly equal in dB, so levels are dense near zero and sparse near the top.
   Quantization is therefore nearest-neighbour against a measured amplitude
   codebook, not a linear sample->index map.
   - Dither (TPDF, scaled to the local codebook step) decorrelates the
     quantization error, turning signal-dependent distortion into benign noise.
   - First-order noise shaping feeds the quantization error back, moving its
     spectrum upward where the ear is less sensitive.
8. Emit an .ays file: 16-byte header + sample stream, one byte per sample or two
   4-bit samples packed per byte.

Output bytes are AY volume values 0..15, ready to write to R8 with bit 4 clear.

Optionally (--preview-wav) also emit a 32-bit float WAV that maps the quantized
indices back through the codebook to the real DAC amplitudes, at the playback
rate, so the AY-degraded result can be auditioned on a PC.

.ays header (little-endian)
    0   4   magic 'AYS1'
    4   1   version = 1
    5   1   channels = 1
    6   1   flags (bit0: 1 = nibble-packed)
    7   1   reserved = 0
    8   2   sample rate, Hz
    10  4   logical sample count (per channel, pre-packing)
    14  2   reserved = 0
    16  ..  data
"""

import argparse
import struct
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly, sosfiltfilt

# Normalized fixed-volume DAC tables for register values 0..15.
# Values were taken from ayumi library by Peter Sovietov https://github.com/true-grue
AY_DAC_TABLE = np.array([
    0.0,
    0.00999465934234,
    0.0144502937362,
    0.0210574502174,
    0.0307011520562,
    0.0455481803616,
    0.0644998855573,
    0.107362478065,
    0.126588845655,
    0.20498970016,
    0.292210269322,
    0.372838941024,
    0.492530708782,
    0.635324635691,
    0.805584802014,
    1.0,
])
YM_DAC_TABLE = np.array([
    0.0,
    0.00772106507973,
    0.0139620050355,
    0.0200198367285,
    0.029694056611,
    0.0403906309606,
    0.0583352407111,
    0.0777752346075,
    0.111085679408,
    0.148485542077,
    0.211551079576,
    0.281101701381,
    0.400427252613,
    0.53443198291,
    0.75800717174,
    1.0,
])
DAC_TABLES = {
    "ay": AY_DAC_TABLE,
    "ym": YM_DAC_TABLE,
}


def fixed_volume_codebook(chip):
    return DAC_TABLES[chip]


def load_mono(path):
    sig, rate = sf.read(path, dtype="float64", always_2d=True)
    return sig.mean(axis=1), rate


def dc_block(sig, rate, cutoff):
    sos_b, sos_a = butter(2, cutoff / (rate / 2.0), btype="highpass")
    return filtfilt(sos_b, sos_a, sig)


def to_rate(sig, src_rate, dst_rate):
    if src_rate == dst_rate:
        return sig
    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(sig, dst_rate // g, src_rate // g)


def pre_emphasis(sig, rate, gain_db, corner_hz):
    """First-order high-shelf: flat below corner_hz, +gain_db above.

    Brightens HF before quantization to perceptually offset the AY's coarse top
    codes and the noise shaper masking quiet treble. There is no playback-side
    de-emphasis (the AY plays raw bytes), so this is a permanent tonal change.
    Audio-EQ-Cookbook high-shelf biquad with shelf slope S=1 (Q ~= 0.707);
    zero-phase via filtfilt.
    """
    if gain_db == 0.0:
        return sig
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * corner_hz / rate
    cw, sw = np.cos(w0), np.sin(w0)
    # shelf slope S = 1  ->  the cookbook term (A + 1/A)*(1/S - 1) + 2  ==  2
    alpha = sw / 2.0 * np.sqrt(2.0)
    tsa = 2.0 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) + (A - 1) * cw + tsa),
                  -2 * A * ((A - 1) + (A + 1) * cw),
                  A * ((A + 1) + (A - 1) * cw - tsa)])
    a = np.array([(A + 1) - (A - 1) * cw + tsa,
                  2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - tsa])
    return filtfilt(b / a[0], a / a[0], sig)


def steep_lpf(sig, src_rate, cutoff_hz, order):
    """Zero-phase high-order Butterworth low-pass (SOS), applied at src_rate.

    cutoff_hz is derived from the destination rate (just below playback Nyquist)
    but the filter runs at the source rate, so it hard-cuts content that would
    otherwise fold or sit in the resampler FIR's transition band. No-op when the
    cutoff is at/above the source Nyquist (no headroom to filter).
    """
    nyq = src_rate / 2.0
    if cutoff_hz >= nyq:
        return sig
    sos = butter(order, cutoff_hz / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, sig)


def lower_envelope(sig, rate, attack_hz, release_hz, smooth_hz):
    """Track how far the waveform dips below zero, over time (the trough depth).

    An asymmetric one-pole follower on -sig: fast attack catches each trough,
    slow release lets it ride a decaying tail rather than snapping back up
    between cycles. A final low-pass smooths zipper noise. Returns a
    non-negative envelope (0 where the signal never goes below zero).
    """
    neg = -sig
    atk = float(np.exp(-2.0 * np.pi * attack_hz / rate))
    rel = float(np.exp(-2.0 * np.pi * release_hz / rate))
    env = np.empty_like(neg)
    s = 0.0
    for i, v in enumerate(neg):
        if v > s:
            s = atk * s + (1.0 - atk) * v
        else:
            s = rel * s + (1.0 - rel) * v
        env[i] = s
    env = np.maximum(env, 0.0)
    if smooth_hz > 0:
        b, a = butter(2, smooth_hz / (rate / 2.0), btype="lowpass")
        env = np.maximum(filtfilt(b, a, env), 0.0)
    return env


def antidc_bend(sig, rate, attack_hz, release_hz, smooth_hz):
    """Lift the waveform's lower envelope to ~0 so its bottom rides at zero.

    Unlike static normalization (which fixes silence at mid-DAC and leaves a
    loud, noisy pedestal during quiet passages), this adds the time-varying
    lower envelope back to the signal: y = sig + env. The trough now hugs zero,
    so quiet tails decay toward code 0 (true silence) while the full bipolar
    waveform shape is preserved (no half-wave clamp / overdrive). The cost is
    some added sub-bass, since the envelope itself is a slow component.
    """
    env = lower_envelope(sig, rate, attack_hz, release_hz, smooth_hz)
    return sig + env


def range_for_normalization(sig, rate, edge_ms, percentile=100.0):
    """Lo/hi for level mapping, excluding filter/resampler edge transients.

    With percentile < 100 the span is taken from a high percentile of the
    signal's deviation about its midpoint, not the absolute peak. This parks
    the loud body of the signal in the dense mid-codes while letting rare
    transients overshoot (the caller clips them). The range stays centred on
    the true midpoint so the unipolar mapping remains symmetric.
    """
    edge = int(round(rate * edge_ms / 1000.0))
    if edge > 0 and len(sig) > edge * 4:
        sig = sig[edge:-edge]
    lo, hi = float(np.min(sig)), float(np.max(sig))
    if percentile >= 100.0 or len(sig) == 0:
        return lo, hi
    mid = (lo + hi) / 2.0
    half = float(np.percentile(np.abs(sig - mid), percentile))
    return mid - half, mid + half


def local_step(values, codebook):
    """Codebook gap bracketing each value; used to scale dither per sample."""
    j = np.searchsorted(codebook, values)
    lo = codebook[np.clip(j - 1, 0, len(codebook) - 1)]
    hi = codebook[np.clip(j, 0, len(codebook) - 1)]
    step = hi - lo
    step[step <= 0] = codebook[1] - codebook[0]
    return step


def add_dither(values, rng, codebook):
    """Add TPDF dither inside the codebook range, preserving exact endpoints."""
    out = values.copy()
    active = (values > codebook[0]) & (values < codebook[-1])
    if np.any(active):
        d = rng.random(np.count_nonzero(active)) + rng.random(np.count_nonzero(active)) - 1.0
        out[active] += d * local_step(values[active], codebook)
    return out


def sigma_delta_quantize(dac, max_level, codebook):
    """First-order sigma-delta using only the labels around each input sample."""
    codebook = codebook[:max_level + 1]
    out = np.empty(len(dac), dtype=np.uint8)
    if len(codebook) == 1:
        out.fill(0)
        return out

    err = 0.0
    last = len(codebook) - 1
    for i, x in enumerate(dac):
        j = int(np.searchsorted(codebook, x, side="left"))
        if j <= 0:
            lo = hi = 0
        elif j > last:
            lo = hi = last
        elif codebook[j] == x:
            lo = hi = j
        else:
            lo = j - 1
            hi = j

        if lo == hi:
            k = lo
            err = 0.0
        else:
            u = x + err
            if abs(u - codebook[lo]) <= abs(u - codebook[hi]):
                k = lo
            else:
                k = hi
            err = u - codebook[k]
        out[i] = k
    return out


def quantize(dac, dither, shaping, rng, max_level, codebook, adjacent_only,
             shaper=(1.0,)):
    codebook = codebook[:max_level + 1]
    bounds = (codebook[:-1] + codebook[1:]) / 2.0
    if dither and shaping:
        return sigma_delta_quantize(dac, max_level, codebook)

    # Only the adjacent-only clamp needs the undithered nearest index.
    nearest = np.searchsorted(bounds, dac) if adjacent_only else None

    if not shaping:
        t = add_dither(dac, rng, codebook) if dither else dac
        idx = np.searchsorted(bounds, t)
        if adjacent_only:
            idx = np.minimum(np.maximum(idx, nearest - 1), nearest + 1)
        return idx.astype(np.uint8)

    # FIR error-feedback noise shaping: u[n] = x[n] - sum_k a_k * e[n-k],
    # e[n] = y[n] - u[n]. A single-tap [1.0] shaper is first-order feedback;
    # higher-order taps push the quantization noise spectrum upward, where the
    # ear is less sensitive. Sequential dependency, so this runs as a scalar loop.
    cb = codebook.tolist()
    bnd = bounds.tolist()
    base_step = cb[1] - cb[0] if len(cb) > 1 else 0.0
    taps = [float(a) for a in shaper] or [1.0]
    hist = [0.0] * len(taps)  # hist[0] = e[n-1], hist[1] = e[n-2], ...
    out = np.empty(len(dac), dtype=np.uint8)
    for i, x in enumerate(dac):
        fb = 0.0
        for a, e in zip(taps, hist):
            fb += a * e
        u = x - fb
        t = u
        if dither:
            if cb[0] < u < cb[-1]:
                j = np.searchsorted(cb, u)
                lo = cb[max(j - 1, 0)]
                hi = cb[min(j, len(cb) - 1)]
                step = hi - lo if hi > lo else base_step
                t = u + (rng.random() + rng.random() - 1.0) * step
        k = int(np.searchsorted(bnd, t))
        k = min(max(k, 0), len(cb) - 1)
        if adjacent_only:
            n = int(nearest[i])
            k = min(max(k, n - 1), n + 1)
        out[i] = k
        hist = [cb[k] - u] + hist[:-1]
    return out


def pack(idx, nibble):
    if not nibble:
        return idx.tobytes()
    # Low nibble holds the earlier sample; pad an odd tail with a zero sample.
    if len(idx) % 2:
        idx = np.append(idx, np.uint8(0))
    return (idx[0::2] | (idx[1::2] << 4)).astype(np.uint8).tobytes()


def write_ays(path, idx, rate, nibble):
    flags = 1 if nibble else 0
    header = struct.pack("<4sBBBBHIH", b"AYS1", 1, 1, flags, 0,
                          rate, len(idx), 0)
    Path(path).write_bytes(header + pack(idx, nibble))


def decode_to_float(idx, codebook):
    """AY indices -> raw unipolar DAC voltage (0..1) via the measured codebook.

    No DC removal: the output is exactly what the AY DAC emits, so silence and
    the signal floor sit at 0.0 and everything is positive. A speaker can't
    reproduce DC anyway, so the offset is inaudible and only affects the visual.
    """
    return codebook[idx]


def write_preview_wav(path, idx, rate, codebook):
    sf.write(path, decode_to_float(idx, codebook), rate, subtype="FLOAT")


def main():
    ap = argparse.ArgumentParser(description="WAV -> AY sample (.ays), 1 channel")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("-r", "--rate", type=int, default=15556,
                    help="playback sample rate in Hz (cpu_clock / loop_cycles)")
    ap.add_argument("-l", "--level", type=int, default=15,
                    help="maximum AY DAC volume level to emit (0..15)")
    ap.add_argument("--chip", choices=sorted(DAC_TABLES), default="ay",
                    help="DAC curve to use for quantization and preview")
    ap.add_argument("--headroom-db", type=float, default=0.0,
                    help="attenuation below the selected AY DAC level, in dB")
    ap.add_argument("--hpf-hz", type=float, default=20.0,
                    help="DC-blocking high-pass cutoff in Hz")
    ap.add_argument("--pre-emphasis-db", type=float, default=0.0,
                    help="high-shelf HF boost in dB above the corner; 0 = off "
                         "(default). 3-6 dB perceptually offsets quantization "
                         "noise masking quiet treble. No playback de-emphasis: "
                         "output is permanently brighter")
    ap.add_argument("--pre-emphasis-hz", type=float, default=400.0,
                    help="high-shelf corner frequency in Hz; content above this "
                         "is boosted by --pre-emphasis-db")
    ap.add_argument("--lpf-margin", type=float, default=0.45,
                    help="explicit steep anti-alias low-pass cutoff as a "
                         "fraction of the playback rate (default 0.45 -> "
                         "0.45*rate, just below Nyquist). Hard-cuts HF the "
                         "resampler FIR leaves in the transition band, and any "
                         "energy pre-emphasis pushed up. 0 disables")
    ap.add_argument("--lpf-order", type=int, default=8,
                    help="Butterworth order of the explicit anti-alias low-pass")
    ap.add_argument("--norm-edge-ms", type=float, default=10.0,
                    help="milliseconds to ignore at each edge when measuring "
                         "normalization range, avoiding filter edge transients")
    ap.add_argument("--norm-percentile", type=float, default=100.0,
                    help="map this percentile of the signal magnitude to full "
                         "scale instead of the absolute peak (e.g. 99.5). "
                         "Parks the loud body of the signal in the dense "
                         "mid-codes; rare transient peaks soft-clip. ~97-98 "
                         "suits percussion; steady/periodic tones (sine, "
                         "triangle) benefit from ~85, which keeps the loud body "
                         "off the sparse top codes (worth ~+1.5 dB A-weighted). "
                         "100 = absolute peak (original behaviour)")
    ap.add_argument("--map", choices=["bipolar", "antidc"], default="bipolar",
                    help="DAC mapping. 'bipolar': static range -> [0, full], "
                         "silence sits at mid-DAC (best fidelity on sustained "
                         "material, but quiet tails ride a loud pedestal). "
                         "'antidc': lift the waveform's lower envelope to ~0 so "
                         "the bottom rides at code 0 and quiet tails decay to "
                         "silence, preserving the full waveform shape (best for "
                         "decaying one-shots; adds some sub-bass)")
    ap.add_argument("--antidc-attack-hz", type=float, default=400.0,
                    help="anti-DC envelope follower attack cutoff (Hz); higher "
                         "tracks sharp troughs faster")
    ap.add_argument("--antidc-release-hz", type=float, default=15.0,
                    help="anti-DC envelope follower release cutoff (Hz); lower "
                         "rides decaying tails more smoothly")
    ap.add_argument("--antidc-smooth-hz", type=float, default=60.0,
                    help="anti-DC envelope post-smoothing low-pass cutoff (Hz); "
                         "0 disables")
    ap.add_argument("--dither", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--noise-shaping", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--shaper", default="1.0",
                    help="comma-separated FIR error-feedback coefficients for "
                         "noise shaping. '1.0' = first-order; '1.6,-0.6' = a "
                         "2nd-order high-pass shaper that pushes more noise into "
                         "the upper band. Ignored when both dither and "
                         "noise-shaping are on (sigma-delta path)")
    ap.add_argument("--adjacent-dither", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="limit dithered output to the nearest codebook value "
                         "or one adjacent code")
    ap.add_argument("--pack", choices=["byte", "nibble"], default="byte")
    ap.add_argument("--preview-wav", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="also write a 32-bit float WAV of the AY-quantized "
                         "result (codebook amplitudes at the playback rate) for "
                         "PC audition; PATH optional, defaults to "
                         "<output>.preview.wav")
    ap.add_argument("--emphasis-wav", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="also write a float WAV of the pre-emphasized "
                         "(brightened) signal at the playback rate, for "
                         "measuring quantization SNR against the brightened "
                         "reference via aysnr.py; PATH optional, defaults to "
                         "<output>.emphasis.wav")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not 1 <= args.rate <= 65535:
        sys.exit("rate must fit in uint16 (1..65535)")
    level = min(max(args.level, 0), 15)
    codebook = fixed_volume_codebook(args.chip)

    if args.dither and args.noise_shaping and args.shaper != "1.0":
        print("note: --shaper is ignored with the default sigma-delta quantizer "
              "(dither + noise-shaping both on)", file=sys.stderr)

    try:
        sig, src_rate = load_mono(args.input)
    except Exception as e:
        sys.exit(f"cannot read {args.input}: {e}")
    sig = dc_block(sig, src_rate, args.hpf_hz)
    sig = pre_emphasis(sig, src_rate, args.pre_emphasis_db, args.pre_emphasis_hz)
    if args.lpf_margin > 0:
        sig = steep_lpf(sig, src_rate, args.lpf_margin * args.rate, args.lpf_order)
    sig = to_rate(sig, src_rate, args.rate)
    bright = sig  # post-resample, pre-normalization: the brightened reference

    # Map the signal's full range onto the unipolar DAC span [0, full].
    # The AY DAC is logarithmic, so centering a bipolar signal at mid-scale
    # (0.5) would strand it on the few coarse high codes and never touch the
    # dense low codes. Anchoring the minimum at 0 instead spreads the signal
    # across all 16 levels and yields a near-symmetric decoded swing.
    full = codebook[level] * (10.0 ** (-args.headroom_db / 20.0))
    if args.map == "antidc":
        # Lift the lower envelope to ~0 so the trough rides at code 0; quiet
        # tails decay to silence instead of parking on a loud mid-DAC pedestal.
        bent = antidc_bend(sig, args.rate, args.antidc_attack_hz,
                           args.antidc_release_hz, args.antidc_smooth_hz)
        # The bent signal already rides from ~0 upward, so scale its positive
        # peak (a high percentile, ignoring edge transients) to full scale.
        edge = int(round(args.rate * max(args.norm_edge_ms, 0.0) / 1000.0))
        body = bent[edge:-edge] if edge > 0 and len(bent) > edge * 4 else bent
        pct = min(max(args.norm_percentile, 0.0), 100.0)
        hi = float(np.percentile(body, pct)) if len(body) else 0.0
        span = hi
        dac = bent / span * full if span > 0 else np.zeros_like(bent)
    else:
        lo, hi = range_for_normalization(sig, args.rate,
                                         max(args.norm_edge_ms, 0.0),
                                         args.norm_percentile)
        span = hi - lo
        dac = (sig - lo) / span * full if span > 0 else np.zeros_like(sig)
    dac = np.clip(dac, 0.0, full)

    try:
        shaper = tuple(float(c) for c in args.shaper.split(",") if c.strip())
    except ValueError:
        sys.exit("--shaper must be comma-separated numbers, e.g. '1.6,-0.6'")
    if not shaper:
        shaper = (1.0,)

    rng = np.random.default_rng(args.seed)
    idx = quantize(dac, args.dither, args.noise_shaping, rng, level, codebook,
                   args.adjacent_dither, shaper)

    out = args.output or str(Path(args.input).with_suffix(".ays"))
    write_ays(out, idx, args.rate, args.pack == "nibble")

    dur = len(idx) / args.rate
    print(f"{out}: {len(idx)} samples, {args.rate} Hz, {dur:.2f} s, "
          f"{args.pack}-packed")

    if args.preview_wav is not None:
        wav_path = args.preview_wav or str(Path(out).with_suffix(".preview.wav"))
        write_preview_wav(wav_path, idx, args.rate, codebook)
        print(f"{wav_path}: AY-preview WAV, {args.rate} Hz, 32-bit float")

    if args.emphasis_wav is not None:
        emph_path = args.emphasis_wav or str(Path(out).with_suffix(".emphasis.wav"))
        sf.write(emph_path, bright, args.rate, subtype="FLOAT")
        print(f"{emph_path}: pre-emphasized reference WAV, {args.rate} Hz, "
              f"32-bit float")


if __name__ == "__main__":
    main()
