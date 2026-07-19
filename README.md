# the_splat — a live predictive cortex on a Gabor-splat manifold

**PerceptionLab / Antti Luode (Helsinki), with several AI collaborators. 2026.**

> Do not hype. Do not lie. Just show.

This is the full description of the `the_splat` live-cortex system as it stands
at **V5 (the octave cascade)**. It is one file, `the_splatV5.py`, that runs a
real predictive-coding loop live on a webcam using a pre-trained 31 MB Gabor
splat autoencoder as its appearance manifold. Everything below describes what is
actually in the code, why each piece is there, what has been verified and how,
and what is still open.

---

## 1. The one-paragraph version

A small autoencoder (`SplatVAE`) was trained on celeba faces. Its decoder turns
a 128-D latent `z` into a set of oriented **Gabor packets** — little localized
wavelets with position, size, orientation, spatial frequency, and colour — which
sum into an image. That renderer is a **manifold**: a smooth map from a
low-dimensional `z` to the space of faces it learned. The live cortex never
looks at the whole camera frame. It **holds** a `z`, predicts where it should go
from its own recent motion, reads a **sparse afferent** from the frame (a
handful of tracked feature points and where they moved), and corrects `z` by
asking the manifold to *move the same way the world moved*. A precision term
decides how much to trust the afferent frame-to-frame. The result is a belief —
a rendered face — that tracks your pose and lighting in real time, held in
lockstep with the physical world by a trickle of numbers rather than a
full-frame neural net. V5 splits that afferent into **frequency octaves** (a
V1→V2→V4-style coarse-to-fine cascade) with a live graphical EQ over the bands.

---

## 2. Why this exists — the question being asked

Large interactive video models (e.g. Wan-Streamer, 2026) achieve real-time
audio-visual interaction by brute force: encode the whole frame into thousands
of tokens, push them through a billion-parameter transformer, regenerate every
160 ms, burning a supercomputer to track a face. They ask *how large a manifold
can get*.

`the_splat` asks the opposite, biological question: **what is the algorithmic
minimum required to keep an internal hallucination locked to physical reality?**
The thalamic channel into cortex is narrow (~10⁶ fibres feeding ~10¹⁰ neurons);
perception is mostly prediction corrected by a sparse error signal. This system
is a working, watchable model of that idea at 31 MB, on one desktop GPU.

The lineage of falsifiable builds:

- **V2** — proved the loop closes through *real rendered pixels*, and surfaced
  Takens' genericity condition (a symmetric scalar observable can't resolve
  direction; the loop locked to a mirror solution until the observable was made
  complex). State was 1-D (a phase).
- **V2-live** — the full 128-D celeba latent held by *K sparse colour probes*.
  Confirmed live: sparse holds the manifold; a full encoder tracks better;
  tracking scales with K. Found the **frontal-face attractor** — point the
  camera at a table and it renders a face, because the manifold has no "not a
  face" direction. That is predictive coding's core claim in 31 MB: perception
  is projection onto the prior.
- **V3** — replaced colour probes with **Lucas-Kanade flow probes**
  (displacements, not appearances). Beat colour probes and held under lighting
  drift, because brightness constancy is only assumed frame-to-frame. Live, the
  *shoulders* became a steerable channel.
- **V4** — split the packets into **two frequency bands** and routed each band's
  flow to its own packets. The `band_diagnostic` confirmed the field had split
  *itself*, unsupervised, into a low-freq shading/luminance channel and a
  high-freq oriented **contour/edge** channel — the V1 simple-cell
  decomposition, exactly as Barlow's 1961 efficient-coding hypothesis predicts
  for a localized-frequency basis under reconstruction pressure.
- **V5 (this file)** — generalized to **N log-spaced octaves** (a coarse-to-fine
  cascade) with per-octave LK window / weight / precision schedules and a live
  **graphical EQ** over the bands.

---

## 3. The manifold — `SplatVAE` (carried verbatim from the original repo)

The generative model is the original `the_splat` autoencoder, embedded
unchanged so trained checkpoints load `strict=True`.

- **`Encoder`** — a small conv stack, image → `(mu, logvar)` in latent space.
  Used only for **acquisition** (the GIST button); the loop does not use it to
  track.
- **`Decoder`** — an MLP, `z (latent) → raw (N packets × 11 params)`.
- **`GaborRenderer`** — turns `raw` into an image. Each packet `i` has, via
  `activate()`:
  - `px, py` — position (anchored on a grid + learned offset),
  - `sigma ∈ [0.012, 0.152]` — envelope size,
  - `theta` — orientation,
  - **`freq ∈ [1, 16]` — spatial frequency** (this is the axis V4/V5 split on),
  - `coeff` — a 3×2 colour/quadrature tensor.

  The image is `sigmoid(Σ_i env_i · (a·cos − b·sin))`, an anisotropic Gabor sum.
  Typical trained field: `image_size=128, packets=512, latent=128, hidden=512`.

- **`render_probes(raw, pxy)`** — evaluates the field at arbitrary points
  `pxy` instead of the pixel grid. Cost `O(K·N)` not `O(H·W·N)`. **Verified to
  match the full render at grid coordinates to float precision.** This is what
  makes the sparse afferent cheap.

- **`load_v1(path)`** — infers `image_size, num_packets, latent, hidden` from
  the checkpoint itself and loads `strict=True`. **Verified bit-identical**
  (max render diff `0.0`) against the repo's original `SplatVAE`.

---

## 4. The afferent — sparse Lucas-Kanade flow

The cortex reads **where tracked things went**, not their colour.

- **`LKFlow`** — torch-native sparse Lucas-Kanade (no OpenCV needed for the core).
  For each point it solves the 2×2 structure-tensor system over a small window,
  iterated, on a 2-level pyramid. Returns per-point flow (in `[0,1]` coords), a
  confidence (min eigenvalue), and a post-fit residual. **Verified to recover a
  known (+2, −1)-pixel shift exactly** (residual `~0.0004`).
- **`good_features`** — a cheap Shi-Tomasi: pick high-gradient points from random
  candidates, keeping a minimum distance from existing probes. Used to seed and
  to re-seed lost probes (the **saccade** — active sensing).

Why flow, not colour: the celeba manifold derives skin tone from webcam
luminance, so colour probes couple the belief to the *lighting*. Flow only
assumes brightness constancy frame-to-frame, so slow lighting drift passes
through untouched. (V3 `--selftest` [B] measured colour probes degrading *below
open-loop* under lighting drift, while flow held.)

---

## 5. The octave split — V5's core addition

`octave_bands(vae, n_oct)` splits the packets into `n_oct` **log-spaced
frequency octaves** by the field's own trained `freq`, using quantile edges so
each band holds a comparable count. Log spacing matches the roughly
octave-spaced frequency channels of visual cortex. On a real 512-packet field
this yields (verified in NumPy at true dims) four clean bands of 128 packets
each, spanning freq `1.3–6.0 / 6.0–8.3 / 8.4–11.0 / 11.0–15.3`, a full partition
with no overlap and none dropped.

- **`render_probes_subset` / `render_full_subset`** — render using only one
  band's packets, with the anchor buffer **sliced to that band**. (This fixed a
  real crash in V4 where the verbatim `activate()` added the full 512-row anchor
  to a 256-packet subset. Verified in NumPy at 512 packets: the sliced subset
  render equals the full render restricted to those packets — numbers unchanged,
  only the shape bug gone.)
- **`octave_diagnostic`** — writes `FULL | O0 | O1 | … | O_{N-1}` for several
  random `z`. This is the empirical test of whether the field learned a genuine
  cascade: O0 shading, O_{N-1} fine contours, middle bands interpolating.

---

## 6. The cortex — `OctaveCortex`

Holds `z`; runs one loop tick per frame. The math per tick:

```
# 1. prior flow — predict z from its own recent motion + a weak leak to a slow prior
vel     = z - z_prev
z_pred  = z + beta_mom * vel
z_pred  = z_pred - leak * (z_pred - z_prior)

# 2. per octave i (skipped if inactive, EQ-muted, or empty):
d_i, conf_i, res_i = LK_octave_i(prev_frame, frame, probes_i)   # sparse afferent
prec_i  = sig_ref^2 / (sig_ref^2 + roughness(res_i)^2)          # reliability
prec_i  = min(prec_i, prec_cap_i)                               # per-octave trust cap
eff_i   = weight_i * eq_i * prec_i                              # effective gain
# "what octave i rendered at p under old z must appear at p+d under new z":
loss   += eff_i * || R_i(p_i + d_i ; z_pred) - R_i(p_i ; z) ||^2

# 3. correction — one gradient step through the decoder, trust-region clamped
z      <- z_pred - clamp(eta * dloss/dz, dz_clamp)

# 4. probes ride their octave's flow, gated by that octave's precision;
#    lost probes re-seed to high-gradient features (the saccade)
p_i    <- clamp(p_i + prec_i * d_i)
```

Per-octave schedules (octave 0 = lowest freq → N−1 = highest):

| octave | probes | LK window        | weight | prec cap | role                     |
|-------:|-------:|:-----------------|-------:|---------:|:-------------------------|
| 0 low  |   14   | 13 px on 2× pyr  |  1.00  |   1.00   | shading / pose / envelope |
| 1      |   11   | 11 px on 2× pyr  |  0.75  |   0.80   | major placement           |
| 2      |    8   |  9 px full-res   |  0.50  |   0.60   | orientation refine        |
| 3 high |    5   |  5 px full-res   |  0.25  |   0.40   | fine contours (near-immovable) |

(Verified monotone in NumPy: low octaves get more probes, higher weight, higher
trust.) The design intent: **low octaves are world-driven** (reliable
high-variance low-frequency motion the manifold should honour), **high octaves
are prior-dominated** (the strong, low-variance face-contour prior, steered only
by strong evidence). The lowest octave orients everything; each higher octave
refines within the basin below it.

**Acquisition vs holding.** The loop *holds*; it does not *find*. On its own the
sparse afferent can only correct within the current basin. `bootstrap()` (the
GIST button) runs the encoder once to place `z` in the right basin. The cascade
hypothesis is that higher octaves can inherit orientation from the low ones,
reducing how often a full GIST is needed — that is claim **[E]**.

---

## 7. The graphical EQ — the "sigh" idea, folded in

Antti's earlier FFT tool taught that low frequencies carry the *gist* by
filtering pixels in Fourier space. The splat field **already is** a frequency
decomposition — each packet is a localized frequency atom — so an EQ over
packet-frequency is a live mixing board on the manifold's octaves. Each octave
has a fader (`eq_i ∈ [0,1]`) that gates **both**:

- its **render** contribution — `belief_render` sums `eq_i · (octave_i − 0.5)`
  over octaves (additive, no division; all-zero EQ is safe flat gray), and
- its **correction** gradient — `eff_i = weight_i · eq_i · prec_i`, so a muted
  octave drops out of the loss entirely.

Pull the top faders down: the gist (pose, shading) survives on the low bands.
Pull the bottom faders down: identity/detail drops, structure holds.

---

## 8. The GUI

Two live panes: **AFFERENT** (the camera frame with octave-coloured probes and
flow vectors — cyan = lowest freq, warm = highest) and **BELIEF**
(`render(dec(z))`). Controls:

- **START / STOP**
- **OCTAVE EQ** — a vertical fader per octave (gates render + correction live)
- **VIEW** — cycles the belief pane through the EQ-mix and each single octave
- **INJECT SLOP** — replaces the frame with noise for ~60 frames; watch precision
  collapse and the belief coast on its prior
- **prec DYN / FIX** — dynamic vs fixed precision
- **GIST** — encoder re-anchor (needs `--model`)
- **K** — probe count (scaled across octaves; low bands get more)

Telemetry per octave: precision, EQ gain, probe count, plus `|dz|` and frame `t`.

---

## 9. The synthetic world and the scorecard

For falsifiable testing without a webcam, `OctaveWorld` renders a hidden `z`
moving on a low-D trajectory: the **low** latent axes move slowly (pose), the
**high** axes faster (detail), with optional luminance drift and injectable
slop. Ground truth exists, so the loop can be scored. `--selftest` prints:

- **[A]** all-octaves tracks pose better than open-loop.
- **[E] cascade inheritance — the headline.** Does the highest octave alone track
  pose *better with the low octaves on* than with them off (inherited
  orientation)? Holds iff `all ≤ high-alone`. If not, the cascade adds nothing
  over independent bands — and the ledger must say so.
- **[B]** the low octave holds pose under luminance drift.
- **[D]** dynamic precision coasts through slop (`dyn ≤ fixed`).

---

## 10. Honest ledger — what is verified, and how

- **[V] carried & re-verified across versions:** strict `.pt` compatibility
  (bit-identical renders); `render_probes` / `*_subset` match the full render
  (anchor sliced) to float precision; LK recovers a known pixel shift exactly;
  colour→flow and lighting-drift results (V3); the V1-like unsupervised
  frequency split (V4 `band_diagnostic`).
- **[V] verified this build, in NumPy at real 512-packet dims:** the octave split
  partitions all packets into log-spaced bands (no overlap/loss); the per-octave
  schedules are monotone; the LK windows scale 13→5 px with a pyramid on the low
  half; the EQ gates a muted octave out of *both* correction and render; the
  belief EQ-mix is additive and safe at all-zero.
- **[UNVERIFIED — run on your GPU]:** the full torch scorecard ([A], **[E]**,
  [B], [D]). The sandbox package proxy blocked installing torch, so these did not
  run here. Seconds on CUDA: `python the_splatV5.py --selftest`.
- **[K] open questions:**
  - The octave split is only as meaningful as the *trained field's* frequency
    organization. On a 2-epoch celeba `.pt` this is empirical — run
    `--diagnostic` to see whether four bands separate cleanly or whether the
    middle bands are muddy (in which case use `--octaves 3`).
  - The persistent held state's *effective* dimensionality that a sparse afferent
    can steer is bounded by the motion's intrinsic dimension (Takens); pose is
    low-D and tracks well, full identity/detail is the frontier.
  - The manifold is the ceiling: a 2-epoch, 31 MB celeba field can't open a mouth
    or turn to profile (not in its training distribution). More/again-trained
    manifolds raise the ceiling with **no change to the loop**.
- **[B] boundary — what this is NOT:** it does not "navigate a learned world" or
  do deepfake-style expression synthesis. It *holds and steers the pose/appearance
  of a face render from a sparse pixel read*. The wire between predict → render →
  correct → move is built and watchable; a richer manifold and a genuinely
  multi-D navigable state are future work.

---

## 11. Running it

```bash
pip install torch numpy pillow          # opencv-python only for --webcam; tkinter ships with Python

# 1) SEE the cascade the field learned (do this first):
python the_splatV5.py --model model.pt --diagnostic

# 2) the falsifiable scorecard (run [E] on your GPU):
python the_splatV5.py --selftest

# 3) live cortex on your webcam:
python the_splatV5.py --model model.pt --webcam

# options:
python the_splatV5.py --octaves 3 ...   # band count (default 4)
python the_splatV5.py                    # synthetic world, no model needed
```

Suggested first live session: START, drop the top EQ faders and move around —
the low-octave belief should hold your pose; raise them and the face detail
returns. INJECT SLOP to watch the belief coast on its prior when the afferent
goes to noise. GIST if the belief falls out of its basin.

---

## 12. Files

- `the_splatV5.py` — the whole system (one file).
- `README.md` — this document.
- (produced on run) `octave_diagnostic.png` — the FULL | O0 | … | O_{N-1} grid.
- your `model.pt` — a trained `the_splat` SplatVAE checkpoint.

---

## 13. The idea in one line

A handful of moving points, a 128-D latent, a small Gabor renderer, and a
precision-weighted gradient loop keep an internal hallucination locked to a face
in real time — on one desktop GPU, in less than the size of an MP3 — and the
manifold organizes itself, unsupervised, into the same frequency cascade the
visual cortex uses. That last fact is the point: efficient coding predicted it in
1961, and here it is in the code.
