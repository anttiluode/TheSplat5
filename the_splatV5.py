"""
the_splatV5.py  —  the OCTAVE CASCADE: N frequency bands, a V1->V2->V4 style
                   coarse-to-fine hierarchy on the splat manifold, with a live
                   graphical EQ over the bands.

PerceptionLab / Antti Luode (Helsinki), with Claude (Sonnet 5). July 2026.

    Do not hype. Do not lie. Just show.

WHERE THIS COMES FROM
---------------------
V4's band_diagnostic showed the 2-epoch celeba field split ITSELF, unsupervised,
into a shading/luminance channel (low-freq packets) and a clean oriented
contour/edge channel (high-freq packets) — the V1 simple-cell decomposition,
derived from reconstruction pressure on a Gabor basis. That is Barlow's 1961
efficient-coding prediction turning up in the code 65 years later: an efficient
code of natural images under a localized-frequency basis has to organize this
way. V4 split at the median into TWO bands. Cortex is a CASCADE — V1 (fine
oriented edges) -> V2/V4 (progressively larger receptive fields, coarser
integration). V5 generalizes the split to N log-spaced frequency octaves, each
with its own LK window, precision cap, and correction weight, lowest-first.

THE HIERARCHY (octave i, i=0 lowest freq .. N-1 highest):
  - LK window shrinks with octave    (coarse=big window on pyramid; fine=tiny)
  - correction weight falls with octave (world-driven low band trusted; high
    band is the strong, near-immovable prior — steered only by strong evidence)
  - precision cap falls with octave   (V4's prec_fine_cap, generalized per band)
  - probe count falls with octave     (low bands drive global structure)
The lowest octave orients everything; each higher octave refines within the
basin the one below established. Acquisition is still the encoder's job (GIST);
the cascade lets higher octaves INHERIT the low octave's orientation, which is
the falsifiable question below.

THE GRAPHICAL EQ (Antti's idea, folded in):
  The old FFT "sigh" tool taught that low frequencies carry the gist by
  filtering pixels in Fourier space. The splat field ALREADY is that
  decomposition — each Gabor packet is a localized frequency atom. So an EQ over
  packet-frequency is a live mixing board on the manifold's octaves: pull a
  band's slider down and that octave stops rendering AND stops correcting. It is
  the hierarchy with a fader per level — watch the gist survive with only the
  bottom sliders up, watch the identity/detail live in the top sliders.

FALSIFIABLE CLAIMS (--selftest):
  [A] cascade tracks pose >= two-band >= open loop.
  [E] CASCADE INHERITANCE — the headline: with the low octaves ON, the HIGH
      octave alone becomes steerable (its pose error drops vs high-octave-only
      with low octaves OFF), because it inherits orientation from below. If
      high-only-with-low-on is no better than high-only-alone, the cascade
      bought nothing over independent bands and we say so.
  [B] low octaves hold pose under luminance drift (carried from V4).
  [D] dynamic precision still coasts through slop.

CARRIED OVER VERBATIM (verified in V2/V3/V4, unchanged here):
  the_splat v1 SplatVAE (your model.pt loads strict, bit-identical renders),
  render_probes / render_full_subset (subset render matches full, anchor sliced),
  LKFlow (recovers a known pixel shift exactly), the precision/prior/saccade loop.
V5 ADDS: octave_bands() (log-spaced freq quantiles -> list of index tensors),
  an OctaveCortex looping over bands with per-octave configs and an EQ gain
  vector, an EQ-gated render, and a GUI with a per-octave slider bank.

RUN
    python the_splatV5.py --selftest
    python the_splatV5.py --model "face model trained 2 epochs/model.pt" --diagnostic
    python the_splatV5.py --model "face model trained 2 epochs/model.pt" --webcam
    python the_splatV5.py --octaves 4 ...    # choose band count (default 4)
GUI: START; per-octave EQ sliders (gain 0..1, gates render + correction);
  BANDS ALL/<octave> to isolate; VIEW cycles belief full/per-octave; INJECT
  SLOP; precision DYN/FIXED; GIST; probe K. Octave colors run cool->warm
  (cyan lowest -> magenta highest).
"""

from __future__ import annotations
import argparse, math, os, sys, time, colorsys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_PARAMS = 11

#  render_probes() (sparse evaluation; touches no state_dict key).           #
# ========================================================================== #

class GaborRenderer(nn.Module):
    def __init__(self, image_size=128, num_packets=512, chunk=64, use_checkpoint=True):
        super().__init__()
        self.H = self.W = image_size
        self.N = num_packets
        self.chunk = chunk
        self.use_checkpoint = use_checkpoint
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
        side = int(math.ceil(math.sqrt(num_packets)))
        ax = torch.linspace(0.08, 0.92, side)
        anch = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:num_packets]
        anch = torch.clamp(anch, 1e-3, 1 - 1e-3)
        self.register_buffer("anchor_logit", torch.log(anch / (1 - anch)))

    def activate(self, raw):
        a_px = self.anchor_logit[:, 0][None]
        a_py = self.anchor_logit[:, 1][None]
        px = torch.sigmoid(a_px + raw[..., 0])
        py = torch.sigmoid(a_py + raw[..., 1])
        sigma = 0.012 + 0.14 * torch.sigmoid(raw[..., 2])
        theta = raw[..., 3]
        freq = 1.0 + 15.0 * torch.sigmoid(raw[..., 4])
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)
        return px, py, sigma, theta, freq, coeff

    def _render_chunk(self, px, py, sigma, theta, freq, coeff):
        px_ = px[..., None, None]; py_ = py[..., None, None]; s_ = sigma[..., None, None]
        th = theta[..., None, None]; f_ = freq[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None, None]
            b = coeff[:, :, c, 1][..., None, None]
            chans.append((env * (a * cos - b * sin)).sum(dim=1))
        return torch.stack(chans, dim=1)

    def forward(self, raw):
        with torch.amp.autocast("cuda", enabled=False):
            px, py, sigma, theta, freq, coeff = self.activate(raw.float())
            B = raw.shape[0]
            out = torch.zeros(B, 3, self.H, self.W, device=raw.device)
            for i in range(0, self.N, self.chunk):
                sl = slice(i, i + self.chunk)
                args = (px[:, sl], py[:, sl], sigma[:, sl], theta[:, sl], freq[:, sl], coeff[:, sl])
                if self.use_checkpoint and self.training:
                    out = out + checkpoint(self._render_chunk, *args, use_reentrant=False)
                else:
                    out = out + self._render_chunk(*args)
            return torch.sigmoid(out)

    def render_probes(self, raw, pxy):
        """Evaluate the field only at coords pxy (M,2) xy in [0,1] -> (B,M,3).
        Same math as _render_chunk with the pixel grid replaced by the points.
        Verified to match the full render at grid coords to float precision."""
        px, py, sigma, theta, freq, coeff = self.activate(raw.float())
        qx = pxy[:, 0][None, None, :]
        qy = pxy[:, 1][None, None, :]
        px_ = px[..., None]; py_ = py[..., None]; s_ = sigma[..., None]
        th = theta[..., None]; f_ = freq[..., None]
        dx = qx - px_; dy = qy - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None]
            b = coeff[:, :, c, 1][..., None]
            chans.append((env * (a * cos - b * sin)).sum(dim=1))
        return torch.sigmoid(torch.stack(chans, dim=-1))


class Encoder(nn.Module):
    def __init__(self, image_size=64, latent=128, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], 3, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent=128, num_packets=256, hidden=512):
        super().__init__()
        self.N = num_packets
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, num_packets * K_PARAMS))
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].weight.data *= 0.1

    def forward(self, z):
        return self.net(z).view(-1, self.N, K_PARAMS)


class SplatVAE(nn.Module):
    def __init__(self, image_size=64, latent=128, num_packets=256, chunk=64):
        super().__init__()
        self.enc = Encoder(image_size, latent)
        self.dec = Decoder(latent, num_packets)
        self.ren = GaborRenderer(image_size, num_packets, chunk)
        self.latent = latent

    def forward(self, x):
        mu, lv = self.enc(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.ren(self.dec(z)), mu, lv

    @torch.no_grad()
    def generate(self, z):
        return self.ren(self.dec(z))


def load_v1(path, device=DEVICE):
    """Load a real the_splat checkpoint; hyperparams inferred, strict=True."""
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict) or "ren.GX" not in sd:
        raise ValueError(f"{path} is not a the_splat v1 SplatVAE state_dict")
    image_size = sd["ren.GX"].shape[-1]
    num_packets = sd["ren.anchor_logit"].shape[0]
    latent = sd["enc.fc_mu.weight"].shape[0]
    hidden = sd["dec.net.0.weight"].shape[0]
    assert sd["dec.net.4.weight"].shape[0] // K_PARAMS == num_packets
    model = SplatVAE(image_size, latent, num_packets).to(device)
    model.dec.net[0] = nn.Linear(latent, hidden)
    model.dec.net[2] = nn.Linear(hidden, hidden)
    model.dec.net[4] = nn.Linear(hidden, num_packets * K_PARAMS)
    model.dec.to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[load_v1] {path}: image_size={image_size} packets={num_packets} "
          f"latent={latent} hidden={hidden}  (strict load OK)")
    return model


# ========================================================================== #
#  LUCAS-KANADE FLOW AT K POINTS  —  torch, no cv2, two pyramid levels       #
# ========================================================================== #

class LKFlow:
    """Sparse LK: for each point, solve the 2x2 structure-tensor system over a
    small window, iterated, on a 2-level pyramid. Returns per-point flow (in
    [0,1] coords), a confidence (min eigenvalue of the structure tensor) and
    the post-fit residual (used for precision)."""

    def __init__(self, win=9, iters=3, device=DEVICE):
        self.win = win
        self.iters = iters
        r = win // 2
        oy, ox = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1),
                                indexing="ij")
        self.offs = torch.stack([ox, oy], -1).reshape(-1, 2).float().to(device)  # (W2,2) px

    @staticmethod
    def gray(img):                       # (3,H,W) -> (H,W)
        return img.mean(0)

    @staticmethod
    def grads(g):                        # central differences, per pixel step
        gx = torch.zeros_like(g); gy = torch.zeros_like(g)
        gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
        gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
        return gx, gy

    @staticmethod
    def _sample(field, coords):          # field (H,W), coords (K,W2,2) xy [0,1]
        H = field.shape[-1]
        grid = coords[None] * 2 - 1                              # (1,K,W2,2)
        v = F.grid_sample(field[None, None], grid, align_corners=True,
                          padding_mode="border")
        return v[0, 0]                                           # (K,W2)

    def _level(self, g0, g1, pts, d0):
        """One pyramid level. pts (K,2) [0,1]; d0 initial flow [0,1]. Returns
        d, conf, resid."""
        H = g0.shape[-1]
        px = 1.0 / (H - 1)
        coords0 = pts[:, None, :] + self.offs[None] * px          # (K,W2,2)
        gx, gy = self.grads(g0)
        Ix = self._sample(gx, coords0); Iy = self._sample(gy, coords0)
        I0 = self._sample(g0, coords0)
        Sxx = (Ix * Ix).sum(1); Sxy = (Ix * Iy).sum(1); Syy = (Iy * Iy).sum(1)
        det = Sxx * Syy - Sxy * Sxy
        tr = Sxx + Syy
        conf = 0.5 * (tr - torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0)))  # min eig
        d = d0.clone()
        for _ in range(self.iters):
            I1 = self._sample(g1, coords0 + d[:, None, :])
            It = I1 - I0                                          # (K,W2)
            bx = -(Ix * It).sum(1); by = -(Iy * It).sum(1)
            safe = det.abs() > 1e-9
            dx = torch.where(safe, ( Syy * bx - Sxy * by) / det, torch.zeros_like(det))
            dy = torch.where(safe, (-Sxy * bx + Sxx * by) / det, torch.zeros_like(det))
            d = d + torch.stack([dx, dy], -1) * px                # px -> [0,1]
        I1 = self._sample(g1, coords0 + d[:, None, :])
        resid = (I1 - I0).abs().mean(1)
        return d, conf, resid

    def __call__(self, frame0, frame1, pts):
        g0f, g1f = self.gray(frame0), self.gray(frame1)
        g0c = F.avg_pool2d(g0f[None, None], 2)[0, 0]
        g1c = F.avg_pool2d(g1f[None, None], 2)[0, 0]
        d, _, _ = self._level(g0c, g1c, pts, torch.zeros_like(pts))
        d, conf, resid = self._level(g0f, g1f, pts, d)
        return d, conf, resid


def good_features(frame, n_cand=256, k=1, rng=None, avoid=None, min_d=0.06):
    """Pick k high-gradient locations from n_cand random candidates (a cheap
    Shi-Tomasi). avoid: (M,2) existing points to keep distance from."""
    g = LKFlow.gray(frame)
    gx, gy = LKFlow.grads(g)
    energy = (gx * gx + gy * gy)
    rng = rng or np.random.default_rng()
    cand = torch.tensor(rng.uniform(0.08, 0.92, (n_cand, 2)), dtype=torch.float32,
                        device=frame.device)
    e = LKFlow._sample(energy, cand[:, None, :])[:, 0]
    if avoid is not None and len(avoid):
        dist = torch.cdist(cand, avoid).min(dim=1).values
        e = torch.where(dist > min_d, e, torch.zeros_like(e))
    idx = torch.topk(e, k).indices
    return cand[idx]


# ========================================================================== #
#  THE FLOW CORTEX  —  held z, corrected by where the probes went            #
# ========================================================================== #


def packet_bands(vae, low_frac=0.5):
    """Split packets into (coarse_idx, fine_idx) by their trained spatial
    frequency. Low-freq packets = the envelope (body/pose); high-freq = detail
    (face). On the random stand-in field this is by construction; on a real
    celeba .pt it is an empirical read of the field's own organization."""
    with torch.no_grad():
        # freq for each packet at z=0 baseline (freq depends only on raw[...,4],
        # which the decoder sets; evaluate at the current prior mean is fine, but
        # z=0 gives a stable, state-independent split we can cache).
        raw = vae.dec(torch.zeros(1, vae.latent, device=DEVICE))
        freq = (1.0 + 15.0 * torch.sigmoid(raw[0, :, 4]))          # (N,)
    thresh = torch.quantile(freq, low_frac)
    coarse = (freq <= thresh).nonzero(as_tuple=True)[0]
    fine = (freq > thresh).nonzero(as_tuple=True)[0]
    return coarse, fine, freq


def render_probes_subset(ren, raw, pxy, idx):
    """render_probes but summing ONLY the packets in idx — so a band's flow
    corrects a band's packets. raw (1,N,K).

    NOTE: ren.activate() adds the full (N,2) anchor_logit buffer, so it can only
    be called on the full packet set. Here we inline the identical activation
    math with the anchor SLICED to idx — same numbers, subset-safe. (The verbatim
    v1 class is left untouched.)"""
    sub = raw[:, idx, :].float()
    a_px = ren.anchor_logit[idx, 0][None]
    a_py = ren.anchor_logit[idx, 1][None]
    px = torch.sigmoid(a_px + sub[..., 0])
    py = torch.sigmoid(a_py + sub[..., 1])
    sigma = 0.012 + 0.14 * torch.sigmoid(sub[..., 2])
    theta = sub[..., 3]
    freq = 1.0 + 15.0 * torch.sigmoid(sub[..., 4])
    coeff = torch.tanh(sub[..., 5:11]).reshape(*sub.shape[:2], 3, 2)
    qx = pxy[:, 0][None, None, :]; qy = pxy[:, 1][None, None, :]
    px_ = px[..., None]; py_ = py[..., None]; s_ = sigma[..., None]
    th = theta[..., None]; f_ = freq[..., None]
    dx = qx - px_; dy = qy - py_
    xr = dx * torch.cos(th) + dy * torch.sin(th)
    env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
    cos = torch.cos(2 * math.pi * f_ * xr); sin = torch.sin(2 * math.pi * f_ * xr)
    chans = []
    for c in range(3):
        a = coeff[:, :, c, 0][..., None]; b = coeff[:, :, c, 1][..., None]
        chans.append((env * (a * cos - b * sin)).sum(dim=1))
    # NOTE: no sigmoid here — a band is a partial field; we compare partial
    # renders old-vs-new consistently, so the monotone squash is unnecessary and
    # would compress the very gradients we steer by.
    return torch.stack(chans, dim=-1)


@torch.no_grad()
def render_full_subset(ren, raw, idx):
    """Full HxW render from ONLY the packets in idx. Same math as the verbatim
    forward(), anchor sliced to the subset. Used by the band diagnostic to SHOW
    what each half of the field draws. Returns (3,H,W) in [0,1]."""
    sub = raw[:, idx, :].float()
    a_px = ren.anchor_logit[idx, 0][None]
    a_py = ren.anchor_logit[idx, 1][None]
    px = torch.sigmoid(a_px + sub[..., 0])
    py = torch.sigmoid(a_py + sub[..., 1])
    sigma = 0.012 + 0.14 * torch.sigmoid(sub[..., 2])
    theta = sub[..., 3]
    freq = 1.0 + 15.0 * torch.sigmoid(sub[..., 4])
    coeff = torch.tanh(sub[..., 5:11]).reshape(*sub.shape[:2], 3, 2)
    px_ = px[..., None, None]; py_ = py[..., None, None]; s_ = sigma[..., None, None]
    th = theta[..., None, None]; f_ = freq[..., None, None]
    dx = ren.GX - px_; dy = ren.GY - py_
    xr = dx * torch.cos(th) + dy * torch.sin(th)
    env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
    cos = torch.cos(2 * math.pi * f_ * xr); sin = torch.sin(2 * math.pi * f_ * xr)
    chans = []
    for c in range(3):
        a = coeff[:, :, c, 0][..., None, None]; b = coeff[:, :, c, 1][..., None, None]
        chans.append((env * (a * cos - b * sin)).sum(dim=1))
    return torch.sigmoid(torch.stack(chans, dim=1))[0]




# ========================================================================== #
#  N-OCTAVE BAND SPLIT                                                        #
# ========================================================================== #

def octave_bands(vae, n_oct=4):
    """Split packets into n_oct log-spaced frequency octaves (low -> high) using
    the field's OWN trained freq. Returns (list[LongTensor], freq). Log spacing
    is the biologically-plausible choice: cortical frequency channels are
    roughly octave-spaced. Quantile edges so each band holds a comparable count
    even if the field's freq histogram is lumpy."""
    with torch.no_grad():
        raw = vae.dec(torch.zeros(1, vae.latent, device=DEVICE))
        freq = (1.0 + 15.0 * torch.sigmoid(raw[0, :, 4]))          # (N,)
    lf = torch.log(freq)
    edges = torch.quantile(lf, torch.linspace(0, 1, n_oct + 1, device=DEVICE))
    edges[0] -= 1e-3; edges[-1] += 1e-3
    bands = []
    for i in range(n_oct):
        m = (lf > edges[i]) & (lf <= edges[i + 1])
        bands.append(m.nonzero(as_tuple=True)[0])
    return bands, freq


def octave_colors(n):
    """Cool (cyan, low freq) -> warm (magenta, high freq) hex colors."""
    cols = []
    for i in range(n):
        h = 0.5 - 0.42 * (i / max(1, n - 1))   # 0.5 cyan .. 0.08 warm/magenta-ish
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.85, 1.0)
        cols.append("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
    return cols


def octave_diagnostic(model_path=None, seed=0, n_oct=4, n_samples=5,
                      out="octave_diagnostic.png"):
    """FULL | octave0 | octave1 | ... per sample. Shows the cascade the field
    learned: lowest octave = shading/envelope, highest = fine contours."""
    from PIL import Image, ImageDraw
    vae = load_v1(model_path) if model_path else make_test_vae(seed)
    bands, freq = octave_bands(vae, n_oct)
    for i, b in enumerate(bands):
        fr = freq[b]
        print(f"[octave {i}] {len(b):4d} packets  freq {fr.min():.2f}..{fr.max():.2f}")
    H = vae.ren.H
    rng = torch.Generator(device="cpu").manual_seed(seed)
    def np_img(t):
        return (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
    rows = []
    for s in range(n_samples):
        z = (torch.randn(1, vae.latent, generator=rng) * 1.1).to(DEVICE)
        with torch.no_grad():
            raw = vae.dec(z)
            cols = [np_img(vae.generate(z)[0])]
            for b in bands:
                cols.append(np_img(render_full_subset(vae.ren, raw, b)))
        rows.append(np.concatenate(cols, axis=1))
    grid = np.concatenate(rows, axis=0)
    hdr = Image.new("RGB", (grid.shape[1], 22), (16, 16, 20))
    d = ImageDraw.Draw(hdr)
    names = ["FULL"] + [f"OCT{i} ({'low' if i==0 else 'high' if i==n_oct-1 else 'mid'})"
                        for i in range(n_oct)]
    for i, name in enumerate(names):
        d.text((i * H + 5, 5), name, fill=(140, 230, 200))
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + 22), (16, 16, 20))
    canvas.paste(hdr, (0, 0)); canvas.paste(Image.fromarray(grid), (0, 22))
    canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST)
    canvas.save(out)
    print(f"[octave_diagnostic] wrote {out}  ({n_samples} x [full|{n_oct} octaves])")
    return out


# ========================================================================== #
#  PER-OCTAVE LK  —  window & pyramid level scale with the octave             #
# ========================================================================== #

class OctaveLK:
    """One LK config per octave. Lowest octave: big window on a 2x-downsampled
    pyramid (robust to large, low-freq motion). Highest: small window at full
    res (fine detail). Window/level interpolate between."""
    def __init__(self, n_oct=4, device=DEVICE):
        self.n_oct = n_oct
        self.cfg = []            # (LKFlow, downsample_factor)
        for i in range(n_oct):
            frac = i / max(1, n_oct - 1)            # 0 low .. 1 high
            win = int(round(13 - 8 * frac));  win += (win % 2 == 0)   # 13 -> 5, odd
            down = 2 if frac < 0.5 else 1           # low half on pyramid
            self.cfg.append((LKFlow(win=win, iters=4 if down == 2 else 3, device=device),
                             down))

    def flow(self, oct_i, f0, f1, pts):
        lk, down = self.cfg[oct_i]
        g0, g1 = LKFlow.gray(f0), LKFlow.gray(f1)
        if down == 2:
            g0 = F.avg_pool2d(g0[None, None], 2)[0, 0]
            g1 = F.avg_pool2d(g1[None, None], 2)[0, 0]
        return lk._level(g0, g1, pts, torch.zeros_like(pts))

    def win_of(self, oct_i):
        return self.cfg[oct_i][0].win, self.cfg[oct_i][1]


# ========================================================================== #
#  THE OCTAVE CASCADE CORTEX                                                  #
# ========================================================================== #

class OctaveCortex:
    def __init__(self, vae, n_oct=4, k_low=14, k_high=6, m_hist=8, sig_ref=0.02,
                 eta=20.0, beta_mom=0.5, leak=0.02, dz_clamp=0.25, seed=0):
        self.vae = vae
        self.n_oct = n_oct
        self.rng = np.random.default_rng(seed)
        self.z = torch.zeros(1, vae.latent, device=DEVICE)
        self.z_prev = self.z.clone(); self.z_prior = self.z.clone()
        self.bands, self.freq = octave_bands(vae, n_oct)
        self.lk = OctaveLK(n_oct)
        # per-octave schedules (low -> high)
        fr = np.linspace(0, 1, n_oct)
        self.k = [max(3, int(round(k_low + (k_high - k_low) * f))) for f in fr]  # probes
        self.w = [float(1.0 - 0.75 * f) for f in fr]         # weight 1.0 -> 0.25
        self.prec_cap = [float(1.0 - 0.6 * f) for f in fr]   # cap 1.0 -> 0.4
        self.eq = [1.0] * n_oct                              # graphical EQ gains
        self.active = list(range(n_oct))                     # BANDS isolation
        # probe state per octave
        self.pts = [None] * n_oct
        self.flow = [None] * n_oct
        self.res_hist = [deque(maxlen=m_hist) for _ in range(n_oct)]
        self.prec = [1.0] * n_oct; self.rough = [0.0] * n_oct; self.mag = [0.0] * n_oct
        self.prev_frame = None
        self.m_hist = m_hist; self.sig_ref = sig_ref; self.eta = eta
        self.beta_mom = beta_mom; self.leak = leak; self.dz_clamp = dz_clamp
        self.dynamic_precision = True; self.dz = 0.0
        s = 1.5
        oy, ox = torch.meshgrid(torch.tensor([-s, 0., s]), torch.tensor([-s, 0., s]),
                                indexing="ij")
        self.stencil_px = torch.stack([ox, oy], -1).reshape(-1, 2).to(DEVICE)

    def _stencil(self, pts):
        px = 1.0 / (self.vae.ren.H - 1)
        return (pts[:, None, :] + self.stencil_px[None] * px).reshape(-1, 2).clamp(0, 1)

    def seed(self, frame):
        avoid = None
        for i in range(self.n_oct):
            self.pts[i] = good_features(frame, k=self.k[i], rng=self.rng,
                                        n_cand=250, avoid=avoid, min_d=0.04)
            avoid = self.pts[i] if avoid is None else torch.cat([avoid, self.pts[i]])
            self.res_hist[i].clear()
        self.prev_frame = frame.detach()

    def set_k(self, k_low):
        fr = np.linspace(0, 1, self.n_oct)
        k_high = max(3, k_low // 3)
        self.k = [max(3, int(round(k_low + (k_high - k_low) * f))) for f in fr]
        self.pts = [None] * self.n_oct
        self.flow = [None] * self.n_oct

    def bootstrap(self, frame):
        with torch.no_grad():
            mu, _ = self.vae.enc(frame[None])
        self.z = mu.detach(); self.z_prev = self.z.clone(); self.z_prior = self.z.clone()

    @staticmethod
    def sample_frame(frame, pts):
        g = pts[None, None] * 2 - 1
        return F.grid_sample(frame[None], g, align_corners=True)[0, :, 0, :].T

    def _prec(self, i, r):
        h = self.res_hist[i]; h.append(r)
        if len(h) < 4:
            return 1.0, 0.0
        d2 = np.diff(np.array(h), n=2)
        rough = float(np.sqrt(np.mean(d2 * d2)))
        return float(self.sig_ref**2 / (self.sig_ref**2 + rough**2)), rough

    def step(self, frame):
        frame = frame.detach()
        if self.pts[0] is None or self.prev_frame is None:
            self.seed(frame); return

        # prior flow on z
        vel = self.z - self.z_prev
        z_pred = self.z + self.beta_mom * vel
        z_pred = z_pred - self.leak * (z_pred - self.z_prior)
        z_var = z_pred.detach().clone().requires_grad_(True)
        raw = self.vae.dec(z_var)
        with torch.no_grad():
            raw_old = self.vae.dec(self.z)

        loss = 0.0 * z_var.sum()
        eff_prec = [0.0] * self.n_oct
        for i in range(self.n_oct):
            if i not in self.active or self.eq[i] <= 1e-3 or len(self.bands[i]) == 0:
                self.flow[i] = None; continue
            d, conf, res = self.lk.flow(i, self.prev_frame, frame, self.pts[i])
            self.flow[i] = d.detach(); self.mag[i] = float(d.norm(dim=1).mean())
            if self.dynamic_precision:
                p, self.rough[i] = self._prec(i, float(res.mean()))
            else:
                p = 1.0
            p = min(p, self.prec_cap[i])
            self.prec[i] = p
            eff = self.w[i] * self.eq[i] * p       # weight * EQ gain * precision
            eff_prec[i] = eff
            moved = self._stencil((self.pts[i] + d).clamp(0, 1))
            here = self._stencil(self.pts[i])
            with torch.no_grad():
                ref = render_probes_subset(self.vae.ren, raw_old, here, self.bands[i])[0]
            pred = render_probes_subset(self.vae.ren, raw, moved, self.bands[i])[0]
            loss = loss + eff * F.mse_loss(pred, ref)
        loss.backward()

        with torch.no_grad():
            step = self.eta * z_var.grad
            n = step.norm()
            if n > self.dz_clamp:
                step = step * (self.dz_clamp / n)
            self.dz = float(step.norm())
            self.z_prev = self.z
            self.z = (z_pred - step).detach()
            self.z_prior = 0.995 * self.z_prior + 0.005 * self.z
            # probes ride their octave's flow, gated by that octave's precision
            for i in range(self.n_oct):
                if self.flow[i] is None:
                    continue
                self.pts[i] = (self.pts[i] + self.prec[i] * self.flow[i]).clamp(0.02, 0.98)
                bad = (self.pts[i].min(1).values < 0.04) | (self.pts[i].max(1).values > 0.96)
                if bad.any() and self.prec[i] > 0.5:
                    self.pts[i][bad] = good_features(frame, k=int(bad.sum()),
                                                     rng=self.rng, avoid=self.pts[i][~bad])
        self.prev_frame = frame

    @torch.no_grad()
    def belief_render(self, only_octave=None):
        """Full belief, or (EQ-gated) render of a single octave's packets."""
        raw = self.vae.dec(self.z)
        if only_octave is None:
            # EQ-weighted sum over octaves — the mixing board applied to output
            H = self.vae.ren.H
            out = torch.zeros(3, H, H, device=DEVICE)
            for i in range(self.n_oct):
                if len(self.bands[i]) == 0 or self.eq[i] <= 1e-3:
                    continue
                out = out + self.eq[i] * (render_full_subset(self.vae.ren, raw, self.bands[i]) - 0.5)
            return (out + 0.5).clamp(0, 1)
        return render_full_subset(self.vae.ren, raw, self.bands[only_octave])


# ========================================================================== #
#  SYNTHETIC WORLD  —  low latent axes = pose (slow), high axes = detail (fast)
# ========================================================================== #

class OctaveWorld:
    def __init__(self, vae, seed=0, omega_lo=0.04, omega_hi=0.14, radius=1.1,
                 lum_drift=False):
        self.vae = vae
        g = torch.Generator(device="cpu").manual_seed(seed)
        B = torch.randn(4, vae.latent, generator=g); B, _ = torch.linalg.qr(B.T)
        self.basis = B[:, :4].to(DEVICE)              # 0,1 pose ; 2,3 detail
        self.omega_lo, self.omega_hi, self.radius = omega_lo, omega_hi, radius
        self.t = 0
        self.z0 = torch.randn(1, vae.latent, generator=g).to(DEVICE) * 0.3
        self.lum_drift = lum_drift; self.slop_left = 0

    def _coeff(self, pose_only=False):
        tl, th = self.omega_lo * self.t, self.omega_hi * self.t
        c = [self.radius * math.cos(tl), self.radius * math.sin(tl),
             0.0 if pose_only else 0.5 * self.radius * math.cos(th),
             0.0 if pose_only else 0.5 * self.radius * math.sin(th)]
        return torch.tensor(c, dtype=torch.float32, device=DEVICE)

    def z_true(self):  return self.z0 + (self.basis @ self._coeff())[None]
    def pose_z(self):  return self.z0 + (self.basis @ self._coeff(pose_only=True))[None]
    def inject_slop(self, n=60): self.slop_left = n

    @torch.no_grad()
    def frame(self):
        self.t += 1
        img = self.vae.generate(self.z_true())[0]
        if self.lum_drift:
            img = (img * (0.65 + 0.35 * math.sin(2 * math.pi * self.t / 90))).clamp(0, 1)
        if self.slop_left > 0:
            self.slop_left -= 1; img = torch.rand_like(img)
        return img, self.z_true()


# ========================================================================== #
#  SELFTEST                                                                  #
# ========================================================================== #

def make_test_vae(seed=0, image_size=64, packets=144, latent=64,
                  wscale=8.0, bscale=2.0):
    torch.manual_seed(seed)
    vae = SplatVAE(image_size, latent, packets).to(DEVICE)
    with torch.no_grad():
        vae.dec.net[-1].weight *= wscale
        vae.dec.net[-1].bias.normal_(0, bscale)
    vae.eval(); return vae


def _run(vae, T=170, n_oct=4, active=None, eq=None, dynamic=True, lum_drift=False,
         slop_at=None, score="pose", seed=0):
    world = OctaveWorld(vae, seed=seed, lum_drift=lum_drift)
    ctx = OctaveCortex(vae, n_oct=n_oct, seed=seed)
    ctx.dynamic_precision = dynamic
    if active is not None: ctx.active = list(active)
    if eq is not None: ctx.eq = list(eq)
    f0, z0 = world.frame()
    ctx.z = (z0 + 0.2 * torch.randn_like(z0)).detach()
    ctx.z_prev = ctx.z.clone(); ctx.z_prior = ctx.z.clone()
    ctx.seed(f0)
    errs, after = [], []
    for t in range(T):
        if slop_at and t == slop_at[0]:
            world.inject_slop(slop_at[1] - slop_at[0])
        frame, z_true = world.frame()
        ctx.step(frame)
        if t % 2 == 0:
            with torch.no_grad():
                tgt = world.pose_z() if score == "pose" else z_true
                errs.append(F.mse_loss(vae.generate(ctx.z), vae.generate(tgt)).item())
            if slop_at and t >= slop_at[1] + 20:
                after.append(errs[-1])
    tail = errs[len(errs)//3:]
    return float(np.mean(tail)), (float(np.mean(after)) if after else None)


def _open(vae, seed=0, T=170):
    world = OctaveWorld(vae, seed=seed)
    f0, z0 = world.frame(); z = (z0 + 0.2*torch.randn_like(z0)); zp = z.clone()
    errs = []
    for t in range(T):
        world.frame(); z, zp = z + 0.5*(z-zp), z
        if t % 2 == 0:
            with torch.no_grad():
                errs.append(F.mse_loss(vae.generate(z), vae.generate(world.pose_z())).item())
    return float(np.mean(errs[len(errs)//3:]))


def selftest(seed=0, n_oct=4):
    print(f"\n=== the_splatV5 selftest (seed {seed}, {n_oct} octaves, {DEVICE}) ===")
    vae = make_test_vae(seed)
    bands, freq = octave_bands(vae, n_oct)
    for i, b in enumerate(bands):
        print(f"  octave {i}: {len(b):3d} packets  freq {freq[b].min():.2f}..{freq[b].max():.2f}")

    allb, _ = _run(vae, n_oct=n_oct, score="pose", seed=seed)
    opn = _open(vae, seed=seed)
    print(f"[A] pose: all-octaves {allb:.4f}   open {opn:.4f}")

    # [E] cascade inheritance: highest octave alone, WITH vs WITHOUT low octaves.
    hi = n_oct - 1
    hi_alone, _ = _run(vae, n_oct=n_oct, active=[hi], score="pose", seed=seed)
    with_low, _ = _run(vae, n_oct=n_oct, active=list(range(n_oct)), score="pose", seed=seed)
    lo_alone, _ = _run(vae, n_oct=n_oct, active=[0], score="pose", seed=seed)
    print(f"[E] cascade: high-octave alone {hi_alone:.4f}   "
          f"low-octave alone {lo_alone:.4f}   all (high inherits low) {with_low:.4f}")
    print(f"    inheritance holds if 'all' <= 'high alone' (low orients high).")

    b_all, _ = _run(vae, n_oct=n_oct, lum_drift=True, score="pose", seed=seed)
    b_lo, _ = _run(vae, n_oct=n_oct, active=[0], lum_drift=True, score="pose", seed=seed)
    print(f"[B] lum-drift pose: low-octave {b_lo:.4f}   all {b_all:.4f}   (clean low {lo_alone:.4f})")

    d_dyn, d_dyn_a = _run(vae, n_oct=n_oct, slop_at=(75,120), dynamic=True, score="full", seed=seed)
    d_fix, d_fix_a = _run(vae, n_oct=n_oct, slop_at=(75,120), dynamic=False, score="full", seed=seed)
    print(f"[D] slop: dyn {d_dyn:.4f} (after {d_dyn_a:.4f})   fixed {d_fix:.4f} (after {d_fix_a:.4f})")

    print("\nledger: [A] all-octaves beats open on pose.")
    print("        [E] THE BET: 'all' <= 'high-octave alone' — the high band")
    print("            steers better WITH the low octaves on (inherited orientation).")
    print("            If high-alone <= all, the cascade adds nothing; say so.")
    print("        [B] low octave holds pose under lum-drift. [D] dyn<=fixed in slop.")


# ========================================================================== #
#  GUI  —  per-octave EQ slider bank                                         #
# ========================================================================== #

def run_gui(model_path=None, webcam=False, cam_index=0, n_oct=4):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw

    vae = load_v1(model_path) if model_path else make_test_vae(0)
    has_enc = model_path is not None
    world = None if webcam else OctaveWorld(vae, seed=0)
    cap = None
    if webcam:
        import cv2
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("webcam not available — synthetic world instead")
            world, cap = OctaveWorld(vae, seed=0), None

    ctx = OctaveCortex(vae, n_oct=n_oct)
    cols = octave_colors(n_oct)
    H = vae.ren.H
    running = {"on": False}; slop = {"left": 0}; booted = {"done": False}
    view = {"mode": "full"}          # full | 0..n_oct-1

    root = tk.Tk()
    root.title(f"the_splatV5 — {n_oct}-octave cascade  (cyan=low freq -> warm=high)")
    root.configure(bg="#101014")
    VIEW = 280

    top = tk.Frame(root, bg="#101014"); top.pack(padx=8, pady=6)
    tk.Label(top, text="AFFERENT (octave-colored probes+flow)", fg="#8fd",
             bg="#101014").grid(row=0, column=0)
    tk.Label(top, text="BELIEF  render(dec(z))  [VIEW: EQ-mix / octave]", fg="#fd8",
             bg="#101014").grid(row=0, column=1)
    lab_world = tk.Label(top, bg="#101014"); lab_world.grid(row=1, column=0, padx=4)
    lab_belief = tk.Label(top, bg="#101014"); lab_belief.grid(row=1, column=1, padx=4)

    tele = tk.Label(root, fg="#ddd", bg="#101014", font=("Courier", 9), justify="left")
    tele.pack()

    # --- EQ slider bank: one vertical slider per octave ---
    eqf = tk.LabelFrame(root, text="OCTAVE EQ  (gain gates render + correction)",
                        fg="#8fd", bg="#101014", labelanchor="n")
    eqf.pack(pady=4)
    eq_sliders = []
    def make_eq_cb(i):
        def cb(v): ctx.eq[i] = float(v)
        return cb
    for i in range(n_oct):
        col = tk.Frame(eqf, bg="#101014"); col.grid(row=0, column=i, padx=6)
        s = tk.Scale(col, from_=1.0, to=0.0, resolution=0.05, orient="vertical",
                     length=90, bg="#101014", fg=cols[i], troughcolor="#333",
                     highlightthickness=0, command=make_eq_cb(i))
        s.set(1.0); s.pack()
        tk.Label(col, text=f"O{i}", fg=cols[i], bg="#101014").pack()
        eq_sliders.append(s)

    ctrl = tk.Frame(root, bg="#101014"); ctrl.pack(pady=4)

    def toggle():
        running["on"] = not running["on"]
        b_start.config(text="STOP" if running["on"] else "START")
    def cycle_view():
        seq = ["full"] + list(range(n_oct))
        cur = view["mode"]; view["mode"] = seq[(seq.index(cur) + 1) % len(seq)]
        b_view.config(text=f"VIEW {'MIX' if view['mode']=='full' else 'O'+str(view['mode'])}")
    def do_slop():
        if world: world.inject_slop(60)
        else: slop["left"] = 60
    def toggle_prec():
        ctx.dynamic_precision = not ctx.dynamic_precision
        b_prec.config(text=f"prec {'DYN' if ctx.dynamic_precision else 'FIX'}")
    def do_gist():
        if ctx.prev_frame is not None and has_enc: ctx.bootstrap(ctx.prev_frame)
    def set_k(v): ctx.set_k(int(float(v)))

    b_start = tk.Button(ctrl, text="START", command=toggle, width=6)
    b_view = tk.Button(ctrl, text="VIEW MIX", command=cycle_view)
    b_slop = tk.Button(ctrl, text="INJECT SLOP", command=do_slop)
    b_prec = tk.Button(ctrl, text="prec DYN", command=toggle_prec)
    b_gist = tk.Button(ctrl, text="GIST", command=do_gist,
                       state="normal" if has_enc else "disabled")
    for i, b in enumerate((b_start, b_view, b_slop, b_prec, b_gist)):
        b.grid(row=0, column=i, padx=3)
    tk.Label(ctrl, text="K", fg="#ddd", bg="#101014").grid(row=0, column=5, padx=(10, 0))
    s_k = tk.Scale(ctrl, from_=6, to=48, orient="horizontal", bg="#101014", fg="#ddd",
                   command=set_k, length=110); s_k.set(14); s_k.grid(row=0, column=6)

    def to_photo(img_t, bands=None):
        arr = (img_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((VIEW, VIEW), Image.NEAREST)
        if bands:
            dr = ImageDraw.Draw(im)
            for pts, fl, col in bands:
                if pts is None: continue
                P = pts.cpu().numpy()
                Fl = fl.cpu().numpy() if fl is not None else None
                draw_flow = Fl is not None and len(Fl) == len(P)
                for i, (x, y) in enumerate(P):
                    dr.ellipse([x*VIEW-2, y*VIEW-2, x*VIEW+2, y*VIEW+2], outline=col, width=2)
                    if draw_flow:
                        dr.line([x*VIEW, y*VIEW, (x+Fl[i,0]*8)*VIEW, (y+Fl[i,1]*8)*VIEW],
                                fill=col, width=2)
        return ImageTk.PhotoImage(im)

    def grab():
        if cap is not None:
            import cv2
            ok, fr = cap.read()
            if not ok: return None
            h, w, _ = fr.shape; s = min(h, w)
            fr = fr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            fr = cv2.cvtColor(cv2.resize(fr, (H, H)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(fr).float().permute(2, 0, 1).to(DEVICE) / 255.0
            if slop["left"] > 0:
                slop["left"] -= 1; t = torch.rand_like(t)
            return t
        img, _ = world.frame(); return img

    n = {"t": 0}
    def tick():
        if running["on"]:
            frame = grab()
            if frame is not None:
                if not booted["done"] and has_enc:
                    ctx.prev_frame = frame; ctx.bootstrap(frame); booted["done"] = True
                ctx.step(frame); n["t"] += 1
                overlays = [(ctx.pts[i], ctx.flow[i], cols[i]) for i in range(n_oct)]
                ph_w = to_photo(frame, overlays)
                lab_world.configure(image=ph_w); lab_world.image = ph_w
                if DEVICE == "cuda" or n["t"] % 2 == 0:
                    bimg = ctx.belief_render(None if view["mode"] == "full" else view["mode"])
                    ph_b = to_photo(bimg)
                    lab_belief.configure(image=ph_b); lab_belief.image = ph_b
                tele.config(text=(
                    "  ".join(f"O{i}:p{ctx.prec[i]:.2f}·eq{ctx.eq[i]:.1f}·K{ctx.k[i]}"
                              for i in range(n_oct))
                    + f"   |dz| {ctx.dz:5.3f}  t {n['t']}"))
        root.after(40, tick)
    tick(); root.mainloop()
    if cap is not None: cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--octaves", type=int, default=4)
    args = ap.parse_args()
    if args.diagnostic:
        octave_diagnostic(args.model, args.seed, args.octaves)
    elif args.selftest:
        selftest(args.seed, args.octaves)
    else:
        run_gui(args.model, args.webcam, args.cam, args.octaves)


if __name__ == "__main__":
    main()
