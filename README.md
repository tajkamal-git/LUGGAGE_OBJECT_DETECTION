# SecureScan — Live Luggage Screening Dashboard

A real screening-line style dashboard for your dual-YOLO luggage X-ray detector:
left side continuously screens a batch of images, right side shows the live
result on top and a running security log on the bottom.

---

## Real-time detection — what's actually in here

**Live Camera mode** (new): click "🎥 Live Camera" next to "Load Sample
Batch". It requests your camera (rear-facing on mobile), then captures and
screens frames continuously through the exact same tiered ONNX pipeline as
batch images — same confirmed/possible tiers, same threat alerts, same log.
Pause/Resume keeps the camera warm; the Prev button becomes "Stop Camera"
while live mode is active and fully releases the camera. No backend changes
were needed for this — it reuses `/api/screen` exactly as a batch upload
does, just with captured frames instead of files.

**Honest performance numbers**, benchmarked on this sandbox's CPU before
shipping (your hosting hardware will differ — GPU or a dedicated server
core will be meaningfully faster):

| Frame size | Latency | Throughput |
|---|---|---|
| Full-size X-ray (2700×2200) | ~246 ms | ~4.1 FPS |
| Webcam-typical (640×480) | ~209 ms | ~4.8 FPS |

I also tested running the two ONNX models concurrently via threads expecting
a speedup — there wasn't one (ONNX Runtime's CPU provider already
parallelizes internally, so outer threading just contends for the same
cores). I'm reporting that rather than claiming a gain I couldn't verify.
**A few FPS is genuinely useful for live monitoring** — it's roughly the
pace a human operator scans at anyway — but it's not 30fps video. If you
need higher throughput, GPU inference (`onnxruntime-gpu` + CUDA, on a host
that has a GPU — Vercel's functions don't) is the next lever, not something
fixable in software alone.

---

## Expanding what it can detect — what I tried, and why "detect anything" needs new data

You asked for the broadest possible coverage of prohibited items. I looked
into this seriously rather than just adding labels:

**Real X-ray security datasets exist, but I can't pull them into this
project from here.** SIXray (1,059,231 images: gun/knife/wrench/pliers/
scissors/hammer) and OPIXray are academic-license-only, distributed off of
GitHub — the GitHub repos hold citation/code, not the image data itself,
which is gated behind an academic-use agreement. GDXray is genuinely public
but it's industrial weld/casting X-rays, not luggage security — wrong domain
entirely despite the name. None of this is something I can legitimately
download into a sandboxed environment with restricted network access and
no ability to accept a dataset's usage terms on your behalf.

**What this means practically:** I cannot make this model detect explosives,
lighters, additional weapon types, etc. without real labelled training
images for those classes — and I won't fake that capability by, say,
dropping the confidence threshold until a generic model hallucinates a
match (I tested that approach last round; it produces "train", "airplane",
"cake" on X-ray images — pure noise dressed up as coverage).

**The honest, useful thing I can do: make adding real data as frictionless
as possible once you have it.**
- `train.py` already supports retraining and auto-exports straight to ONNX
  in the format this app expects.
- If you can get access to SIXray, OPIXray, HiXray, CLCXray, or PIDray under
  their academic terms (or label your own images — even a few hundred per
  new class meaningfully helps), convert their annotations to YOLO format
  and drop them into `dataset.yaml` alongside the existing classes.
- Roboflow Universe also hosts several open X-ray threat datasets with
  direct YOLO-format export — worth checking from a machine with normal
  internet access (it isn't reachable from this sandbox).
- Run `python train.py --model yolov8s --epochs 100`, then copy the
  resulting `.onnx` into `models/` and add its class list to
  `inference.py`'s `MODEL_CONFIGS` / `CLASS_INFO`.

I'd rather hand you a fast, honest path to real improvement than a model
that quietly does nothing useful on classes it's never seen.

---

## Detection coverage — what changed and why

You asked for the system to detect "everything" and label it. I looked into
two ways to genuinely expand coverage, tested both, and only kept the one
that actually holds up:

**Tried and rejected: bolting on a generic pretrained model.** I ran a stock
COCO-pretrained YOLO (never touched X-ray data) against your bundled sample
images to see if it could catch additional categories like scissors or
baseball bats. It hallucinated "train", "airplane", "cake", "clock", "book" —
pure domain-mismatch noise, because X-ray silhouettes look nothing like the
natural photos it was trained on. Adding this would have made the tool *look*
more comprehensive while actually being less trustworthy, so it's not in here.

**What's actually in here: tiered confidence detection.** Real YOLO output
has dozens of overlapping candidate boxes per object at decreasing
confidence — that's normal, and NMS already collapses them to one box per
object. Previously, anything that survived NMS below the confidence slider
was silently discarded. Now every surviving candidate is kept and split into:

- **Confirmed** — solid colored box, as before.
- **Possible — needs review** — dashed amber box, shown instead of being
  thrown away. A weak "gun" signal at 15% confidence that used to vanish
  completely now shows up clearly marked as unconfirmed, with its own toast,
  a lighter alert tone, a `⚠ NEEDS REVIEW` badge, and a dashed marker on the
  log row — distinct from a confirmed threat so it can't be confused with one.

**Critical items get a lower bar than the slider asks for.** Weapon-shaped
objects (gun/knife/blade/shuriken) confirm at roughly 55% of whatever the
confidence slider is set to; suspicious items at 85%; safe items at 100%
(unchanged). This mirrors standard security posture — a false alarm on a
knife-shaped object costs a few seconds of operator review, a missed one
doesn't. None of this required new training data; it's the same trained
weights, just no longer throwing away a borderline-but-real signal.

**Honest reference guide.** Click the ⓘ icon next to the sound toggle. It
lists, from the same metadata the detection engine itself uses (so it can't
drift out of sync):
- Every trained class, with an icon, plain-language description, and whether
  it's actually AI-detectable or just a label with no training examples
  behind it yet (bottle / screw / headset / spectacles).
- A separate, clearly-marked list of real prohibited-item categories — explosives,
  flammable liquids, batteries over capacity, undeclared currency — that
  **no image classifier here detects at all**, because they need trace
  detection or document checks, not visual classification. This is the part
  that matters most: being upfront about where AI coverage stops, rather than
  implying a single tool catches everything a real checkpoint has to catch.

---

## Visual design — v2

The interface is built around the idea of an actual checkpoint X-ray monitor
in a dim operations room, not a generic dark dashboard template:

- **Live Feed stage** — the signature element. Viewfinder brackets frame the
  image like a camera monitor, a subtle scanline texture overlays the feed,
  and a small `LIVE · CONF ≥ X%` HUD readout sits in the corner. The scan
  sweep now has a soft cyan glow matching an actual phosphor display.
- **Type system** — Chakra Petch for headings (a technical, HUD-flavoured
  display face), Inter for body text and controls, JetBrains Mono for every
  piece of data: filenames, timestamps, confidence percentages, the frame
  counter. Loaded from Google Fonts with system-font fallbacks, so the app
  still looks correct offline.
- **Signal palette** — cyan for the live/primary state, red for critical
  threats, amber for suspicious items, green for clear — used consistently
  across badges, chips, log rows, and the stage's threat-flash effect.

New functionality that came with the redesign:

| Feature | What it does |
|---|---|
| **Live Camera mode** | Real-time continuous screening from your camera, same tiered pipeline as batch images — see "Real-time detection" above |
| **Tiered detection** | Borderline detections below the confirm threshold are shown dashed/amber for review instead of being discarded — see "Detection coverage" above |
| **Reference guide** | ⓘ icon — honest breakdown of what's AI-detectable vs label-only vs not visually detectable at all |
| **Session stats readout** | Screened / Flagged / Avg ms / Clear rate, live-updating above the stage |
| **Threat flash + alert tone** | The stage frame flashes red and plays a two-tone alert (Web Audio, no audio file) the instant a critical item is found — toggle with the speaker icon top-right |
| **Toast notifications** | Replaced blocking `alert()` popups with non-blocking corner toasts |
| **Log filter + search** | Filter the log by Critical / Suspicious / Safe, or search by filename |
| **Click-to-zoom** | Click any result image to open it full-screen; `Esc` to close |
| **Keyboard shortcuts** | `Space` play/pause · `←` `→` step through the queue · `Esc` close preview |

Everything that already worked — sample batch loading, folder/file
selection, drag-and-drop, play/pause/next/prev, the confidence and speed
controls, CSV export, and the localStorage-backed log — is unchanged
underneath. `app.py` and `inference.py` now carry the tiered-detection logic
described above; the Flask routing, ONNX loading, and Vercel-safe stateless
architecture are otherwise the same as the version you confirmed working.

---

## What was actually wrong with the original version (root-caused, not guessed)

I ran your real `best.pt` files directly and compared them to what the app
produced. Two concrete, verifiable problems:

### 1. The model loader used relative paths
```python
weight_paths = ['runs/detect/train4/weights/best.pt', 'runs/detect/train3/weights/best.pt']
```
This only resolves if the process's **current working directory** happens to
be the project root. Run it via gunicorn, Docker, an IDE, or any serverless
platform (Vercel included) and the working directory is usually different —
the load silently failed and the app fell back to a generic pretrained model
that has never seen an X-ray image. That fallback is the most likely reason
it looked like "not detecting anything."

**Fix:** every path in `inference.py` is now resolved with
`os.path.dirname(os.path.abspath(__file__))`, so it works no matter where or
how the process is launched. I proved this by running the server and
querying it from `/tmp` as the working directory — confirmed working.

### 2. Your two trained models don't cover the same classes — and never did
I inspected the class list **embedded in each checkpoint** (not the
dataset.yaml, which was edited after the fact):

| Model | Classes it actually knows |
|---|---|
| `train3/best.pt` | gun, knife, blade, shuriken, spring, paperclip, zipper (7) |
| `train4/best.pt` | **gun only** (1) — it was trained on a separate `data.yaml` that no longer exists in the repo, declaring a single class |

**Neither model has ever been shown a labelled `bottle`, `screw`, `headset`,
or `spectacles`.** Those 4 classes are listed in `dataset.yaml` but no
training run ever included them. No code fix changes this — it's a data
problem, not a bug. If you scan an image containing only those items, zero
detections is the *correct* output, not a malfunction. To fix it for real,
add labelled images for those classes and retrain with `train.py`.

I exported both real checkpoints to ONNX and ran them against your bundled
sample images to confirm the detections are accurate (see screenshots below
— shuriken, blade, spring, zipper all detected with tight, correct boxes).

---

## The new screening dashboard

Single page, laid out exactly as requested:

```
┌─────────────────────────────────────────┬───────────────────┐
│                                          │   SCAN RESULT      │
│         LIVE SCREENING (60%)             │   (top half)       │
│   • Load Sample Batch / Select Folder    ├───────────────────┤
│   • big stage with scanning animation    │   SCREENING LOG     │
│   • play / pause / next / prev           │   (bottom half)     │
│   • confidence + speed controls          │   filename + level  │
│                                          │   per image, newest  │
│                                          │   on top             │
└─────────────────────────────────────────┴───────────────────┘
```

- **Left (60%)** — click **Load Sample Batch** to immediately screen the 5
  bundled X-ray images, or **Select Folder** / **Select Images** / drag-and-
  drop to point it at real images on your device. Each image is displayed
  with an animated scanning sweep while it's being processed, then it moves
  to the next one automatically (Play/Pause/Next/Prev + adjustable speed).
- **Top-right** — the most recently completed scan: annotated image, a big
  🔴/🟠/🟢 security badge, and a confidence chip per detected object.
- **Bottom-right** — the running log: every screened filename with its
  timestamp and security level, newest first. Click any row to re-open that
  result above. Export the whole log as CSV, or clear it.

The log persists in your browser (`localStorage`) so it survives a page
refresh — no database required.

---

## Why no database / no saved files this time

Vercel's Python functions have an **ephemeral filesystem** — anything a
request writes to disk is not guaranteed to exist for the next request, so
the original "save upload → save result → serve both from disk" pattern
breaks the moment it's deployed there. Every scan is now fully
self-contained instead: image bytes go in, an annotated JPEG comes back as a
base64 data URI in the same response, and the log lives in the browser. This
works identically on Vercel, a normal server, or your laptop.

---

## Why ONNX Runtime instead of PyTorch/Ultralytics

Vercel's Python function size limit is **500MB uncompressed** (raised from
250MB in Feb 2026). I checked actual installed sizes:

| Stack | Size |
|---|---|
| `torch` (default PyPI wheel) | **1.2 GB** |
| `onnxruntime` + `opencv-python-headless` + `numpy` | **~190 MB** |

Shipping `ultralytics`/`torch` risks blowing the limit once you add OpenCV,
pandas, matplotlib and the rest of ultralytics' own dependencies. So
`models/model_a.onnx` and `models/model_b.onnx` are **your real trained
weights**, exported once via `ultralytics.YOLO(...).export(format='onnx')`
and verified to reproduce matching detections — the deployed app never
imports `torch` at all, just `onnxruntime`. `requirements.txt` reflects this;
`requirements-train.txt` (torch + ultralytics) is separate and only needed if
you retrain locally.

---

## Deploying to Vercel

**Update:** if you previously hit *"The pattern 'app.py' defined in
`functions` doesn't match any Serverless Functions inside the `api`
directory"* — that was caused by a `vercel.json` I shipped with a
`functions` block targeting `app.py`. Confirmed against Vercel's current
docs: their zero-config Flask detection (which finds `app.py` at your
project root automatically) and the manual `functions` glob-config key are
two different mechanisms — the glob-matcher only resolves against files
under conventional locations, and a root-level `app.py` picked up by
framework auto-detection isn't one of them. **`vercel.json` has been removed
entirely** — this app needs zero configuration to deploy.

I also moved the static assets (`static/css/style.css`,
`static/js/screening.js`) into a `public/` directory as well, mirroring the
originals. Per Vercel's own Flask docs, `Flask's app.static_folder should
not be used for static files on Vercel — use the public/** directory
instead`, since files there are served directly from Vercel's CDN rather
than round-tripping through the Python function on every request. Both
copies exist side by side: local `python app.py` / gunicorn keeps using
Flask's normal `static/` folder; Vercel's CDN intercepts the same URLs from
`public/` automatically. If you ever edit the CSS or JS, update both copies
(or just the one matching your hosting target).

**Steps:**
1. Push this folder to a GitHub repo.
2. [vercel.com/new](https://vercel.com/new) → import the repo. Vercel
   detects the Flask app from `app.py` automatically — no `vercel.json`
   needed.
3. The Python runtime is currently in **beta** on Vercel — if the build
   still fails, check the build log; the error message usually names the
   exact missing piece.
4. **If inference times out on a cold start** (504 / `FUNCTION_INVOCATION_TIMEOUT`):
   raise the duration from the dashboard — Project → Settings → Functions →
   Function Max Duration — rather than `vercel.json`. That setting applies
   to any function regardless of how Vercel generated it, which sidesteps
   the glob-matching issue above entirely.
5. First request after a deploy/idle period will be a bit slower (cold
   start loading the ONNX models) — subsequent requests reuse the warm
   instance.

**If Vercel's Python beta gives you trouble:** Render, Railway, Fly.io, or
Hugging Face Spaces all run this exact same code with zero changes on a
normal persistent server, which sidesteps the serverless quirks entirely.
Worth keeping as a backup option since the Python runtime is still beta.

## Running locally

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Or with gunicorn (production-style, still local):
```bash
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
```

---

## Project structure

```
├── app.py                  Flask app (Vercel auto-detects this — zero config needed)
├── inference.py             ONNX ensemble engine — no torch needed
├── config.py
├── requirements.txt          Lean runtime deps (deployed)
├── requirements-train.txt    torch + ultralytics (local retraining only)
├── train.py                  Retraining script (not used at runtime)
├── dataset.yaml               Class list, typos fixed, limitations documented
├── models/
│   ├── model_a.onnx            from train3 — 7 classes
│   └── model_b.onnx            from train4 — gun only
├── sample_images/              5 bundled demo X-rays
├── public/                     Mirrors static/ — served directly by
│   └── static/                 Vercel's CDN per their Flask guidance
├── training_reference/         original training run artifacts (plots, .pt
│                                files, results.csv) — kept for your records,
│                                excluded from the Vercel deploy
├── static/
│   ├── css/style.css            (used by local/gunicorn hosting)
│   └── js/screening.js
└── templates/
    ├── base.html
    └── screening.html
```

---

## Detected classes & threat levels

| Class | Level | Confirm threshold | In `train3`? | In `train4`? |
|---|---|---|---|---|
| gun | 🔴 Critical | ~55% of slider | ✅ | ✅ |
| knife | 🔴 Critical | ~55% of slider | ✅ | ❌ |
| blade | 🔴 Critical | ~55% of slider | ✅ | ❌ |
| shuriken | 🔴 Critical | ~55% of slider | ✅ | ❌ |
| spring | 🟠 Suspicious | ~85% of slider | ✅ | ❌ |
| paperclip | 🟠 Suspicious | ~85% of slider | ✅ | ❌ |
| zipper | 🟠 Suspicious | ~85% of slider | ✅ | ❌ |
| bottle | 🟢 Safe | 100% of slider | ❌ never trained | ❌ never trained |
| screw | 🟢 Safe | 100% of slider | ❌ never trained | ❌ never trained |
| headset | 🟢 Safe | 100% of slider | ❌ never trained | ❌ never trained |
| spectacles | 🟢 Safe | 100% of slider | ❌ never trained | ❌ never trained |

Anything that clears half of its confirm threshold but not the full amount
shows up in the "possible — needs review" tier instead of being dropped.

Check this anytime live at `GET /api/model-status` — same data now also
powers the in-app Reference Guide (ⓘ icon), so the two can't drift out of
sync with each other.
top-right pill on the dashboard itself.
