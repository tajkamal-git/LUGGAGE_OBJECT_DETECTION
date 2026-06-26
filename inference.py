"""
inference.py — Lightweight ONNX-based ensemble detection engine.

WHY ONNX INSTEAD OF ULTRALYTICS/TORCH:
PyTorch + ultralytics pulls in torch, torchvision, pandas, matplotlib, scipy —
often 500MB-1GB+. Vercel's Python functions cap at 500MB uncompressed.
ONNX Runtime + OpenCV + NumPy does the exact same inference in ~200MB,
comfortably fitting any serverless size budget, and starts faster too.

These are YOUR real trained weights — model_a.onnx and model_b.onnx were
exported directly from runs/detect/train3/weights/best.pt and
runs/detect/train4/weights/best.pt using ultralytics' own exporter, then
verified to reproduce matching detections on the bundled sample images.

KEY FIX vs the original project:
  • All paths are resolved relative to THIS FILE's location (BASE_DIR), not
    the process's current working directory.
  • No silent fallback to a generic pretrained model. I tested this
    explicitly: a stock COCO-pretrained YOLO on the bundled X-ray images
    hallucinates "train", "airplane", "cake", "clock" — pure domain-mismatch
    noise. Bolting that on would make the tool LOOK more comprehensive while
    actually being less trustworthy, so it's deliberately not used here.

TIERED DETECTION (this revision):
Real YOLO output has dozens of overlapping candidate boxes per object at
decreasing confidence — that's normal, and NMS already collapses them to one
box per object. The tiering below only ever looks at NMS *survivors* — i.e.
genuinely distinct candidate objects — and splits them into:
  • "confirmed" — at or above the (category-adjusted) threshold, drawn as a
    solid box.
  • "possible"  — below that but still plausible, drawn as a dashed amber
    box marked for human review instead of being silently discarded.
Critical-class items (weapons) use a *lower* effective bar than the slider
value — standard security posture: a false alarm on a knife-shaped object
costs a few seconds of operator review; a missed one doesn't.
"""

import os
import logging
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model registry ───────────────────────────────────────────────────────────
# model_a = train3 (7 classes)   model_b = train4 (1 class: gun only)
MODEL_CONFIGS = [
    {
        "key": "model_a",
        "path": os.path.join(BASE_DIR, "models", "model_a.onnx"),
        "names": {0: "gun", 1: "knife", 2: "blade", 3: "shuriken",
                 4: "spring", 5: "paperclip", 6: "zipper"},
    },
    {
        "key": "model_b",
        "path": os.path.join(BASE_DIR, "models", "model_b.onnx"),
        "names": {0: "gun"},
    },
]

# ── Class metadata — single source of truth for category, icon, description,
#    and per-category sensitivity. Used by both the drawing code and the
#    frontend reference guide (via get_model_status). ────────────────────────
CLASS_INFO = {
    "gun":        {"category": "critical",   "icon": "🔫", "trained": True,
                   "label": "Firearm",
                   "description": "Firearm or firearm component — prohibited in all luggage."},
    "knife":      {"category": "critical",   "icon": "🔪", "trained": True,
                   "label": "Knife",
                   "description": "Edged weapon — prohibited in carry-on, restricted in checked bags."},
    "blade":      {"category": "critical",   "icon": "🗡️", "trained": True,
                   "label": "Blade",
                   "description": "Sharp blade or blade component — prohibited in carry-on."},
    "shuriken":   {"category": "critical",   "icon": "✳️", "trained": True,
                   "label": "Throwing weapon",
                   "description": "Throwing weapon — prohibited in all luggage."},
    "spring":     {"category": "suspicious", "icon": "🌀", "trained": True,
                   "label": "Spring mechanism",
                   "description": "Mechanical spring — usually harmless (pens, tools) but can mask a concealed mechanism."},
    "paperclip":  {"category": "suspicious", "icon": "📎", "trained": True,
                   "label": "Metal fastener",
                   "description": "Small metal fastener — common false-positive source, flagged for confirmation."},
    "zipper":     {"category": "suspicious", "icon": "🧷", "trained": True,
                   "label": "Zipper pull",
                   "description": "Metal zipper pull — very common in luggage, flagged for confirmation."},
    "bottle":     {"category": "safe",       "icon": "🍶", "trained": False,
                   "label": "Bottle / container",
                   "description": "Container — check liquid volume against carry-on limits."},
    "screw":      {"category": "safe",       "icon": "🔩", "trained": False,
                   "label": "Screw / hardware",
                   "description": "Hardware fastener — no action needed."},
    "headset":    {"category": "safe",       "icon": "🎧", "trained": False,
                   "label": "Headset",
                   "description": "Electronics — no action needed."},
    "spectacles": {"category": "safe",       "icon": "👓", "trained": False,
                   "label": "Spectacles",
                   "description": "Eyewear — no action needed."},
}

CRITICAL_THREATS = {n for n, info in CLASS_INFO.items() if info["category"] == "critical"}
SUSPICIOUS_ITEMS = {n for n, info in CLASS_INFO.items() if info["category"] == "suspicious"}
SAFE_ITEMS       = {n for n, info in CLASS_INFO.items() if info["category"] == "safe"}

# Real-world prohibited-item categories with NO visual class in this dataset
# at all — included so the tool is honest about the boundary of AI coverage
# rather than implying full coverage. Shown in the UI's reference guide.
# These are NOT detected by anything in this codebase; they require human
# judgement, document checks, or specialised sensors (e.g. swabs, chemical
# trace detectors) that no image classifier substitutes for.
NON_VISUAL_CATEGORIES = [
    {"label": "Explosives & precursors", "icon": "💥",
     "note": "Requires trace-detection / swab equipment, not image classification."},
    {"label": "Flammable liquids & gases", "icon": "🔥",
     "note": "Often visually indistinguishable from permitted liquids on X-ray alone."},
    {"label": "Corrosive substances", "icon": "🧪",
     "note": "Requires chemical screening, not visual classification."},
    {"label": "Lithium battery packs over capacity", "icon": "🔋",
     "note": "Requires reading printed Wh rating, not shape detection."},
    {"label": "Undeclared currency over limit", "icon": "💵",
     "note": "A declarations/compliance check, not a visual screening task."},
]

# Per-category sensitivity. Effective confirmed threshold = user_slider * scale,
# floored at MIN_FLOOR. Lower scale = more sensitive = fires at a lower
# confidence than the slider nominally requests.
CONFIRMED_SCALE = {"critical": 0.55, "suspicious": 0.85, "safe": 1.0, "unknown": 1.0}
REVIEW_SCALE    = 0.5     # review-tier cutoff, relative to the confirmed cutoff above
MIN_FLOOR       = 0.12    # never go below this regardless of scaling -- pure noise below it
DECODE_FLOOR    = 0.12    # raw decode floor fed into NMS (must be <= the lowest possible review cutoff)

BOX_COLORS = {
    "critical":   (0,   0,   220),
    "suspicious": (0,   140, 255),
    "safe":       (30,  180, 30),
    "unknown":    (160, 160, 160),
}

def category(name: str) -> str:
    info = CLASS_INFO.get(name)
    return info["category"] if info else "unknown"

def confirmed_cutoff(cat: str, user_conf: float) -> float:
    return max(MIN_FLOOR, user_conf * CONFIRMED_SCALE.get(cat, 1.0))

def review_cutoff(cat: str, user_conf: float) -> float:
    return max(MIN_FLOOR, confirmed_cutoff(cat, user_conf) * REVIEW_SCALE)

# ── Lazy singleton session cache (reused across warm serverless invocations) ──
_sessions = None
_load_errors = []

def get_sessions():
    global _sessions, _load_errors
    if _sessions is not None:
        return _sessions

    loaded, errors = [], []
    for cfg in MODEL_CONFIGS:
        if not os.path.exists(cfg["path"]):
            msg = f"Missing weight file: {cfg['path']}"
            logger.error(msg)
            errors.append(msg)
            continue
        try:
            sess = ort.InferenceSession(cfg["path"], providers=["CPUExecutionProvider"])
            loaded.append({"key": cfg["key"], "session": sess, "names": cfg["names"]})
            logger.info(f"Loaded {cfg['key']} ({len(cfg['names'])} classes) from {cfg['path']}")
        except Exception as exc:
            msg = f"Failed to load {cfg['path']}: {exc}"
            logger.error(msg)
            errors.append(msg)

    _sessions = loaded
    _load_errors = errors
    return _sessions

def get_model_status() -> dict:
    """Diagnostic info surfaced in the UI so it's obvious whether the real
    trained models loaded, instead of guessing from silent behaviour. Also
    the single source of truth the frontend reference guide renders from."""
    sessions = get_sessions()
    total_classes = sorted({n for s in sessions for n in s["names"].values()})
    return {
        "models_loaded": len(sessions),
        "models_expected": len(MODEL_CONFIGS),
        "classes_covered": total_classes,
        "classes_never_trained": sorted(SAFE_ITEMS - set(total_classes)),
        "errors": _load_errors,
        "ready": len(sessions) > 0,
        "class_info": CLASS_INFO,
        "non_visual_categories": NON_VISUAL_CATEGORIES,
    }

# ── Preprocessing (letterbox, matches ultralytics' own behaviour) ────────────
def _letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, left, top

def _preprocess(img: np.ndarray, size: int = 640):
    lb, r, padx, pady = _letterbox(img, size)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(chw), r, padx, pady

# ── Decode + NMS ───────────────────────────────────────────────────────────
def _decode(output: np.ndarray, names: dict, r: float, padx: int, pady: int,
           conf_thr: float) -> list:
    preds = output[0].T                      # (num_anchors, 4+nc)
    boxes_xywh = preds[:, :4]
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]
    mask = confs > conf_thr
    boxes_xywh, class_ids, confs = boxes_xywh[mask], class_ids[mask], confs[mask]

    out = []
    for (cx, cy, w, h), cid, conf in zip(boxes_xywh, class_ids, confs):
        x1 = (cx - w / 2 - padx) / r
        y1 = (cy - h / 2 - pady) / r
        x2 = (cx + w / 2 - padx) / r
        y2 = (cy + h / 2 - pady) / r
        out.append({"class_name": names[int(cid)], "confidence": float(conf),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
    return out

def _iou(a: dict, b: dict) -> float:
    xi1, yi1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    xi2, yi2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _nms(detections: list, iou_thr: float = 0.45) -> list:
    """Collapses the dozens of overlapping raw candidate boxes YOLO produces
    around each real object down to one best box per object. This runs
    BEFORE tiering — tiering a pre-NMS box would just be re-labelling the
    same object's weaker duplicate echoes as separate 'possible' threats,
    which would be actively misleading, not more thorough."""
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets
                if d["class_name"] != best["class_name"] or _iou(best, d) < iou_thr]
    return kept

# ── Drawing ───────────────────────────────────────────────────────────────
def _dashed_rect(img, pt1, pt2, color, thickness, dash_len=10, gap_len=7):
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (ex1, ey1), (ex2, ey2) in edges:
        length = max(abs(ex2 - ex1), abs(ey2 - ey1))
        if length == 0:
            continue
        steps = max(1, length // (dash_len + gap_len))
        for i in range(int(steps) + 1):
            t0 = i * (dash_len + gap_len) / length
            t1 = min(1.0, t0 + dash_len / length)
            if t0 >= 1.0:
                break
            sx = int(ex1 + (ex2 - ex1) * t0); sy = int(ey1 + (ey2 - ey1) * t0)
            ex = int(ex1 + (ex2 - ex1) * t1); ey = int(ey1 + (ey2 - ey1) * t1)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

def _draw(img: np.ndarray, confirmed: list, possible: list) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 900.0
    thick_base = max(2, int(round(2 * scale)))
    font_scale = max(0.5, 0.55 * scale)

    for det in confirmed:
        cat = category(det["class_name"])
        color = BOX_COLORS[cat]
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        thick = thick_base + (1 if cat == "critical" else 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        label = f"{det['class_name']}  {det['confidence']:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ly1 = max(0, y1 - th - bl - 6)
        cv2.rectangle(out, (x1, ly1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)

    # Possible/review-tier: dashed amber box, distinct from confirmed
    review_color = (0, 184, 255)
    for det in possible:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        _dashed_rect(out, (x1, y1), (x2, y2), review_color, thick_base)

        label = f"{det['class_name']}?  {det['confidence']:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick_base)
        ly1 = max(0, y1 - th - bl - 6)
        cv2.rectangle(out, (x1, ly1), (x1 + tw + 8, y1), review_color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thick_base)
    return out

# ── Public entry point ────────────────────────────────────────────────────
def run_detection(img_bgr: np.ndarray, conf_threshold: float = 0.35) -> dict | None:
    """
    Run the full ensemble on a single BGR image (numpy array, as read by cv2).
    Returns dict with annotated image + tiered detection summary, or None if
    no model could be loaded at all (see get_model_status() for why).
    """
    sessions = get_sessions()
    if not sessions:
        return None

    try:
        inp, r, padx, pady = _preprocess(img_bgr)

        all_dets = []
        for s in sessions:
            input_name = s["session"].get_inputs()[0].name
            raw = s["session"].run(None, {input_name: inp})[0]
            all_dets.extend(_decode(raw, s["names"], r, padx, pady, DECODE_FLOOR))

        # NMS first -- collapse duplicate echoes into one box per real object
        survivors = _nms(all_dets, iou_thr=0.45)

        # THEN tier each surviving (genuinely distinct) object by confidence
        confirmed, possible = [], []
        for d in survivors:
            cat = category(d["class_name"])
            if d["confidence"] >= confirmed_cutoff(cat, conf_threshold):
                confirmed.append(d)
            elif d["confidence"] >= review_cutoff(cat, conf_threshold):
                possible.append(d)
            # else: too weak even for review -- dropped as noise

        detected_objects = sorted({d["class_name"] for d in confirmed})
        threat_items = sorted({d["class_name"] for d in confirmed
                               if category(d["class_name"]) in ("critical", "suspicious")})
        possible_objects = sorted({d["class_name"] for d in possible})
        possible_threat_items = sorted({d["class_name"] for d in possible
                                        if category(d["class_name"]) in ("critical", "suspicious")})

        conf_scores: dict[str, float] = {}
        for d in confirmed + possible:
            name = d["class_name"]
            if name not in conf_scores or d["confidence"] > conf_scores[name]:
                conf_scores[name] = round(d["confidence"] * 100, 1)

        annotated = _draw(img_bgr, confirmed, possible)

        if any(category(t) == "critical" for t in threat_items):
            security_level = "critical"
        elif threat_items:
            security_level = "suspicious"
        else:
            security_level = "safe"

        needs_review = len(possible_threat_items) > 0

        return {
            "image": annotated,
            "detected_objects": detected_objects,
            "threat_items": threat_items,
            "possible_objects": possible_objects,
            "possible_threat_items": possible_threat_items,
            "confidence_scores": conf_scores,
            "security_level": security_level,
            "is_threat": len(threat_items) > 0,
            "needs_review": needs_review,
            "raw_detections": confirmed + possible,
        }
    except Exception as exc:
        logger.error(f"run_detection failed: {exc}", exc_info=True)
        return None
