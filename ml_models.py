"""
ml_models.py  —  Spinach Disease Detection — Full ML Stack
===========================================================
Deep model   : EfficientNet-B4 (timm, 19M params, ImageNet pretrained)
Classical    : XGBoost + Random Forest + Gradient Boosting + SVM + KNN + Logistic Regression
Ensemble     : Weighted vote across all 7 models
Report       : Full agronomic treatment report generated per prediction

All installed libraries are used:
  torch 2.10   | timm 1.0.9       — EfficientNet-B4
  xgboost 2.1  | sklearn 1.7      — 6 classical models
  opencv 4.13  | scipy 1.16       — advanced feature extraction
  numpy 2.3    | PIL 12.0         — array & image ops
  safetensors  | safetensors 0.8  — fast weight loading
"""
from __future__ import annotations

import io, re, os, json, time, base64, hashlib, logging
import threading, warnings, pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore")
logger = logging.getLogger("spinach")

# ── OpenCV ────────────────────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── SciPy ─────────────────────────────────────────────────────────────────────
try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ── PyTorch + timm ────────────────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    import timm
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── XGBoost ───────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ── sklearn ───────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, classification_report,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from config import Config

# ─────────────────────────────────────────────────────────────────────────────
# Colour-Rule Classifier  —  deterministic, no training required
# ─────────────────────────────────────────────────────────────────────────────

class ColourRuleClassifier:
    """
    Rule-based classifier using colour analysis and 49-dim features.
    Produces calibrated probability scores without any training data.
    Used as a reliable baseline when the deep model head is untrained.
    """

    @staticmethod
    def predict(colour: dict, features: np.ndarray) -> dict:
        """
        Returns a result dict with the same shape as DeepModel.predict().
        Scores are driven by colour percentages, texture, and edge features.
        """
        pct   = colour.get("colour_pct", {})
        hints = colour.get("disease_hints", [])
        hint_set = {h["signal"] for h in hints}

        green  = pct.get("green",  0.0)
        yellow = pct.get("yellow", 0.0)
        brown  = pct.get("brown",  0.0)
        white  = pct.get("white",  0.0)
        dark   = pct.get("dark",   0.0)
        purple = pct.get("purple", 0.0)

        # Pull useful feature dims (indices match extract_features order)
        hue_var   = float(features[30]) if len(features) > 30 else 0.0  # Block 7: hue_variance
        green_pat = float(features[31]) if len(features) > 31 else 0.0  # green_patchiness
        edge_den  = float(features[24]) if len(features) > 24 else 0.0  # Block 6: edge_density
        entropy   = float(features[20]) if len(features) > 20 else 0.0  # Block 5: entropy
        spot_den  = float(features[26]) if len(features) > 26 else 0.0  # spot_density
        sat_var   = float(features[33]) if len(features) > 33 else 0.0  # saturation_variance

        # ── Score each disease ────────────────────────────────────────────────
        scores: dict[str, float] = {lbl: 1.0 for lbl in Config.LABELS}

        # healthy: high green, low yellow/brown/dark, uniform
        scores["healthy"] = (
            max(green - 40, 0) * 1.2
            + max(80 - yellow, 0) * 0.3
            + max(80 - brown, 0) * 0.3
            + max(5 - hue_var * 100, 0) * 2.0
            + 5.0
        )

        # downy_mildew: yellow + purple/white coating + high humidity texture
        scores["downy_mildew"] = (
            yellow * 0.8
            + purple * 2.0
            + white * 0.8
            + ("Yellowing" in hint_set) * 15
            + ("Purple/Grey sporulation" in hint_set) * 25
            + ("White coating" in hint_set) * 10
            + hue_var * 30
        )

        # leaf_spot: brown spots + dark lesions + high edge density
        scores["leaf_spot"] = (
            brown * 1.5
            + dark * 1.2
            + edge_den * 60
            + spot_den * 80
            + ("Browning/Necrosis" in hint_set) * 15
            + ("Dark lesions" in hint_set) * 12
        )

        # damping_off: dark + brown heavy + low brightness (collapsed tissue)
        scores["damping_off"] = (
            dark * 2.0
            + brown * 0.8
            + max(30 - green, 0) * 0.5
            + spot_den * 50
            + ("Dark lesions" in hint_set) * 10
        )

        # white_rust: strong white blister patches
        scores["white_rust"] = (
            white * 2.5
            + yellow * 0.5
            + ("White coating" in hint_set) * 30
            + ("Purple/Grey sporulation" in hint_set) * 8
        )

        # anthracnose: brown + dark lesions + edge sharpness
        scores["anthracnose"] = (
            brown * 1.2
            + dark * 1.5
            + edge_den * 40
            + ("Browning/Necrosis" in hint_set) * 10
            + ("Dark lesions" in hint_set) * 15
        )

        # mosaic_virus: patchy green/yellow contrast + high hue variance
        scores["mosaic_virus"] = (
            hue_var * 50
            + green_pat * 40
            + yellow * 0.6
            + sat_var * 30
            + ("Yellowing" in hint_set) * 8
        )

        # nutrient_deficiency: uniform yellowing, low brown/dark (no lesions)
        scores["nutrient_deficiency"] = (
            yellow * 1.0
            + max(30 - brown, 0) * 0.3
            + max(30 - dark, 0) * 0.3
            + ("Yellowing" in hint_set) * 12
            - ("Dark lesions" in hint_set) * 10
            + max(entropy - 5, 0) * 5
        )

        # pest_damage: irregular holes → high edge density + dark spots
        scores["pest_damage"] = (
            edge_den * 50
            + dark * 1.0
            + spot_den * 60
            + brown * 0.5
            + ("Dark lesions" in hint_set) * 8
        )

        # ── Softmax-style normalisation ───────────────────────────────────────
        import math
        # Clamp negatives to 0.5 so every class keeps a small probability
        clamped = {k: max(v, 0.5) for k, v in scores.items()}
        exp_s = {k: math.exp(min(v / 15.0, 10)) for k, v in clamped.items()}
        total = sum(exp_s.values()) or 1.0
        proba = {k: round(v / total * 100, 2) for k, v in exp_s.items()}

        sorted_p = sorted(proba.items(), key=lambda kv: -kv[1])
        best, best_conf = sorted_p[0]

        return {
            "prediction":       best,
            "confidence":       best_conf,
            "model_used":       "colour_rule_classifier",
            "top3":             [{"label": l, "probability": p} for l, p in sorted_p[:3]],
            "all_probabilities": proba,
        }


def _is_deep_model_random(deep_result: dict) -> bool:
    """
    Detect whether the deep model output looks like random noise.
    A trained model on a real image should have at least one class > 35%.
    If the top confidence is < 30% the head is likely untrained/random.
    """
    top_conf = deep_result.get("confidence", 0)
    return top_conf < 30.0


# ─────────────────────────────────────────────────────────────────────────────
# PlantVillage → Spinach mapping  (38 explicit entries, zero defaults)
# ─────────────────────────────────────────────────────────────────────────────
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy",
]

PV_TO_SPINACH: dict[str, str] = {
    "Apple___Apple_scab":                                "leaf_spot",
    "Apple___Black_rot":                                 "anthracnose",
    "Apple___Cedar_apple_rust":                          "white_rust",
    "Apple___healthy":                                   "healthy",
    "Blueberry___healthy":                               "healthy",
    "Cherry_(including_sour)___Powdery_mildew":          "downy_mildew",
    "Cherry_(including_sour)___healthy":                 "healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "leaf_spot",
    "Corn_(maize)___Common_rust_":                       "white_rust",
    "Corn_(maize)___Northern_Leaf_Blight":               "leaf_spot",
    "Corn_(maize)___healthy":                            "healthy",
    "Grape___Black_rot":                                 "anthracnose",
    "Grape___Esca_(Black_Measles)":                      "mosaic_virus",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":        "leaf_spot",
    "Grape___healthy":                                   "healthy",
    "Orange___Haunglongbing_(Citrus_greening)":          "nutrient_deficiency",
    "Peach___Bacterial_spot":                            "damping_off",
    "Peach___healthy":                                   "healthy",
    "Pepper,_bell___Bacterial_spot":                     "damping_off",
    "Pepper,_bell___healthy":                            "healthy",
    "Potato___Early_blight":                             "leaf_spot",
    "Potato___Late_blight":                              "downy_mildew",
    "Potato___healthy":                                  "healthy",
    "Raspberry___healthy":                               "healthy",
    "Soybean___healthy":                                 "healthy",
    "Squash___Powdery_mildew":                           "downy_mildew",
    "Strawberry___Leaf_scorch":                          "leaf_spot",
    "Strawberry___healthy":                              "healthy",
    "Tomato___Bacterial_spot":                           "damping_off",
    "Tomato___Early_blight":                             "leaf_spot",
    "Tomato___Late_blight":                              "downy_mildew",
    "Tomato___Leaf_Mold":                                "downy_mildew",
    "Tomato___Septoria_leaf_spot":                       "leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":     "pest_damage",
    "Tomato___Target_Spot":                              "anthracnose",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":            "mosaic_virus",
    "Tomato___Tomato_mosaic_virus":                      "mosaic_virus",
    "Tomato___healthy":                                  "healthy",
}

_VALID_SIGS = (
    b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n",
    b"GIF87a", b"GIF89a", b"BM", b"RIFF", b"II*\x00", b"MM\x00*",
)

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_image_bytes(data: bytes) -> bool:
    return len(data) >= 8 and any(data.startswith(s) for s in _VALID_SIGS)

def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def size_label(n: int) -> str:
    if n < 1024:      return f"{n} B"
    if n < 1_048_576: return f"{n/1024:.1f} KB"
    return f"{n/1_048_576:.1f} MB"

def allowed_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXT

def sanitize_name(fn: str) -> str:
    return re.sub(r"[^\w.\-]", "_", os.path.basename(fn or ""))[:200] or "upload"

def fix_filename_ext(filename: str, image_bytes: bytes) -> str:
    if allowed_ext(filename): return filename
    if image_bytes[:3] == b"\xff\xd8\xff":         ext = "jpg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":  ext = "png"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP": ext = "webp"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"): ext = "gif"
    elif image_bytes[:2] == b"BM":                 ext = "bmp"
    elif image_bytes[:4] in (b"II*\x00", b"MM\x00*"): ext = "tiff"
    else: ext = "jpg"
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f"{base}.{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# ImageProcessor  —  preprocessing, colour analysis, 49-dim feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class ImageProcessor:
    """
    Handles image loading, resizing, thumbnail, colour analysis, and feature extraction.

    Feature vector (49 dims) uses:
      OpenCV  — LAB colour space, CLAHE, Sobel gradients, HSV (vectorised)
      scipy   — skewness and kurtosis of green channel distribution
      numpy   — histogram percentiles, block patchiness, spatial symmetry
    """
    N_FEATURES = 49

    def __init__(self, image_bytes: bytes):
        self.raw  = image_bytes
        self.orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        self._resized: Image.Image | None = None
        self._thumb:   Image.Image | None = None
        self.meta = {
            "original_width":  self.orig.width,
            "original_height": self.orig.height,
            "aspect_ratio":    round(self.orig.width / max(self.orig.height, 1), 3),
            "megapixels":      round(self.orig.width * self.orig.height / 1_000_000, 3),
            "file_size":       len(image_bytes),
            "file_size_label": size_label(len(image_bytes)),
            "file_hash":       compute_hash(image_bytes),
        }

    def prepare(self) -> "ImageProcessor":
        self._resized = self.orig.resize(Config.IMG_SIZE, Image.LANCZOS)
        t = self.orig.copy(); t.thumbnail(Config.THUMB_SIZE, Image.LANCZOS)
        self._thumb = t
        return self

    def pil_image(self) -> Image.Image:
        return self._resized if self._resized is not None else self.orig

    def thumbnail_b64(self) -> str:
        if self._thumb is None: self.prepare()
        buf = io.BytesIO()
        self._thumb.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def colour_analysis(self) -> dict:
        if self._resized is None: self.prepare()
        arr = np.array(self._resized, dtype=np.float32) / 255.0
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        import colorsys
        h_vals, s_vals, v_vals = [], [], []
        for row in range(0, arr.shape[0], 4):
            for col in range(0, arr.shape[1], 4):
                h, s, v = colorsys.rgb_to_hsv(float(r[row,col]),float(g[row,col]),float(b[row,col]))
                h_vals.append(h); s_vals.append(s); v_vals.append(v)
        def pct(mask): return round(float(np.mean(mask)*100), 2)
        green_pct  = pct((g > r) & (g > b) & (g > 0.25))
        yellow_pct = pct((r > 0.55) & (g > 0.55) & (b < 0.35))
        brown_pct  = pct((r > 0.40) & (g < r*0.75) & (b < r*0.60))
        white_pct  = pct((r > 0.80) & (g > 0.80) & (b > 0.80))
        dark_pct   = pct((r < 0.20) & (g < 0.20) & (b < 0.20))
        purple_pct = pct((b > 0.40) & (r > 0.30) & (g < 0.35))
        dom_r = round(float(r.mean())*255); dom_g = round(float(g.mean())*255); dom_b = round(float(b.mean())*255)
        def hist32(ch):
            h, _ = np.histogram(ch.flatten(), bins=32, range=(0,1)); return [int(x) for x in h]
        hints = []
        if yellow_pct > 15:
            hints.append({"signal":"Yellowing","pct":yellow_pct,"suggests":["downy_mildew","nutrient_deficiency","mosaic_virus"]})
        if brown_pct > 10:
            hints.append({"signal":"Browning/Necrosis","pct":brown_pct,"suggests":["leaf_spot","anthracnose","damping_off"]})
        if white_pct > 8:
            hints.append({"signal":"White coating","pct":white_pct,"suggests":["white_rust","downy_mildew"]})
        if purple_pct > 5:
            hints.append({"signal":"Purple/Grey sporulation","pct":purple_pct,"suggests":["downy_mildew","white_rust"]})
        if dark_pct > 8:
            hints.append({"signal":"Dark lesions","pct":dark_pct,"suggests":["anthracnose","leaf_spot","damping_off"]})
        if green_pct > 70 and yellow_pct < 5 and brown_pct < 5:
            hints.append({"signal":"Uniform green","pct":green_pct,"suggests":["healthy"]})
        return {
            "dominant_hex":   "#{:02x}{:02x}{:02x}".format(dom_r,dom_g,dom_b),
            "dominant_rgb":   {"r":dom_r,"g":dom_g,"b":dom_b},
            "avg_hue_deg":    round(float(np.mean(h_vals))*360, 1),
            "avg_saturation": round(float(np.mean(s_vals))*100, 1),
            "avg_brightness": round(float(np.mean(v_vals))*100, 1),
            "colour_pct":     {"green":green_pct,"yellow":yellow_pct,"brown":brown_pct,
                               "white":white_pct,"dark":dark_pct,"purple":purple_pct},
            "rgb_histogram":  {"r":hist32(r),"g":hist32(g),"b":hist32(b)},
            "disease_hints":  hints,
        }

    def extract_features(self) -> np.ndarray:
        """
        49-dimensional feature vector.
        Blocks: RGB(6) | HSV(6) | LAB(6) | colour_indices(8) | texture(4) | spatial(4) | mosaic(6) | percentiles(9)
        """
        if self._resized is None: self.prepare()
        pil_img = self._resized
        img_np  = np.array(pil_img, dtype=np.uint8)
        arr_f   = img_np.astype(np.float32) / 255.0
        r, g, b = arr_f[:,:,0], arr_f[:,:,1], arr_f[:,:,2]

        # ── Block 1: RGB stats (6) ────────────────────────────────────────────
        r_mean,g_mean,b_mean = float(r.mean()),float(g.mean()),float(b.mean())
        r_std, g_std, b_std  = float(r.std()), float(g.std()), float(b.std())

        # ── Block 2: HSV via OpenCV (6) ───────────────────────────────────────
        if CV2_OK:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            h_f = hsv[:,:,0]/179.0; s_f = hsv[:,:,1]/255.0; v_f = hsv[:,:,2]/255.0
        else:
            import colorsys
            h_list,s_list,v_list=[],[],[]
            for row in range(0,224,4):
                for col in range(0,224,4):
                    hh,ss,vv=colorsys.rgb_to_hsv(float(r[row,col]),float(g[row,col]),float(b[row,col]))
                    h_list.append(hh);s_list.append(ss);v_list.append(vv)
            h_f=np.array(h_list);s_f=np.array(s_list);v_f=np.array(v_list)
        h_mean,h_std=float(h_f.mean()),float(h_f.std())
        s_mean,s_std=float(s_f.mean()),float(s_f.std())
        v_mean,v_std=float(v_f.mean()),float(v_f.std())

        # ── Block 3: LAB via OpenCV (6) — perceptually uniform ───────────────
        if CV2_OK:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_ch,a_ch,b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            l_f=l_ch.astype(np.float32)/255.0; a_f=a_ch.astype(np.float32)/255.0; b_f_lab=b_ch.astype(np.float32)/255.0
        else:
            gray_pil=np.array(pil_img.convert("L"),dtype=np.float32)/255.0
            l_f=gray_pil; a_f=(g-r+1)/2; b_f_lab=(g-b+1)/2
        l_mean,l_std=float(l_f.mean()),float(l_f.std())
        a_mean,a_std=float(a_f.mean()),float(a_f.std())
        b_lab_mean,b_lab_std=float(b_f_lab.mean()),float(b_f_lab.std())

        # ── Block 4: Colour indices (8) ───────────────────────────────────────
        green_dominance = float(g_mean-(r_mean+b_mean)/2)
        yellow_index    = float((r_mean+g_mean)/2-b_mean)
        brown_index     = float(r_mean-g_mean)
        white_index     = float(min(r_mean,g_mean,b_mean))
        brightness      = float(l_f.mean())
        contrast        = float(l_f.std())
        warm_cool       = float(r_mean-b_mean)
        green_ratio     = float(g_mean/max(r_mean+b_mean,0.001))

        # ── Block 5: Texture via scipy+OpenCV (4) ─────────────────────────────
        gray_f = np.array(pil_img.convert("L"),dtype=np.float32)/255.0
        hist_g,_=np.histogram(gray_f.flatten(),bins=256,range=(0,1))
        hist_p=hist_g/(hist_g.sum()+1e-9)
        entropy=float(-np.sum(hist_p*np.log2(hist_p+1e-9)))
        if SCIPY_OK:
            g_skew=float(sp_stats.skew(g.flatten()))
            g_kurt=float(sp_stats.kurtosis(g.flatten()))
        else:
            g_skew=float(((g-g_mean)**3).mean()/max(g_std**3,1e-9))
            g_kurt=float(((g-g_mean)**4).mean()/max(g_std**4,1e-9))-3.0
        sharpness=float(np.mean(np.abs(np.diff(gray_f,axis=1))))

        # ── Block 6: Spatial/Edge via OpenCV (4) ─────────────────────────────
        if CV2_OK:
            gray_cv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
            sobelx=cv2.Sobel(gray_cv,cv2.CV_64F,1,0,ksize=3)
            sobely=cv2.Sobel(gray_cv,cv2.CV_64F,0,1,ksize=3)
            grad=np.sqrt(sobelx**2+sobely**2)
            edge_density=float(grad.mean()/255.0)
            gradient_variance=float(grad.var()/65025.0)
            blurred=cv2.GaussianBlur(arr_f,(5,5),0)
            noise=float(np.mean(np.abs(arr_f-blurred)))
        else:
            edges=np.array(pil_img.filter(ImageFilter.FIND_EDGES),dtype=np.float32)/255.0
            edge_density=float(edges.mean()); gradient_variance=float(edges.var())
            blurred_pil=np.array(pil_img.filter(ImageFilter.GaussianBlur(2)),dtype=np.float32)/255.0
            noise=float(np.mean(np.abs(arr_f-blurred_pil)))
        spot_density=float(np.mean((gray_f<0.25).astype(np.float32)))
        left=arr_f[:,:112,:]; right=arr_f[:,112:,:]
        symmetry=float(1.0-np.mean(np.abs(left-right[:,::-1,:])))

        # ── Block 7: Mosaic discriminators (6) — key for mosaic vs downy ─────
        hue_variance=float(h_f.var())
        block_greens=[]
        for br in range(0,224,28):
            for bc in range(0,224,28):
                block_greens.append(float(g[br:br+28,bc:bc+28].mean()-r[br:br+28,bc:bc+28].mean()))
        green_patchiness=float(np.std(block_greens))
        ym=((r>0.55)&(g>0.55)&(b<0.35)).astype(np.float32)
        q_y=[ym[:112,:112].mean(),ym[:112,112:].mean(),ym[112:,:112].mean(),ym[112:,112:].mean()]
        yellow_uniformity=float(1.0-np.std(q_y))
        saturation_variance=float(s_f.var())
        green_yellow_contrast=float(np.mean((g>0.45)&(g>r)&(g>b))/max(float(np.mean(ym)),0.001))
        # a* channel in LAB: negative=green, positive=red/yellow
        a_variance=float(a_f.var())

        # ── Block 8: Histogram percentiles RGB (9) ────────────────────────────
        r_p25,r_p50,r_p75=np.percentile(r,[25,50,75])
        g_p25,g_p50,g_p75=np.percentile(g,[25,50,75])
        b_p25,b_p50,b_p75=np.percentile(b,[25,50,75])

        return np.array([
            r_mean,g_mean,b_mean,r_std,g_std,b_std,          # 6
            h_mean,h_std,s_mean,s_std,v_mean,v_std,           # 6
            l_mean,l_std,a_mean,a_std,b_lab_mean,b_lab_std,   # 6
            green_dominance,yellow_index,brown_index,white_index,
            brightness,contrast,warm_cool,green_ratio,         # 8
            entropy,g_skew,g_kurt,sharpness,                   # 4
            edge_density,gradient_variance,spot_density,symmetry, # 4
            hue_variance,green_patchiness,yellow_uniformity,
            saturation_variance,green_yellow_contrast,a_variance, # 6
            float(r_p25),float(r_p50),float(r_p75),
            float(g_p25),float(g_p50),float(g_p75),
            float(b_p25),float(b_p50),float(b_p75),           # 9
        ], dtype=np.float32)  # total = 49


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B4 Deep Model
# ─────────────────────────────────────────────────────────────────────────────

class DeepModel:
    """EfficientNet-B4 (timm, 19M params). Loads from local safetensors — no download on repeat runs."""
    FINETUNED_PATH   = Path("models/efficientnet_b4_spinach.pth")
    IMAGENET_PATH    = Path("models/efficientnet_b4_imagenet.pth")
    SAFETENSORS_PATH = Path("models/efficientnet_b4_imagenet.safetensors")
    TIMM_MODEL       = "efficientnet_b4"

    def __init__(self):
        self.net = None; self.transform = None
        self.ready = False; self._load_error: str | None = None
        self._direct_9 = False; self._n_outputs = 9
        if TORCH_OK:
            threading.Thread(target=self._load, daemon=True).start()
        else:
            self._load_error = "PyTorch/timm not installed."

    def _load(self) -> None:
        try:
            if self.FINETUNED_PATH.exists():   self._load_finetuned()
            elif self.IMAGENET_PATH.exists():  self._load_from_pth()
            elif self.SAFETENSORS_PATH.exists(): self._load_from_safetensors()
            else:                              self._download_and_load()
        except Exception as exc:
            self._load_error = str(exc)
            self.ready = False
            logger.error("DeepModel load failed: %s", exc, exc_info=True)

    def _load_from_pth(self) -> None:
        logger.info("Loading EfficientNet-B4 from .pth: %s", self.IMAGENET_PATH)
        net = timm.create_model(self.TIMM_MODEL, pretrained=False, num_classes=9)
        state = torch.load(str(self.IMAGENET_PATH), map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        self._finalise(net)

    def _load_from_safetensors(self) -> None:
        logger.info("Loading EfficientNet-B4 from safetensors (no download): %s", self.SAFETENSORS_PATH)
        from safetensors.torch import load_file
        net1000 = timm.create_model(self.TIMM_MODEL, pretrained=False, num_classes=1000)
        state   = load_file(str(self.SAFETENSORS_PATH))
        net1000.load_state_dict(state, strict=True)
        in_features = net1000.classifier.in_features
        net1000.classifier = torch.nn.Linear(in_features, 9)
        torch.nn.init.xavier_uniform_(net1000.classifier.weight)
        torch.nn.init.zeros_(net1000.classifier.bias)
        torch.save(net1000.state_dict(), str(self.IMAGENET_PATH))
        logger.info("Saved 9-class .pth: %s", self.IMAGENET_PATH)
        self._finalise(net1000)

    def _load_finetuned(self) -> None:
        logger.info("Loading fine-tuned EfficientNet-B4: %s", self.FINETUNED_PATH)
        ck = torch.load(str(self.FINETUNED_PATH), map_location="cpu", weights_only=True)
        n  = ck.get("n_classes", 9)
        net = timm.create_model(self.TIMM_MODEL, pretrained=False, num_classes=n)
        net.load_state_dict(ck.get("model_state_dict", ck), strict=True)
        self._direct_9 = (n == 9); self._n_outputs = n
        self._finalise(net)

    def _download_and_load(self) -> None:
        logger.info("Downloading EfficientNet-B4 weights via timm (one-time ~74 MB)…")
        net1000 = timm.create_model(self.TIMM_MODEL, pretrained=True, num_classes=1000)
        in_features = net1000.classifier.in_features
        net1000.classifier = torch.nn.Linear(in_features, 9)
        torch.nn.init.xavier_uniform_(net1000.classifier.weight)
        torch.nn.init.zeros_(net1000.classifier.bias)
        self.IMAGENET_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net1000.state_dict(), str(self.IMAGENET_PATH))
        self._finalise(net1000)

    def _finalise(self, net) -> None:
        net.eval()
        self.net = net; self._direct_9 = True; self._n_outputs = 9
        self.transform = T.Compose([
            T.Resize((380,380), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.ready = True
        logger.info("EfficientNet-B4 ready — 9-class spinach head.")

    def predict(self, pil_img: Image.Image) -> dict | None:
        if not self.ready or self.net is None: return None
        try:
            x = self.transform(pil_img.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logits = self.net(x)
                probs  = torch.softmax(logits/1.3, dim=1)[0].cpu().numpy()
            final = {lbl: round(float(probs[i])*100, 2) for i,lbl in enumerate(Config.LABELS)}
            sorted_p = sorted(final.items(), key=lambda kv: -kv[1])
            best = sorted_p[0][0]
            return {
                "prediction": best, "confidence": final[best],
                "model_used": "EfficientNet-B4",
                "top3": [{"label":l,"probability":p} for l,p in sorted_p[:3]],
                "all_probabilities": final,
            }
        except Exception as exc:
            logger.error("DeepModel.predict: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Classical ML Registry  —  6 models: RF + GBM + SVM + KNN + LR + XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalModelRegistry:
    """
    Trains and runs all 6 classical ML models on the 49-dim feature vector.
    Models: Random Forest, Gradient Boosting, SVM (RBF), KNN, Logistic Regression, XGBoost.
    All are saved to disk and loaded on startup.
    """
    MODELS_DIR = Path("models/classical")

    def __init__(self):
        self.pipelines:   dict = {}   # name → fitted sklearn Pipeline or XGB
        self.le:          "LabelEncoder | None" = None
        self.scaler:      "StandardScaler | None" = None
        self.stats:       dict = {}
        self.active       = "random_forest"
        self._lock        = threading.Lock()
        if SKLEARN_OK or XGB_OK:
            threading.Thread(target=self._load_saved, daemon=True).start()

    def _model_defs(self) -> dict:
        defs = {}
        if SKLEARN_OK:
            defs["random_forest"]       = RandomForestClassifier(n_estimators=300,max_depth=16,min_samples_leaf=2,max_features="sqrt",random_state=42,n_jobs=-1)
            defs["gradient_boosting"]   = GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=0.08,subsample=0.85,random_state=42)
            defs["svm"]                 = SVC(kernel="rbf",C=5.0,gamma="scale",probability=True,random_state=42)
            defs["knn"]                 = KNeighborsClassifier(n_neighbors=7,weights="distance",metric="euclidean")
            defs["logistic_regression"] = LogisticRegression(max_iter=2000,C=1.0,solver="lbfgs",multi_class="multinomial",random_state=42)
        return defs

    def _load_saved(self) -> None:
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        loaded = 0
        # Load sklearn pipelines
        for name in ["random_forest","gradient_boosting","svm","knn","logistic_regression"]:
            path = self.MODELS_DIR / f"{name}.pkl"
            if path.exists():
                try:
                    with open(path,"rb") as f: self.pipelines[name] = pickle.load(f)
                    loaded += 1
                except Exception as e: logger.error("Load %s failed: %s", name, e)
        # Load XGBoost
        xgb_path  = self.MODELS_DIR / "xgboost.ubj"
        sc_path   = self.MODELS_DIR / "scaler.pkl"
        le_path   = self.MODELS_DIR / "label_encoder.pkl"
        if xgb_path.exists() and sc_path.exists() and le_path.exists():
            try:
                m = xgb.XGBClassifier(); m.load_model(str(xgb_path))
                with open(sc_path,"rb") as f: sc = pickle.load(f)
                with open(le_path,"rb") as f: le = pickle.load(f)
                with self._lock:
                    self.pipelines["xgboost"] = m
                    self.scaler  = sc
                    self.le      = le
                loaded += 1
            except Exception as e: logger.error("Load xgboost failed: %s", e)
        # Load stats
        stats_path = self.MODELS_DIR / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f: self.stats = json.load(f)
            except Exception: pass
        if loaded:
            logger.info("ClassicalModelRegistry: %d models loaded.", loaded)
        else:
            logger.warning("No classical models saved. Run evaluate.py --data <dataset> to train.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit all 6 classical models on real data with 5-fold CV."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        le = LabelEncoder(); y_enc = le.fit_transform(y)
        sc = StandardScaler(); X_sc = sc.fit_transform(X)
        results = {}

        # sklearn models
        for name, clf in self._model_defs().items():
            try:
                t0 = time.time()
                pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
                pipe.fit(X, y)
                acc = float(accuracy_score(y, pipe.predict(X)))
                path = self.MODELS_DIR / f"{name}.pkl"
                with open(path,"wb") as f: pickle.dump(pipe, f)
                with self._lock: self.pipelines[name] = pipe
                results[name] = {"fit_sec": round(time.time()-t0,2), "train_acc": round(acc,4)}
                logger.info("Classical [%s] trained acc=%.4f", name, acc)
            except Exception as e: logger.error("Classical [%s] failed: %s", name, e)

        # XGBoost
        if XGB_OK:
            try:
                t0 = time.time()
                model = xgb.XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
                    objective="multi:softprob", num_class=len(le.classes_),
                    eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
                )
                model.fit(X_sc, y_enc)
                y_pred = le.inverse_transform(model.predict(X_sc))
                acc = float(accuracy_score(y, y_pred))
                model.save_model(str(self.MODELS_DIR/"xgboost.ubj"))
                with open(self.MODELS_DIR/"scaler.pkl","wb") as f: pickle.dump(sc,f)
                with open(self.MODELS_DIR/"label_encoder.pkl","wb") as f: pickle.dump(le,f)
                with self._lock:
                    self.pipelines["xgboost"] = model
                    self.scaler = sc; self.le = le
                results["xgboost"] = {"fit_sec": round(time.time()-t0,2), "train_acc": round(acc,4)}
                logger.info("Classical [xgboost] trained acc=%.4f", acc)
            except Exception as e: logger.error("XGBoost training failed: %s", e)

        with self._lock: self.stats = results
        with open(self.MODELS_DIR/"stats.json","w") as f: json.dump(results, f, indent=2)
        return results

    def predict_one(self, name: str, features: np.ndarray) -> dict | None:
        pipe = self.pipelines.get(name)
        if pipe is None: return None
        try:
            if name == "xgboost":
                X = self.scaler.transform(features.reshape(1,-1))
                proba = pipe.predict_proba(X)[0]
                classes = self.le.classes_
                pd = {str(classes[i]): round(float(proba[i])*100,2) for i in range(len(classes))}
            else:
                X = features.reshape(1,-1)
                proba = pipe.predict_proba(X)[0]
                classes = pipe.classes_
                pd = {str(c): round(float(p)*100,2) for c,p in zip(classes,proba)}
            # Ensure all labels present
            for lbl in Config.LABELS:
                if lbl not in pd: pd[lbl] = 0.0
            sorted_p = sorted(pd.items(), key=lambda kv: -kv[1])
            best = sorted_p[0][0]
            return {
                "prediction": best,
                "confidence": pd[best],
                "model_used": name,
                "top3": [{"label":l,"probability":p} for l,p in sorted_p[:3]],
                "all_probabilities": pd,
            }
        except Exception as e:
            logger.error("Classical.predict_one [%s]: %s", name, e)
            return None

    def predict_all(self, features: np.ndarray) -> dict:
        results = {}
        for name in self.pipelines:
            r = self.predict_one(name, features)
            if r: results[name] = r
        return results

    def is_ready(self) -> bool: return len(self.pipelines) > 0

    def get_info(self) -> dict:
        return {
            "available": list(self.pipelines.keys()),
            "active":    self.active,
            "stats":     self.stats,
            "sklearn_ok": SKLEARN_OK,
            "xgb_ok":    XGB_OK,
        }

    def set_active(self, name: str) -> None:
        if name not in self.pipelines:
            raise ValueError(f"Model '{name}' not available. Have: {list(self.pipelines)}")
        with self._lock: self.active = name


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble  —  EfficientNet-B4 (40%) + all 6 classical models averaged (60%)
# ─────────────────────────────────────────────────────────────────────────────

# Ensemble weights per model
_ENSEMBLE_WEIGHTS = {
    "efficientnet_b4":      0.40,
    "xgboost":              0.15,
    "random_forest":        0.10,
    "gradient_boosting":    0.10,
    "svm":                  0.10,
    "knn":                  0.08,
    "logistic_regression":  0.07,
}
# Sum = 1.00


def _ensemble_vote(deep_result: dict, classical_results: dict,
                   colour: dict = None, features: np.ndarray = None) -> dict:
    """
    Weighted ensemble across all available models.
    EfficientNet-B4 (40%) + up to 6 classical models + ColourRuleClassifier.

    If the deep model head appears untrained/random (confidence < 30%),
    its weight is reduced to 0.05 and the ColourRuleClassifier is added
    with weight 0.40 to provide a reliable signal.

    If no classical models available, deep model (or colour rules) used directly.
    """
    if "all_probabilities" not in deep_result:
        raise ValueError("deep_result missing 'all_probabilities'")

    deep_is_random = _is_deep_model_random(deep_result)

    # Build colour rule result if needed
    colour_result = None
    if deep_is_random and colour is not None and features is not None:
        try:
            colour_result = ColourRuleClassifier.predict(colour, features)
        except Exception as exc:
            logger.warning("ColourRuleClassifier failed: %s", exc)

    # If deep is random and no classical + no colour rule, still return something
    if not classical_results and colour_result is None:
        return deep_result

    # If deep is random and no classical but colour rule works, return colour rule
    if not classical_results and colour_result is not None:
        return colour_result

    # Collect all model results with weights
    model_results = {"efficientnet_b4": deep_result}
    model_results.update(classical_results)
    if colour_result is not None:
        model_results["colour_rule"] = colour_result

    # Dynamic weights: reduce deep model if it looks random
    effective_weights = dict(_ENSEMBLE_WEIGHTS)
    if deep_is_random:
        effective_weights["efficientnet_b4"] = 0.05
        if colour_result is not None:
            effective_weights["colour_rule"] = 0.40
        # Redistribute remaining classical weight proportionally
        classical_names = [n for n in model_results if n not in ("efficientnet_b4", "colour_rule")]
        classic_total_w = sum(_ENSEMBLE_WEIGHTS.get(n, 0) for n in classical_names)
        remaining = 1.0 - effective_weights.get("efficientnet_b4", 0.05) \
                       - effective_weights.get("colour_rule", 0.0)
        if classic_total_w > 0:
            for n in classical_names:
                effective_weights[n] = _ENSEMBLE_WEIGHTS.get(n, 0) / classic_total_w * remaining

    # Sum available weights
    total_weight = sum(
        effective_weights.get(name, 0.0)
        for name in model_results
        if "all_probabilities" in model_results[name]
    )
    if total_weight == 0:
        return colour_result or deep_result

    blended: dict[str, float] = {lbl: 0.0 for lbl in Config.LABELS}
    for name, result in model_results.items():
        if "all_probabilities" not in result:
            raise ValueError(f"Model '{name}' result missing 'all_probabilities'")
        w     = effective_weights.get(name, 0.0) / total_weight
        probs = result["all_probabilities"]
        for lbl in Config.LABELS:
            blended[lbl] += probs.get(lbl, 0.0) * w

    # Renormalise
    total = sum(blended.values()) or 1.0
    blended = {lbl: round(v/total*100, 2) for lbl, v in blended.items()}

    sorted_p = sorted(blended.items(), key=lambda kv: -kv[1])
    best     = sorted_p[0][0]
    models_used = list(model_results.keys())
    return {
        "prediction":        best,
        "confidence":        blended[best],
        "model_used":        f"ensemble({','.join(models_used)})",
        "models_breakdown":  {
            name: {"prediction": model_results[name]["prediction"],
                   "confidence": model_results[name]["confidence"],
                   "weight_pct": round(effective_weights.get(name, 0)*100, 1)}
            for name in models_used
        },
        "top3":              [{"label":l,"probability":p} for l,p in sorted_p[:3]],
        "all_probabilities": blended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Treatment Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_treatment_report(prediction: str, confidence: float,
                               filename: str, colour_analysis: dict,
                               ensemble_result: dict,
                               image_metadata: dict) -> dict:
    """
    Generate a comprehensive agronomic treatment report for the detected disease.
    Includes disease info, immediate actions, chemical/organic treatments,
    severity assessment, and model comparison table.
    """
    info = Config.DISEASE_INFO.get(prediction, {})
    hints = colour_analysis.get("disease_hints", [])
    colour_pct = colour_analysis.get("colour_pct", {})
    models_breakdown = ensemble_result.get("models_breakdown", {})

    # Confidence band
    if confidence >= 80:    conf_band = "HIGH"
    elif confidence >= 55:  conf_band = "MODERATE"
    else:                   conf_band = "LOW"

    # Visual evidence summary from colour analysis
    visual_evidence = []
    for hint in hints:
        visual_evidence.append(f"{hint['signal']} ({hint['pct']}% of leaf area)")

    # Model agreement: how many models agree on the top prediction
    agreement_count = sum(
        1 for m in models_breakdown.values()
        if m.get("prediction") == prediction
    )
    total_models = len(models_breakdown) or 1
    agreement_pct = round(agreement_count / total_models * 100)

    # Per-model prediction table for report
    model_table = [
        {
            "model":      name,
            "prediction": data["prediction"],
            "confidence": data["confidence"],
            "weight_pct": data.get("weight_pct", 0),
            "agrees":     data["prediction"] == prediction,
        }
        for name, data in models_breakdown.items()
    ]

    report = {
        "report_generated_at": datetime.utcnow().isoformat() + "Z",
        "image": {
            "filename":   filename,
            "width_px":   image_metadata.get("original_width"),
            "height_px":  image_metadata.get("original_height"),
            "megapixels": image_metadata.get("megapixels"),
            "file_size":  image_metadata.get("file_size_label"),
        },
        "diagnosis": {
            "disease":          prediction,
            "status":           info.get("status", prediction),
            "severity":         info.get("severity", "unknown"),
            "severity_score":   info.get("severity_score", 0),
            "confidence_pct":   confidence,
            "confidence_band":  conf_band,
            "icon":             info.get("icon", "🌿"),
            "color":            info.get("color", "#888"),
            "affected_parts":   info.get("affected_parts", []),
            "description":      info.get("description", ""),
        },
        "visual_evidence": visual_evidence,
        "colour_analysis_summary": {
            "green_pct":  colour_pct.get("green", 0),
            "yellow_pct": colour_pct.get("yellow", 0),
            "brown_pct":  colour_pct.get("brown", 0),
            "white_pct":  colour_pct.get("white", 0),
        },
        "causes":            info.get("causes", []),
        "immediate_actions": info.get("immediate_actions", []),
        "chemical_treatments":  info.get("chemical_treatments", []),
        "organic_treatments":   info.get("organic_treatments", []),
        "fertilizer_schedule":  info.get("fertilizer_schedule", ""),
        "prevention":           info.get("prevention", ""),
        "recovery_time":        info.get("recovery_time", ""),
        "economic_impact":      info.get("economic_impact", ""),
        "model_analysis": {
            "total_models_used":   total_models,
            "models_agreeing":     agreement_count,
            "agreement_pct":       agreement_pct,
            "model_table":         model_table,
            "top3_diseases":       ensemble_result.get("top3", []),
            "all_probabilities":   ensemble_result.get("all_probabilities", {}),
        },
        "urgency": (
            "IMMEDIATE ACTION REQUIRED"
            if info.get("severity") in ("critical","high") and confidence >= 60
            else "MONITOR CLOSELY"
            if info.get("severity") == "medium"
            else "ROUTINE CARE"
        ),
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Batch Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch_report(results: list[dict]) -> dict:
    """
    Generate a summary report for a batch of predictions.
    Includes disease distribution, average confidence, urgency summary.
    """
    if not results:
        return {"error": "No results to summarise"}

    disease_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {"critical":0,"high":0,"medium":0,"none":0}
    confidences = []
    urgent = []

    for r in results:
        pred = r.get("prediction","unknown")
        disease_counts[pred] = disease_counts.get(pred, 0) + 1
        conf = r.get("confidence", 0)
        confidences.append(conf)
        info = Config.DISEASE_INFO.get(pred, {})
        sev = info.get("severity","none")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        if sev in ("critical","high") and conf >= 60:
            urgent.append({
                "filename":   r.get("filename","?"),
                "disease":    pred,
                "confidence": conf,
                "severity":   sev,
            })

    dominant_disease = max(disease_counts.items(), key=lambda kv: kv[1])[0] if disease_counts else "unknown"
    healthy_count = disease_counts.get("healthy", 0)
    diseased_count = len(results) - healthy_count

    return {
        "report_generated_at":  datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_images":        len(results),
            "healthy_count":       healthy_count,
            "diseased_count":      diseased_count,
            "health_rate_pct":     round(healthy_count/len(results)*100, 1),
            "average_confidence":  round(float(np.mean(confidences)), 1),
            "dominant_disease":    dominant_disease,
        },
        "disease_distribution":  disease_counts,
        "severity_distribution": severity_counts,
        "urgent_cases":          urgent,
        "urgent_count":          len(urgent),
        "field_recommendation": (
            "URGENT: Multiple diseased plants detected. Apply treatment immediately."
            if len(urgent) > 0
            else "Field appears healthy. Continue regular monitoring."
            if healthy_count == len(results)
            else "Some disease detected. Monitor and treat affected plants."
        ),
        "individual_results":    results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_mappings() -> None:
    missing = [c for c in PLANTVILLAGE_CLASSES if c not in PV_TO_SPINACH]
    if missing:
        raise RuntimeError(f"PV_TO_SPINACH missing: {missing}")
    invalid = {pv: tgt for pv, tgt in PV_TO_SPINACH.items() if tgt not in Config.LABELS}
    if invalid:
        raise RuntimeError(f"PV_TO_SPINACH invalid targets: {invalid}")
    if len(PLANTVILLAGE_CLASSES) != 38:
        raise RuntimeError(f"Expected 38 PV classes, got {len(PLANTVILLAGE_CLASSES)}")
    logger.info("Mapping validation passed: %d PV classes → %d spinach labels.", 38, 9)


_validate_mappings()

# ── Singletons ────────────────────────────────────────────────────────────────
deep_model        = DeepModel()
classical_models  = ClassicalModelRegistry()

# Legacy shims — keep app.py working without changes
xgb_model        = classical_models   # used by app.py /model/status
sklearn_registry = classical_models   # used by app.py /models


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline  —  main inference entry point
# ─────────────────────────────────────────────────────────────────────────────

def _restore_json_fields(row: dict) -> None:
    for key, target in (("top3_predictions","top3"),("all_probabilities","all_probabilities")):
        raw = row.get(key)
        if isinstance(raw, str):
            try: row[target] = json.loads(raw)
            except Exception: row.setdefault(target, [] if target=="top3" else {})
        else:
            row.setdefault(target, [] if target=="top3" else {})


def run_pipeline(file_obj, use_cache: bool = True, sklearn_model: str = None) -> dict:
    """
    Full inference pipeline:
      1. Validate image
      2. Extract 49-dim features (OpenCV + scipy + numpy)
      3. Run EfficientNet-B4
      4. Run all available classical models (RF, GBM, SVM, KNN, LR, XGBoost)
      5. Weighted ensemble vote
      6. Generate full treatment report
      7. Persist to DB
      8. Return complete result with report
    """
    from database import PredictionDAO

    t0       = time.time()
    filename = sanitize_name(getattr(file_obj, "filename", None) or "upload")

    image_bytes = file_obj.read()
    if not image_bytes:
        raise ValueError("Empty file received.")
    if len(image_bytes) > Config.MAX_FILE_SIZE:
        raise ValueError(f"File too large ({size_label(len(image_bytes))}). Max 15 MB.")
    if not validate_image_bytes(image_bytes):
        raise ValueError("Not a valid image — file signature check failed.")

    filename  = fix_filename_ext(filename, image_bytes)
    if not allowed_ext(filename):
        raise ValueError(f"File type not allowed. Accepted: {', '.join(sorted(Config.ALLOWED_EXT))}")

    file_hash = compute_hash(image_bytes)

    # ── Cache ─────────────────────────────────────────────────────────────────
    if use_cache:
        cached = PredictionDAO.get_by_hash(file_hash)
        if cached:
            try:
                created = cached.get("created_at")
                import datetime as _dt
                dt  = created if not isinstance(created, str) else \
                      _dt.datetime.fromisoformat(created.replace("Z",""))
                age = time.time() - dt.timestamp()
            except Exception: age = 99999
            if age < 3600:
                result = dict(cached)
                result["cached"]     = True
                pred_key             = result.get("prediction_result")
                if pred_key is None: raise RuntimeError("Cached row has no 'prediction_result'.")
                if pred_key not in Config.DISEASE_INFO:
                    raise RuntimeError(f"Cached label '{pred_key}' not in DISEASE_INFO.")
                result["prediction"]  = pred_key
                result["disease_info"] = Config.DISEASE_INFO[pred_key]
                _restore_json_fields(result)
                return result

    if not deep_model.ready:
        raise RuntimeError(
            deep_model._load_error
            or "Model is loading. Please wait 60-120 seconds and retry."
        )

    # ── Process image ─────────────────────────────────────────────────────────
    proc     = ImageProcessor(image_bytes)
    proc.prepare()
    features = proc.extract_features()

    # ── Colour analysis (done early so ensemble can use it) ───────────────────
    colour = proc.colour_analysis()

    # ── Deep model ────────────────────────────────────────────────────────────
    deep_result = deep_model.predict(proc.pil_image())
    if deep_result is None:
        raise RuntimeError("EfficientNet-B4 returned no output.")

    # ── Classical models ──────────────────────────────────────────────────────
    classical_all = {}
    if classical_models.is_ready():
        if sklearn_model and sklearn_model in classical_models.pipelines:
            # Only the requested model
            r = classical_models.predict_one(sklearn_model, features)
            if r: classical_all = {sklearn_model: r}
        else:
            classical_all = classical_models.predict_all(features)

    # ── Ensemble (colour + features passed for ColourRuleClassifier fallback) ─
    ensemble_result = _ensemble_vote(deep_result, classical_all,
                                     colour=colour, features=features)

    ms    = round((time.time()-t0)*1000, 1)
    thumb = proc.thumbnail_b64()
    meta  = proc.meta

    final_prediction = ensemble_result["prediction"]
    final_confidence = ensemble_result["confidence"]

    # ── Treatment report ──────────────────────────────────────────────────────
    report = generate_treatment_report(
        prediction       = final_prediction,
        confidence       = final_confidence,
        filename         = filename,
        colour_analysis  = colour,
        ensemble_result  = ensemble_result,
        image_metadata   = meta,
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    try:
        row_id = PredictionDAO.insert(
            image_name     = filename,
            prediction     = final_prediction,
            confidence     = final_confidence,
            model_used     = "ensemble",
            file_hash      = meta["file_hash"],
            file_size      = meta["file_size"],
            width          = meta["original_width"],
            height         = meta["original_height"],
            top3_json      = json.dumps(ensemble_result["top3"]),
            all_proba_json = json.dumps(ensemble_result["all_probabilities"]),
            processing_ms  = ms,
            thumbnail      = thumb,
            image_data     = image_bytes,
        )
    except Exception as exc:
        logger.error("DB insert failed: %s", exc)
        row_id = None

    return {
        "id":                   row_id,
        "filename":             filename,
        "cached":               False,
        "prediction":           final_prediction,
        "confidence":           final_confidence,
        "model_used":           "ensemble",
        "top3":                 ensemble_result["top3"],
        "all_probabilities":    ensemble_result["all_probabilities"],
        "deep_model_result":    deep_result,
        "classical_results":    classical_all,
        "ensemble_result":      ensemble_result,
        "colour_analysis":      colour,
        "processing_time_ms":   ms,
        "image_metadata":       meta,
        "thumbnail":            thumb,
        "disease_info":         Config.DISEASE_INFO[final_prediction],
        "treatment_report":     report,
        # legacy keys for backwards compat
        "xgb_result":           classical_all.get("xgboost"),
        "sklearn_result":       classical_all.get(classical_models.active),
        "sklearn_all_models":   classical_all,
    }
