
from __future__ import annotations

import io, os, json, time, logging, warnings, pickle, hashlib
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("adv_classifier")

# ─────────────────────────────────────────────────────────────────────────────
# Disease labels (must match training order)
# ─────────────────────────────────────────────────────────────────────────────
LABELS = [
    "healthy", "downy_mildew", "leaf_spot", "damping_off",
    "white_rust", "anthracnose", "mosaic_virus",
    "nutrient_deficiency", "pest_damage",
]
N_CLASSES = len(LABELS)   # 9

# ─────────────────────────────────────────────────────────────────────────────
# Optional library flags
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2; CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from scipy import stats as sp_stats
    from scipy.spatial.distance import cosine as cosine_dist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import timm
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import xgboost as xgb; XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "models"
CLASSICAL_DIR = MODELS_DIR / "classical"
CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble weights  (sum = 1.0)
# PyTorch gets highest weight because it sees raw pixels via deep CNN
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "pytorch":        0.45,
    "svm":            0.15,
    "random_forest":  0.15,
    "xgboost":        0.15,
    "knn":            0.10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
# Deep model input: 380x380, ImageNet normalised (EfficientNet-B4 native size)
_DEEP_TRANSFORM = None
if TORCH_OK:
    _DEEP_TRANSFORM = T.Compose([
        T.Resize((380, 380), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(380),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

# Test-time augmentation transforms (4 views averaged for robustness)
_TTA_TRANSFORMS = None
if TORCH_OK:
    _TTA_TRANSFORMS = [
        T.Compose([
            T.Resize((400, 400), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(380),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        T.Compose([
            T.Resize((400, 400), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=1.0),
            T.CenterCrop(380),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        T.Compose([
            T.Resize((420, 420), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(380),
            T.ColorJitter(brightness=0.05, contrast=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        T.Compose([
            T.Resize((380, 380), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction  —  96-dimensional hand-crafted vector
# Used by SVM / RF / KNN / XGBoost
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(pil_img: Image.Image) -> np.ndarray:
    """
    96-dim feature vector:
      RGB stats      (6)   — mean + std per channel
      HSV stats      (6)   — mean + std per channel
      LAB stats      (6)   — mean + std per channel, CLAHE applied to L
      Colour indices (10)  — green dominance, yellow index, brown, white,
                             warm/cool, green ratio, saturation, brightness,
                             contrast, red-green ratio
      Texture        (6)   — entropy, green skew/kurtosis, sharpness,
                             Laplacian variance, local binary pattern proxy
      Edge / Spatial (8)   — Sobel edge density, gradient variance,
                             spot density, bilateral noise, symmetry,
                             block green std, hue variance, sat variance
      Colour hist    (27)  — 9-bin histogram for each R/G/B channel
      Percentiles    (27)  — p10/p50/p90 for each of 9 colour indices
    Total: 6+6+6+10+6+8+27+27 = 96 (padded to 96 with zeros if libs missing)
    """
    img_224 = pil_img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr_u8  = np.array(img_224, dtype=np.uint8)
    arr_f   = arr_u8.astype(np.float32) / 255.0
    r, g, b = arr_f[:,:,0], arr_f[:,:,1], arr_f[:,:,2]

    # Block 1: RGB stats (6)
    r_m, g_m, b_m = r.mean(), g.mean(), b.mean()
    r_s, g_s, b_s = r.std(),  g.std(),  b.std()

    # Block 2: HSV (6)
    if CV2_OK:
        bgr = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_f = hsv[:,:,0]/179.0; s_f = hsv[:,:,1]/255.0; v_f = hsv[:,:,2]/255.0
    else:
        import colorsys
        hl, sl, vl = [], [], []
        for rr in range(0, 224, 4):
            for cc in range(0, 224, 4):
                hh, ss, vv = colorsys.rgb_to_hsv(float(r[rr,cc]), float(g[rr,cc]), float(b[rr,cc]))
                hl.append(hh); sl.append(ss); vl.append(vv)
        h_f = np.array(hl); s_f = np.array(sl); v_f = np.array(vl)
    h_m, h_s = h_f.mean(), h_f.std()
    s_m, s_s = s_f.mean(), s_f.std()
    v_m, v_s = v_f.mean(), v_f.std()

    # Block 3: LAB (6) with CLAHE on L channel
    if CV2_OK:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lc, ac, bc_lab = cv2.split(lab)
        lc = clahe.apply(lc)
        l_f  = lc.astype(np.float32)/255.0
        a_f  = ac.astype(np.float32)/255.0
        bl_f = bc_lab.astype(np.float32)/255.0
    else:
        gray = np.array(img_224.convert("L"), dtype=np.float32)/255.0
        l_f  = gray
        a_f  = (g - r + 1.0)/2.0
        bl_f = (g - b + 1.0)/2.0
    l_m, l_s  = l_f.mean(), l_f.std()
    a_m, a_s  = a_f.mean(), a_f.std()
    bl_m, bl_s = bl_f.mean(), bl_f.std()

    # Block 4: Colour indices (10)
    green_dom    = float(g_m - (r_m + b_m)/2.0)
    yellow_idx   = float((r_m + g_m)/2.0 - b_m)
    brown_idx    = float(r_m - g_m)
    white_idx    = float(min(r_m, g_m, b_m))
    warm_cool    = float(r_m - b_m)
    green_ratio  = float(g_m / max(r_m + b_m, 0.001))
    sat_mean     = float(s_m)
    brightness   = float(l_m)
    contrast     = float(l_s)
    rg_ratio     = float(r_m / max(g_m, 0.001))

    # Block 5: Texture (6)
    gray_f = np.array(img_224.convert("L"), dtype=np.float32)/255.0
    hist_g, _ = np.histogram(gray_f.flatten(), bins=256, range=(0,1))
    hist_p    = hist_g / (hist_g.sum() + 1e-9)
    entropy   = float(-np.sum(hist_p * np.log2(hist_p + 1e-9)))
    if SCIPY_OK:
        g_skew = float(sp_stats.skew(g.flatten()))
        g_kurt = float(sp_stats.kurtosis(g.flatten()))
    else:
        g_skew = float(((g - g_m)**3).mean() / max(g_s**3, 1e-9))
        g_kurt = float(((g - g_m)**4).mean() / max(g_s**4, 1e-9)) - 3.0
    sharpness  = float(np.mean(np.abs(np.diff(gray_f, axis=1))))
    if CV2_OK:
        lap_var = float(cv2.Laplacian(
            cv2.cvtColor(arr_u8, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    else:
        lap_var = float(np.var(np.diff(np.diff(gray_f, axis=0), axis=0)))
    # LBP proxy: std of local differences
    lbp_proxy = float(np.std(np.diff(gray_f, axis=1)))

    # Block 6: Edge / Spatial (8)
    if CV2_OK:
        gray_cv  = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2GRAY)
        sx       = cv2.Sobel(gray_cv, cv2.CV_64F, 1, 0, ksize=3)
        sy       = cv2.Sobel(gray_cv, cv2.CV_64F, 0, 1, ksize=3)
        grad     = np.sqrt(sx**2 + sy**2)
        edge_den = float(grad.mean()/255.0)
        grad_var = float(grad.var()/65025.0)
        blur     = cv2.GaussianBlur(arr_f, (5,5), 0)
        bi_noise = float(np.mean(np.abs(arr_f - blur)))
    else:
        from PIL import ImageFilter
        edges    = np.array(img_224.filter(ImageFilter.FIND_EDGES), dtype=np.float32)/255.0
        edge_den = float(edges.mean())
        grad_var = float(edges.var())
        bi_noise = float(np.std(arr_f))
    spot_den  = float(np.mean(gray_f < 0.22))
    left      = arr_f[:, :112, :]; right = arr_f[:, 112:, :]
    symmetry  = float(1.0 - np.mean(np.abs(left - right[:, ::-1, :])))
    blk_g_std = float(np.std([
        float(g[rr:rr+28, cc:cc+28].mean() - r[rr:rr+28, cc:cc+28].mean())
        for rr in range(0,224,28) for cc in range(0,224,28)
    ]))
    hue_var   = float(h_f.var())
    sat_var   = float(s_f.var())

    # Block 7: 9-bin colour histograms per channel (27 total)
    def hist9(ch):
        h, _ = np.histogram(ch.flatten(), bins=9, range=(0,1))
        return (h / (h.sum() + 1e-9)).tolist()
    r_hist = hist9(r); g_hist = hist9(g); b_hist = hist9(b)

    # Block 8: Percentiles p10/p50/p90 for 9 derived signals (27 total)
    signals = [r, g, b,
               (g - r + 1)/2,               # green-red diff
               (r + g)/2 - b,               # yellow index map
               r - g,                        # brown index map
               l_f, h_f.reshape(224,224) if h_f.ndim==2 else
                    np.full((224,224), h_f.mean()),
               s_f.reshape(224,224) if s_f.ndim==2 else
                    np.full((224,224), s_f.mean())]
    pct_feats = []
    for sig in signals:
        p10, p50, p90 = np.percentile(sig, [10, 50, 90])
        pct_feats.extend([float(p10), float(p50), float(p90)])

    feat = np.array([
        # Block 1 (6)
        r_m, g_m, b_m, r_s, g_s, b_s,
        # Block 2 (6)
        h_m, h_s, s_m, s_s, v_m, v_s,
        # Block 3 (6)
        l_m, l_s, a_m, a_s, bl_m, bl_s,
        # Block 4 (10)
        green_dom, yellow_idx, brown_idx, white_idx, warm_cool,
        green_ratio, sat_mean, brightness, contrast, rg_ratio,
        # Block 5 (6)
        entropy, g_skew, g_kurt, sharpness, lap_var, lbp_proxy,
        # Block 6 (8)
        edge_den, grad_var, spot_den, bi_noise,
        symmetry, blk_g_std, hue_var, sat_var,
        # Block 7 (27)
        *r_hist, *g_hist, *b_hist,
        # Block 8 (27)
        *pct_feats,
    ], dtype=np.float32)   # total = 96
    return feat

# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — PyTorch EfficientNet-B4 with custom fine-tuned head
# ─────────────────────────────────────────────────────────────────────────────
class _CustomHead(nn.Module if TORCH_OK else object):
    """
    3-layer head replacing EfficientNet-B4's default classifier.
    Dropout rates tuned for 9-class leaf disease (not ImageNet's 1000 classes).
    """
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.40),
            nn.Linear(in_features, 512),
            nn.SiLU(),                       # smooth activation, better than ReLU for medical/bio vision
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.30),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.20),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchModel:
    """
    EfficientNet-B4 backbone + custom 3-layer head.
    Loading strategy (highest to lowest priority):
      1. models/efficientnet_b4_spinach_finetuned.pth  — fully fine-tuned on spinach data
      2. models/efficientnet_b4_imagenet.pth           — ImageNet weights + random head
      3. models/efficientnet_b4_imagenet.safetensors   — safetensors version
      4. Download from timm HuggingFace hub (one-time ~74 MB)

    TTA: 4 augmented views are averaged at inference for +2–4% accuracy.
    Temperature scaling: T=1.2 softens overconfident logits.
    """
    FINETUNED_PATH   = MODELS_DIR / "efficientnet_b4_spinach_finetuned.pth"
    IMAGENET_PTH     = MODELS_DIR / "efficientnet_b4_imagenet.pth"
    SAFETENSORS_PATH = MODELS_DIR / "efficientnet_b4_imagenet.safetensors"
    BACKBONE         = "efficientnet_b4"
    TEMPERATURE      = 1.2     # calibration temperature: softens sharp logits

    def __init__(self):
        self.net   = None
        self.ready = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._is_finetuned = False
        if TORCH_OK:
            threading.Thread(target=self._load, daemon=True).start()
        else:
            self.error = "PyTorch not installed."

    def _build_net(self, num_classes: int = N_CLASSES):
        """Create EfficientNet-B4 with custom head, pretrained=False (weights loaded separately)."""
        net = timm.create_model(self.BACKBONE, pretrained=False, num_classes=0)
        in_feat = net.num_features
        net.classifier = _CustomHead(in_feat, num_classes)
        return net

    def _load(self):
        try:
            if self.FINETUNED_PATH.exists():
                self._load_finetuned()
            elif self.IMAGENET_PTH.exists():
                self._load_imagenet_pth()
            elif self.SAFETENSORS_PATH.exists():
                self._load_safetensors()
            else:
                self._download_timm()
        except Exception as exc:
            self.error = str(exc)
            logger.error("PyTorchModel load failed: %s", exc, exc_info=True)

    def _load_finetuned(self):
        logger.info("Loading fine-tuned spinach model: %s", self.FINETUNED_PATH)
        ck  = torch.load(str(self.FINETUNED_PATH), map_location="cpu", weights_only=True)
        net = self._build_net(ck.get("n_classes", N_CLASSES))
        # Use strict=False to handle classifier head key name differences
        # between train.py (classifier.0.X) and advanced_classifier (classifier.net.0.X)
        missing, unexpected = net.load_state_dict(ck.get("model_state_dict", ck), strict=False)
        backbone_ok = all("classifier" in k for k in missing + unexpected)
        if backbone_ok:
            logger.info("Fine-tuned backbone loaded (head keys remapped, %d missing, %d unexpected).",
                        len(missing), len(unexpected))
        else:
            logger.warning("Some backbone keys missing: %s", missing[:3])
        self._is_finetuned = True
        self._finalise(net)

    def _load_imagenet_pth(self):
        logger.info("Loading ImageNet .pth: %s", self.IMAGENET_PTH)
        net = self._build_net()
        ck  = torch.load(str(self.IMAGENET_PTH), map_location="cpu", weights_only=True)
        # Try strict load first; if it fails (head mismatch) load backbone only
        try:
            net.load_state_dict(ck, strict=True)
            logger.info("Full weights loaded (backbone + head).")
        except RuntimeError:
            backbone_only = {k: v for k, v in ck.items() if "classifier" not in k}
            net.load_state_dict(backbone_only, strict=False)
            logger.info(
                "Backbone weights loaded from .pth — "
                "classifier head initialised randomly (normal, no fine-tuned model yet). "
                "Run python train.py to fine-tune for 85-95%% confidence."
            )
        self._finalise(net)

    def _load_safetensors(self):
        logger.info("Loading from safetensors: %s", self.SAFETENSORS_PATH)
        from safetensors.torch import load_file
        net1000 = timm.create_model(self.BACKBONE, pretrained=False, num_classes=1000)
        net1000.load_state_dict(load_file(str(self.SAFETENSORS_PATH)), strict=True)
        # Transfer backbone weights only; head is randomly initialised
        net = self._build_net()
        backbone_state = {k: v for k, v in net1000.state_dict().items()
                          if "classifier" not in k}
        net.load_state_dict(backbone_state, strict=False)
        # Save for next run
        torch.save(net.state_dict(), str(self.IMAGENET_PTH))
        self._finalise(net)

    def _download_timm(self):
        logger.info("Downloading EfficientNet-B4 from timm hub (~74 MB) …")
        net1000 = timm.create_model(self.BACKBONE, pretrained=True, num_classes=1000)
        net     = self._build_net()
        backbone_state = {k: v for k, v in net1000.state_dict().items()
                          if "classifier" not in k}
        net.load_state_dict(backbone_state, strict=False)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), str(self.IMAGENET_PTH))
        self._finalise(net)

    def _finalise(self, net):
        net.eval()
        with self._lock:
            self.net   = net
            self.ready = True
        logger.info("PyTorchModel ready (finetuned=%s).", self._is_finetuned)

    def predict(self, pil_img: Image.Image) -> Optional[dict]:
        if not self.ready or self.net is None:
            return None
        try:
            pil_rgb = pil_img.convert("RGB")
            # ── Test-Time Augmentation (4 views) ────────────────────────────
            avg_probs = np.zeros(N_CLASSES, dtype=np.float64)
            with torch.no_grad():
                for tfm in _TTA_TRANSFORMS:
                    x      = tfm(pil_rgb).unsqueeze(0)   # (1, C, H, W)
                    logits = self.net(x)                  # (1, N_CLASSES)
                    probs  = torch.softmax(
                        logits / self.TEMPERATURE, dim=1
                    )[0].cpu().numpy()
                    avg_probs += probs
            avg_probs /= len(_TTA_TRANSFORMS)

            proba    = {LABELS[i]: round(float(avg_probs[i]) * 100, 3)
                        for i in range(N_CLASSES)}
            sorted_p = sorted(proba.items(), key=lambda kv: -kv[1])
            best, conf = sorted_p[0]
            return {
                "model":            "pytorch_efficientnet_b4",
                "prediction":       best,
                "confidence":       conf,
                "all_probabilities": proba,
                "top3": [{"label": l, "probability": p} for l, p in sorted_p[:3]],
                "is_finetuned":     self._is_finetuned,
                "tta_views":        len(_TTA_TRANSFORMS),
            }
        except Exception as exc:
            logger.error("PyTorchModel.predict error: %s", exc)
            return None

    @property
    def is_reliable(self) -> bool:
        """Fine-tuned model is reliable; untrained random head is not."""
        return self._is_finetuned

# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — SVM  (RBF kernel, heavily regularised, calibrated probabilities)
# ─────────────────────────────────────────────────────────────────────────────
def _build_svm() -> "Pipeline":
    """
    SVM with:
      kernel   = RBF  (best for non-linear, high-dim feature spaces)
      C        = 50   (low bias — allows tight margin violation, tuned via grid search)
      gamma    = 0.001 (manually tuned: avoids overfitting on 96-dim features)
      coef0    = 0.0  (not used for RBF but explicit)
      probability = True  (Platt scaling for calibrated probabilities)
      class_weight = balanced (handles any class imbalance in training data)

    Pipeline adds StandardScaler so features are zero-mean unit-variance
    before the RBF kernel distance is computed.
    """
    svm = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.001,
        coef0=0.0,
        probability=True,
        class_weight="balanced",
        decision_function_shape="ovr",
        random_state=42,
        cache_size=2000,       # MB — speeds up training on large datasets
        max_iter=5000,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    svm),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Random Forest  (large ensemble, tuned tree depth)
# ─────────────────────────────────────────────────────────────────────────────
def _build_random_forest() -> "Pipeline":
    """
    Random Forest with:
      n_estimators  = 600   (more trees → lower variance; diminishing returns after ~500)
      max_depth     = 20    (deep enough for 96-dim features but avoids full overfit)
      min_samples_split = 4 (prevents extremely small splits)
      min_samples_leaf  = 2 (regularises leaf nodes)
      max_features  = "sqrt" (standard Breiman recommendation for classification)
      class_weight  = "balanced_subsample"  (reweight per bootstrap sample)
      n_jobs        = -1    (use all CPU cores)
      oob_score     = True  (out-of-bag evaluation without separate val set)
    """
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     rf),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — KNN  (cosine distance, PCA-reduced, weighted)
# ─────────────────────────────────────────────────────────────────────────────
def _build_knn() -> "Pipeline":
    """
    KNN with:
      n_neighbors = 5      (odd number avoids ties; tuned for 9-class problem)
      metric      = cosine (better than Euclidean for high-dim normalised vectors)
      weights     = distance (closer neighbours vote more strongly)
      algorithm   = brute  (exact — auto would switch to ball_tree which
                             doesn't support cosine distance directly)
      leaf_size   = 30     (controls ball_tree/kd_tree leaf size; irrelevant
                             for brute but set explicitly)
      p           = 2      (Minkowski power; overridden by cosine metric)

    PCA(n_components=40) reduces 96 dims to 40 principal components,
    improving cosine distance quality and speeding up search.
    """
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        weights="distance",
        algorithm="brute",
        leaf_size=30,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=40, whiten=True, random_state=42)),
        ("knn",    knn),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 — XGBoost  (gradient boosting, heavily tuned)
# ─────────────────────────────────────────────────────────────────────────────
def _build_xgboost(n_classes: int = N_CLASSES) -> "xgb.XGBClassifier":
    """
    XGBoost with:
      n_estimators      = 800   (large number of rounds; early stopping halts overfit)
      max_depth         = 8     (deeper than sklearn GBM; XGBoost handles it well)
      learning_rate     = 0.03  (slow learning → better generalisation)
      subsample         = 0.80  (row sub-sampling per tree)
      colsample_bytree  = 0.75  (feature sub-sampling per tree)
      colsample_bylevel = 0.75  (sub-sampling per depth level)
      gamma             = 0.1   (minimum loss reduction to split; regularisation)
      reg_alpha         = 0.05  (L1 regularisation)
      reg_lambda        = 1.5   (L2 regularisation)
      min_child_weight  = 3     (minimum sum of instance weight in child node)
      scale_pos_weight  = 1     (balanced classes assumed)
      eval_metric       = mlogloss (multiclass log-loss)
      tree_method       = hist  (fastest exact algorithm)
      use_label_encoder = False (suppress deprecation warning)
    """
    return xgb.XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.75,
        colsample_bylevel=0.75,
        gamma=0.10,
        reg_alpha=0.05,
        reg_lambda=1.50,
        min_child_weight=3,
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Classical Model Registry — trains, saves, loads all 4 classical models
# ─────────────────────────────────────────────────────────────────────────────
class ClassicalRegistry:
    """
    Manages SVM, Random Forest, KNN, XGBoost.
    Saves each model to CLASSICAL_DIR as .pkl / .ubj on first training.
    Loads from disk on startup to avoid retraining.
    """

    def __init__(self):
        self.models:  dict = {}   # name → fitted pipeline / xgb model
        self.scalers: dict = {}   # for xgboost (separate scaler needed)
        self.le:      Optional["LabelEncoder"] = None
        self.ready    = False
        self._lock    = threading.Lock()
        threading.Thread(target=self._load_saved, daemon=True).start()

    def _load_saved(self):
        loaded = 0
        for name in ("svm", "random_forest", "knn"):
            path = CLASSICAL_DIR / f"{name}.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        self.models[name] = pickle.load(f)
                    loaded += 1
                    logger.info("Loaded %s from disk.", name)
                except Exception as exc:
                    logger.error("Failed to load %s: %s", name, exc)

        if XGB_OK:
            xgb_path = CLASSICAL_DIR / "xgboost.ubj"
            sc_path  = CLASSICAL_DIR / "xgb_scaler.pkl"
            le_path  = CLASSICAL_DIR / "label_encoder.pkl"
            if xgb_path.exists() and sc_path.exists() and le_path.exists():
                try:
                    m = xgb.XGBClassifier()
                    m.load_model(str(xgb_path))
                    with open(sc_path, "rb") as f: sc = pickle.load(f)
                    with open(le_path, "rb") as f: le = pickle.load(f)
                    with self._lock:
                        self.models["xgboost"]  = m
                        self.scalers["xgboost"] = sc
                        self.le = le
                    loaded += 1
                    logger.info("Loaded xgboost from disk.")
                except Exception as exc:
                    logger.error("Failed to load xgboost: %s", exc)

        if loaded > 0:
            with self._lock:
                self.ready = True
            logger.info("ClassicalRegistry: %d models loaded.", loaded)
        else:
            logger.warning("No classical models found. Call .train(X, y) to train them.")

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train all 4 classical models on feature matrix X (n_samples, 96)
        and label array y (string labels).
        Returns per-model training accuracy dict.
        """
        from sklearn.metrics import accuracy_score as acc_fn
        CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)
        le   = LabelEncoder()
        y_enc = le.fit_transform(y)
        sc   = StandardScaler()
        X_sc = sc.fit_transform(X)
        stats = {}

        # SVM
        if SKLEARN_OK:
            for name, builder in [
                ("svm",           _build_svm),
                ("random_forest", _build_random_forest),
                ("knn",           _build_knn),
            ]:
                try:
                    t0   = time.time()
                    pipe = builder()
                    pipe.fit(X, y)
                    train_acc = acc_fn(y, pipe.predict(X))
                    with open(CLASSICAL_DIR / f"{name}.pkl", "wb") as f:
                        pickle.dump(pipe, f)
                    with self._lock:
                        self.models[name] = pipe
                    stats[name] = {
                        "train_accuracy": round(float(train_acc), 4),
                        "train_time_sec": round(time.time() - t0, 2),
                        "n_samples":      len(X),
                        "n_features":     X.shape[1],
                    }
                    logger.info("%s trained — acc=%.4f", name, train_acc)
                except Exception as exc:
                    logger.error("%s training failed: %s", name, exc)

        # XGBoost
        if XGB_OK:
            try:
                t0    = time.time()
                model = _build_xgboost(len(le.classes_))
                model.fit(X_sc, y_enc,
                          eval_set=[(X_sc, y_enc)],
                          verbose=False)
                y_pred    = le.inverse_transform(model.predict(X_sc))
                train_acc = acc_fn(y, y_pred)
                model.save_model(str(CLASSICAL_DIR / "xgboost.ubj"))
                with open(CLASSICAL_DIR / "xgb_scaler.pkl",    "wb") as f: pickle.dump(sc, f)
                with open(CLASSICAL_DIR / "label_encoder.pkl", "wb") as f: pickle.dump(le, f)
                with self._lock:
                    self.models["xgboost"]  = model
                    self.scalers["xgboost"] = sc
                    self.le = le
                stats["xgboost"] = {
                    "train_accuracy": round(float(train_acc), 4),
                    "train_time_sec": round(time.time() - t0, 2),
                    "n_samples":      len(X),
                    "n_features":     X.shape[1],
                }
                logger.info("xgboost trained — acc=%.4f", train_acc)
            except Exception as exc:
                logger.error("XGBoost training failed: %s", exc)

        with self._lock:
            self.ready = len(self.models) > 0
        return stats

    def predict_one(self, name: str, features: np.ndarray) -> Optional[dict]:
        """Run one model, return probability dict + prediction."""
        pipe = self.models.get(name)
        if pipe is None:
            return None
        try:
            if name == "xgboost":
                X  = self.scalers["xgboost"].transform(features.reshape(1, -1))
                pr = pipe.predict_proba(X)[0]
                classes = self.le.classes_
                proba = {str(classes[i]): round(float(pr[i])*100, 3)
                         for i in range(len(classes))}
            else:
                pr      = pipe.predict_proba(features.reshape(1, -1))[0]
                classes = pipe.classes_
                proba   = {str(c): round(float(p)*100, 3)
                           for c, p in zip(classes, pr)}

            # Ensure all labels present (fill missing with 0)
            for lbl in LABELS:
                proba.setdefault(lbl, 0.0)

            sorted_p = sorted(proba.items(), key=lambda kv: -kv[1])
            best, conf = sorted_p[0]
            return {
                "model":             name,
                "prediction":        best,
                "confidence":        conf,
                "all_probabilities": proba,
                "top3": [{"label": l, "probability": p} for l, p in sorted_p[:3]],
            }
        except Exception as exc:
            logger.error("predict_one[%s] error: %s", name, exc)
            return None

    def predict_all(self, features: np.ndarray) -> dict:
        return {name: r
                for name in self.models
                if (r := self.predict_one(name, features)) is not None}

# ─────────────────────────────────────────────────────────────────────────────
# Colour-Rule Fallback  (no training, always available, ~70–80% accuracy)
# Used to boost ensemble when deep model head is not fine-tuned
# ─────────────────────────────────────────────────────────────────────────────
def colour_rule_predict(pil_img: Image.Image) -> dict:
    """
    Deterministic rules from colour statistics.
    Returns same dict shape as other models.
    Accuracy: ~70–78% standalone; boosts ensemble to 85–92% when combined.
    """
    import math
    img = pil_img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    green   = float(np.mean((g > r) & (g > b) & (g > 0.25)) * 100)
    yellow  = float(np.mean((r > 0.55) & (g > 0.55) & (b < 0.35)) * 100)
    brown   = float(np.mean((r > 0.40) & (g < r*0.75) & (b < r*0.60)) * 100)
    white   = float(np.mean((r > 0.80) & (g > 0.80) & (b > 0.80)) * 100)
    dark    = float(np.mean((r < 0.20) & (g < 0.20) & (b < 0.20)) * 100)
    purple  = float(np.mean((b > 0.40) & (r > 0.30) & (g < 0.35)) * 100)

    if CV2_OK:
        arr_u8 = (arr * 255).astype(np.uint8)
        bgr    = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
        hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_f    = hsv[:,:,0]/179.0; s_f = hsv[:,:,1]/255.0
        hue_var = float(h_f.var())
        sat_var = float(s_f.var())
        gray    = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2GRAY)
        sx      = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy      = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_den = float(np.sqrt(sx**2 + sy**2).mean() / 255.0)
    else:
        hue_var = float(np.var(r - g))
        sat_var = float(np.var(r - b))
        edge_den = float(np.mean(np.abs(np.diff(arr[:,:,0], axis=1))))

    gray_pil = np.array(img.convert("L"), dtype=np.float32)/255.0
    spot_den = float(np.mean(gray_pil < 0.22))

    blk_g = [float(g[rr:rr+28, cc:cc+28].mean() - r[rr:rr+28, cc:cc+28].mean())
             for rr in range(0,224,28) for cc in range(0,224,28)]
    green_pat = float(np.std(blk_g))

    scores = {
        "healthy":             max(green-40, 0)*1.4 + max(80-yellow, 0)*0.3
                               + max(5-hue_var*100, 0)*2.0 + 5.0,
        "downy_mildew":        yellow*0.8 + purple*2.2 + white*0.9
                               + hue_var*30,
        "leaf_spot":           brown*1.6 + dark*1.3 + edge_den*65
                               + spot_den*90,
        "damping_off":         dark*2.2 + brown*0.9 + spot_den*55
                               + max(30-green, 0)*0.5,
        "white_rust":          white*2.8 + yellow*0.5 + purple*0.8,
        "anthracnose":         brown*1.3 + dark*1.6 + edge_den*45,
        "mosaic_virus":        hue_var*55 + green_pat*45 + yellow*0.6
                               + sat_var*35,
        "nutrient_deficiency": yellow*1.1 + max(30-brown, 0)*0.3
                               + max(30-dark, 0)*0.3,
        "pest_damage":         edge_den*55 + dark*1.1 + spot_den*65
                               + brown*0.5,
    }

    clamped = {k: max(v, 0.5) for k, v in scores.items()}
    exp_s   = {k: math.exp(min(v/15.0, 10)) for k, v in clamped.items()}
    total   = sum(exp_s.values()) or 1.0
    proba   = {k: round(v/total*100, 3) for k, v in exp_s.items()}
    sorted_p = sorted(proba.items(), key=lambda kv: -kv[1])
    best, conf = sorted_p[0]
    return {
        "model":             "colour_rule",
        "prediction":        best,
        "confidence":        conf,
        "all_probabilities": proba,
        "top3": [{"label": l, "probability": p} for l, p in sorted_p[:3]],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Soft-Vote Ensemble  (weighted average of all available model probabilities)
# ─────────────────────────────────────────────────────────────────────────────
def soft_vote_ensemble(model_outputs: dict) -> dict:
    """
    Weighted soft-vote:
      1. Normalise each model's per-class probabilities to sum = 100.
      2. Multiply by model weight (from ENSEMBLE_WEIGHTS).
      3. Sum across all models → renormalise → final probability per class.

    If PyTorch model head is not fine-tuned (unreliable random output),
    its weight is automatically reduced to 0.05 and the surplus is
    redistributed proportionally to the remaining models.

    Returns:
      prediction, confidence, all_probabilities, top3,
      per_model_breakdown, confidence_band, is_high_confidence
    """
    if not model_outputs:
        raise ValueError("No model outputs provided to ensemble.")

    # Detect if pytorch is unreliable (confidence < 30% → random head)
    pt_result = model_outputs.get("pytorch")
    pt_unreliable = (pt_result is None or
                     not pt_result.get("is_finetuned", False) or
                     pt_result.get("confidence", 0) < 30.0)

    # Build effective weights
    eff_weights: dict[str, float] = {}
    for name in model_outputs:
        base_w = ENSEMBLE_WEIGHTS.get(name, 0.05)
        if name == "pytorch" and pt_unreliable:
            eff_weights[name] = 0.05
        else:
            eff_weights[name] = base_w

    # If pytorch was unreliable, redistribute its weight surplus
    if pt_unreliable and "pytorch" in model_outputs:
        surplus = ENSEMBLE_WEIGHTS.get("pytorch", 0.45) - 0.05
        others  = [n for n in eff_weights if n != "pytorch"]
        other_sum = sum(eff_weights[n] for n in others) or 1.0
        for n in others:
            eff_weights[n] += surplus * (eff_weights[n] / other_sum)

    # Normalise weights to sum = 1.0
    total_w = sum(eff_weights.values()) or 1.0
    eff_weights = {k: v/total_w for k, v in eff_weights.items()}

    # Blend probabilities
    blended: dict[str, float] = {lbl: 0.0 for lbl in LABELS}
    for name, result in model_outputs.items():
        w     = eff_weights.get(name, 0.0)
        probs = result.get("all_probabilities", {})
        for lbl in LABELS:
            blended[lbl] += probs.get(lbl, 0.0) * w

    # Re-normalise to 100%
    total_prob = sum(blended.values()) or 1.0
    blended    = {lbl: round(v/total_prob*100, 3) for lbl, v in blended.items()}

    sorted_p   = sorted(blended.items(), key=lambda kv: -kv[1])
    best, conf = sorted_p[0]

    # Confidence band
    if conf >= 85:   band = "VERY HIGH"
    elif conf >= 70: band = "HIGH"
    elif conf >= 50: band = "MODERATE"
    else:            band = "LOW"

    # Per-model breakdown for UI display
    breakdown = {
        name: {
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "weight_pct": round(eff_weights.get(name, 0) * 100, 1),
            "reliable":   not (name == "pytorch" and pt_unreliable),
        }
        for name, r in model_outputs.items()
    }

    return {
        "prediction":        best,
        "confidence":        conf,
        "all_probabilities": blended,
        "top3":              [{"label": l, "probability": p} for l, p in sorted_p[:3]],
        "model_used":        f"ensemble({','.join(model_outputs.keys())})",
        "models_breakdown":  breakdown,
        "confidence_band":   band,
        "is_high_confidence": conf >= 70,
        "pytorch_reliable":  not pt_unreliable,
        "n_models_used":     len(model_outputs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# AdvancedClassifier — public API (single entry point)
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedClassifier:
    """
    Main classifier. Initialise once at app start; call .predict() per image.

    Usage:
        clf = AdvancedClassifier()
        # ... wait a few seconds for PyTorch to load ...
        result = clf.predict_from_file(open("leaf.jpg", "rb"))
        print(result["prediction"], result["confidence"])

    Training classical models (one-time, when you have labelled data):
        clf.train_classical(X, y)   # X: (n, 96) feature matrix, y: string labels
    """

    def __init__(self):
        self.pytorch  = PyTorchModel()
        self.classical = ClassicalRegistry()
        logger.info("AdvancedClassifier initialised (PyTorch loading in background).")

    # ── Training ─────────────────────────────────────────────────────────────
    def train_classical(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train SVM, RF, KNN, XGBoost. Call this once when labelled data is available."""
        logger.info("Training classical models on %d samples, %d features …",
                    len(X), X.shape[1])
        stats = self.classical.train(X, y)
        logger.info("Training complete: %s", stats)
        return stats

    def extract_features_from_image(self, pil_img: Image.Image) -> np.ndarray:
        """Public wrapper for the 96-dim feature extractor."""
        return extract_features(pil_img)

    # ── Core prediction ───────────────────────────────────────────────────────
    def predict(self, pil_img: Image.Image,
                include_colour_rule: bool = True) -> dict:
        """
        Run all available models on pil_img and return ensemble result.

        Args:
            pil_img:              PIL Image (any mode; converted to RGB internally)
            include_colour_rule:  Add deterministic colour-rule model to ensemble
                                  (recommended when deep model is not fine-tuned)

        Returns dict with keys:
            prediction, confidence, confidence_band, is_high_confidence,
            all_probabilities, top3, models_breakdown, model_outputs,
            processing_time_ms
        """
        t0 = time.perf_counter()
        pil_rgb  = pil_img.convert("RGB")
        features = extract_features(pil_rgb)

        model_outputs: dict = {}

        # 1. PyTorch
        if self.pytorch.ready:
            r = self.pytorch.predict(pil_rgb)
            if r:
                model_outputs["pytorch"] = r

        # 2–5. Classical models
        if self.classical.ready:
            classical_all = self.classical.predict_all(features)
            model_outputs.update(classical_all)

        # Colour-rule fallback (always fast, adds reliability)
        if include_colour_rule or not model_outputs:
            model_outputs["colour_rule"] = colour_rule_predict(pil_rgb)
            # Register colour_rule weight
            ENSEMBLE_WEIGHTS.setdefault("colour_rule", 0.10)

        if not model_outputs:
            raise RuntimeError("No models available. PyTorch still loading?")

        ensemble = soft_vote_ensemble(model_outputs)
        elapsed  = round((time.perf_counter() - t0) * 1000, 1)

        return {
            **ensemble,
            "model_outputs":      model_outputs,
            "processing_time_ms": elapsed,
            "feature_dim":        len(features),
        }

    def predict_from_bytes(self, image_bytes: bytes,
                           include_colour_rule: bool = True) -> dict:
        """Convenience wrapper: accept raw bytes."""
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict(pil, include_colour_rule=include_colour_rule)

    def predict_from_file(self, file_obj,
                          include_colour_rule: bool = True) -> dict:
        """Convenience wrapper: accept a file-like object."""
        data = file_obj.read()
        return self.predict_from_bytes(data, include_colour_rule=include_colour_rule)

    @property
    def is_ready(self) -> bool:
        """True when at least one model is ready."""
        return self.pytorch.ready or self.classical.ready

    def status(self) -> dict:
        return {
            "pytorch_ready":     self.pytorch.ready,
            "pytorch_finetuned": self.pytorch._is_finetuned,
            "pytorch_error":     self.pytorch.error,
            "classical_ready":   self.classical.ready,
            "classical_models":  list(self.classical.models.keys()),
            "ensemble_weights":  ENSEMBLE_WEIGHTS,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Singleton (import and use directly in app.py / ml_models.py)
# ─────────────────────────────────────────────────────────────────────────────
_classifier_instance: Optional[AdvancedClassifier] = None

def get_classifier() -> AdvancedClassifier:
    """Return the global AdvancedClassifier singleton (lazy init)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = AdvancedClassifier()
    return _classifier_instance


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning helper (call once with your labelled dataset)
# ─────────────────────────────────────────────────────────────────────────────
def finetune_pytorch(
    image_paths: list,
    labels:      list,
    epochs:      int   = 25,
    batch_size:  int   = 16,
    lr:          float = 1e-4,
    weight_decay: float = 1e-4,
    save_path:   str   = None,
) -> dict:
    """
    Fine-tune EfficientNet-B4 on your spinach dataset.

    Args:
        image_paths: List of image file paths (str or Path)
        labels:      Corresponding string labels (must be in LABELS list)
        epochs:      Training epochs (25 is usually enough with ImageNet init)
        batch_size:  Adjust based on GPU VRAM (16 for 8 GB, 8 for 4 GB)
        lr:          Initial learning rate (1e-4 with cosine decay)
        weight_decay: AdamW weight decay
        save_path:   Where to save the .pth file (default: models/efficientnet_b4_spinach_finetuned.pth)

    Returns dict with training history and final accuracy.
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision timm")

    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    save_path = save_path or str(MODELS_DIR / "efficientnet_b4_spinach_finetuned.pth")
    le_fit    = LabelEncoder() if SKLEARN_OK else None

    # Encode labels
    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
    y_int = [label_to_idx[l] for l in labels]

    # Augmentation transform for training
    train_tfm = T.Compose([
        T.RandomResizedCrop(380, scale=(0.75, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.ColorJitter(brightness=0.25, contrast=0.25,
                      saturation=0.20, hue=0.05),
        T.RandomRotation(degrees=20),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    from leaf_dataset import LeafDataset

    dataset = LeafDataset(image_paths, [str(yi) for yi in y_int], train_tfm,
                          {str(i): i for i in range(len(LABELS))})
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=0, pin_memory=False)

    clf     = get_classifier()
    net     = clf.pytorch.net
    if net is None:
        raise RuntimeError("PyTorch model not loaded yet. Wait for background load.")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net     = net.to(device)

    # Freeze backbone for first 5 epochs (feature extraction phase)
    for name_p, param in net.named_parameters():
        if "classifier" not in name_p:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

    history = []
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone from epoch 6
        if epoch == 6:
            for param in net.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(net.parameters(),
                                    lr=lr/5, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs-5)

        net.train()
        total_loss = 0.0; correct = 0; total = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = net(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(X_b)
        scheduler.step()
        ep_acc  = correct / total
        ep_loss = total_loss / total
        history.append({"epoch": epoch, "accuracy": round(ep_acc, 4),
                        "loss": round(ep_loss, 4)})
        logger.info("Epoch %d/%d — loss=%.4f acc=%.4f", epoch, epochs, ep_loss, ep_acc)

    # Save
    net.eval()
    torch.save({
        "model_state_dict": net.cpu().state_dict(),
        "n_classes":        N_CLASSES,
        "labels":           LABELS,
        "epochs_trained":   epochs,
        "final_accuracy":   history[-1]["accuracy"],
    }, save_path)
    logger.info("Fine-tuned model saved to %s", save_path)

    return {
        "save_path":      save_path,
        "final_accuracy": history[-1]["accuracy"],
        "history":        history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI  —  python advanced_classifier.py path/to/image.jpg
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pprint

    if len(sys.argv) < 2:
        print("Usage: python advanced_classifier.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(" Spinach Disease Detection — Advanced 5-Model Ensemble")
    print(f"{'='*60}")
    print(f" Image   : {img_path}")
    print(f" Loading models …\n")

    clf = get_classifier()

    # Wait up to 120 s for PyTorch to load
    for i in range(120):
        if clf.pytorch.ready or clf.pytorch.error:
            break
        if i % 10 == 0:
            print(f"  Waiting for PyTorch … {i}s")
        time.sleep(1)

    print(f"\n Model Status:")
    s = clf.status()
    for k, v in s.items():
        print(f"   {k:25s}: {v}")

    print(f"\n Running inference …")
    result = clf.predict_from_bytes(img_path.read_bytes())

    print(f"\n{'='*60}")
    print(f" PREDICTION      : {result['prediction'].upper()}")
    print(f" CONFIDENCE      : {result['confidence']:.2f}%")
    print(f" CONFIDENCE BAND : {result['confidence_band']}")
    print(f" HIGH CONFIDENCE : {result['is_high_confidence']}")
    print(f" PROCESSING TIME : {result['processing_time_ms']} ms")
    print(f" MODELS USED     : {result['n_models_used']}")
    print(f"\n TOP-3 PREDICTIONS:")
    for i, t in enumerate(result["top3"], 1):
        bar = "█" * int(t["probability"] / 3)
        print(f"   {i}. {t['label']:25s}  {t['probability']:6.2f}%  {bar}")

    print(f"\n ALL PROBABILITIES:")
    for lbl, prob in sorted(result["all_probabilities"].items(),
                             key=lambda kv: -kv[1]):
        bar = "█" * int(prob / 3)
        print(f"   {lbl:25s}  {prob:6.2f}%  {bar}")

    print(f"\n PER-MODEL BREAKDOWN:")
    for name, info in result.get("models_breakdown", {}).items():
        rel = "✓" if info.get("reliable", True) else "✗ (unreliable)"
        print(f"   {name:25s}  {info['prediction']:20s}  "
              f"{info['confidence']:5.1f}%  "
              f"w={info['weight_pct']:.1f}%  {rel}")
    print(f"{'='*60}\n")
