"""
app.py  —  Spinach Disease Detection API
==========================================
All routes wired to advanced_classifier (5-model ensemble)
+ full treatment report + sharing + feedback + analytics
"""
from __future__ import annotations
import os, io, re, csv, ssl, json, time, base64, logging, threading, urllib.request
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler

from flask import Flask, request, jsonify, g, Response, send_from_directory
from flask_cors import CORS

from config import Config
from database import close_db, execute, PredictionDAO, FeedbackDAO
from ml_models import (
    deep_model, classical_models, sklearn_registry, xgb_model,
    run_pipeline, generate_batch_report,
    validate_image_bytes, sanitize_name, allowed_ext, ImageProcessor,
    _ENSEMBLE_WEIGHTS,
)

# ── Advanced classifier (5-model ensemble) ────────────────────────────────────
try:
    from advanced_classifier import get_classifier, extract_features, colour_rule_predict
    _adv_clf = get_classifier()
    ADV_OK = True
except Exception as _e:
    _adv_clf = None
    ADV_OK = False
    logging.getLogger("spinach").warning("advanced_classifier not available: %s", _e)


def setup_logging():
    log = logging.getLogger("spinach")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(funcName)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); log.addHandler(ch)
    try:
        fh = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=3)
        fh.setFormatter(fmt); log.addHandler(fh)
    except Exception: pass
    return log

logger = setup_logging()

app = Flask(__name__)
app.config["SECRET_KEY"]         = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_FILE_SIZE
CORS(app, resources={r"/*": {"origins": "*"}})
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
app.teardown_appcontext(close_db)

_rate_store: dict = {}
_rate_lock = threading.Lock()

def rate_limit(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        ip  = request.remote_addr or "unknown"
        now = time.time()
        with _rate_lock:
            _rate_store.setdefault(ip, [])
            _rate_store[ip] = [t for t in _rate_store[ip] if now-t < Config.RATE_WIN]
            if len(_rate_store[ip]) >= Config.RATE_LIMIT:
                return error_resp("Rate limit exceeded.", 429)
            _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return wrapped

threading.Thread(
    target=lambda: [time.sleep(120) or _rate_store.clear() for _ in iter(int,1)],
    daemon=True).start()

@app.before_request
def _before(): g.t0 = time.time()

@app.after_request
def _after(resp):
    ms = (time.time() - g.get("t0", time.time())) * 1000
    resp.headers["X-Response-Time"] = f"{ms:.1f}ms"
    resp.headers["X-Model-Version"]  = Config.MODEL_VERSION
    return resp

def ok(data, status=200):
    return jsonify({"status":"success","data":data,
                    "ts":datetime.utcnow().isoformat()+"Z"}), status

def error_resp(msg, status=400, details=None):
    body = {"status":"error","error":msg,"ts":datetime.utcnow().isoformat()+"Z"}
    if details: body["details"] = details
    return jsonify(body), status

def _clean_row(row):
    if not row: return None
    out = {}
    for k, v in (row.items() if hasattr(row,"items") else row):
        if isinstance(v, datetime): out[k] = v.isoformat()
        elif isinstance(v, (bytes,bytearray)): out[k] = f"<binary {len(v)} bytes>"
        else: out[k] = v
    return out

def _clean_rows(rows): return [_clean_row(r) for r in (rows or [])]


# ─────────────────────────────────────────────────────────────────────────────
# Advanced pipeline — merges ml_models result with advanced_classifier
# ─────────────────────────────────────────────────────────────────────────────
def _run_advanced_pipeline(file_obj, use_cache=True, sklearn_model=None):
    """
    Run full 5-model ensemble:
    1. ml_models.run_pipeline  (EfficientNet + 6 classical + ColourRule)
    2. advanced_classifier     (SVM + RF + KNN + XGBoost + PyTorch TTA)
    3. Merge both probability dicts with weighted average
    Returns enriched result dict with all fields from both pipelines.
    """
    # Core result from existing pipeline
    result = run_pipeline(file_obj, use_cache=use_cache, sklearn_model=sklearn_model)

    # If advanced classifier is available, blend its probabilities in
    if ADV_OK and _adv_clf:
        try:
            from PIL import Image
            import numpy as np
            img_bytes = None
            # Try to get image bytes from result thumbnail (base64)
            thumb = result.get("thumbnail","")
            if thumb and thumb.startswith("data:"):
                img_bytes = base64.b64decode(thumb.split(",",1)[1])
            # If we have image bytes, run advanced classifier
            if img_bytes:
                pil_img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                adv_out  = _adv_clf.predict(pil_img, include_colour_rule=True)
                # Blend probabilities: ml_models 55% + advanced 45%
                ml_probs  = result.get("all_probabilities", {})
                adv_probs = adv_out.get("all_probabilities", {})
                blended   = {}
                for lbl in Config.LABELS:
                    blended[lbl] = round(
                        ml_probs.get(lbl,0)*0.55 + adv_probs.get(lbl,0)*0.45, 3)
                total = sum(blended.values()) or 1.0
                blended = {k: round(v/total*100,3) for k,v in blended.items()}
                sorted_p  = sorted(blended.items(), key=lambda kv:-kv[1])
                best, conf = sorted_p[0]
                result["prediction"]        = best
                result["confidence"]        = conf
                result["all_probabilities"] = blended
                result["top3"]              = [{"label":l,"probability":p}
                                                for l,p in sorted_p[:3]]
                result["advanced_result"]   = adv_out
                result["model_used"]        = "hybrid_ensemble_7model"
                # Update disease_info if prediction changed
                if best in Config.DISEASE_INFO:
                    result["disease_info"] = Config.DISEASE_INFO[best]
        except Exception as exc:
            logger.warning("advanced_classifier blend failed: %s", exc)

    # Always attach full treatment_report fields at top level for easy UI access
    tr = result.get("treatment_report", {})
    result["urgency"]           = tr.get("urgency","ROUTINE CARE")
    result["confidence_band"]   = tr.get("diagnosis",{}).get("confidence_band","")
    result["agreement_pct"]     = tr.get("model_analysis",{}).get("agreement_pct",0)
    result["visual_evidence"]   = tr.get("visual_evidence",[])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prediction routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@rate_limit
def predict():
    if "image" not in request.files:
        return error_resp("Field 'image' is required.")
    sklearn_model = request.form.get("sklearn_model","").strip() or None
    try:
        data = _run_advanced_pipeline(request.files["image"],
                                      sklearn_model=sklearn_model)
        return ok(data)
    except ValueError as exc: return error_resp(str(exc), 400)
    except RuntimeError as exc: return error_resp(str(exc), 503)
    except Exception as exc:
        logger.error("predict error: %s", exc, exc_info=True)
        return error_resp(f"Prediction failed: {exc}", 500)


@app.route("/predict/batch", methods=["POST"])
@rate_limit
def predict_batch():
    files = request.files.getlist("images[]")
    if not files: return error_resp("No images provided under 'images[]'.")
    files = files[:20]
    results, errors = [], []
    for f in files:
        try:
            r = _run_advanced_pipeline(f)
            results.append(r)
        except Exception as exc:
            errors.append({"filename": sanitize_name(f.filename or "?"),
                           "error": str(exc)})
    report = generate_batch_report(results) if results else {}
    return ok({
        "total":     len(files),
        "succeeded": len(results),
        "failed":    len(errors),
        "results":   results,
        "errors":    errors,
        "batch_report": report,
    })


@app.route("/predict/url", methods=["POST"])
@rate_limit
def predict_url():
    body = request.get_json(silent=True, force=True) or {}
    url  = (body.get("url") or request.form.get("url") or "").strip()
    if not url: return error_resp("'url' is required.")
    if not url.startswith(("http://","https://")):
        return error_resp("URL must start with http:// or https://")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0",
                                               "Accept":"image/*,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            image_bytes = resp.read(Config.MAX_FILE_SIZE+1)
    except Exception as exc:
        return error_resp(f"Could not fetch image: {exc}", 400)
    if len(image_bytes) > Config.MAX_FILE_SIZE:
        return error_resp("Remote image exceeds 15 MB limit.")
    if not validate_image_bytes(image_bytes):
        return error_resp("URL does not point to a valid image.")
    raw_name = sanitize_name(url.split("?")[0].split("/")[-1]) or "url_image"

    class _UrlFile:
        filename = raw_name
        _d = image_bytes
        def read(self): return self._d

    try:
        data = _run_advanced_pipeline(_UrlFile(), use_cache=False)
        return ok(data)
    except ValueError as exc: return error_resp(str(exc), 400)
    except RuntimeError as exc: return error_resp(str(exc), 503)
    except Exception as exc:
        logger.error("predict_url error: %s", exc, exc_info=True)
        return error_resp(f"URL prediction failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# History / Search / Export
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
@rate_limit
def history():
    try:
        page     = max(1,int(request.args.get("page",1)))
        per_page = min(100,max(1,int(request.args.get("per_page",10))))
        label    = request.args.get("label","").strip() or None
        model    = request.args.get("model","").strip() or None
        data     = PredictionDAO.paginate(page,per_page,label,model)
        data["data"] = _clean_rows(data["data"])
        return ok(data)
    except Exception as exc: return error_resp("Failed to load history.",500,str(exc))

@app.route("/history/<int:pid>", methods=["GET"])
@rate_limit
def get_prediction(pid):
    row = PredictionDAO.get_by_id(pid)
    if not row: return error_resp(f"Prediction #{pid} not found.",404)
    return ok({"prediction":_clean_row(row)})

@app.route("/history/<int:pid>", methods=["DELETE"])
@rate_limit
def delete_prediction(pid):
    if not PredictionDAO.get_by_id(pid):
        return error_resp(f"Prediction #{pid} not found.",404)
    PredictionDAO.delete(pid)
    return ok({"deleted":True,"id":pid})

@app.route("/search", methods=["GET"])
@rate_limit
def search():
    q = request.args.get("q","").strip()
    if len(q)<2: return error_resp("Query must be at least 2 characters.")
    limit   = min(100,max(1,int(request.args.get("limit",30))))
    results = PredictionDAO.search(q,limit)
    return ok({"results":_clean_rows(results),"count":len(results),"query":q})

@app.route("/export/csv", methods=["GET"])
@rate_limit
def export_csv():
    try:
        label = request.args.get("label","").strip() or None
        model = request.args.get("model","").strip() or None
        rows  = PredictionDAO.export_rows(label,model)
        buf   = io.StringIO()
        w     = csv.writer(buf)
        w.writerow(["id","filename","prediction","confidence","model",
                    "file_size","width","height","processing_ms","created_at"])
        for r in rows:
            w.writerow([r["id"],r["image_name"],r["prediction_result"],
                        round(float(r["confidence"] or 0),2),r["model_used"],
                        r["file_size"],r["original_width"],r["original_height"],
                        round(float(r["processing_time_ms"] or 0),1),r["created_at"]])
        return Response(buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":"attachment; filename=spinach_detections.csv"})
    except Exception as exc: return error_resp("CSV export failed.",500,str(exc))

@app.route("/export/json", methods=["GET"])
@rate_limit
def export_json():
    try:
        label = request.args.get("label","").strip() or None
        rows  = PredictionDAO.export_rows(label,None)
        return Response(
            json.dumps({"data":_clean_rows(rows),"count":len(rows)}, default=str),
            mimetype="application/json",
            headers={"Content-Disposition":"attachment; filename=spinach_detections.json"})
    except Exception as exc: return error_resp("JSON export failed.",500,str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
@rate_limit
def stats():
    try:
        data = PredictionDAO.stats()
        data["model_ready"]        = deep_model.ready
        data["advanced_ready"]     = ADV_OK and bool(_adv_clf)
        data["feedback"]           = FeedbackDAO.accuracy()
        data["labels"]             = Config.LABELS
        return ok(data)
    except Exception as exc: return error_resp("Stats failed.",500,str(exc))

@app.route("/stats/timeline", methods=["GET"])
@rate_limit
def stats_timeline():
    try:
        days = min(90,max(1,int(request.args.get("days",7))))
        return ok({"timeline":PredictionDAO.timeline(days),"days":days})
    except Exception as exc: return error_resp("Timeline failed.",500,str(exc))

@app.route("/stats/labels", methods=["GET"])
@rate_limit
def stats_labels():
    try:
        rows = execute(
            "SELECT prediction_result AS label, COUNT(*) AS cnt, "
            "AVG(confidence) AS avg_conf, MIN(confidence) AS min_conf, "
            "MAX(confidence) AS max_conf "
            "FROM predictions GROUP BY prediction_result ORDER BY cnt DESC",
            fetch=True) or []
        data = [{"label":r["label"],"count":r["cnt"],
                 "avg_confidence":round(float(r["avg_conf"] or 0),1),
                 "min_confidence":round(float(r["min_conf"] or 0),1),
                 "max_confidence":round(float(r["max_conf"] or 0),1)}
                for r in rows]
        return ok({"labels":data})
    except Exception as exc: return error_resp("Label stats failed.",500,str(exc))

@app.route("/stats/confidence", methods=["GET"])
@rate_limit
def stats_confidence():
    """Returns confidence distribution histogram (10 buckets 0-100)."""
    try:
        rows = execute(
            "SELECT FLOOR(confidence/10)*10 AS bucket, COUNT(*) AS cnt "
            "FROM predictions GROUP BY bucket ORDER BY bucket",
            fetch=True) or []
        buckets = {f"{int(r['bucket'])}-{int(r['bucket'])+10}": r["cnt"]
                   for r in rows if r["bucket"] is not None}
        return ok({"confidence_distribution": buckets})
    except Exception as exc: return error_resp("Confidence stats failed.",500,str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Disease info + Treatment report
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/disease-info", methods=["GET"])
def disease_info_all():
    return ok({"diseases":Config.DISEASE_INFO,"labels":Config.LABELS})

@app.route("/disease-info/<string:label>", methods=["GET"])
def disease_info_one(label):
    info = Config.DISEASE_INFO.get(label)
    if not info: return error_resp(f"No info for '{label}'.",404)
    return ok({"label":label,**info})

@app.route("/treatment/<string:label>", methods=["GET"])
def treatment_guide(label):
    """Return full treatment guide for a disease label."""
    info = Config.DISEASE_INFO.get(label)
    if not info: return error_resp(f"No treatment info for '{label}'.",404)
    return ok({
        "label":               label,
        "status":              info.get("status",""),
        "severity":            info.get("severity",""),
        "severity_score":      info.get("severity_score",0),
        "description":         info.get("description",""),
        "affected_parts":      info.get("affected_parts",[]),
        "causes":              info.get("causes",[]),
        "immediate_actions":   info.get("immediate_actions",[]),
        "chemical_treatments": info.get("chemical_treatments",[]),
        "organic_treatments":  info.get("organic_treatments",[]),
        "fertilizer_schedule": info.get("fertilizer_schedule",""),
        "prevention":          info.get("prevention",""),
        "recovery_time":       info.get("recovery_time",""),
        "economic_impact":     info.get("economic_impact",""),
    })

@app.route("/report/<int:pid>", methods=["GET"])
@rate_limit
def get_report(pid):
    """Return full treatment report for a saved prediction."""
    row = PredictionDAO.get_by_id(pid)
    if not row: return error_resp(f"Prediction #{pid} not found.",404)
    label  = row.get("prediction_result","")
    info   = Config.DISEASE_INFO.get(label,{})
    conf   = float(row.get("confidence",0))
    report = {
        "id":            pid,
        "filename":      row.get("image_name",""),
        "prediction":    label,
        "confidence":    conf,
        "created_at":    str(row.get("created_at","")),
        "disease_info":  info,
        "urgency": (
            "IMMEDIATE ACTION REQUIRED"
            if info.get("severity") in ("critical","high") and conf>=60
            else "MONITOR CLOSELY" if info.get("severity")=="medium"
            else "ROUTINE CARE"),
    }
    return ok(report)


# ─────────────────────────────────────────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
@rate_limit
def submit_feedback():
    try:
        body    = request.get_json(silent=True,force=True) or {}
        pid     = body.get("prediction_id")
        correct = (body.get("correct_label") or "").strip()
        comment = (body.get("comment") or "")[:500]
        if not isinstance(pid,int): return error_resp("'prediction_id' (integer) required.")
        if correct not in Config.LABELS:
            return error_resp(f"Invalid label. Valid: {Config.LABELS}")
        if not PredictionDAO.get_by_id(pid):
            return error_resp(f"Prediction #{pid} not found.",404)
        FeedbackDAO.insert(pid,correct,comment)
        return ok({"message":"Feedback saved.","prediction_id":pid,"correct_label":correct})
    except Exception as exc: return error_resp("Feedback failed.",500,str(exc))

@app.route("/feedback", methods=["GET"])
@rate_limit
def get_feedback():
    try:
        limit = min(200,max(1,int(request.args.get("limit",50))))
        rows  = FeedbackDAO.get_all(limit)
        return ok({"feedback":_clean_rows(rows),"count":len(rows)})
    except Exception as exc: return error_resp("Failed to load feedback.",500,str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# Analyze (color + features)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
@rate_limit
def analyze():
    if "image" not in request.files: return error_resp("Field 'image' is required.")
    f    = request.files["image"]
    name = sanitize_name(f.filename or "upload")
    if not allowed_ext(name): return error_resp("File type not allowed.")
    data = f.read()
    if not data or not validate_image_bytes(data): return error_resp("Invalid image.")
    try:
        proc = ImageProcessor(data); proc.prepare()
        colour   = proc.colour_analysis()
        features = proc.extract_features()
        feature_names = [
            "r_mean","g_mean","b_mean","r_std","g_std","b_std",
            "h_mean","h_std","s_mean","s_std","v_mean","v_std",
            "l_mean","l_std","a_mean","a_std","bl_mean","bl_std",
            "green_dom","yellow_idx","brown_idx","white_idx","brightness",
            "contrast","warm_cool","green_ratio","entropy","g_skew","g_kurt",
            "sharpness","edge_density","grad_var","spot_density","symmetry",
            "hue_variance","green_patchiness","yellow_uniformity",
            "sat_variance","green_yellow_contrast","a_variance",
            "r_p25","r_p50","r_p75","g_p25","g_p50","g_p75",
            "b_p25","b_p50","b_p75",
        ]
        named = {n:round(float(v),5) for n,v in zip(feature_names,features.tolist())}
        # Advanced 96-dim features if available
        adv_features = None
        if ADV_OK:
            try:
                from PIL import Image
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                adv_features = extract_features(pil).tolist()
            except Exception: pass
        return ok({
            "filename":        name,
            "metadata":        proc.meta,
            "colour_analysis": colour,
            "features_49":     named,
            "features_96":     adv_features,
            "thumbnail":       proc.thumbnail_b64(),
        })
    except Exception as exc: return error_resp("Analysis failed.",500,str(exc))

@app.route("/analyze/color", methods=["POST"])
@rate_limit
def analyze_color():
    if "image" not in request.files: return error_resp("Field 'image' is required.")
    f    = request.files["image"]
    name = sanitize_name(f.filename or "upload")
    if not allowed_ext(name): return error_resp("File type not allowed.")
    data = f.read()
    if not data or not validate_image_bytes(data): return error_resp("Invalid image.")
    try:
        proc = ImageProcessor(data); proc.prepare()
        return ok({"filename":name,"metadata":proc.meta,
                   "colour_analysis":proc.colour_analysis(),
                   "thumbnail":proc.thumbnail_b64()})
    except Exception as exc: return error_resp("Color analysis failed.",500,str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# Images / Models / Health
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/image/<int:pid>", methods=["GET"])
def serve_image(pid):
    row = PredictionDAO.get_image_data(pid)
    if not row or not row.get("image_data"):
        return error_resp("Image not found.",404)
    data = row["image_data"]
    if isinstance(data,str):
        try: data = base64.b64decode(data)
        except Exception: return error_resp("Image data corrupted.",500)
    mime = ("image/jpeg" if data[:3]==b"\xff\xd8\xff" else
            "image/png"  if data[:8]==b"\x89PNG\r\n\x1a\n" else "image/jpeg")
    return Response(data, mimetype=mime,
        headers={"Content-Disposition":f'inline; filename="{row["image_name"]}"',
                 "Cache-Control":"public, max-age=86400"})

@app.route("/models", methods=["GET"])
@rate_limit
def list_models():
    adv_status = {}
    if ADV_OK and _adv_clf:
        try: adv_status = _adv_clf.status()
        except Exception: pass
    return ok({
        "deep_model": {"name":"EfficientNet-B4","ready":deep_model.ready,
                       "error":deep_model._load_error},
        "classical_models": classical_models.get_info(),
        "advanced_classifier": adv_status,
        "sklearn": classical_models.get_info(),
    })

@app.route("/models/active", methods=["PUT"])
@rate_limit
def set_active_model():
    try:
        body = request.get_json(silent=True,force=True) or {}
        name = (body.get("model") or "").strip()
        if not name: return error_resp("'model' is required.")
        sklearn_registry.set_active(name)
        return ok({"active_model":name,"message":f"Active model set to '{name}'"})
    except ValueError as exc: return error_resp(str(exc),404)

@app.route("/health", methods=["GET"])
def health():
    db_ok = False
    try:
        row   = execute("SELECT 1 AS v",fetchone=True)
        db_ok = row is not None
    except Exception: pass
    return jsonify({
        "status":           "healthy" if db_ok else "degraded",
        "db_ok":            db_ok,
        "model_ready":      deep_model.ready,
        "model_loading":    not deep_model.ready and deep_model._load_error is None,
        "model_error":      deep_model._load_error,
        "classical_ready":  classical_models.is_ready(),
        "advanced_ready":   ADV_OK and bool(_adv_clf),
        "classical_models": list(classical_models.pipelines.keys()),
        "version":          Config.MODEL_VERSION,
        "ts":               datetime.utcnow().isoformat()+"Z",
    }), 200 if db_ok else 503

@app.route("/model/status", methods=["GET"])
def model_status():
    from ml_models import TORCH_OK
    try:
        import torch; tv = torch.__version__
    except Exception: tv = None
    imagenet_path  = deep_model.IMAGENET_PATH
    finetuned_path = deep_model.FINETUNED_PATH
    active_path    = finetuned_path if finetuned_path.exists() else imagenet_path
    adv_status = {}
    if ADV_OK and _adv_clf:
        try: adv_status = _adv_clf.status()
        except Exception: pass
    return jsonify({
        "model_ready":            deep_model.ready,
        "model_error":            deep_model._load_error,
        "torch_available":        TORCH_OK,
        "torch_version":          tv,
        "finetuned_exists":       finetuned_path.exists(),
        "imagenet_cached":        imagenet_path.exists(),
        "model_file_exists":      active_path.exists(),
        "model_file_size_mb":     round(active_path.stat().st_size/1_048_576,1) if active_path.exists() else None,
        "classical_models_ready": classical_models.is_ready(),
        "classical_models":       list(classical_models.pipelines.keys()),
        "advanced_classifier":    adv_status,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Training routes  (trigger model training from UI)
# ─────────────────────────────────────────────────────────────────────────────
_train_status = {"running": False, "progress": "", "result": None, "error": None}

def _resolve_dataset_dir(path: str) -> str:
    """
    Given a user-supplied path, find the actual folder that contains
    disease-labelled subfolders (e.g. Apple___healthy/).
    Handles:
      - Extra quotes around path
      - Downloaded ZIP extract adds one extra wrapper folder
      - Common nested structures like color/, segmented/, PlantVillage/
    """
    import pathlib
    p = pathlib.Path(path.strip().strip('"').strip("'").strip())

    if not p.exists():
        return str(p)  # let the caller raise the error with the clean path

    # Check if this folder ITSELF has disease subfolders
    def _has_images(folder):
        for sub in folder.iterdir():
            if sub.is_dir():
                imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.jpeg")) + list(sub.glob("*.png"))
                if imgs:
                    return True
        return False

    if _has_images(p):
        return str(p)

    # Try common nested names
    for sub_name in ["color", "Color", "segmented", "PlantVillage", "plantvillage",
                     "plant_village", "images", "data", "dataset"]:
        candidate = p / sub_name
        if candidate.exists() and _has_images(candidate):
            logger.info("Auto-detected dataset subfolder: %s", candidate)
            return str(candidate)

    # Try any single subdirectory that has image folders inside
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    for sub in subdirs:
        if _has_images(sub):
            logger.info("Auto-detected dataset subfolder: %s", sub)
            return str(sub)

    # Return original — let downstream code report error
    return str(p)

@app.route("/train/status", methods=["GET"])
def train_status():
    return ok(_train_status)

@app.route("/train/start", methods=["POST"])
@rate_limit
def train_start():
    """
    Start model training in background thread.
    Body: { data_dir, epochs, batch_size, max_per_class,
            skip_pytorch, skip_classical }
    """
    global _train_status
    if _train_status["running"]:
        return error_resp("Training already running.", 409)
    body = request.get_json(silent=True, force=True) or {}
    data_dir = (body.get("data_dir") or "").strip().strip('"').strip("'").strip()
    if not data_dir:
        return error_resp("'data_dir' is required — path to PlantVillage folder.", 400)

    # Auto-detect actual image folder inside the given path
    data_dir = _resolve_dataset_dir(data_dir)

    if not os.path.isdir(data_dir):
        return error_resp(
            f"Directory not found: {data_dir}. "
            "Paste the folder path WITHOUT quotes, e.g.: C:\\Users\\shyam\\Downloads\\PlantVillage",
            400
        )

    epochs        = int(body.get("epochs", 25))
    batch_size    = int(body.get("batch_size", 16))
    max_per_class = int(body.get("max_per_class", 800))
    skip_pytorch  = bool(body.get("skip_pytorch", False))
    skip_classical= bool(body.get("skip_classical", False))

    def _run():
        global _train_status
        _train_status = {"running": True, "progress": "Starting…",
                          "result": None, "error": None}
        try:
            from evaluate import run_training, build_plantvillage_dataset

            _train_status["progress"] = "Scanning dataset folder…"
            # Quick check before full run
            paths, lbls = build_plantvillage_dataset(data_dir, max_per_class=10)
            n_cls = len(set(lbls))
            _train_status["progress"] = f"Dataset found: {n_cls} classes detected. Building full dataset…"

            if not skip_pytorch:
                _train_status["progress"] = "Training EfficientNet-B4 (this takes 40–60 min on CPU)…"
            else:
                _train_status["progress"] = "Training classical models (SVM / RF / KNN / XGBoost)…"

            result = run_training(
                data_dir, epochs=epochs, batch_size=batch_size,
                max_per_class=max_per_class,
                skip_pytorch=skip_pytorch, skip_classical=skip_classical)

            _train_status["running"]  = False
            _train_status["progress"] = "Complete ✅ — restart the server to activate new models"
            _train_status["result"]   = result

            # Reload models after training
            if not skip_pytorch:
                try:
                    from ml_models import deep_model as dm
                    threading.Thread(target=dm._load, daemon=True).start()
                except Exception: pass
            if not skip_classical:
                try:
                    from advanced_classifier import get_classifier
                    clf = get_classifier()
                    threading.Thread(target=clf.classical._load_saved, daemon=True).start()
                except Exception: pass

        except Exception as exc:
            _train_status["running"]  = False
            _train_status["progress"] = f"Failed: {exc}"
            _train_status["error"]    = str(exc)
            logger.error("Training failed: %s", exc, exc_info=True)

    threading.Thread(target=_run, daemon=True).start()
    return ok({"message": "Training started.", "status": "running"})

@app.route("/", methods=["GET"])
@app.route("/ui", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)),"index.html")

@app.errorhandler(400)
def bad_request(e): return error_resp("Bad request.",400)
@app.errorhandler(404)
def not_found(e): return error_resp(f"Not found: {request.path}",404)
@app.errorhandler(413)
def too_large(e): return error_resp("File too large — max 15 MB.",413)
@app.errorhandler(429)
def too_many(e): return error_resp("Too many requests.",429)
@app.errorhandler(500)
def server_error(e):
    logger.error("500: %s",e)
    return error_resp("Internal server error.",500)

if __name__ == "__main__":
    app.config["START_TIME"] = time.time()
    print(f"\n{'='*56}")
    print("  🌿  Spinach Disease Detection — 7-Model Ensemble API")
    print(f"  URL: http://{Config.HOST}:{Config.PORT}/")
    print(f"  Advanced classifier: {'✅ loaded' if ADV_OK else '⚠️  not loaded'}")
    print(f"{'='*56}\n")
    app.run(host=Config.HOST,port=Config.PORT,debug=Config.DEBUG,
            threaded=True,use_reloader=False)
