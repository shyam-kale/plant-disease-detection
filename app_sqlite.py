import os, io, re, uuid, time, json, logging, hashlib, threading, warnings, csv, sqlite3
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler

import numpy as np
from flask import Flask, request, jsonify, g, Response, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageStat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    DB_PATH       = os.environ.get("DB_PATH", "crophealth.db")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 15 * 1024 * 1024))
    ALLOWED_EXT   = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "heic"}
    IMG_SIZE      = (256, 256)
    THUMB_SIZE    = (100, 100)
    LOG_FILE      = "app.log"
    LOG_MAX_BYTES = 10 * 1024 * 1024
    LOG_BACKUP    = 5
    RATE_LIMIT    = 200
    RATE_WIN      = 60
    SECRET_KEY    = os.environ.get("SECRET_KEY", "crophealth-secret-2024")
    DEBUG         = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    HOST          = os.environ.get("HOST", "0.0.0.0")
    PORT          = int(os.environ.get("PORT", 5000))
    FEATURE_VER   = "v3.1"

    LABELS = [
        "healthy", "leaf_blight", "powdery_mildew", "rust", "leaf_spot",
        "bacterial_wilt", "mosaic_virus", "downy_mildew", "anthracnose",
        "root_rot", "nutrient_deficiency", "pest_damage",
    ]

    DISEASE_INFO = {
        "healthy": {
            "status": "Healthy Plant",
            "severity": "none",
            "severity_score": 0,
            "color": "#22c55e",
            "icon": "🌱",
            "affected_parts": [],
            "description": "The plant appears healthy with no visible signs of disease, pest damage, or nutrient stress.",
            "causes": ["Optimal soil conditions", "Balanced nutrition", "Adequate irrigation"],
            "immediate_actions": ["Continue current management practices"],
            "chemical_treatments": [],
            "organic_treatments": ["Apply compost tea monthly", "Neem oil spray every 3 weeks"],
            "fertilizer_schedule": "Balanced NPK 10-10-10 @ 5g/L every 3 weeks",
            "prevention": "Maintain crop rotation, proper spacing, weekly inspection",
            "recovery_time": "N/A — plant is healthy",
            "economic_impact": "None",
        },
        "leaf_blight": {
            "status": "Leaf Blight",
            "severity": "high",
            "severity_score": 75,
            "color": "#ef4444",
            "icon": "🍂",
            "affected_parts": ["Leaves", "Stems"],
            "description": "Leaf blight causes large, irregular brown to tan necrotic lesions with yellow halos.",
            "causes": ["Fungal infection", "Prolonged leaf wetness", "Poor air circulation"],
            "immediate_actions": ["Remove infected leaves", "Avoid overhead irrigation", "Apply fungicide"],
            "chemical_treatments": ["Copper Oxychloride 50% WP @ 3g/L", "Mancozeb 75% WP @ 2.5g/L"],
            "organic_treatments": ["Bordeaux mixture 1%", "Trichoderma viride @ 5g/L"],
            "fertilizer_schedule": "Reduce nitrogen; apply Potassium Sulphate @ 3g/L",
            "prevention": "Use disease-free seeds, crop rotation, preventive copper spray",
            "recovery_time": "7–14 days with treatment",
            "economic_impact": "20–40% yield loss if untreated",
        },
        "powdery_mildew": {
            "status": "Powdery Mildew",
            "severity": "medium",
            "severity_score": 50,
            "color": "#f59e0b",
            "icon": "🌫️",
            "affected_parts": ["Leaves", "Shoots", "Buds"],
            "description": "White to gray powdery fungal colonies on leaf surfaces.",
            "causes": ["Dry weather with moderate humidity", "Excess nitrogen", "Dense canopy"],
            "immediate_actions": ["Remove infected leaves", "Increase airflow", "Apply sulphur fungicide"],
            "chemical_treatments": ["Wettable Sulphur 80% WP @ 2g/L", "Hexaconazole 5% SC @ 2ml/L"],
            "organic_treatments": ["Baking soda spray", "Neem oil 5% EC @ 5ml/L", "Milk spray 40%"],
            "fertilizer_schedule": "Reduce nitrogen; apply Calcium Nitrate @ 2g/L",
            "prevention": "Plant resistant varieties, proper spacing, avoid late irrigation",
            "recovery_time": "10–21 days with treatment",
            "economic_impact": "10–30% yield reduction",
        },
        "rust": {
            "status": "Rust Disease",
            "severity": "high",
            "severity_score": 70,
            "color": "#ef4444",
            "icon": "🟤",
            "affected_parts": ["Leaves", "Stems", "Pods"],
            "description": "Orange-brown to reddish-brown pustules on leaf undersides.",
            "causes": ["Fungal infection", "Cool moist weather", "Wind dispersal"],
            "immediate_actions": ["Apply fungicide immediately", "Remove infected material"],
            "chemical_treatments": ["Propiconazole 25% EC @ 1ml/L", "Tebuconazole 25.9% EC @ 1ml/L"],
            "organic_treatments": ["Sulphur dust 80% WP @ 3g/L", "Neem oil 5% EC @ 5ml/L"],
            "fertilizer_schedule": "Apply Potassium Sulphate @ 5g/L; avoid excess nitrogen",
            "prevention": "Plant rust-resistant varieties, preventive fungicide",
            "recovery_time": "14–21 days",
            "economic_impact": "30–70% yield loss",
        },
        "leaf_spot": {
            "status": "Leaf Spot",
            "severity": "medium",
            "severity_score": 45,
            "color": "#f59e0b",
            "icon": "🔴",
            "affected_parts": ["Leaves"],
            "description": "Circular to irregular necrotic spots with distinct dark borders.",
            "causes": ["Fungal/bacterial pathogens", "Wet foliage", "Infected seeds"],
            "immediate_actions": ["Remove infected leaves", "Switch to drip irrigation", "Apply fungicide"],
            "chemical_treatments": ["Chlorothalonil 75% WP @ 2g/L", "Carbendazim 50% WP @ 1g/L"],
            "organic_treatments": ["Bordeaux mixture 0.5%", "Trichoderma harzianum @ 5g/L"],
            "fertilizer_schedule": "Balanced NPK; increase Potassium",
            "prevention": "Use certified seeds, seed treatment, crop rotation",
            "recovery_time": "7–14 days",
            "economic_impact": "15–25% yield loss",
        },
        "bacterial_wilt": {
            "status": "Bacterial Wilt",
            "severity": "critical",
            "severity_score": 95,
            "color": "#dc2626",
            "icon": "⚠️",
            "affected_parts": ["Vascular system", "Stems", "Roots"],
            "description": "Sudden, irreversible wilting despite adequate soil moisture.",
            "causes": ["Soil-borne bacterium", "Contaminated tools", "Waterlogged conditions"],
            "immediate_actions": ["Remove and destroy infected plants", "Disinfect tools", "Stop irrigation"],
            "chemical_treatments": ["Copper Oxychloride 50% WP @ 3g/L soil drench (suppressive only)"],
            "organic_treatments": ["Trichoderma viride @ 5g/kg soil", "Soil solarization"],
            "fertilizer_schedule": "Do not fertilize infected plants",
            "prevention": "Use resistant varieties, 3-year rotation, avoid waterlogging",
            "recovery_time": "No recovery — remove infected plants",
            "economic_impact": "Up to 100% loss in affected areas",
        },
        "mosaic_virus": {
            "status": "Mosaic Virus",
            "severity": "high",
            "severity_score": 72,
            "color": "#ef4444",
            "icon": "🦠",
            "affected_parts": ["Leaves", "Fruits", "Whole plant"],
            "description": "Mottled yellow-green patterns on leaves, leaf distortion, stunted growth.",
            "causes": ["Aphid vectors", "Infected seeds", "Contaminated tools"],
            "immediate_actions": ["Remove infected plants", "Control aphids", "Disinfect tools"],
            "chemical_treatments": ["Thiamethoxam 25% WG @ 0.3g/L (aphid control)", "Imidacloprid 17.8% SL @ 0.5ml/L"],
            "organic_treatments": ["Neem oil 5% EC @ 5ml/L", "Yellow sticky traps"],
            "fertilizer_schedule": "Avoid excess nitrogen; apply Zinc Sulphate @ 0.5g/L",
            "prevention": "Use virus-free seeds, control aphids preventively",
            "recovery_time": "No cure — remove to prevent spread",
            "economic_impact": "25–50% yield loss",
        },
        "downy_mildew": {
            "status": "Downy Mildew",
            "severity": "high",
            "severity_score": 68,
            "color": "#ef4444",
            "icon": "💧",
            "affected_parts": ["Leaves", "Stems", "Fruits"],
            "description": "Angular yellow patches with grayish-purple sporulation on undersides.",
            "causes": ["Oomycete pathogen", "Cool temperatures with high humidity", "Poor air circulation"],
            "immediate_actions": ["Apply oomycete-specific fungicide", "Remove infected leaves", "Improve ventilation"],
            "chemical_treatments": ["Metalaxyl 8% + Mancozeb 64% WP @ 2.5g/L", "Fosetyl-Al 80% WP @ 2.5g/L"],
            "organic_treatments": ["Bordeaux mixture 1%", "Potassium phosphonate @ 3ml/L"],
            "fertilizer_schedule": "Apply Calcium Chloride @ 2g/L; reduce nitrogen",
            "prevention": "Plant resistant varieties, good drainage, preventive copper sprays",
            "recovery_time": "10–18 days",
            "economic_impact": "30–60% yield loss",
        },
        "anthracnose": {
            "status": "Anthracnose",
            "severity": "medium",
            "severity_score": 55,
            "color": "#f59e0b",
            "icon": "🌑",
            "affected_parts": ["Leaves", "Stems", "Fruits", "Seeds"],
            "description": "Dark, sunken, water-soaked lesions with salmon-pink spore masses.",
            "causes": ["Fungal infection", "Warm humid weather", "Infected seeds"],
            "immediate_actions": ["Remove infected parts", "Apply systemic fungicide", "Avoid wetting foliage"],
            "chemical_treatments": ["Azoxystrobin 23% SC @ 1ml/L", "Carbendazim 50% WP @ 1g/L"],
            "organic_treatments": ["Trichoderma asperellum @ 5g/L", "Hot water seed treatment"],
            "fertilizer_schedule": "Balanced NPK; apply Silicon @ 1g/L",
            "prevention": "Use disease-free seeds, crop rotation, remove debris",
            "recovery_time": "10–14 days",
            "economic_impact": "15–35% yield loss",
        },
        "root_rot": {
            "status": "Root Rot",
            "severity": "critical",
            "severity_score": 88,
            "color": "#dc2626",
            "icon": "🌿",
            "affected_parts": ["Roots", "Crown", "Lower stem"],
            "description": "Progressive yellowing, wilting, stunted growth. Roots appear brown, soft, mushy.",
            "causes": ["Soil-borne pathogens", "Waterlogged soil", "Overwatering"],
            "immediate_actions": ["Reduce irrigation", "Improve drainage", "Apply fungicide soil drench"],
            "chemical_treatments": ["Metalaxyl 35% WS @ 2g/L soil drench", "Fosetyl-Al 80% WP @ 3g/L"],
            "organic_treatments": ["Trichoderma harzianum @ 5g/kg soil", "Cinnamon powder @ 5g/L"],
            "fertilizer_schedule": "Avoid nitrogen; apply Calcium Superphosphate @ 3g/L",
            "prevention": "Ensure proper drainage, use raised beds, avoid overwatering",
            "recovery_time": "14–28 days if caught early",
            "economic_impact": "40–80% plant mortality",
        },
        "nutrient_deficiency": {
            "status": "Nutrient Deficiency",
            "severity": "medium",
            "severity_score": 40,
            "color": "#f59e0b",
            "icon": "🌾",
            "affected_parts": ["Leaves", "Whole plant"],
            "description": "Chlorosis, necrosis, purple discoloration, or stunted growth indicating nutrient imbalance.",
            "causes": ["Insufficient fertilization", "Soil pH imbalance", "Poor root development"],
            "immediate_actions": ["Conduct soil test", "Correct pH", "Apply targeted foliar spray"],
            "chemical_treatments": ["Urea 1% foliar spray", "Ferrous Sulphate 0.5%", "NPK 19:19:19 @ 5g/L"],
            "organic_treatments": ["Vermicompost @ 2 tonnes/acre", "Seaweed extract @ 3ml/L"],
            "fertilizer_schedule": "NPK 120:60:60 kg/ha basal + 60kg N top-dress",
            "prevention": "Soil testing, maintain pH 6.0–7.0, balanced fertilizers",
            "recovery_time": "5–10 days for foliar; 2–4 weeks for soil",
            "economic_impact": "10–40% yield reduction",
        },
        "pest_damage": {
            "status": "Pest Damage",
            "severity": "medium",
            "severity_score": 52,
            "color": "#f59e0b",
            "icon": "🐛",
            "affected_parts": ["Leaves", "Stems", "Fruits", "Roots"],
            "description": "Irregular holes, chewed margins, stippling, leaf curling, tunneling, or galls.",
            "causes": ["Chewing insects", "Sucking insects", "Boring insects", "Absence of natural predators"],
            "immediate_actions": ["Identify pest", "Remove infested parts", "Install sticky traps", "Apply pesticide"],
            "chemical_treatments": ["Spinosad 45% SC @ 0.3ml/L", "Imidacloprid 17.8% SL @ 0.5ml/L"],
            "organic_treatments": ["Neem oil 5% EC @ 5ml/L", "Bacillus thuringiensis @ 2g/L"],
            "fertilizer_schedule": "Avoid excess nitrogen; apply Silicon @ 1g/L",
            "prevention": "IPM, regular scouting, conservation of natural enemies",
            "recovery_time": "7–14 days after pest control",
            "economic_impact": "15–45% yield loss",
        },
    }

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def setup_logging():
    log = logging.getLogger("crophealth")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(funcName)s:%(lineno)d — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); log.addHandler(ch)
    try:
        fh = RotatingFileHandler(Config.LOG_FILE, maxBytes=Config.LOG_MAX_BYTES, backupCount=Config.LOG_BACKUP)
        fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); log.addHandler(fh)
    except Exception: pass
    return log

logger = setup_logging()

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_FILE_SIZE
CORS(app, resources={r"/*": {"origins": "*"}})
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# SQLITE DATABASE
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(Config.DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            model_used TEXT NOT NULL DEFAULT 'random_forest',
            file_hash TEXT NOT NULL DEFAULT '',
            file_size INTEGER NOT NULL DEFAULT 0,
            original_width INTEGER NOT NULL DEFAULT 0,
            original_height INTEGER NOT NULL DEFAULT 0,
            top3_predictions TEXT,
            all_probabilities TEXT,
            feature_vector TEXT,
            processing_time_ms REAL NOT NULL DEFAULT 0,
            feature_version TEXT NOT NULL DEFAULT 'v3.1',
            thumbnail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_result ON predictions(prediction_result)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON predictions(file_hash)")
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized at %s", Config.DB_PATH)

init_db()

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(Config.DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db: db.close()

def execute_query(sql, params=None, fetch=False, fetchone=False, commit=False):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(sql, params or ())
        if commit: conn.commit()
        if fetchone:
            row = cur.fetchone()
            return dict(row) if row else None
        if fetch:
            return [dict(row) for row in cur.fetchall()]
        return cur.lastrowid
    except Exception as e:
        conn.rollback()
        logger.error("DB error: %s | SQL: %.120s", e, sql)
        raise
    finally:
        cur.close()

# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────
_rate_store: dict = {}
_rate_lock = threading.Lock()

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        with _rate_lock:
            _rate_store.setdefault(ip, [])
            _rate_store[ip] = [t for t in _rate_store[ip] if now - t < Config.RATE_WIN]
            if len(_rate_store[ip]) >= Config.RATE_LIMIT:
                return jsonify({"error": "Rate limit exceeded."}), 429
            _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated

threading.Thread(target=lambda: [time.sleep(120) or _rate_store.clear() for _ in iter(int, 1)], daemon=True).start()

# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────
@app.before_request
def before_request():
    g.t0 = time.time()
    g.rid = str(uuid.uuid4())[:8]

@app.after_request
def after_request(response):
    ms = (time.time() - g.get("t0", time.time())) * 1000
    response.headers["X-Request-ID"] = g.get("rid", "-")
    response.headers["X-Response-Time"] = f"{ms:.1f}ms"
    return response

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(fn): return "." in fn and fn.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXT
def sanitize_filename(fn): return re.sub(r"[^\w.\-]", "_", os.path.basename(fn))[:200] or "upload"
def compute_hash(data): return hashlib.sha256(data).hexdigest()
def size_label(n):
    if n < 1024: return f"{n}B"
    if n < 1048576: return f"{n/1024:.1f}KB"
    return f"{n/1048576:.1f}MB"

def validate_image(data):
    sigs = [b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM", b"RIFF", b"II*\x00", b"MM\x00*"]
    return len(data) >= 8 and any(data.startswith(s) for s in sigs)

def success_response(data, status=200):
    return jsonify({"status": "success", "data": data, "ts": datetime.utcnow().isoformat() + "Z"}), status

def error_response(msg, status=400, details=None):
    body = {"status": "error", "error": msg, "ts": datetime.utcnow().isoformat() + "Z"}
    if details: body["details"] = details
    return jsonify(body), status

def serialize_row(row):
    if not row: return None
    out = {}
    for k, v in (row.items() if hasattr(row, 'items') else dict(row).items()):
        if isinstance(v, str) and ('T' in v or '-' in v):
            try: out[k] = v
            except: out[k] = v
        else: out[k] = v
    return out

def serialize_rows(rows): return [serialize_row(r) for r in (rows or [])]

# [REST OF THE CODE CONTINUES WITH ImageProcessor, ModelRegistry, etc. - SAME AS ORIGINAL]
# Due to length limits, I'll create this as a separate file

