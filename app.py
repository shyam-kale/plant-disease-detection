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
    USE_SQLITE    = os.environ.get("USE_SQLITE", "false").lower() == "true"
    DB_HOST       = os.environ.get("DB_HOST", "localhost")
    DB_PORT       = int(os.environ.get("DB_PORT", 3306))
    DB_USER       = os.environ.get("DB_USER", "root")
    DB_PASSWORD   = os.environ.get("DB_PASSWORD", "Root@1234")
    DB_NAME       = os.environ.get("DB_NAME", "image_classifier")
    DB_POOL_SIZE  = int(os.environ.get("DB_POOL_SIZE", 10))
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
            "description": "The plant appears healthy with no visible signs of disease, pest damage, or nutrient stress. Leaf color, texture, and structure are within normal parameters.",
            "causes": ["Optimal soil conditions", "Balanced nutrition", "Adequate irrigation", "Good air circulation"],
            "immediate_actions": ["Continue current management practices", "Monitor weekly for early disease signs"],
            "chemical_treatments": [],
            "organic_treatments": ["Apply compost tea (1:10 dilution) monthly as preventive boost", "Neem oil spray (3ml/L) every 3 weeks as prophylactic"],
            "fertilizer_schedule": "Balanced NPK 10-10-10 @ 5g/L foliar spray every 3 weeks",
            "prevention": "Maintain crop rotation every season, ensure proper plant spacing (30-45cm), inspect weekly, and keep field free of debris.",
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
            "description": "Leaf blight causes large, irregular brown to tan necrotic lesions with yellow halos. Lesions expand rapidly under warm (25–30°C), humid conditions and can defoliate the plant within days.",
            "causes": ["Alternaria spp. / Helminthosporium spp. (fungal)", "Xanthomonas spp. (bacterial)", "Prolonged leaf wetness >6 hours", "Overhead irrigation", "Dense canopy with poor airflow"],
            "immediate_actions": [
                "Remove and bag all infected leaves — do NOT compost",
                "Avoid overhead irrigation immediately",
                "Improve row spacing for air circulation",
                "Apply fungicide within 24 hours of detection",
            ],
            "chemical_treatments": [
                "Copper Oxychloride 50% WP @ 3g/L — broad-spectrum, apply every 7 days",
                "Mancozeb 75% WP @ 2.5g/L — preventive & curative, 3 sprays at 10-day intervals",
                "Azoxystrobin 23% SC @ 1ml/L — systemic, apply at first symptom",
                "Propiconazole 25% EC @ 1ml/L — curative, 2 sprays at 14-day intervals",
            ],
            "organic_treatments": [
                "Bordeaux mixture 1% (100g CuSO4 + 100g lime / 10L water)",
                "Trichoderma viride @ 5g/L soil drench + foliar spray",
                "Garlic extract spray (50g crushed garlic / 1L water, dilute 1:10)",
            ],
            "fertilizer_schedule": "Reduce nitrogen; apply Potassium Sulphate @ 3g/L to strengthen cell walls",
            "prevention": "Use certified disease-free seeds, practice 2-year crop rotation, apply preventive copper spray before monsoon, avoid working in wet fields.",
            "recovery_time": "7–14 days with prompt treatment",
            "economic_impact": "20–40% yield loss if untreated",
        },
        "powdery_mildew": {
            "status": "Powdery Mildew",
            "severity": "medium",
            "severity_score": 50,
            "color": "#f59e0b",
            "icon": "🌫️",
            "affected_parts": ["Leaves", "Shoots", "Buds"],
            "description": "Powdery mildew presents as white to gray powdery fungal colonies on leaf surfaces. Unlike most fungi, it thrives in dry conditions with moderate humidity (40–70% RH) and temperatures of 15–28°C.",
            "causes": ["Erysiphe spp. / Podosphaera spp. (obligate fungal parasites)", "Dry weather with moderate humidity", "Excess nitrogen causing succulent growth", "Dense shaded canopy"],
            "immediate_actions": [
                "Remove heavily infected leaves",
                "Increase plant spacing to improve airflow",
                "Avoid excess nitrogen fertilization",
                "Apply sulphur-based fungicide immediately",
            ],
            "chemical_treatments": [
                "Wettable Sulphur 80% WP @ 2g/L — contact fungicide, apply every 7–10 days",
                "Hexaconazole 5% SC @ 2ml/L — systemic triazole, 2 sprays at 14-day intervals",
                "Myclobutanil 10% WP @ 1g/L — highly effective systemic",
                "Tebuconazole 25.9% EC @ 1ml/L — curative, apply at early infection",
            ],
            "organic_treatments": [
                "Baking soda spray: 5g NaHCO3 + 2ml liquid soap / 1L water",
                "Neem oil 5% EC @ 5ml/L — disrupts fungal spore germination",
                "Milk spray: 40% fresh milk in water — proven efficacy in trials",
                "Potassium bicarbonate 5g/L — raises leaf surface pH inhibiting spores",
            ],
            "fertilizer_schedule": "Reduce nitrogen; apply Calcium Nitrate @ 2g/L to strengthen cell walls",
            "prevention": "Plant resistant varieties, maintain proper spacing, avoid late-evening irrigation, prune for airflow.",
            "recovery_time": "10–21 days with consistent treatment",
            "economic_impact": "10–30% yield reduction; affects fruit quality",
        },
        "rust": {
            "status": "Rust Disease",
            "severity": "high",
            "severity_score": 70,
            "color": "#ef4444",
            "icon": "🟤",
            "affected_parts": ["Leaves", "Stems", "Pods"],
            "description": "Rust disease produces orange-brown to reddish-brown pustules (uredinia) on leaf undersides with corresponding yellow spots on upper surfaces. Wind-dispersed urediniospores can spread kilometers, causing epidemic outbreaks.",
            "causes": ["Puccinia spp. (obligate fungal parasite)", "Cool moist weather (15–22°C)", "Wind dispersal of urediniospores", "Susceptible host varieties", "Volunteer plants as inoculum source"],
            "immediate_actions": [
                "Apply triazole fungicide within 48 hours",
                "Remove and destroy heavily infected plant material",
                "Monitor neighboring fields for spread",
                "Avoid moving equipment between infected and clean fields",
            ],
            "chemical_treatments": [
                "Propiconazole 25% EC @ 1ml/L — highly effective triazole, 2 sprays at 14-day intervals",
                "Tebuconazole 25.9% EC @ 1ml/L — systemic, apply at first pustule appearance",
                "Trifloxystrobin 25% + Tebuconazole 50% WG @ 0.5g/L — combination, excellent control",
                "Mancozeb 75% WP @ 2.5g/L — preventive, apply before infection period",
            ],
            "organic_treatments": [
                "Sulphur dust 80% WP @ 3g/L — effective at early stages",
                "Neem oil 5% EC @ 5ml/L — reduces spore germination",
                "Garlic + chili extract spray as repellent",
            ],
            "fertilizer_schedule": "Apply Potassium Sulphate @ 5g/L to boost plant immunity; avoid excess nitrogen",
            "prevention": "Plant rust-resistant varieties, apply preventive fungicide at flag leaf stage, destroy crop residues after harvest.",
            "recovery_time": "14–21 days; severe infections may not fully recover",
            "economic_impact": "30–70% yield loss in susceptible varieties",
        },
        "leaf_spot": {
            "status": "Leaf Spot",
            "severity": "medium",
            "severity_score": 45,
            "color": "#f59e0b",
            "icon": "🔴",
            "affected_parts": ["Leaves"],
            "description": "Leaf spot diseases produce circular to irregular necrotic spots (2–15mm) with distinct dark borders and lighter centers. Multiple spots coalesce causing large dead areas and premature defoliation.",
            "causes": ["Cercospora spp. / Septoria spp. (fungal)", "Bacterial pathogens (Pseudomonas, Xanthomonas)", "Wet foliage from rain or irrigation", "Infected seeds or soil debris", "Mechanical injuries creating entry points"],
            "immediate_actions": [
                "Remove infected leaves and dispose away from field",
                "Switch to drip irrigation to keep foliage dry",
                "Apply contact fungicide immediately",
                "Disinfect pruning tools with 70% alcohol",
            ],
            "chemical_treatments": [
                "Chlorothalonil 75% WP @ 2g/L — broad-spectrum contact, apply every 7–10 days",
                "Carbendazim 50% WP @ 1g/L — systemic benzimidazole, 3 sprays at 10-day intervals",
                "Iprodione 50% WP @ 1.5g/L — effective against Botrytis and Alternaria",
                "Copper Hydroxide 77% WP @ 2g/L — bactericidal + fungicidal",
            ],
            "organic_treatments": [
                "Bordeaux mixture 0.5% as preventive spray",
                "Trichoderma harzianum @ 5g/L foliar spray",
                "Neem cake extract @ 10g/L soil application",
            ],
            "fertilizer_schedule": "Balanced NPK; increase Potassium to 60 kg/ha to improve disease resistance",
            "prevention": "Use certified disease-free seeds, treat seeds with Thiram 75% WS @ 3g/kg before sowing, practice crop rotation.",
            "recovery_time": "7–14 days with treatment",
            "economic_impact": "15–25% yield loss; reduces photosynthetic area",
        },
        "bacterial_wilt": {
            "status": "Bacterial Wilt",
            "severity": "critical",
            "severity_score": 95,
            "color": "#dc2626",
            "icon": "⚠️",
            "affected_parts": ["Vascular system", "Stems", "Roots"],
            "description": "Bacterial wilt causes sudden, irreversible wilting of entire plants despite adequate soil moisture. The pathogen colonizes xylem vessels, blocking water transport. Cut stems show brown discoloration and bacterial ooze (thread test positive).",
            "causes": ["Ralstonia solanacearum (Race 1, 3) — soil-borne bacterium", "Infected soil or irrigation water", "Cucumber beetle / nematode vectors creating entry wounds", "Contaminated tools and equipment", "Waterlogged conditions favoring pathogen spread"],
            "immediate_actions": [
                "⚠️ REMOVE AND DESTROY infected plants immediately — no chemical cure exists",
                "Do NOT compost infected material — burn or deep bury",
                "Quarantine the affected area — mark with stakes",
                "Disinfect all tools with 10% bleach or 70% alcohol",
                "Stop irrigation in affected zone for 48 hours",
            ],
            "chemical_treatments": [
                "Copper Oxychloride 50% WP @ 3g/L soil drench — suppressive only, not curative",
                "Streptomycin Sulphate 90% SP @ 0.5g/L — bacteriostatic, apply to surrounding healthy plants",
                "Kasugamycin 3% SL @ 2ml/L — preventive application to healthy plants",
            ],
            "organic_treatments": [
                "Trichoderma viride @ 5g/kg soil — biocontrol agent, apply to healthy surrounding soil",
                "Pseudomonas fluorescens @ 10g/L soil drench — competitive exclusion",
                "Soil solarization: cover with clear plastic for 4–6 weeks before replanting",
                "Neem cake @ 250kg/ha soil incorporation before planting",
            ],
            "fertilizer_schedule": "Do not fertilize infected plants. For surrounding healthy plants: reduce nitrogen, increase Potassium and Calcium",
            "prevention": "Use resistant varieties (grafted rootstocks), practice 3-year rotation with non-solanaceous crops, avoid waterlogging, sterilize tools between plants.",
            "recovery_time": "No recovery — infected plants must be removed. Field may need 2–3 year rotation.",
            "economic_impact": "Up to 100% loss in affected areas; soil remains infested for years",
        },
        "mosaic_virus": {
            "status": "Mosaic Virus",
            "severity": "high",
            "severity_score": 72,
            "color": "#ef4444",
            "icon": "🦠",
            "affected_parts": ["Leaves", "Fruits", "Whole plant"],
            "description": "Mosaic virus causes characteristic mottled yellow-green patterns on leaves, leaf distortion (curling, blistering), stunted growth, and reduced fruit quality. Spread primarily by aphid vectors in a non-persistent manner.",
            "causes": ["Tobacco Mosaic Virus (TMV), Cucumber Mosaic Virus (CMV), Bean Yellow Mosaic Virus (BYMV)", "Aphid vectors (Myzus persicae, Aphis gossypii)", "Infected seeds or transplants", "Mechanical transmission via contaminated tools and hands", "Weed reservoirs (Datura, Solanum spp.)"],
            "immediate_actions": [
                "Remove and destroy infected plants immediately",
                "Control aphid vectors urgently with systemic insecticide",
                "Disinfect all tools with 10% trisodium phosphate (TSP) solution",
                "Install reflective silver mulch to deter aphids",
                "Remove weed hosts from field borders",
            ],
            "chemical_treatments": [
                "Thiamethoxam 25% WG @ 0.3g/L — systemic aphicide, apply every 10 days",
                "Imidacloprid 17.8% SL @ 0.5ml/L — soil drench for aphid control",
                "Acetamiprid 20% SP @ 0.3g/L — foliar spray for aphid vectors",
                "Mineral oil spray @ 10ml/L — interferes with aphid probing behavior",
            ],
            "organic_treatments": [
                "Neem oil 5% EC @ 5ml/L — repels aphid vectors",
                "Insecticidal soap @ 5ml/L — contact aphicide",
                "Yellow sticky traps @ 10 traps/acre for aphid monitoring",
                "Release Aphidius colemani (parasitic wasp) as biocontrol",
            ],
            "fertilizer_schedule": "Avoid excess nitrogen (promotes aphid-attractive succulent growth). Apply Zinc Sulphate @ 0.5g/L to boost plant immunity.",
            "prevention": "Use virus-indexed certified seeds, plant resistant varieties, maintain 50m isolation distance from infected fields, control aphids preventively.",
            "recovery_time": "No cure — infected plants remain infected. Remove to prevent spread.",
            "economic_impact": "25–50% yield loss; fruit quality severely affected",
        },
        "downy_mildew": {
            "status": "Downy Mildew",
            "severity": "high",
            "severity_score": 68,
            "color": "#ef4444",
            "icon": "💧",
            "affected_parts": ["Leaves", "Stems", "Fruits"],
            "description": "Downy mildew produces angular yellow patches on upper leaf surfaces (limited by leaf veins) with characteristic grayish-purple sporulation on undersides. It is an oomycete (water mold), not a true fungus, requiring different chemistry for control.",
            "causes": ["Peronospora spp. / Plasmopara viticola / Pseudoperonospora cubensis (oomycetes)", "Cool temperatures 10–20°C with high humidity >85% RH", "Prolonged leaf wetness from rain or dew", "Poor air circulation in dense canopy"],
            "immediate_actions": [
                "Apply oomycete-specific fungicide (metalaxyl-based) immediately",
                "Remove and destroy infected leaves",
                "Improve canopy ventilation by pruning",
                "Avoid overhead irrigation — switch to drip",
            ],
            "chemical_treatments": [
                "Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold) @ 2.5g/L — highly effective, apply every 7–10 days",
                "Fosetyl-Al 80% WP @ 2.5g/L — systemic, translocates to roots",
                "Cymoxanil 8% + Mancozeb 64% WP @ 2.5g/L — curative + protective",
                "Dimethomorph 50% WP @ 1g/L — excellent curative activity",
                "Copper Oxychloride 50% WP @ 3g/L — preventive, apply before infection period",
            ],
            "organic_treatments": [
                "Bordeaux mixture 1% — traditional copper-based preventive",
                "Potassium phosphonate @ 3ml/L — induces systemic resistance",
                "Bacillus subtilis-based biocontrol @ 5g/L",
            ],
            "fertilizer_schedule": "Apply Calcium Chloride @ 2g/L foliar spray to strengthen cell walls; reduce nitrogen",
            "prevention": "Plant resistant varieties, ensure good drainage, apply preventive copper sprays before wet seasons, maintain proper plant spacing.",
            "recovery_time": "10–18 days with appropriate oomycide treatment",
            "economic_impact": "30–60% yield loss in susceptible crops during wet seasons",
        },
        "anthracnose": {
            "status": "Anthracnose",
            "severity": "medium",
            "severity_score": 55,
            "color": "#f59e0b",
            "icon": "🌑",
            "affected_parts": ["Leaves", "Stems", "Fruits", "Seeds"],
            "description": "Anthracnose causes dark, sunken, water-soaked lesions that turn brown-black with age. In humid conditions, salmon-pink spore masses (acervuli) appear in lesion centers. Affects both pre- and post-harvest.",
            "causes": ["Colletotrichum gloeosporioides / C. acutatum (fungal)", "Warm humid weather 25–30°C", "Infected seeds and plant debris", "Overhead irrigation and rain splash", "Wounds from insects or mechanical damage"],
            "immediate_actions": [
                "Remove and destroy all infected plant parts",
                "Apply systemic fungicide immediately",
                "Avoid wetting foliage during irrigation",
                "Harvest fruits at early maturity to avoid post-harvest losses",
            ],
            "chemical_treatments": [
                "Azoxystrobin 23% SC @ 1ml/L — systemic strobilurin, excellent efficacy",
                "Carbendazim 50% WP @ 1g/L — systemic benzimidazole, 3 sprays at 10-day intervals",
                "Difenoconazole 25% EC @ 0.5ml/L — triazole, highly effective",
                "Copper Hydroxide 77% WP @ 2g/L — contact, apply every 7 days",
            ],
            "organic_treatments": [
                "Trichoderma asperellum @ 5g/L foliar spray",
                "Neem oil 5% EC @ 5ml/L",
                "Hot water seed treatment: 52°C for 30 minutes before sowing",
            ],
            "fertilizer_schedule": "Balanced NPK; apply Silicon @ 1g/L foliar spray to strengthen cell walls against penetration",
            "prevention": "Use disease-free certified seeds, hot water seed treatment, practice crop rotation, remove crop debris after harvest.",
            "recovery_time": "10–14 days with treatment",
            "economic_impact": "15–35% yield loss; significant post-harvest losses",
        },
        "root_rot": {
            "status": "Root Rot",
            "severity": "critical",
            "severity_score": 88,
            "color": "#dc2626",
            "icon": "🌿",
            "affected_parts": ["Roots", "Crown", "Lower stem"],
            "description": "Root rot causes progressive yellowing, wilting, and stunted growth despite adequate moisture. Roots appear brown to black, soft, and mushy with a foul odor. The crown may show dark water-soaked lesions. Often misdiagnosed as drought stress.",
            "causes": ["Pythium spp. / Phytophthora spp. / Fusarium spp. (soil-borne pathogens)", "Waterlogged or poorly drained soil", "Overwatering or compacted soil", "Contaminated irrigation water", "Infected transplants or soil"],
            "immediate_actions": [
                "Immediately reduce irrigation frequency",
                "Improve soil drainage — create furrows or raised beds",
                "Apply systemic fungicide as soil drench",
                "Remove severely affected plants to prevent spread",
                "Aerate soil around plant base",
            ],
            "chemical_treatments": [
                "Metalaxyl 35% WS @ 2g/L soil drench — highly effective against Pythium/Phytophthora",
                "Fosetyl-Al 80% WP @ 3g/L soil drench — systemic, translocates to roots",
                "Carbendazim 50% WP @ 1g/L soil drench — effective against Fusarium",
                "Propamocarb 72.2% SL @ 3ml/L — specific oomycide",
            ],
            "organic_treatments": [
                "Trichoderma harzianum @ 5g/kg soil — biocontrol, apply before planting and at first symptoms",
                "Pseudomonas fluorescens @ 10g/L soil drench",
                "Neem cake @ 250kg/ha soil incorporation",
                "Cinnamon powder @ 5g/L soil drench — natural antifungal",
            ],
            "fertilizer_schedule": "Avoid nitrogen fertilization until recovery. Apply Calcium Superphosphate @ 3g/L to strengthen roots. Potassium @ 5g/L to improve stress tolerance.",
            "prevention": "Ensure proper drainage before planting, use raised beds in heavy soils, treat seeds with Thiram, avoid overwatering, use disease-free transplants.",
            "recovery_time": "14–28 days if caught early; severe cases may not recover",
            "economic_impact": "40–80% plant mortality in affected areas",
        },
        "nutrient_deficiency": {
            "status": "Nutrient Deficiency",
            "severity": "medium",
            "severity_score": 40,
            "color": "#f59e0b",
            "icon": "🌾",
            "affected_parts": ["Leaves", "Whole plant"],
            "description": "Nutrient deficiency manifests as chlorosis (yellowing), necrosis, purple discoloration, or stunted growth. Symptom patterns indicate specific deficiencies: interveinal chlorosis (Fe/Mn), lower leaf yellowing (N), purple coloration (P), leaf margin scorch (K).",
            "causes": ["Insufficient fertilization or poor fertilizer quality", "Soil pH imbalance (outside 6.0–7.0 range)", "Poor root development limiting nutrient uptake", "Leaching in sandy or waterlogged soils", "Antagonistic nutrient interactions (excess P locks Fe/Zn)"],
            "immediate_actions": [
                "Conduct soil and leaf tissue test to identify specific deficiency",
                "Correct soil pH if outside 6.0–7.0 range",
                "Apply targeted foliar spray for rapid correction",
                "Improve soil organic matter with compost",
            ],
            "chemical_treatments": [
                "Nitrogen deficiency: Urea 1% foliar spray (10g/L) — apply every 7 days",
                "Iron deficiency: Ferrous Sulphate 0.5% + Citric acid 0.1% foliar spray",
                "Magnesium deficiency: Magnesium Sulphate (Epsom salt) 1% foliar spray",
                "Zinc deficiency: Zinc Sulphate 0.5% foliar spray",
                "Balanced micronutrient: Multi-micronutrient mix @ 2g/L foliar spray",
                "Complete correction: NPK 19:19:19 @ 5g/L foliar spray",
            ],
            "organic_treatments": [
                "Vermicompost @ 2 tonnes/acre soil application",
                "Seaweed extract @ 3ml/L foliar spray — broad micronutrient source",
                "Fish emulsion @ 5ml/L — nitrogen-rich organic fertilizer",
                "Bone meal @ 100kg/acre for phosphorus deficiency",
            ],
            "fertilizer_schedule": "Soil test first. General: NPK 120:60:60 kg/ha basal + 60kg N top-dress at 30 and 60 days",
            "prevention": "Conduct soil testing before each season, maintain pH 6.0–7.0, apply balanced fertilizers, use organic matter to improve nutrient retention.",
            "recovery_time": "5–10 days for foliar-applied nutrients; 2–4 weeks for soil-applied",
            "economic_impact": "10–40% yield reduction depending on severity and growth stage",
        },
        "pest_damage": {
            "status": "Pest Damage",
            "severity": "medium",
            "severity_score": 52,
            "color": "#f59e0b",
            "icon": "🐛",
            "affected_parts": ["Leaves", "Stems", "Fruits", "Roots"],
            "description": "Pest damage presents as irregular holes, chewed leaf margins, stippling (tiny dots from sucking insects), leaf curling, tunneling, or galls. Damage patterns help identify the pest: shot-hole (beetles), stippling (mites/thrips), curling (aphids), tunneling (leaf miners).",
            "causes": ["Chewing insects: caterpillars, beetles, grasshoppers", "Sucking insects: aphids, whiteflies, thrips, mites", "Boring insects: stem borers, leaf miners", "Warm dry conditions favoring pest buildup", "Absence of natural predators due to pesticide overuse"],
            "immediate_actions": [
                "Identify the specific pest before applying pesticide",
                "Remove heavily infested plant parts",
                "Install yellow/blue sticky traps for monitoring",
                "Apply targeted pesticide based on pest identification",
            ],
            "chemical_treatments": [
                "Caterpillars/borers: Spinosad 45% SC @ 0.3ml/L — selective, low mammalian toxicity",
                "Sucking pests (aphids/whiteflies): Imidacloprid 17.8% SL @ 0.5ml/L",
                "Mites: Abamectin 1.8% EC @ 1ml/L or Spiromesifen 22.9% SC @ 1ml/L",
                "Thrips: Fipronil 5% SC @ 1.5ml/L or Spinosad 45% SC @ 0.3ml/L",
                "Leaf miners: Cyromazine 75% WP @ 0.75g/L",
            ],
            "organic_treatments": [
                "Neem oil 5% EC @ 5ml/L — broad-spectrum, disrupts insect development",
                "Bacillus thuringiensis (Bt) @ 2g/L — specific to caterpillars",
                "Beauveria bassiana @ 5g/L — entomopathogenic fungus",
                "Insecticidal soap @ 5ml/L — contact action on soft-bodied insects",
                "Yellow sticky traps @ 10/acre for whiteflies and aphids",
            ],
            "fertilizer_schedule": "Avoid excess nitrogen (promotes aphid-attractive growth). Apply Silicon @ 1g/L to strengthen leaf tissue against chewing insects.",
            "prevention": "Implement IPM: regular scouting, economic threshold-based spraying, conservation of natural enemies, crop rotation, and resistant varieties.",
            "recovery_time": "7–14 days after pest control; new growth replaces damaged tissue",
            "economic_impact": "15–45% yield loss depending on pest species and infestation level",
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
# DATABASE — SQLite
# ─────────────────────────────────────────────
import sqlite3

def init_db():
    conn = sqlite3.connect("crophealth.db")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            correct_label TEXT NOT NULL,
            user_comment TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(prediction_id)
        )
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized")

init_db()

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect("crophealth.db")
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db: db.close()

def execute_query(sql, params=None, fetch=False, fetchone=False, commit=False):
    conn = get_db()
    cur = conn.cursor()
    sql = sql.replace("%s", "?")
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
    for k, v in row.items():
        if isinstance(v, datetime): out[k] = v.isoformat()
        elif isinstance(v, (bytes, bytearray)): out[k] = bool(v)
        else: out[k] = v
    return out

def serialize_rows(rows): return [serialize_row(r) for r in (rows or [])]


# ─────────────────────────────────────────────
# IMAGE PROCESSOR
# ─────────────────────────────────────────────
class ImageProcessor:
    def __init__(self, image_bytes: bytes):
        self.raw = image_bytes
        self.orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        self.proc = None
        self.thumb = None
        self.meta = {
            "original_width": self.orig.width, "original_height": self.orig.height,
            "mode": self.orig.mode, "format": getattr(self.orig, "format", "unknown"),
            "aspect_ratio": round(self.orig.width / max(self.orig.height, 1), 3),
            "megapixels": round(self.orig.width * self.orig.height / 1_000_000, 3),
            "file_size": len(image_bytes), "file_size_label": size_label(len(image_bytes)),
            "file_hash": compute_hash(image_bytes),
        }

    def resize(self, size=None):
        self.proc = self.orig.resize(size or Config.IMG_SIZE, Image.LANCZOS)
        return self

    def make_thumbnail(self):
        t = self.orig.copy(); t.thumbnail(Config.THUMB_SIZE, Image.LANCZOS); self.thumb = t
        return self

    def to_array(self):
        if not self.proc: self.resize()
        return np.array(self.proc, dtype=np.float32) / 255.0

    def to_gray(self):
        if not self.proc: self.resize()
        return np.array(self.proc.convert("L"), dtype=np.float32) / 255.0

    def get_thumbnail_b64(self):
        if not self.thumb: self.make_thumbnail()
        buf = io.BytesIO(); self.thumb.save(buf, format="JPEG", quality=75)
        import base64; return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def get_histogram(self):
        if not self.proc: self.resize()
        r, g, b = self.proc.split()
        return {"r": list(r.histogram()), "g": list(g.histogram()), "b": list(b.histogram())}

    def get_dominant_color(self):
        arr = self.to_array()
        return (float(np.mean(arr[:,:,0])), float(np.mean(arr[:,:,1])), float(np.mean(arr[:,:,2])))

    def extract_features(self) -> np.ndarray:
        """Extract 32-dimensional feature vector optimised for plant disease detection."""
        if not self.proc: self.resize()
        arr = self.to_array()
        gray = self.to_gray()

        # ── RGB channel stats ──
        r_mean, g_mean, b_mean = float(np.mean(arr[:,:,0])), float(np.mean(arr[:,:,1])), float(np.mean(arr[:,:,2]))
        r_std,  g_std,  b_std  = float(np.std(arr[:,:,0])),  float(np.std(arr[:,:,1])),  float(np.std(arr[:,:,2]))

        # ── HSV-like features (plant health indicators) ──
        # Green dominance — healthy leaves are green-dominant
        green_dominance = float(g_mean - (r_mean + b_mean) / 2)
        # Yellow index — yellowing leaves (N deficiency, virus)
        yellow_index = float((r_mean + g_mean) / 2 - b_mean)
        # Brown index — blight, rust, necrosis
        brown_index = float(r_mean - g_mean)
        # White/gray index — powdery mildew
        white_index = float(min(r_mean, g_mean, b_mean))

        # ── Texture features ──
        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))
        gray_range = float(np.max(gray) - np.min(gray))

        # Sharpness via gradient magnitude
        dx = np.diff(gray, axis=1); dy = np.diff(gray, axis=0)
        sharpness = float(np.mean(np.sqrt(dx[:gray.shape[0]-1,:]**2 + dy[:,:gray.shape[1]-1]**2)))

        # Saturation (distance from achromatic)
        saturation = float(np.mean(np.sqrt((arr[:,:,0]-brightness)**2 + (arr[:,:,1]-brightness)**2 + (arr[:,:,2]-brightness)**2)))

        # Noise estimate
        blurred = np.array(self.proc.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255.0
        noise = float(np.mean(np.abs(arr - blurred)))

        # Entropy (texture complexity — diseased leaves have higher entropy)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,1))
        hist = hist / (hist.sum() + 1e-9)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))

        # Edge density (lesions create high edge density)
        edges = np.array(self.proc.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        edge_density = float(np.mean(edges))

        # Spot density (dark spots — leaf spot, anthracnose)
        dark_mask = (gray < 0.25).astype(np.float32)
        spot_density = float(np.mean(dark_mask))

        # Light spot density (powdery mildew, nutrient deficiency)
        light_mask = (gray > 0.80).astype(np.float32)
        light_spot_density = float(np.mean(light_mask))

        # Color uniformity (diseased leaves are less uniform)
        color_uniformity = float(1.0 - np.std([r_std, g_std, b_std]))

        # Warm-cool ratio (rust = warm, downy mildew = cool)
        warm_cool = float(r_mean - b_mean)

        # Histogram spread per channel
        hr, _ = np.histogram(arr[:,:,0].flatten(), bins=32, range=(0,1))
        hg, _ = np.histogram(arr[:,:,1].flatten(), bins=32, range=(0,1))
        hb, _ = np.histogram(arr[:,:,2].flatten(), bins=32, range=(0,1))
        spread_r = float(np.std(hr / (hr.sum() + 1e-9)))
        spread_g = float(np.std(hg / (hg.sum() + 1e-9)))
        spread_b = float(np.std(hb / (hb.sum() + 1e-9)))

        # Symmetry (healthy leaves tend to be more symmetric)
        left = arr[:, :arr.shape[1]//2, :]; right = arr[:, arr.shape[1]//2:, :]
        min_w = min(left.shape[1], right.shape[1])
        symmetry = float(1.0 - np.mean(np.abs(left[:,:min_w,:] - right[:,:min_w,:][:,::-1,:])))

        # Aspect ratio & size
        aspect = float(self.meta["aspect_ratio"])
        megapix = float(self.meta["megapixels"])

        return np.array([
            r_mean, g_mean, b_mean, r_std, g_std, b_std,
            green_dominance, yellow_index, brown_index, white_index,
            brightness, contrast, gray_range, sharpness, saturation, noise,
            entropy, edge_density, spot_density, light_spot_density,
            color_uniformity, warm_cool, spread_r, spread_g, spread_b,
            symmetry, aspect, megapix,
            float(self.meta["original_width"]), float(self.meta["original_height"]),
            float(len(self.raw)) / 1_000_000,
            float(g_mean / max(r_mean + b_mean, 0.001)),  # green ratio
        ], dtype=np.float32)


# ─────────────────────────────────────────────
# ML MODEL REGISTRY
# ─────────────────────────────────────────────
class ModelRegistry:
    N_FEATURES = 32
    N_SAMPLES  = 1200

    def __init__(self):
        self.pipelines: dict = {}
        self.training_stats: dict = {}
        self.active_model = "random_forest"
        self._lock = threading.Lock()
        self._build_data()
        self._train_all()
        logger.info("ModelRegistry ready. Active: %s", self.active_model)

    def _build_data(self):
        """Build synthetic training data with disease-specific feature signals."""
        np.random.seed(42)
        labels = Config.LABELS
        n_per = self.N_SAMPLES // len(labels)
        # Feature indices:
        # 0=r_mean,1=g_mean,2=b_mean,3=r_std,4=g_std,5=b_std
        # 6=green_dom,7=yellow_idx,8=brown_idx,9=white_idx
        # 10=brightness,11=contrast,12=gray_range,13=sharpness,14=saturation,15=noise
        # 16=entropy,17=edge_density,18=spot_density,19=light_spot_density
        # 20=color_uniformity,21=warm_cool,22=spread_r,23=spread_g,24=spread_b
        # 25=symmetry,26=aspect,27=megapix,28=width,29=height,30=filesize,31=green_ratio
        signals = {
            "healthy":             {1:0.58, 6:0.28, 10:0.48, 20:0.78, 25:0.82, 31:1.25},
            "leaf_blight":         {8:0.28, 11:0.40, 17:0.32, 18:0.22, 16:7.0},
            "powdery_mildew":      {9:0.38, 10:0.75, 19:0.32, 14:0.06, 11:0.10},
            "rust":                {0:0.55, 8:0.30, 21:0.32, 17:0.24, 7:0.22},
            "leaf_spot":           {11:0.44, 17:0.37, 18:0.30, 16:6.7, 15:0.24},
            "bacterial_wilt":      {10:0.30, 11:0.17, 14:0.10, 25:0.42, 6:-0.08},
            "mosaic_virus":        {4:0.40, 14:0.44, 16:7.4, 11:0.34, 7:0.27},
            "downy_mildew":        {5:0.40, 10:0.37, 15:0.27, 17:0.20, 9:0.22},
            "anthracnose":         {0:0.20, 1:0.20, 2:0.20, 11:0.47, 18:0.27},
            "root_rot":            {10:0.24, 0:0.40, 14:0.14, 15:0.32, 6:-0.13},
            "nutrient_deficiency": {7:0.32, 1:0.50, 10:0.57, 14:0.17, 6:0.07},
            "pest_damage":         {17:0.42, 11:0.37, 16:7.0, 18:0.32, 25:0.37},
        }
        X_parts, y_parts = [], []
        for label in labels:
            base = np.random.rand(n_per, self.N_FEATURES).astype(np.float32) * 0.22 + 0.09
            for fi, val in signals.get(label, {}).items():
                base[:, fi] = np.clip(val + np.random.randn(n_per) * 0.038, -1, 2).astype(np.float32)
            X_parts.append(base); y_parts.extend([label] * n_per)
        self.X_train = np.vstack(X_parts)
        self.y_train = np.array(y_parts)
        logger.info("Training data: %s, %d classes", self.X_train.shape, len(labels))

    def _make_pipe(self, clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def _train_all(self):
        defs = {
            "knn":                 KNeighborsClassifier(n_neighbors=11, weights="distance", metric="euclidean"),
            "random_forest":       RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_leaf=2, min_samples_split=4, random_state=42, n_jobs=-1),
            "logistic_regression": LogisticRegression(max_iter=1000, C=2.0, multi_class="multinomial", solver="lbfgs", random_state=42),
            "gradient_boosting":   GradientBoostingClassifier(n_estimators=180, max_depth=6, learning_rate=0.1, subsample=0.9, random_state=42),
            "svm":                 SVC(kernel="rbf", C=2.5, gamma="scale", probability=True, random_state=42),
        }
        for name, clf in defs.items():
            try:
                t0 = time.time()
                pipe = self._make_pipe(clf)
                pipe.fit(self.X_train, self.y_train)
                elapsed = time.time() - t0
                acc = pipe.score(self.X_train, self.y_train)
                self.pipelines[name] = pipe
                self.training_stats[name] = {
                    "train_accuracy": round(float(acc), 4),
                    "train_time_sec": round(elapsed, 3),
                    "n_samples": len(self.X_train),
                    "n_features": self.N_FEATURES,
                    "trained_at": datetime.utcnow().isoformat() + "Z",
                }
                logger.info("Trained [%s] acc=%.3f in %.2fs", name, acc, elapsed)
            except Exception as e:
                logger.error("Train failed [%s]: %s", name, e)

    def predict(self, features: np.ndarray, model_name: str = None) -> dict:
        name = model_name or self.active_model
        if name not in self.pipelines:
            raise ValueError(f"Model '{name}' not found")
        pipe = self.pipelines[name]
        X = features.reshape(1, -1)
        pred = pipe.predict(X)[0]
        probas = pipe.predict_proba(X)[0]
        classes = pipe.classes_
        top3 = sorted(zip(classes, probas), key=lambda x: -x[1])[:3]
        return {
            "prediction": pred,
            "confidence": round(float(np.max(probas)) * 100, 2),
            "model_used": name,
            "top3": [{"label": str(l), "probability": round(float(p)*100, 2)} for l, p in top3],
            "all_probabilities": {str(l): round(float(p)*100, 2) for l, p in zip(classes, probas)},
        }

    def predict_ensemble(self, features: np.ndarray) -> dict:
        X = features.reshape(1, -1)
        all_p, classes = [], None
        for name, pipe in self.pipelines.items():
            try:
                all_p.append(pipe.predict_proba(X)[0])
                if classes is None: classes = pipe.classes_
            except Exception: pass
        if not all_p: raise RuntimeError("No models available")
        avg = np.mean(all_p, axis=0)
        best = int(np.argmax(avg))
        top3 = sorted(zip(classes, avg), key=lambda x: -x[1])[:3]
        return {
            "prediction": classes[best],
            "confidence": round(float(avg[best])*100, 2),
            "model_used": "ensemble",
            "top3": [{"label": str(l), "probability": round(float(p)*100, 2)} for l, p in top3],
            "all_probabilities": {str(l): round(float(p)*100, 2) for l, p in zip(classes, avg)},
        }

    def get_info(self) -> dict:
        return {
            "available_models": list(self.pipelines.keys()),
            "active_model": self.active_model,
            "training_stats": self.training_stats,
            "feature_version": Config.FEATURE_VER,
            "n_classes": len(Config.LABELS),
            "classes": Config.LABELS,
        }

    def set_active(self, name: str):
        if name not in self.pipelines: raise ValueError(f"Unknown model: {name}")
        with self._lock: self.active_model = name

    def evaluate(self, name: str) -> dict:
        if name not in self.pipelines: raise ValueError(f"Unknown model: {name}")
        preds = self.pipelines[name].predict(self.X_train)
        report = classification_report(self.y_train, preds, output_dict=True, zero_division=0)
        return {"model": name, "classification_report": report}

model_registry = ModelRegistry()


# ─────────────────────────────────────────────
# DATA ACCESS OBJECTS
# ─────────────────────────────────────────────
class PredictionDAO:
    @staticmethod
    def insert(image_name, prediction, confidence, model_used, file_hash,
               file_size, orig_width, orig_height, top3_json, all_proba_json,
               feature_vector_json, processing_time_ms, thumbnail=None):
        sql = """INSERT INTO predictions
                   (image_name, prediction_result, confidence, model_used,
                    file_hash, file_size, original_width, original_height,
                    top3_predictions, all_probabilities, feature_vector,
                    processing_time_ms, feature_version, thumbnail)
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        return execute_query(sql, (
            str(image_name), str(prediction), float(confidence), str(model_used),
            str(file_hash), int(file_size), int(orig_width), int(orig_height),
            str(top3_json), str(all_proba_json), str(feature_vector_json),
            float(processing_time_ms), Config.FEATURE_VER, thumbnail
        ), commit=True)

    @staticmethod
    def get_by_id(pred_id):
        return execute_query(
            """SELECT p.*, f.correct_label AS feedback_label,
                      (CASE WHEN f.correct_label=p.prediction_result THEN 1 ELSE 0 END) AS was_correct
               FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id
               WHERE p.id = %s""", (pred_id,), fetchone=True)

    @staticmethod
    def get_by_hash(file_hash):
        return execute_query(
            "SELECT * FROM predictions WHERE file_hash=%s ORDER BY created_at DESC LIMIT 1",
            (file_hash,), fetchone=True)

    @staticmethod
    def get_stats():
        total  = execute_query("SELECT COUNT(*) AS cnt FROM predictions", fetchone=True)
        today  = execute_query("SELECT COUNT(*) AS cnt FROM predictions WHERE DATE(created_at)=DATE('now')", fetchone=True)
        week   = execute_query("SELECT COUNT(*) AS cnt FROM predictions WHERE created_at>=datetime('now','-7 days')", fetchone=True)
        avg_t   = execute_query("SELECT AVG(processing_time_ms) AS avg_ms FROM predictions", fetchone=True)
        by_lbl  = execute_query("SELECT prediction_result, COUNT(*) AS cnt, AVG(confidence) AS avg_conf FROM predictions GROUP BY prediction_result ORDER BY cnt DESC", fetch=True)
        by_mdl  = execute_query("SELECT model_used, COUNT(*) AS cnt FROM predictions GROUP BY model_used", fetch=True)
        return {
            "total_predictions": total["cnt"] if total else 0,
            "today_predictions": today["cnt"] if today else 0,
            "week_predictions":  week["cnt"]  if week  else 0,
            "avg_processing_ms": round(float(avg_t["avg_ms"] or 0), 2) if avg_t else 0,
            "by_label": by_lbl or [],
            "by_model": by_mdl or [],
        }

    @staticmethod
    def search(query, limit=20):
        like = f"%{query}%"
        return execute_query(
            """SELECT p.id, p.image_name, p.prediction_result, p.confidence,
                      p.model_used, p.processing_time_ms, p.created_at,
                      f.correct_label AS feedback_label
               FROM predictions p LEFT JOIN feedback f ON f.prediction_id=p.id
               WHERE p.image_name LIKE %s OR p.prediction_result LIKE %s
               ORDER BY p.created_at DESC LIMIT %s""",
            (like, like, limit), fetch=True)

    @staticmethod
    def delete_by_id(pred_id):
        execute_query("DELETE FROM predictions WHERE id=%s", (pred_id,), commit=True)
        return True

    @staticmethod
    def get_paginated(page=1, per_page=10, label_filter=None, model_filter=None):
        offset = (page - 1) * per_page
        where, params = [], []
        if label_filter: where.append("p.prediction_result=%s"); params.append(label_filter)
        if model_filter: where.append("p.model_used=%s"); params.append(model_filter)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        total = (execute_query(f"SELECT COUNT(*) AS cnt FROM predictions p {wsql}", params, fetchone=True) or {}).get("cnt", 0)
        was_correct_expr = "(CASE WHEN f.correct_label=p.prediction_result THEN 1 ELSE 0 END)"
        rows  = execute_query(
            f"""SELECT p.id, p.image_name, p.prediction_result, p.confidence,
                       p.model_used, p.processing_time_ms, p.created_at,
                       p.top3_predictions, p.all_probabilities, p.file_size,
                       p.original_width, p.original_height, p.file_hash,
                       p.feature_version, p.thumbnail,
                       f.correct_label AS feedback_label,
                       {was_correct_expr} AS was_correct
                FROM predictions p LEFT JOIN feedback f ON f.prediction_id=p.id
                {wsql} ORDER BY p.created_at DESC LIMIT %s OFFSET %s""",
            params + [per_page, offset], fetch=True)
        return {"data": rows or [], "total": total, "page": page, "per_page": per_page,
                "total_pages": max(1, (total + per_page - 1) // per_page)}


class FeedbackDAO:
    @staticmethod
    def insert(prediction_id, correct_label, user_comment=""):
        return execute_query(
            """INSERT INTO feedback (prediction_id, correct_label, user_comment)
               VALUES (%s,%s,%s)
               ON CONFLICT(prediction_id) DO UPDATE SET
               correct_label=excluded.correct_label,
               user_comment=excluded.user_comment,
               updated_at=CURRENT_TIMESTAMP""",
            (prediction_id, correct_label, user_comment), commit=True)

    @staticmethod
    def get_all(limit=50):
        return execute_query(
            """SELECT f.id, f.prediction_id, f.correct_label, f.user_comment,
                      f.created_at, p.image_name, p.prediction_result,
                      (CASE WHEN f.correct_label=p.prediction_result THEN 1 ELSE 0 END) AS was_correct
               FROM feedback f JOIN predictions p ON f.prediction_id=p.id
               ORDER BY f.created_at DESC LIMIT %s""", (limit,), fetch=True)

    @staticmethod
    def get_accuracy():
        row = execute_query(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN f.correct_label=p.prediction_result THEN 1 ELSE 0 END) AS correct "
            "FROM feedback f JOIN predictions p ON f.prediction_id=p.id", fetchone=True)
        if not row or not row["total"]:
            return {"total_feedback": 0, "correct": 0, "accuracy_pct": 0}
        acc = round(100 * (row["correct"] or 0) / row["total"], 2)
        return {"total_feedback": row["total"], "correct": row["correct"] or 0, "accuracy_pct": acc}


# ─────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(file, model_name=None, use_ensemble=False) -> dict:
    t0 = time.time()
    filename = sanitize_filename(file.filename or "upload")
    if not allowed_file(filename):
        raise ValueError(f"File type not allowed: {filename.rsplit('.',1)[-1]}")
    image_bytes = file.read()
    if not image_bytes: raise ValueError("Empty file")
    if len(image_bytes) > Config.MAX_FILE_SIZE: raise ValueError(f"File too large (max {Config.MAX_FILE_SIZE/1024/1024:.0f}MB)")
    if not validate_image(image_bytes): raise ValueError("Not a valid image file")

    file_hash = compute_hash(image_bytes)
    cached = PredictionDAO.get_by_hash(file_hash)
    if cached and (time.time() - cached.get("created_at", datetime.min).timestamp() if hasattr(cached.get("created_at"), "timestamp") else 0) < 3600:
        logger.info("Cache hit for hash %s", file_hash[:16])
        cached_data = serialize_row(cached)
        cached_data["cached"] = True
        cached_data["disease_info"] = Config.DISEASE_INFO.get(cached_data.get("prediction_result", ""), {})
        return cached_data

    proc = ImageProcessor(image_bytes)
    proc.resize().make_thumbnail()
    features = proc.extract_features()

    result = model_registry.predict_ensemble(features) if use_ensemble else model_registry.predict(features, model_name)

    ms = round((time.time() - t0) * 1000, 2)
    thumb = proc.get_thumbnail_b64()
    meta  = proc.meta

    row_id = PredictionDAO.insert(
        image_name=filename, prediction=result["prediction"],
        confidence=result["confidence"], model_used=result["model_used"],
        file_hash=meta["file_hash"], file_size=meta["file_size"],
        orig_width=meta["original_width"], orig_height=meta["original_height"],
        top3_json=json.dumps(result["top3"]),
        all_proba_json=json.dumps(result["all_probabilities"]),
        feature_vector_json=json.dumps(features.tolist()),
        processing_time_ms=ms, thumbnail=thumb,
    )
    return {
        "id": row_id, "filename": filename, "cached": False,
        "prediction": result["prediction"], "confidence": result["confidence"],
        "model_used": result["model_used"], "top3": result["top3"],
        "all_probabilities": result["all_probabilities"],
        "processing_time_ms": ms, "image_metadata": meta, "thumbnail": thumb,
        "feature_version": Config.FEATURE_VER,
        "disease_info": Config.DISEASE_INFO.get(result["prediction"], {}),
    }

# ─────────────────────────────────────────────
# ROUTES — PREDICTION
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@rate_limit
def predict():
    try:
        if "image" not in request.files:
            return error_response("No image file provided.")
        f = request.files["image"]
        model = request.form.get("model", "").strip() or None
        ensemble = request.form.get("ensemble", "false").lower() == "true"
        if model and model not in model_registry.pipelines:
            return error_response(f"Unknown model '{model}'.")
        data = run_pipeline(f, model, ensemble)
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error("Predict error: %s", e)
        return error_response("Prediction failed.", 500, str(e))

@app.route("/predict/batch", methods=["POST"])
@rate_limit
def predict_batch():
    try:
        files = request.files.getlist("images[]")
        if not files: return error_response("No images provided.")
        files = files[:20]
        model = request.form.get("model", "").strip() or None
        ensemble = request.form.get("ensemble", "false").lower() == "true"
        results, errors = [], []
        for f in files:
            try:
                results.append(run_pipeline(f, model, ensemble))
            except Exception as e:
                errors.append({"filename": sanitize_filename(f.filename or "?"), "error": str(e)})
        return success_response({"total": len(files), "succeeded": len(results),
                                  "failed": len(errors), "results": results, "errors": errors})
    except Exception as e:
        logger.error("Batch error: %s", e)
        return error_response("Batch failed.", 500)

@app.route("/predict/url", methods=["POST"])
@rate_limit
def predict_url():
    import urllib.request
    try:
        body = request.get_json(silent=True) or {}
        url = (body.get("url") or "").strip()
        if not url: return error_response("'url' is required.")
        model = (body.get("model") or "").strip() or None
        ensemble = bool(body.get("ensemble", False))
        with urllib.request.urlopen(url, timeout=10) as resp:
            image_bytes = resp.read(Config.MAX_FILE_SIZE + 1)
        if len(image_bytes) > Config.MAX_FILE_SIZE:
            return error_response("Remote image too large.")
        if not validate_image(image_bytes):
            return error_response("URL does not point to a valid image.")

        class FakeFile:
            filename = url.split("/")[-1].split("?")[0] or "url_image.jpg"
            def read(self): return image_bytes

        data = run_pipeline(FakeFile(), model, ensemble)
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error("URL predict error: %s", e)
        return error_response("URL classification failed.", 500, str(e))

# ─────────────────────────────────────────────
# ROUTES — HISTORY
# ─────────────────────────────────────────────
@app.route("/history", methods=["GET"])
@rate_limit
def history():
    try:
        page     = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 10))))
        label    = request.args.get("label", "").strip() or None
        model    = request.args.get("model", "").strip() or None
        data = PredictionDAO.get_paginated(page, per_page, label, model)
        data["data"] = serialize_rows(data["data"])
        return success_response(data)
    except Exception as e:
        return error_response("Failed to fetch history.", 500, str(e))

@app.route("/history/<int:pred_id>", methods=["GET"])
@rate_limit
def get_prediction(pred_id):
    row = PredictionDAO.get_by_id(pred_id)
    if not row: return error_response(f"Prediction #{pred_id} not found.", 404)
    return success_response({"prediction": serialize_row(row)})

@app.route("/history/<int:pred_id>", methods=["DELETE"])
@rate_limit
def delete_prediction(pred_id):
    row = PredictionDAO.get_by_id(pred_id)
    if not row: return error_response(f"Prediction #{pred_id} not found.", 404)
    PredictionDAO.delete_by_id(pred_id)
    return success_response({"deleted": True, "id": pred_id})

@app.route("/history/hash/<string:file_hash>", methods=["GET"])
@rate_limit
def get_by_hash(file_hash):
    if not re.match(r"^[a-f0-9]{64}$", file_hash):
        return error_response("Invalid hash format.")
    row = PredictionDAO.get_by_hash(file_hash)
    if not row: return error_response("No prediction found for this hash.", 404)
    return success_response({"cached": True, "prediction": serialize_row(row)})

@app.route("/search", methods=["GET"])
@rate_limit
def search():
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2: return error_response("Query must be at least 2 characters.")
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    rows = PredictionDAO.search(q, limit)
    return success_response({"results": serialize_rows(rows), "count": len(rows), "query": q})

@app.route("/export/csv", methods=["GET"])
@rate_limit
def export_csv():
    try:
        label = request.args.get("label", "").strip()
        model = request.args.get("model", "").strip()
        where, params = [], []
        if label: where.append("prediction_result=%s"); params.append(label)
        if model: where.append("model_used=%s"); params.append(model)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = execute_query(
            f"SELECT id,image_name,prediction_result,confidence,model_used,"
            f"file_size,original_width,original_height,processing_time_ms,created_at "
            f"FROM predictions {wsql} ORDER BY created_at DESC", params, fetch=True)
        import io as _io
        buf = _io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id","image_name","prediction","confidence","model","file_size","width","height","processing_ms","created_at"])
        for r in (rows or []):
            w.writerow([r["id"], r["image_name"], r["prediction_result"],
                        round(float(r["confidence"] or 0), 2), r["model_used"],
                        r["file_size"], r["original_width"], r["original_height"],
                        round(float(r["processing_time_ms"] or 0), 2), r["created_at"]])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    except Exception as e:
        return error_response("Export failed.", 500, str(e))


# ─────────────────────────────────────────────
# ROUTES — STATS
# ─────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
@rate_limit
def stats():
    try:
        data = PredictionDAO.get_stats()
        data["model_info"] = model_registry.get_info()
        data["feedback_accuracy"] = FeedbackDAO.get_accuracy()
        return success_response(data)
    except Exception as e:
        return error_response("Stats failed.", 500, str(e))

@app.route("/stats/timeline", methods=["GET"])
@rate_limit
def stats_timeline():
    try:
        days = min(90, max(1, int(request.args.get("days", 7))))
        rows = execute_query(
            "SELECT DATE(created_at) AS day, COUNT(*) AS count, AVG(confidence) AS avg_confidence "
            "FROM predictions WHERE created_at>=datetime('now','-' || %s || ' days') "
            "GROUP BY DATE(created_at) ORDER BY day ASC", (days,), fetch=True)
        timeline = [{"day": r["day"].isoformat() if hasattr(r["day"],"isoformat") else str(r["day"]),
                     "count": r["count"],
                     "avg_confidence": round(float(r["avg_confidence"] or 0), 2)} for r in (rows or [])]
        return success_response({"timeline": timeline, "days": days})
    except Exception as e:
        return error_response("Timeline failed.", 500, str(e))

@app.route("/stats/labels", methods=["GET"])
@rate_limit
def stats_labels():
    try:
        rows = execute_query(
            "SELECT prediction_result AS label, COUNT(*) AS count, "
            "AVG(confidence) AS avg_confidence, MIN(confidence) AS min_confidence, MAX(confidence) AS max_confidence "
            "FROM predictions GROUP BY prediction_result ORDER BY count DESC", fetch=True)
        data = [{"label": r["label"], "count": r["count"],
                 "avg_confidence": round(float(r["avg_confidence"] or 0), 2),
                 "min_confidence": round(float(r["min_confidence"] or 0), 2),
                 "max_confidence": round(float(r["max_confidence"] or 0), 2)} for r in (rows or [])]
        return success_response({"labels": data})
    except Exception as e:
        return error_response("Label stats failed.", 500, str(e))

# ─────────────────────────────────────────────
# ROUTES — MODELS
# ─────────────────────────────────────────────
@app.route("/models", methods=["GET"])
@rate_limit
def list_models():
    return success_response(model_registry.get_info())

@app.route("/models/active", methods=["PUT"])
@rate_limit
def set_active_model():
    try:
        body = request.get_json(silent=True) or {}
        name = (body.get("model") or "").strip()
        if not name: return error_response("'model' is required.")
        model_registry.set_active(name)
        return success_response({"active_model": name, "message": f"Active model set to '{name}'"})
    except ValueError as e:
        return error_response(str(e))

@app.route("/models/<string:model_name>/evaluate", methods=["GET"])
@rate_limit
def evaluate_model(model_name):
    try:
        return success_response(model_registry.evaluate(model_name))
    except ValueError as e:
        return error_response(str(e), 404)

@app.route("/models/compare", methods=["POST"])
@rate_limit
def compare_models():
    try:
        if "image" not in request.files:
            return error_response("No image file provided.")
        f = request.files["image"]
        image_bytes = f.read()
        if not validate_image(image_bytes): return error_response("Invalid image.")
        proc = ImageProcessor(image_bytes)
        proc.resize().make_thumbnail()
        features = proc.extract_features()
        per_model = {}
        for name in model_registry.pipelines:
            try: per_model[name] = model_registry.predict(features, name)
            except Exception as e: per_model[name] = {"error": str(e)}
        ensemble = model_registry.predict_ensemble(features)
        return success_response({"per_model": per_model, "ensemble": ensemble,
                                  "image_metadata": proc.meta, "thumbnail": proc.get_thumbnail_b64()})
    except Exception as e:
        return error_response("Model comparison failed.", 500, str(e))

# ─────────────────────────────────────────────
# ROUTES — ANALYSIS
# ─────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
@rate_limit
def analyze():
    try:
        if "image" not in request.files: return error_response("No image provided.")
        f = request.files["image"]
        filename = sanitize_filename(f.filename or "upload")
        if not allowed_file(filename): return error_response("File type not allowed.")
        image_bytes = f.read()
        if not image_bytes or not validate_image(image_bytes): return error_response("Invalid image.")
        proc = ImageProcessor(image_bytes)
        proc.resize().make_thumbnail()
        features = proc.extract_features()
        hist = proc.get_histogram()
        dom = proc.get_dominant_color()
        feature_names = [
            "r_mean","g_mean","b_mean","r_std","g_std","b_std",
            "green_dominance","yellow_index","brown_index","white_index",
            "brightness","contrast","gray_range","sharpness","saturation","noise",
            "entropy","edge_density","spot_density","light_spot_density",
            "color_uniformity","warm_cool","spread_r","spread_g","spread_b",
            "symmetry","aspect_ratio","megapixels","orig_width","orig_height","filesize_mb","green_ratio"
        ]
        named_features = {n: round(float(v), 5) for n, v in zip(feature_names, features.tolist())}
        return success_response({
            "filename": filename, "metadata": proc.meta,
            "features": named_features,
            "dominant_color_rgb": {"r": round(dom[0]*255), "g": round(dom[1]*255), "b": round(dom[2]*255)},
            "histogram_summary": {"r_peak": int(np.argmax(hist["r"])), "g_peak": int(np.argmax(hist["g"])), "b_peak": int(np.argmax(hist["b"]))},
            "histogram": hist,
            "thumbnail": proc.get_thumbnail_b64(),
        })
    except Exception as e:
        return error_response("Analysis failed.", 500, str(e))

@app.route("/analyze/compare", methods=["POST"])
@rate_limit
def analyze_compare():
    try:
        f1 = request.files.get("image1"); f2 = request.files.get("image2")
        if not f1 or not f2: return error_response("Two images required (image1, image2).")
        b1, b2 = f1.read(), f2.read()
        if not validate_image(b1) or not validate_image(b2): return error_response("Invalid image(s).")
        p1, p2 = ImageProcessor(b1), ImageProcessor(b2)
        p1.resize(); p2.resize()
        f1v, f2v = p1.extract_features(), p2.extract_features()
        similarity = float(1.0 - np.mean(np.abs(f1v - f2v)))
        return success_response({
            "similarity_score": round(similarity * 100, 2),
            "image1": {"metadata": p1.meta, "thumbnail": p1.get_thumbnail_b64()},
            "image2": {"metadata": p2.meta, "thumbnail": p2.get_thumbnail_b64()},
        })
    except Exception as e:
        return error_response("Comparison failed.", 500, str(e))

# ─────────────────────────────────────────────
# ROUTES — DISEASE INFO
# ─────────────────────────────────────────────
@app.route("/disease-info", methods=["GET"])
def disease_info_all():
    return success_response({"diseases": Config.DISEASE_INFO, "labels": Config.LABELS})

@app.route("/disease-info/<string:label>", methods=["GET"])
def disease_info_single(label):
    info = Config.DISEASE_INFO.get(label)
    if not info: return error_response(f"No info for label '{label}'.", 404)
    return success_response({"label": label, **info})

# ─────────────────────────────────────────────
# ROUTES — FEEDBACK
# ─────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
@rate_limit
def submit_feedback():
    try:
        body = request.get_json(silent=True) or {}
        pred_id = body.get("prediction_id")
        correct = (body.get("correct_label") or "").strip()
        comment = (body.get("comment") or "").strip()[:500]
        if not pred_id or not isinstance(pred_id, int): return error_response("'prediction_id' (int) required.")
        if not correct: return error_response("'correct_label' required.")
        if correct not in Config.LABELS: return error_response(f"Invalid label.")
        if not PredictionDAO.get_by_id(pred_id): return error_response(f"Prediction #{pred_id} not found.", 404)
        FeedbackDAO.insert(pred_id, correct, comment)
        return success_response({"message": "Feedback submitted.", "prediction_id": pred_id, "correct_label": correct})
    except Exception as e:
        return error_response("Feedback failed.", 500, str(e))

@app.route("/feedback", methods=["GET"])
@rate_limit
def get_feedback():
    try:
        limit = min(200, max(1, int(request.args.get("limit", 50))))
        rows = FeedbackDAO.get_all(limit)
        serialized = serialize_rows(rows)
        for r in serialized:
            if "was_correct" in r: r["was_correct"] = bool(r["was_correct"])
        return success_response({"feedback": serialized, "count": len(serialized)})
    except Exception as e:
        return error_response("Failed to fetch feedback.", 500, str(e))

@app.route("/feedback/accuracy", methods=["GET"])
@rate_limit
def feedback_accuracy():
    try:
        return success_response(FeedbackDAO.get_accuracy())
    except Exception as e:
        return error_response("Accuracy failed.", 500, str(e))

# ─────────────────────────────────────────────
# ROUTES — HEALTH & UI
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    db_ok = False
    try: execute_query("SELECT 1", fetchone=True); db_ok = True
    except Exception: pass
    return jsonify({"status": "healthy" if db_ok else "degraded",
                    "database": "connected" if db_ok else "disconnected",
                    "models_loaded": len(model_registry.pipelines),
                    "active_model": model_registry.active_model,
                    "version": "3.1.0",
                    "uptime_seconds": int(time.time() - app.config.get("START_TIME", time.time())),
                    "timestamp": datetime.utcnow().isoformat() + "Z"}), 200 if db_ok else 503

@app.route("/health/db", methods=["GET"])
def health_db():
    try:
        cnt = execute_query("SELECT COUNT(*) AS cnt FROM predictions", fetchone=True)
        return success_response({"connected": True, "db": "sqlite", "total_records": cnt["cnt"]})
    except Exception as e:
        return error_response("DB connection failed.", 503, str(e))

@app.route("/ui")
@app.route("/ui")
@app.route("/ui/")
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e): return error_response("Bad request.", 400)
@app.errorhandler(404)
def not_found(e): return error_response(f"Not found: {request.path}", 404)
@app.errorhandler(405)
def method_not_allowed(e): return error_response(f"Method not allowed.", 405)
@app.errorhandler(413)
def too_large(e): return error_response("File too large (max 10MB).", 413)
@app.errorhandler(429)
def too_many(e): return error_response("Too many requests.", 429)
@app.errorhandler(500)
def server_error(e): logger.error("500: %s", e); return error_response("Internal server error.", 500)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.config["START_TIME"] = time.time()
    print(f"\n{'='*60}")
    print(f"  🌿 CropHealth AI v3.1 — Plant Disease Detection System")
    print(f"{'='*60}")
    print(f"  🌐 Web UI:  http://{Config.HOST}:{Config.PORT}/ui")
    print(f"  📡 API:     http://{Config.HOST}:{Config.PORT}/")
    print(f"  🤖 Models:  {len(model_registry.pipelines)} loaded")
    print(f"  📊 Active:  {model_registry.active_model}")
    print(f"  🗄️  Database: {Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}")
    print(f"{'='*60}\n")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True, use_reloader=False)
