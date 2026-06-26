import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    DB_HOST            = os.environ.get("DB_HOST")
    DB_PORT            = int(os.environ.get("DB_PORT", 3306))
    DB_USER            = os.environ.get("DB_USER")
    DB_PASSWORD        = os.environ.get("DB_PASSWORD")
    DB_NAME            = os.environ.get("DB_NAME")
    DB_POOL_SIZE       = int(os.environ.get("DB_POOL_SIZE", 10))
    DB_CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", 10))

    UPLOAD_FOLDER  = os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_FILE_SIZE  = int(os.environ.get("MAX_FILE_SIZE", 15 * 1024 * 1024))
    ALLOWED_EXT    = {"png", "jpg", "jpeg", "bmp", "webp", "tiff", "heic"}
    IMG_SIZE       = (224, 224)
    THUMB_SIZE     = (120, 120)
    RATE_LIMIT     = int(os.environ.get("RATE_LIMIT", 200))
    RATE_WIN       = 60
    SECRET_KEY     = os.environ.get("SECRET_KEY", os.urandom(32).hex())
    DEBUG          = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    HOST           = os.environ.get("HOST", "0.0.0.0")
    PORT           = int(os.environ.get("PORT", 5000))
    MODEL_VERSION  = "v1.0"

    LABELS = [
        "healthy",
        "downy_mildew",
        "leaf_spot",
        "damping_off",
        "white_rust",
        "anthracnose",
        "mosaic_virus",
        "nutrient_deficiency",
        "pest_damage",
    ]

    DISEASE_INFO = {
        "healthy": {
            "status": "Healthy Spinach",
            "severity": "none",
            "severity_score": 0,
            "color": "#22c55e",
            "icon": "🌱",
            "affected_parts": [],
            "description": "The spinach plant shows no signs of disease, stress, or pest damage. Leaves are dark green, firm, and uniform with no lesions, discoloration, or wilting.",
            "causes": [
                "Optimal growing conditions",
                "Balanced NPK nutrition",
                "Adequate soil moisture",
                "Good air circulation between rows",
            ],
            "immediate_actions": [
                "Continue current management practices",
                "Scout weekly for early disease signs",
                "Maintain irrigation schedule",
            ],
            "chemical_treatments": [],
            "organic_treatments": [
                "Compost tea (1:10 dilution) monthly as preventive",
                "Neem oil 0.5% spray every 3 weeks as prophylactic",
            ],
            "fertilizer_schedule": "NPK 15-15-15 @ 20 kg/acre split in 2 doses — at planting and 3 weeks after",
            "prevention": "Rotate spinach with non-leafy crops every season. Maintain 20–25 cm row spacing for airflow. Remove crop debris after each harvest.",
            "recovery_time": "N/A — plant is healthy",
            "economic_impact": "None",
        },
        "downy_mildew": {
            "status": "Downy Mildew",
            "severity": "high",
            "severity_score": 80,
            "color": "#ef4444",
            "icon": "💧",
            "affected_parts": ["Leaves (upper and lower surface)", "Petioles"],
            "description": "Downy mildew (Peronospora farinosa f.sp. spinaciae) is the most destructive spinach disease worldwide. Upper leaf surface develops pale yellow angular patches bounded by leaf veins. Grayish-purple sporulation appears on the undersurface under humid conditions. Leaves eventually turn brown and collapse.",
            "causes": [
                "Peronospora farinosa f.sp. spinaciae (oomycete)",
                "Cool temperatures 10–18°C with relative humidity above 85%",
                "Prolonged leaf wetness from overhead irrigation or dew",
                "Dense planting restricting airflow",
                "Infected seed lots carrying oospores",
            ],
            "immediate_actions": [
                "Apply metalaxyl-based oomycide within 24 hours of detection",
                "Remove and destroy heavily infected leaves immediately",
                "Switch from overhead to drip irrigation",
                "Increase row spacing to improve canopy ventilation",
                "Do not work in field when plants are wet",
            ],
            "chemical_treatments": [
                "Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold) @ 2.5 g/L — every 7 days, max 3 sprays",
                "Cymoxanil 8% + Mancozeb 64% WP @ 2.5 g/L — curative and protective combination",
                "Fosetyl-Al 80% WP @ 3 g/L — systemic, applied as soil drench",
                "Dimethomorph 50% WP @ 1 g/L — excellent curative activity at early stage",
                "Copper Oxychloride 50% WP @ 3 g/L — preventive, apply before rain periods",
            ],
            "organic_treatments": [
                "Bordeaux mixture 1% (100 g CuSO4 + 100 g lime per 10 L water) weekly",
                "Bacillus subtilis biocontrol (Serenade) @ 5 g/L foliar spray",
                "Potassium phosphonate @ 3 ml/L — induces systemic resistance",
            ],
            "fertilizer_schedule": "Reduce nitrogen to limit leaf succulence. Apply Calcium Chloride @ 2 g/L foliar to strengthen cell walls. Potassium Sulphate @ 3 g/L to boost immunity.",
            "prevention": "Use certified downy-mildew-resistant varieties (Sp75, Lazio, Emilia). Treat seeds with Thiram 75% WS. Avoid overhead irrigation in cool seasons. Apply preventive copper spray before forecasted rain.",
            "recovery_time": "14–21 days with prompt oomycide treatment; severe infections may not recover",
            "economic_impact": "Up to 100% field loss in susceptible varieties during cool humid seasons",
        },
        "leaf_spot": {
            "status": "Spinach Leaf Spot",
            "severity": "medium",
            "severity_score": 50,
            "color": "#f59e0b",
            "icon": "🔴",
            "affected_parts": ["Leaves", "Petioles"],
            "description": "Spinach leaf spot caused by Cercospora beticola or Alternaria spp. produces circular to oval tan or brown spots (3–10 mm) with reddish-purple borders. Centers dry out and fall away creating a shot-hole effect. Heavy infection leads to premature defoliation and unmarketable leaves.",
            "causes": [
                "Cercospora beticola or Alternaria spp. (fungal pathogens)",
                "Warm temperatures 25–30°C with leaf wetness",
                "Overhead irrigation splashing spores between plants",
                "Infected crop debris remaining in soil after harvest",
                "Mechanical injuries from cultivation creating entry points",
            ],
            "immediate_actions": [
                "Remove and bag infected leaves — do not compost",
                "Switch to drip irrigation to keep foliage dry",
                "Apply contact fungicide within 48 hours of detection",
                "Sanitize harvesting tools with 70% isopropyl alcohol",
            ],
            "chemical_treatments": [
                "Chlorothalonil 75% WP @ 2 g/L — broad-spectrum contact, every 7–10 days",
                "Azoxystrobin 23% SC @ 1 ml/L — systemic strobilurin, excellent efficacy",
                "Carbendazim 50% WP @ 1 g/L — systemic, 3 sprays at 10-day intervals",
                "Copper Hydroxide 77% WP @ 2 g/L — bactericidal and fungicidal",
            ],
            "organic_treatments": [
                "Bordeaux mixture 0.5% as preventive spray",
                "Trichoderma harzianum @ 5 g/L foliar spray biweekly",
                "Neem oil 2% + 0.1% sticker — weekly protective spray",
            ],
            "fertilizer_schedule": "Balanced NPK. Increase Potassium to 40 kg/ha to improve disease resistance. Avoid excess nitrogen which promotes soft, susceptible growth.",
            "prevention": "Use certified disease-free seeds. Apply Thiram 75% WS @ 3 g/kg as seed treatment. Maintain 2-year crop rotation. Clear debris immediately after harvest.",
            "recovery_time": "7–14 days with consistent fungicide treatment",
            "economic_impact": "15–30% yield loss; reduces market value significantly due to lesioned leaves",
        },
        "damping_off": {
            "status": "Damping Off",
            "severity": "critical",
            "severity_score": 90,
            "color": "#dc2626",
            "icon": "⚠️",
            "affected_parts": ["Stem at soil line", "Roots", "Seedlings"],
            "description": "Damping off causes rapid seedling collapse and death. Pre-emergence damping off kills seeds before they emerge. Post-emergence damping off causes water-soaked lesions at the soil line — the stem pinches and the seedling topples over. Most devastating in cool, wet, poorly drained seedbeds.",
            "causes": [
                "Pythium spp., Rhizoctonia solani, Phytophthora spp. (soil-borne pathogens)",
                "Fusarium oxysporum — spinach wilt complex",
                "Overwatering and waterlogged seedbeds",
                "Cold soil temperatures below 15°C slowing seedling establishment",
                "High seedling density reducing airflow",
            ],
            "immediate_actions": [
                "Stop overhead watering immediately",
                "Improve drainage — create raised beds or furrows",
                "Apply Metalaxyl soil drench to affected and surrounding areas",
                "Remove collapsed seedlings and surrounding infected soil",
                "Thin seedlings to improve air circulation",
            ],
            "chemical_treatments": [
                "Metalaxyl 35% WS @ 2 g/L soil drench — highly effective against Pythium and Phytophthora",
                "Carbendazim 50% WP @ 1 g/L soil drench — effective against Rhizoctonia and Fusarium",
                "Thiram 75% WP @ 3 g/kg — seed treatment before sowing",
                "Propamocarb 72.2% SL @ 3 ml/L soil drench — specific oomycide",
            ],
            "organic_treatments": [
                "Trichoderma harzianum @ 5 g/kg soil incorporated before sowing",
                "Pseudomonas fluorescens @ 10 g/L soil drench at sowing",
                "Neem cake @ 250 kg/ha soil incorporation — antagonistic to soil pathogens",
                "Cinnamon powder @ 5 g/L soil drench — natural antifungal for Pythium",
            ],
            "fertilizer_schedule": "Do not fertilize affected seedbeds. For surrounding healthy areas: light phosphorus application to promote root development. Avoid nitrogen until seedlings are established.",
            "prevention": "Use raised beds with well-drained soil. Treat seeds with Thiram or Captan before sowing. Avoid over-irrigation. Ensure soil temperature is above 15°C before sowing. Reduce seeding density.",
            "recovery_time": "No recovery for affected seedlings. Replant with treated seeds on well-drained beds after 7 days.",
            "economic_impact": "Total stand loss possible — 50–100% seedling mortality in affected patches",
        },
        "white_rust": {
            "status": "White Rust",
            "severity": "high",
            "severity_score": 75,
            "color": "#f97316",
            "icon": "🟡",
            "affected_parts": ["Leaf underside", "Petioles", "Stems", "Inflorescences"],
            "description": "White rust (Albugo occidentalis) produces bright white to creamy blister-like pustules on the undersurface of spinach leaves. The upper surface shows corresponding pale yellow patches. Infected inflorescences become swollen and distorted (staghead). The disease spreads rapidly in cool, wet weather.",
            "causes": [
                "Albugo occidentalis (oomycete closely related to Peronospora)",
                "Cool moist conditions 10–22°C with high humidity",
                "Infected seed carrying oospores",
                "Windborne zoospores from infected neighbouring fields",
                "Surface water from rain or irrigation enabling zoospore movement",
            ],
            "immediate_actions": [
                "Apply oomycete-specific fungicide immediately",
                "Remove and destroy infected inflorescences to stop sporulation",
                "Reduce irrigation frequency",
                "Harvest marketable leaves before disease spreads further",
            ],
            "chemical_treatments": [
                "Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L — primary choice for white rust",
                "Fosetyl-Al 80% WP @ 3 g/L — systemic, applied every 10 days",
                "Copper Oxychloride 50% WP @ 3 g/L — preventive protectant",
                "Mancozeb 75% WP @ 2.5 g/L — contact protectant before infection period",
            ],
            "organic_treatments": [
                "Bordeaux mixture 1% weekly during high-risk cool wet periods",
                "Potassium bicarbonate 5 g/L — raises leaf surface pH to inhibit spores",
                "Bacillus subtilis @ 5 g/L foliar — induces systemic resistance",
            ],
            "fertilizer_schedule": "Reduce nitrogen to avoid excessive leaf succulence. Apply Potassium Sulphate @ 3 g/L to improve plant immunity. Calcium Chloride @ 2 g/L strengthens cell walls.",
            "prevention": "Use white-rust-resistant varieties where available. Treat seeds with metalaxyl. Avoid overhead irrigation. Implement 2-year crop rotation with non-host crops.",
            "recovery_time": "10–18 days with metalaxyl-based treatment at early stage",
            "economic_impact": "30–60% marketable yield loss; infected plants are unmarketable",
        },
        "anthracnose": {
            "status": "Spinach Anthracnose",
            "severity": "medium",
            "severity_score": 55,
            "color": "#8b5cf6",
            "icon": "🌑",
            "affected_parts": ["Leaves", "Petioles", "Stems"],
            "description": "Anthracnose caused by Colletotrichum dematium produces water-soaked circular lesions that enlarge and turn tan to brown with dark borders. Under humid conditions, salmon-pink spore masses (acervuli) develop in lesion centers. Can cause significant postharvest losses during storage and transit.",
            "causes": [
                "Colletotrichum dematium (fungal pathogen)",
                "Warm humid weather 24–30°C",
                "Infected seeds or crop residues in soil",
                "Rain splash and overhead irrigation dispersing conidia",
                "Wounds from insects or mechanical damage as entry points",
            ],
            "immediate_actions": [
                "Remove and destroy all infected plant material",
                "Apply systemic fungicide immediately at first symptom",
                "Switch to drip irrigation — avoid wetting foliage",
                "Harvest at early maturity to prevent postharvest losses",
            ],
            "chemical_treatments": [
                "Azoxystrobin 23% SC @ 1 ml/L — highly effective strobilurin",
                "Carbendazim 50% WP @ 1 g/L — systemic benzimidazole, 3 sprays at 10-day intervals",
                "Difenoconazole 25% EC @ 0.5 ml/L — triazole, excellent efficacy",
                "Copper Hydroxide 77% WP @ 2 g/L — contact, apply every 7 days",
            ],
            "organic_treatments": [
                "Trichoderma asperellum @ 5 g/L foliar spray biweekly",
                "Neem oil 2% + 0.1% sticker weekly spray",
                "Hot water seed treatment at 52°C for 20 minutes before sowing",
            ],
            "fertilizer_schedule": "Balanced NPK. Apply Silicon @ 1 g/L foliar spray to strengthen cell walls against fungal penetration. Potassium at 40 kg/ha promotes disease resistance.",
            "prevention": "Use disease-free certified seeds. Treat seeds with hot water. Practice 2-year rotation. Remove all crop residues after harvest. Avoid overhead irrigation.",
            "recovery_time": "10–14 days with consistent fungicide treatment",
            "economic_impact": "20–35% yield and postharvest losses in susceptible humid conditions",
        },
        "mosaic_virus": {
            "status": "Spinach Mosaic Virus",
            "severity": "high",
            "severity_score": 70,
            "color": "#ef4444",
            "icon": "🦠",
            "affected_parts": ["Leaves", "Growing points", "Whole plant"],
            "description": "Spinach Mosaic Virus (SpMV, a potyvirus) causes characteristic mosaic mottling and interveinal chlorosis on leaves. Infected plants show leaf puckering, distortion, stunted growth, and reduced leaf size. The virus is primarily transmitted by Myzus persicae (green peach aphid) in a non-persistent manner.",
            "causes": [
                "Spinach Mosaic Virus (SpMV — Potyvirus)",
                "Beet Mosaic Virus (BtMV) as secondary cause",
                "Myzus persicae (green peach aphid) as primary vector",
                "Infected transplants or neighbouring infected crops",
                "Mechanical transmission via contaminated cutting tools and hands",
            ],
            "immediate_actions": [
                "Remove and destroy infected plants immediately — no cure exists",
                "Apply systemic aphicide to control aphid vector population urgently",
                "Install reflective silver mulch to deter aphid landing",
                "Sanitize all cutting tools with 10% trisodium phosphate (TSP) solution",
                "Remove weed hosts especially Chenopodium spp. from field borders",
            ],
            "chemical_treatments": [
                "Thiamethoxam 25% WG @ 0.3 g/L — systemic aphicide, every 10 days",
                "Imidacloprid 17.8% SL @ 0.5 ml/L — soil drench for long-term aphid control",
                "Acetamiprid 20% SP @ 0.3 g/L — foliar spray for aphid vectors",
                "Mineral oil spray @ 10 ml/L — interferes with aphid probing, reduces virus transmission",
            ],
            "organic_treatments": [
                "Neem oil 2% @ 5 ml/L — repels aphid vectors",
                "Insecticidal soap 2% — contact aphicide for soft-bodied insects",
                "Yellow sticky traps @ 10 per acre for aphid monitoring and mass trapping",
                "Release Aphidius colemani (parasitic wasp) as biocontrol at 1000 per acre",
            ],
            "fertilizer_schedule": "Avoid excess nitrogen which creates succulent growth attractive to aphids. Apply Zinc Sulphate @ 0.5 g/L foliar to boost plant immunity.",
            "prevention": "Plant certified virus-indexed seeds. Choose SpMV-resistant varieties. Maintain 50 m distance from infected or weed-infested fields. Scout weekly for aphid colonies.",
            "recovery_time": "No cure — infected plants remain infected. Remove promptly to prevent spread.",
            "economic_impact": "25–50% yield loss; early infection causes near-total loss of that planting",
        },
        "nutrient_deficiency": {
            "status": "Nutrient Deficiency",
            "severity": "medium",
            "severity_score": 40,
            "color": "#eab308",
            "icon": "🌾",
            "affected_parts": ["Leaves", "Growing points", "Whole plant"],
            "description": "Spinach is a heavy feeder especially for nitrogen, iron, and magnesium. Nitrogen deficiency causes uniform light green to yellow coloring starting from older leaves. Iron deficiency produces interveinal chlorosis on young leaves with green veins and yellow tissue. Magnesium deficiency shows interveinal chlorosis on older leaves. Potassium deficiency causes leaf margin scorch.",
            "causes": [
                "Insufficient fertilizer application or poor fertilizer quality",
                "Soil pH outside 6.0–7.0 locking nutrients especially iron at pH above 7.5",
                "Waterlogging reducing root oxygen and nutrient uptake",
                "High soil phosphorus antagonizing zinc and iron uptake",
                "Rapid growth exhausting soil reserves in sandy or leached soils",
            ],
            "immediate_actions": [
                "Take soil and leaf tissue samples for laboratory analysis",
                "Correct soil pH if outside 6.0–7.0 using lime to raise or sulphur to lower",
                "Apply targeted foliar spray for rapid visual correction within days",
                "Improve irrigation management to prevent waterlogging",
            ],
            "chemical_treatments": [
                "Nitrogen deficiency: Urea 1% foliar spray (10 g/L water) every 7 days",
                "Iron deficiency: Ferrous Sulphate 0.5% + Citric acid 0.1% foliar chelate spray",
                "Magnesium deficiency: Magnesium Sulphate (Epsom salt) 1% foliar spray",
                "Zinc deficiency: Zinc Sulphate 0.5% foliar spray every 10 days",
                "Boron deficiency: Borax 0.2% foliar spray",
                "General micronutrient correction: NPK 19:19:19 @ 5 g/L + micronutrient mix @ 2 g/L",
            ],
            "organic_treatments": [
                "Vermicompost @ 2 tonnes/acre soil application before planting",
                "Seaweed extract @ 3 ml/L foliar — broad micronutrient source",
                "Fish emulsion @ 5 ml/L foliar — nitrogen-rich, rapid uptake",
                "Bone meal @ 100 kg/acre incorporated for phosphorus and calcium",
            ],
            "fertilizer_schedule": "Spinach NPK: 80:40:40 kg/ha. Apply 40 kg N at planting + 40 kg N top-dress at 3 weeks. Full P and K as basal dose. Foliar micronutrients at 14-day intervals.",
            "prevention": "Conduct soil testing before every planting season. Maintain pH 6.5–7.0. Apply organic matter FYM @ 10 t/ha to improve nutrient retention and microbial activity.",
            "recovery_time": "3–7 days for foliar-applied nutrients; 2–3 weeks for soil-applied corrections",
            "economic_impact": "10–40% yield reduction; nitrogen deficiency affects leaf quality and market value",
        },
        "pest_damage": {
            "status": "Pest Damage",
            "severity": "medium",
            "severity_score": 55,
            "color": "#f97316",
            "icon": "🐛",
            "affected_parts": ["Leaves", "Stems", "Roots"],
            "description": "Spinach is attacked by a range of pests. Leaf miners (Liriomyza spp.) create characteristic serpentine tunnels in leaf tissue. Aphids (Myzus persicae) cluster on undersides causing leaf curl and virus transmission. Cutworms damage stems at soil level overnight. Flea beetles create tiny shot-holes in young leaves. Spider mites cause stippling and leaf bronzing under hot dry conditions.",
            "causes": [
                "Leaf miners: Liriomyza spp. — larvae tunnel between leaf surfaces",
                "Aphids: Myzus persicae, Macrosiphum euphorbiae — sucking damage and virus vectors",
                "Cutworms: Agrotis spp. — nocturnal stem cutting at soil level",
                "Flea beetles: Phyllotreta spp. — shot-hole feeding on young leaves",
                "Spider mites: Tetranychus urticae — hot dry conditions trigger outbreaks",
            ],
            "immediate_actions": [
                "Identify the specific pest accurately before applying any pesticide",
                "Remove and destroy heavily infested plant parts",
                "Install yellow sticky traps at canopy height for monitoring and mass trapping",
                "Apply targeted pesticide based on confirmed pest identification",
            ],
            "chemical_treatments": [
                "Leaf miners: Cyromazine 75% WP @ 0.75 g/L — selective, disrupts larval development",
                "Aphids: Thiamethoxam 25% WG @ 0.3 g/L — systemic neonicotinoid",
                "Cutworms: Chlorpyrifos 20% EC @ 2.5 ml/L soil drench around stem base",
                "Flea beetles and chewing pests: Spinosad 45% SC @ 0.3 ml/L — selective, low toxicity",
                "Spider mites: Abamectin 1.8% EC @ 1 ml/L or Spiromesifen 22.9% SC @ 1 ml/L",
            ],
            "organic_treatments": [
                "Neem oil 2% + 0.1% sticker — broad-spectrum repellent and insecticide",
                "Bacillus thuringiensis var. kurstaki @ 2 g/L — specific to caterpillars and moths",
                "Beauveria bassiana @ 5 g/L — entomopathogenic fungus for multiple pest species",
                "Insecticidal soap 2% — contact action on aphids and spider mites",
                "Diatomaceous earth dusting at soil line for cutworm and flea beetle control",
            ],
            "fertilizer_schedule": "Avoid excess nitrogen which produces soft aphid-attractive leaf tissue. Apply Silicon @ 1 g/L foliar spray to strengthen cell walls against chewing and leaf-mining insects.",
            "prevention": "Implement IPM strategy with weekly scouting at economic threshold. Use row covers for leaf miners and flea beetles during establishment. Intercrop with coriander or dill to attract natural enemies. Maintain field hygiene.",
            "recovery_time": "7–14 days after pest control; new growth replaces damaged tissue within 2 weeks",
            "economic_impact": "15–40% marketable yield loss; leaf-mined and aphid-infested leaves are unmarketable",
        },
    }
