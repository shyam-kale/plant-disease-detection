# 🌿 CropHealth AI - Plant Disease Detection System

Advanced AI-powered plant disease detection system with real-time analysis, batch processing, and comprehensive analytics.

## ✨ Features

- **🔍 Real-time Disease Detection** - Upload plant images for instant disease identification
- **📦 Batch Processing** - Analyze up to 20 images simultaneously
- **🌐 URL Classification** - Detect diseases from remote image URLs
- **🤖 Multiple ML Models** - 5 trained models (Random Forest, SVM, KNN, Gradient Boosting, Logistic Regression)
- **⚖️ Ensemble Predictions** - Combine all models for maximum accuracy
- **📊 Advanced Analytics** - Comprehensive charts, timelines, and statistics
- **🔬 Image Analysis** - Deep feature extraction and visualization
- **💾 Smart Caching** - Duplicate detection with 1-hour cache
- **📈 Performance Tracking** - Monitor speed, accuracy, and model performance
- **🗄️ Database Storage** - MySQL backend with full history
- **📤 CSV Export** - Export detection history with filters
- **🎨 Modern UI** - Dark theme with responsive design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MySQL 8.0+
- pip

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd project
```

2. **Install dependencies**
```bash
pip install flask flask-cors mysql-connector-python numpy scikit-learn pillow
```

3. **Setup database**
```bash
mysql -u root -p < schema.sql
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your database credentials
```

5. **Run the application**
```bash
python app.py
```

6. **Open browser**
```
http://localhost:5000/ui
```

## 📋 API Endpoints

### Prediction
- `POST /predict` - Single image classification
- `POST /predict/batch` - Batch image processing (up to 20)
- `POST /predict/url` - Classify from URL

### History & Stats
- `GET /history` - Paginated detection history
- `GET /history/{id}` - Get specific prediction
- `DELETE /history/{id}` - Delete prediction
- `GET /stats` - Overall statistics
- `GET /stats/timeline` - Timeline data
- `GET /stats/labels` - Disease distribution
- `GET /export/csv` - Export to CSV

### Models
- `GET /models` - List all models
- `PUT /models/active` - Set active model
- `POST /models/compare` - Compare all models
- `GET /models/{name}/evaluate` - Model evaluation

### Analysis
- `POST /analyze` - Deep image analysis
- `POST /analyze/compare` - Compare two images

### Disease Info
- `GET /disease-info` - All disease information
- `GET /disease-info/{label}` - Specific disease info

### Feedback
- `POST /feedback` - Submit feedback
- `GET /feedback` - Get all feedback
- `GET /feedback/accuracy` - Accuracy metrics

### Health
- `GET /health` - System health check
- `GET /health/db` - Database health

## 🎯 Supported Diseases

1. ✅ Healthy
2. 🍂 Leaf Blight
3. 🌫️ Powdery Mildew
4. 🟤 Rust
5. 🔴 Leaf Spot
6. ⚠️ Bacterial Wilt
7. 🦠 Mosaic Virus
8. 💧 Downy Mildew
9. 🌑 Anthracnose
10. 🌿 Root Rot
11. 🌾 Nutrient Deficiency
12. 🐛 Pest Damage

## ⌨️ Keyboard Shortcuts

- `Ctrl+K` - Quick search
- `Ctrl+U` - Upload page
- `Ctrl+H` - History
- `Ctrl+D` - Dashboard
- `Esc` - Close sidebar/blur

## 🔧 Configuration

Edit `.env` file:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=image_classifier
DB_POOL_SIZE=10

HOST=0.0.0.0
PORT=5000
MAX_FILE_SIZE=15728640  # 15MB
RATE_LIMIT=200
```

## 📊 Performance

- **Processing Speed**: ~50-200ms per image
- **Batch Capacity**: 20 images
- **Cache Duration**: 1 hour
- **Rate Limit**: 200 requests/minute
- **Max File Size**: 15MB
- **Supported Formats**: PNG, JPG, JPEG, WEBP, BMP, TIFF, HEIC

## 🛠️ Tech Stack

**Backend:**
- Flask (Web framework)
- scikit-learn (ML models)
- NumPy (Numerical computing)
- Pillow (Image processing)
- MySQL (Database)

**Frontend:**
- Vanilla JavaScript
- Chart.js (Visualizations)
- CSS3 (Styling)

## 📝 License

MIT License - feel free to use for commercial or personal projects.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Version:** 3.1.0  
**Last Updated:** 2024  
**Status:** Production Ready ✅
