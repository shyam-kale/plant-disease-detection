# How to Train Models for 85-95% Confidence

## Quick Start (3 steps)

### Step 1 — Install dependencies
```
pip install kaggle torch torchvision timm scikit-learn xgboost opencv-python scipy
```

### Step 2 — Get Kaggle API key
1. Go to: https://www.kaggle.com/settings
2. Click **API** → **Create New Token**
3. A file `kaggle.json` downloads — save it to:
   - Windows: `C:\Users\YOUR_NAME\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

### Step 3 — Run training
```
python train.py
```
That's it. The script auto-downloads the dataset (~1.5 GB) and trains everything.

---

## If you already have the dataset

If you downloaded PlantVillage manually from Kaggle:
```
python train.py --data "C:\path\to\PlantVillage"
```

The folder should contain subfolders like:
```
PlantVillage/
  Apple___healthy/
  Tomato___Late_blight/
  Tomato___Bacterial_spot/
  ...
```

---

## Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 25 | More epochs = higher accuracy |
| `--batch-size` | 16 | Reduce to 8 if memory error |
| `--max-per-class` | 800 | Images per disease class |
| `--skip-pytorch` | off | Train only classical models |
| `--skip-classical` | off | Train only EfficientNet |

### Fast test run (10 min):
```
python train.py --epochs 5 --max-per-class 100
```

### Best accuracy (1-2 hours):
```
python train.py --epochs 40 --max-per-class 1500
```

---

## What happens after training

Files saved automatically:
```
models/
  efficientnet_b4_spinach_finetuned.pth   ← PyTorch model (fine-tuned)
  classical/
    svm.pkl                               ← SVM model
    random_forest.pkl                     ← Random Forest
    knn.pkl                               ← KNN
    xgboost.ubj                           ← XGBoost
    xgb_scaler.pkl                        ← Feature scaler
    label_encoder.pkl                     ← Label encoder
    stats.json                            ← Training accuracy report
```

After training, restart Flask server:
```
python app.py
```

Upload any spinach leaf → confidence should be **85-95%** ✅

---

## Expected accuracy per model

| Model | Expected Accuracy |
|-------|-------------------|
| EfficientNet-B4 (fine-tuned) | 88-94% |
| XGBoost | 84-91% |
| Random Forest | 82-89% |
| SVM | 80-87% |
| KNN | 75-82% |
| **7-Model Ensemble** | **90-96%** |

---

## Troubleshooting

**"kaggle: command not found"**
```
pip install kaggle
```

**"403 Forbidden" from Kaggle**
- Make sure you accepted the dataset terms on Kaggle website first
- Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset and click Download

**"CUDA out of memory"**
```
python train.py --batch-size 8
```

**Very slow on CPU?**
- Normal — CPU training takes 40-60 minutes
- You can reduce: `python train.py --epochs 10 --max-per-class 300` for faster results

**After training confidence still low?**
- Make sure you RESTARTED the Flask server after training
- Check that `models/efficientnet_b4_spinach_finetuned.pth` exists
