# 🚀 Deploy CropHealth AI to Render

## ✅ Step 1: GitHub (COMPLETED!)
Your code is now at: https://github.com/shyam-kale/plant-disease-detection

## 📋 Step 2: Deploy on Render

### 2.1 Create Web Service
1. Go to https://render.com
2. Sign up or log in
3. Click **"New +"** → **"Web Service"**
4. Click **"Connect account"** to link GitHub
5. Select repository: **shyam-kale/plant-disease-detection**
6. Click **"Connect"**

### 2.2 Configure Service
Render will auto-detect settings from `render.yaml`, but verify:

- **Name:** `crophealth-ai` (or your choice)
- **Environment:** `Python 3`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Instance Type:** `Free`

### 2.3 Add Environment Variables
Click **"Advanced"** → **"Add Environment Variable"**

Add these:
```
USE_SQLITE = true
SECRET_KEY = crophealth-secret-2024-render
FLASK_DEBUG = false
```

### 2.4 Deploy!
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for build
3. Your app will be live at: `https://crophealth-ai.onrender.com/ui`

## 🎯 What's Included

✅ **12 Disease Detection Models**
- Healthy, Leaf Blight, Powdery Mildew, Rust, Leaf Spot
- Bacterial Wilt, Mosaic Virus, Downy Mildew, Anthracnose
- Root Rot, Nutrient Deficiency, Pest Damage

✅ **Features**
- Single image classification
- Batch processing (up to 20 images)
- URL classification
- Model comparison
- Analytics dashboard
- Image analysis tools
- CSV export
- Smart caching

✅ **No Database Setup Required**
- Uses SQLite (file-based)
- Perfect for free tier
- Data persists during session

## ⚠️ Important Notes

### Free Tier Limitations:
- Service spins down after 15 min of inactivity
- First request after spin-down takes ~30 seconds
- 750 hours/month free

### First Load:
- ML models load on first request (~2-3 seconds)
- Subsequent requests are fast (~100-200ms)

### Data Persistence:
- SQLite database resets on each deploy
- For production, upgrade to paid tier with persistent disk

## 🔧 Troubleshooting

**Build fails?**
- Check logs in Render dashboard
- Verify Python version (should be 3.11.0)

**App crashes?**
- Ensure `USE_SQLITE=true` is set
- Check environment variables

**Slow to load?**
- Normal on free tier after inactivity
- Service wakes up on first request

## 📊 Monitor Your App

In Render dashboard:
- View logs
- Check metrics
- Monitor health checks
- See deployment history

## 🎉 Success!

Once deployed, test your app:
1. Visit: `https://your-app-name.onrender.com/ui`
2. Upload a plant image
3. Get instant disease detection!

---

**Deployment Time:** 5-10 minutes  
**Cost:** FREE  
**Difficulty:** Easy ⭐

Need help? Check Render docs or open an issue on GitHub.
