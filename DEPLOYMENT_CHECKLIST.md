# Deployment Checklist

## Pre-Deployment Validation

### 1. Run Validation Script
```bash
cd project
python validate_deployment.py
```

All checks must pass before deploying.

### 2. Version Compatibility Matrix

| Package | Version | Python Requirement |
|---------|---------|-------------------|
| Python | 3.11.9 | Required |
| Flask | 3.0.3 | 3.8+ |
| numpy | 1.26.4 | 3.9+ |
| scikit-learn | 1.5.2 | 3.9+ |
| Pillow | 10.4.0 | 3.8+ |
| gunicorn | 22.0.0 | 3.7+ |

### 3. Configuration Files Checklist

- [ ] `runtime.txt` specifies `python-3.11.9`
- [ ] `.python-version` specifies `3.11.9`
- [ ] `render.yaml` PYTHON_VERSION is `3.11.9`
- [ ] `requirements.txt` has compatible versions
- [ ] All three files are in sync

### 4. Common Issues & Solutions

#### Issue: Python 3.14 being used instead of 3.11
**Solution:** Ensure all three files (runtime.txt, .python-version, render.yaml) specify 3.11.9

#### Issue: scikit-learn Cython compilation errors
**Symptoms:** `'int_t' is not a type identifier`
**Solution:** Use scikit-learn 1.5.2+ with numpy 1.26.4+

#### Issue: numpy compatibility errors
**Solution:** Use numpy 1.26.4 (compatible with both Python 3.11 and scikit-learn 1.5.2)

#### Issue: Build timeout
**Solution:** Added `--upgrade pip` and `--workers 2 --timeout 120` in render.yaml

### 5. Render.com Specific Settings

```yaml
buildCommand: pip install --upgrade pip && pip install -r requirements.txt
startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

**Why these settings:**
- `--upgrade pip`: Ensures latest pip with better dependency resolution
- `--workers 2`: Optimal for Render's free tier
- `--timeout 120`: Prevents worker timeout during ML model loading

### 6. Environment Variables

Required in Render dashboard:
- `PYTHON_VERSION=3.11.9`
- `USE_SQLITE=true`
- `SECRET_KEY` (auto-generated)
- `FLASK_DEBUG=false`

### 7. Post-Deployment Verification

After deployment succeeds, test these endpoints:

```bash
# Health check
curl https://your-app.onrender.com/api/health

# Database health
curl https://your-app.onrender.com/api/health/db

# Models list
curl https://your-app.onrender.com/api/models
```

### 8. Troubleshooting Failed Builds

If build fails:

1. Check Python version in build logs
2. Verify all requirements install successfully
3. Look for Cython compilation errors
4. Check for version conflicts

**Quick fix command:**
```bash
# Update all critical files at once
echo "3.11.9" > .python-version
echo "python-3.11.9" > runtime.txt
# Then update render.yaml PYTHON_VERSION to 3.11.9
```

### 9. Local Testing Before Deploy

```bash
# Create clean virtual environment
python3.11 -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run validation
python validate_deployment.py

# Test app locally
python app.py
```

### 10. Monitoring After Deployment

- Check Render logs for any warnings
- Monitor memory usage (ML models can be heavy)
- Test prediction endpoint with sample images
- Verify database operations work correctly

## Emergency Rollback

If deployment fails and you need to rollback:

1. Go to Render dashboard
2. Select your service
3. Click "Manual Deploy" → "Deploy previous version"
4. Or revert the commit in Git and push

## Success Criteria

✅ Build completes without errors
✅ All dependencies install successfully
✅ Application starts without crashes
✅ Health endpoints return 200 OK
✅ Prediction endpoint works with test image
✅ Database operations function correctly

---

**Last Updated:** 2026-03-25
**Validated Configuration:** Python 3.11.9 + scikit-learn 1.5.2 + numpy 1.26.4
