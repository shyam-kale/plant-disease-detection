#!/usr/bin/env python3
"""
Deployment Validation Script
Checks all dependencies and configurations before deployment
"""

import sys
import subprocess
import re

def check_python_version():
    """Ensure Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 11:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.11+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_requirements():
    """Validate requirements.txt versions"""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        # Check for problematic versions
        issues = []
        
        if 'numpy==1.24' in content or 'numpy==1.25' in content:
            issues.append("numpy version too old (use 1.26.4+)")
        
        if 'scikit-learn==1.3' in content or 'scikit-learn==1.2' in content:
            issues.append("scikit-learn version too old (use 1.5.2+)")
        
        if 'Flask==2.' in content:
            issues.append("Flask version too old (use 3.0.3+)")
        
        if issues:
            print("❌ Requirements.txt issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ requirements.txt - All versions compatible")
        return True
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_runtime_files():
    """Check runtime.txt and .python-version consistency"""
    try:
        with open('runtime.txt', 'r') as f:
            runtime = f.read().strip()
        
        with open('.python-version', 'r') as f:
            pyversion = f.read().strip()
        
        # Extract version numbers
        runtime_ver = runtime.replace('python-', '')
        
        if runtime_ver != pyversion:
            print(f"❌ Version mismatch: runtime.txt ({runtime_ver}) != .python-version ({pyversion})")
            return False
        
        if not runtime_ver.startswith('3.11'):
            print(f"❌ Python version {runtime_ver} not recommended. Use 3.11.9")
            return False
        
        print(f"✅ Runtime files consistent: Python {runtime_ver}")
        return True
    except Exception as e:
        print(f"❌ Error checking runtime files: {e}")
        return False

def check_render_yaml():
    """Validate render.yaml configuration"""
    try:
        with open('render.yaml', 'r') as f:
            content = f.read()
        
        issues = []
        
        if 'PYTHON_VERSION' in content:
            match = re.search(r'PYTHON_VERSION.*?value:\s*([0-9.]+)', content)
            if match:
                version = match.group(1)
                if not version.startswith('3.11'):
                    issues.append(f"PYTHON_VERSION {version} should be 3.11.9")
        
        if 'pip install -r requirements.txt' in content and 'pip install --upgrade pip' not in content:
            issues.append("Missing 'pip install --upgrade pip' in buildCommand")
        
        if issues:
            print("❌ render.yaml issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ render.yaml - Configuration valid")
        return True
    except Exception as e:
        print(f"❌ Error checking render.yaml: {e}")
        return False

def test_imports():
    """Test critical imports"""
    try:
        import numpy
        import sklearn
        import flask
        import PIL
        
        print(f"✅ Package imports successful:")
        print(f"   - numpy: {numpy.__version__}")
        print(f"   - scikit-learn: {sklearn.__version__}")
        print(f"   - flask: {flask.__version__}")
        print(f"   - Pillow: {PIL.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    print("=" * 50)
    print("  CropHealth AI - Deployment Validation")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Requirements File", check_requirements),
        ("Runtime Files", check_runtime_files),
        ("Render Config", check_render_yaml),
        ("Package Imports", test_imports),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        results.append(check_func())
        print()
    
    print("=" * 50)
    if all(results):
        print("✅ ALL CHECKS PASSED - Ready for deployment!")
        print("=" * 50)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before deploying")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
