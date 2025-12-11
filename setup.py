"""
Quick Setup Script
Helps you set up the project quickly.
"""

import os
import json


def create_directory_structure():
    """Create data directories."""
    os.makedirs("data/resumes", exist_ok=True)
    os.makedirs("data/job_descriptions", exist_ok=True)
    print("✅ Created data directories")


def create_manual_scores_template():
    """Create template for manual scores."""
    if not os.path.exists("manual_scores.json"):
        template = {
            "scores": [5.0] * 20,
            "instructions": "Replace these with your actual manual scores (1-10 scale) for each of the 20 resume-JD pairs"
        }
        with open("manual_scores.json", "w") as f:
            json.dump(template, f, indent=2)
        print("✅ Created manual_scores.json template")
    else:
        print("ℹ️  manual_scores.json already exists")


def create_env_file():
    """Create .env file from example."""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as src:
                content = src.read()
            with open(".env", "w") as dst:
                dst.write(content)
            print("✅ Created .env file from template")
            print("⚠️  IMPORTANT: Edit .env and add your GEMINI_API_KEY")
        else:
            print("⚠️  .env.example not found")
    else:
        print("ℹ️  .env already exists")


def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    packages = [
        ("sklearn", "scikit-learn"),
        ("spacy", "spacy"),
        ("sentence_transformers", "sentence-transformers"),
        ("google.generativeai", "google-generativeai"),
        ("scipy", "scipy"),
        ("numpy", "numpy"),
    ]
    
    missing = []
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - NOT INSTALLED")
            missing.append(package_name)
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all at once:")
        print("  pip install -r requirements.txt")
        print("  python -m spacy download en_core_web_sm")
    else:
        print("\n✅ All dependencies installed!")
        
        # Check spaCy model
        try:
            import spacy
            spacy.load("en_core_web_sm")
            print("✅ spaCy model 'en_core_web_sm' is installed")
        except:
            print("⚠️  spaCy model not found. Install with:")
            print("  python -m spacy download en_core_web_sm")


def main():
    """Run setup."""
    print("="*60)
    print("SETUP SCRIPT - Resume-JD Matching Project")
    print("="*60)
    
    print("\n📁 Setting up directories...")
    create_directory_structure()
    
    print("\n📝 Creating configuration files...")
    create_manual_scores_template()
    create_env_file()
    
    check_dependencies()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Add your 20 resume-JD pairs to data/resumes/ and data/job_descriptions/")
    print("2. Edit manual_scores.json with your ground truth scores")
    print("3. Edit .env and add your GEMINI_API_KEY")
    print("4. Run: python test_suite.py")
    print("="*60)


if __name__ == "__main__":
    main()
