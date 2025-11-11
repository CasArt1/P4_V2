"""
Windows Setup Script for NVDA Trading Strategy
Handles Windows Long Path issues and sets up the environment
"""

import subprocess
import sys
import os

def enable_long_paths_message():
    """Display instructions for enabling long paths on Windows"""
    print("\n" + "="*60)
    print("⚠️  WINDOWS LONG PATH ISSUE DETECTED")
    print("="*60)
    print("\nOption 1: Enable Long Paths (Recommended)")
    print("-" * 60)
    print("Run this in PowerShell as Administrator:")
    print('New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force')
    print("\nThen restart your terminal and try again.")
    print("\nOption 2: Use Shorter Path")
    print("-" * 60)
    print("Move your project to a shorter path like:")
    print("  C:\\projects\\nvda-trading\\")
    print("="*60 + "\n")

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("📦 Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upgrade pip: {e}")
        return False

def install_packages_incrementally():
    """Install packages one by one to identify problematic ones"""
    
    # Core packages first
    core_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
    ]
    
    # ML and API packages
    ml_packages = [
        "tensorflow",
        "mlflow",
        "fastapi",
        "uvicorn[standard]",
    ]
    
    # Visualization packages
    viz_packages = [
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
    ]
    
    # Domain-specific packages
    domain_packages = [
        "yfinance",
        "ta",
        "pandas-ta",
        "scipy",
        "python-dotenv",
        "requests",
    ]
    
    all_package_groups = [
        ("Core Packages", core_packages),
        ("ML & API Packages", ml_packages),
        ("Visualization Packages", viz_packages),
        ("Domain Packages", domain_packages),
    ]
    
    failed_packages = []
    
    for group_name, packages in all_package_groups:
        print(f"\n{'='*60}")
        print(f"Installing {group_name}...")
        print('='*60)
        
        for package in packages:
            print(f"\n📦 Installing {package}...", end=" ")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package, "--no-cache-dir"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("✅")
            except subprocess.CalledProcessError:
                print("❌")
                failed_packages.append(package)
    
    return failed_packages

def create_project_structure():
    """Create the project folder structure"""
    print("\n" + "="*60)
    print("📁 Creating Project Structure...")
    print("="*60)
    
    folders = [
        "data",
        "features",
        "models",
        "models/saved_models",
        "api",
        "backtesting",
        "drift_monitoring",
        "notebooks",
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}/")
    
    print("\n✅ Project structure created successfully!")

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 NVDA Trading Strategy - Windows Setup")
    print("="*60)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("\n⚠️  WARNING: Not in a virtual environment!")
        print("It's recommended to create one first:")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Step 1: Upgrade pip
    if not upgrade_pip():
        print("\n❌ Setup failed at pip upgrade step.")
        return
    
    # Step 2: Install packages
    print("\n" + "="*60)
    print("📦 Installing Python Packages...")
    print("="*60)
    print("This may take 5-10 minutes. Please be patient...")
    
    failed = install_packages_incrementally()
    
    # Step 3: Report results
    print("\n" + "="*60)
    print("📊 Installation Summary")
    print("="*60)
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to install:")
        for package in failed:
            print(f"  - {package}")
        print("\nYou can try installing them manually later:")
        for package in failed:
            print(f"  pip install {package}")
        
        if len(failed) > 3:
            enable_long_paths_message()
    else:
        print("\n✅ All packages installed successfully!")
    
    # Step 4: Create project structure
    create_project_structure()
    
    # Step 5: Final instructions
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\n✅ Next Steps:")
    print("1. Run: python 01_data_collection.py")
    print("2. This will download 15 years of NVDA data")
    print("3. Check the roadmap: trading_strategy_roadmap.md")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()