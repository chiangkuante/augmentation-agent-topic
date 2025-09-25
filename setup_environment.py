#!/usr/bin/env python3
"""
Environment setup script for augmentation-agent-topic.

This script helps users set up their environment configuration
and validates the setup.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional

def print_banner():
    """Print setup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║            AUGMENTATION AGENT TOPIC - SETUP                 ║
║                Environment Configuration                     ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def detect_package_manager() -> str:
    """Detect available package manager."""
    import subprocess
    import shutil

    managers = []

    # Check for mamba
    if shutil.which('mamba'):
        try:
            result = subprocess.run(['mamba', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                managers.append('mamba')
        except:
            pass

    # Check for conda
    if shutil.which('conda'):
        try:
            result = subprocess.run(['conda', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                managers.append('conda')
        except:
            pass

    # Check for pip (always available with Python)
    if shutil.which('pip'):
        managers.append('pip')

    return managers

def check_required_packages() -> Dict[str, bool]:
    """Check if required packages are installed."""
    required_packages = [
        ('python-dotenv', 'dotenv'),  # (display_name, import_name)
        ('openai', 'openai'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('bertopic', 'bertopic'),
        ('sentence-transformers', 'sentence_transformers'),
        ('transformers', 'transformers'),
        ('umap-learn', 'umap'),
        ('hdbscan', 'hdbscan'),
        ('torch', 'torch'),
        ('plotly', 'plotly'),
        ('nltk', 'nltk')
    ]

    results = {}
    print("\n📦 Checking required packages...")

    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {display_name}")
            results[display_name] = True
        except ImportError:
            print(f"  ❌ {display_name}")
            results[display_name] = False

    return results

def create_environment_file(env_type: str = "development") -> bool:
    """Create .env file from template."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    template_file = project_root / f".env.{env_type}"
    example_file = project_root / ".env.example"

    # Check if .env already exists
    if env_file.exists():
        response = input(f"\n⚠️  .env file already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("📝 Keeping existing .env file")
            return True

    # Use specific environment template if exists, otherwise use example
    source_file = template_file if template_file.exists() else example_file

    if not source_file.exists():
        print(f"❌ Template file not found: {source_file}")
        return False

    try:
        shutil.copy2(source_file, env_file)
        print(f"✅ Created .env file from {source_file.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def validate_environment() -> Dict[str, bool]:
    """Validate environment configuration."""
    print("\n🔍 Validating environment configuration...")

    try:
        # Import after ensuring dotenv is available
        from config.env_loader import load_environment
        env_loader = load_environment()

        is_valid, errors = env_loader.validate_config()

        if is_valid:
            print("✅ Environment configuration is valid")
        else:
            print("❌ Environment configuration errors:")
            for error in errors:
                print(f"    - {error}")

        return {"valid": is_valid, "errors": errors}

    except Exception as e:
        print(f"❌ Failed to validate environment: {e}")
        return {"valid": False, "errors": [str(e)]}

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")

    project_root = Path(__file__).parent
    directories = [
        "data",
        "results",
        "results/topics",
        "results/scores",
        "results/optimization",
        "results/resilience",
        "results/checkpoints",
        "logs",
        "tests",
        "notebooks"
    ]

    for dir_path in directories:
        full_path = project_root / dir_path
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path}")
        except Exception as e:
            print(f"  ❌ {dir_path}: {e}")

def get_api_key_setup() -> Optional[str]:
    """Interactive API key setup."""
    print("\n🔑 OpenAI API Key Setup")
    print("You need an OpenAI API key to use this system.")
    print("Get your key from: https://platform.openai.com/api-keys")

    current_key = os.getenv("OPENAI_API_KEY")
    if current_key and current_key != "your-openai-api-key-here":
        print(f"✅ API key already set: {current_key[:8]}...{current_key[-4:]}")
        return current_key

    while True:
        api_key = input("\nEnter your OpenAI API key (or 'skip' to continue without): ").strip()

        if api_key.lower() == 'skip':
            print("⚠️  Skipping API key setup - you'll need to set OPENAI_API_KEY manually")
            return None

        if api_key.startswith('sk-') and len(api_key) > 20:
            return api_key

        print("❌ Invalid API key format. Should start with 'sk-' and be longer than 20 characters")

def update_env_file_with_api_key(api_key: str):
    """Update .env file with API key."""
    env_file = Path(__file__).parent / ".env"

    if not env_file.exists():
        print("❌ .env file not found")
        return False

    try:
        # Read current content
        with open(env_file, 'r') as f:
            content = f.read()

        # Replace API key line
        lines = content.split('\n')
        updated_lines = []
        api_key_found = False

        for line in lines:
            if line.startswith('OPENAI_API_KEY='):
                updated_lines.append(f'OPENAI_API_KEY={api_key}')
                api_key_found = True
            else:
                updated_lines.append(line)

        # Add API key if not found
        if not api_key_found:
            updated_lines.insert(1, f'OPENAI_API_KEY={api_key}')

        # Write back
        with open(env_file, 'w') as f:
            f.write('\n'.join(updated_lines))

        print("✅ Updated .env file with API key")
        return True

    except Exception as e:
        print(f"❌ Failed to update .env file: {e}")
        return False

def show_next_steps():
    """Show next steps to user."""
    print(f"\n🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Place your data file in the data/ directory:")
    print("   - Copy corpus_semantic_chunks.csv to data/")

    print("\n2. Test your setup:")
    print("   python src/main.py --validate-config")
    print("   python src/main.py --show-config")

    print("\n3. Run a small test:")
    print("   python src/main.py --sample-size 100 --budget 5.0 --max-iterations 2")

    print("\n4. Run the full pipeline:")
    print("   python src/main.py --budget 50.0")

    print("\n📚 For more options:")
    print("   python src/main.py --help")

def suggest_installation_method(missing_packages: list, available_managers: list):
    """Suggest the best installation method based on available managers."""
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\n📋 Installation suggestions:")

    if 'mamba' in available_managers:
        print("🎯 Recommended (Mamba):")
        print("  mamba env create -f mamba-environment.yml")
        print("  mamba activate augmentation-agent-topic")
        print("\n🔄 Alternative (Mamba + pip):")
        print("  mamba create -n augmentation-agent-topic python=3.10")
        print("  mamba activate augmentation-agent-topic")
        print("  mamba install pandas numpy scikit-learn python-dotenv")
        print("  pip install bertopic openai transformers")

    elif 'conda' in available_managers:
        print("🎯 Recommended (Conda):")
        print("  conda env create -f environment.yml")
        print("  conda activate augmentation-agent-topic")

    if 'pip' in available_managers:
        print("🐍 Pure pip installation:")
        print("  python -m venv venv")
        print("  # Activate venv (platform dependent)")
        print("  pip install -r requirements.txt")

    print("\n📚 For detailed instructions, see: INSTALLATION.md")

def main():
    """Main setup function."""
    print_banner()

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Detect package managers
    available_managers = detect_package_manager()
    print(f"\n🔍 Available package managers: {', '.join(available_managers)}")

    # Step 3: Check packages
    package_results = check_required_packages()
    missing_packages = [pkg for pkg, installed in package_results.items() if not installed]

    if missing_packages:
        suggest_installation_method(missing_packages, available_managers)

        response = input("\nContinue setup anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            sys.exit(1)

    # Step 3: Choose environment
    print("\n🌍 Environment Selection:")
    print("1. Development (recommended for testing)")
    print("2. Production (for full runs)")
    print("3. Custom (use .env.example template)")

    while True:
        choice = input("\nSelect environment (1-3): ").strip()
        if choice == '1':
            env_type = "development"
            break
        elif choice == '2':
            env_type = "production"
            break
        elif choice == '3':
            env_type = "example"
            break
        else:
            print("❌ Please enter 1, 2, or 3")

    # Step 4: Create .env file
    if not create_environment_file(env_type):
        sys.exit(1)

    # Step 5: Setup API key
    api_key = get_api_key_setup()
    if api_key:
        update_env_file_with_api_key(api_key)

    # Step 6: Setup directories
    setup_directories()

    # Step 7: Validate setup
    validation_result = validate_environment()

    if validation_result["valid"]:
        show_next_steps()
        print(f"\n✅ Setup completed successfully!")
    else:
        print(f"\n⚠️  Setup completed with warnings.")
        print("Please review the configuration errors above.")

    return 0 if validation_result["valid"] else 1

if __name__ == "__main__":
    sys.exit(main())