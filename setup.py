#!/usr/bin/env python3
"""
Setup script for Web Task Automator
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Web Task Automator...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: You're not in a virtual environment.")
        print("   It's recommended to create a virtual environment first:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print()
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install Playwright browsers
    if not run_command("playwright install chromium", "Installing Playwright browsers"):
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo test the installation, run:")
    print("  python browser_automation.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
