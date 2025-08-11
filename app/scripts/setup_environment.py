#!/usr/bin/env python
"""
개발 환경 설정 도우미 스크립트
"""
import os
import sys
import subprocess

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        'uploads',
        'cache',
        'predictions',
        'logs',
        'holidays'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_requirements():
    """requirements.txt 설치"""
    print("\n📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully")

def check_gpu():
    """GPU 사용 가능 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎮 GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("\n💻 GPU not available, using CPU")
    except ImportError:
        print("\n⚠️  PyTorch not installed yet")

def main():
    print("🔧 Setting up MOPJ Prediction Server Environment...")
    print("="*50)
    
    # 디렉토리 생성
    setup_directories()
    
    # 의존성 설치
    try:
        install_requirements()
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        print("Please install manually: pip install -r requirements.txt")
    
    # GPU 확인
    check_gpu()
    
    print("\n✅ Environment setup complete!")
    print("Run 'python run.py' to start the server")

if __name__ == '__main__':
    main()
