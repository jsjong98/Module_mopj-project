#!/usr/bin/env python
"""
Flask 애플리케이션 실행 스크립트
"""
import os
import sys
import argparse
import logging

# 현재 스크립트의 디렉토리 (app 폴더)
current_dir = os.path.dirname(os.path.abspath(__file__))

# app 폴더가 이미 현재 디렉토리이므로, 부모 디렉토리를 Python 경로에 추가
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 이제 app 모듈을 임포트할 수 있음
from app import app, logger

def parse_arguments():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='MOPJ Price Prediction Flask Server')
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='호스트 주소 (기본값: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='포트 번호 (기본값: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='프로덕션 모드로 실행 (디버그 비활성화, 0.0.0.0 바인딩)'
    )
    return parser.parse_args()

def check_dependencies():
    """필수 의존성 확인"""
    required_modules = [
        'flask',
        'flask_cors',
        'pandas',
        'numpy',
        'torch',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ 다음 모듈이 설치되지 않았습니다: {', '.join(missing_modules)}")
        logger.error("pip install -r requirements.txt 명령으로 설치해주세요.")
        sys.exit(1)

def print_startup_info(host, port, debug_mode):
    """서버 시작 정보 출력"""
    print("\n" + "="*60)
    print("🚀 MOPJ Price Prediction Server Starting...")
    print("="*60)
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug Mode: {'ON' if debug_mode else 'OFF'}")
    print(f"📁 Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 Max File Size: {app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024:.0f} MB")
    print(f"🏠 Working Directory: {os.getcwd()}")
    print("="*60)
    print(f"🌐 Server URL: http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/api/docs")
    print("="*60)
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")

def main():
    """메인 실행 함수"""
    # 커맨드라인 인자 파싱
    args = parse_arguments()
    
    # 프로덕션 모드 설정
    if args.production:
        host = '0.0.0.0'  # 모든 네트워크 인터페이스에서 접근 가능
        debug_mode = False
        logger.info("🏭 Running in PRODUCTION mode")
    else:
        host = args.host
        debug_mode = args.debug
        if debug_mode:
            logger.info("🐛 Running in DEBUG mode")
    
    port = args.port
    
    # 의존성 확인
    check_dependencies()
    
    # 시작 정보 출력
    print_startup_info(host, port, debug_mode)
    
    try:
        # Flask 앱 실행
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            use_reloader=debug_mode,  # 디버그 모드에서만 자동 리로드
            threaded=True  # 멀티스레드 활성화
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        logger.info("Server shutdown gracefully")
    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}")
        raise

if __name__ == '__main__':
    main()