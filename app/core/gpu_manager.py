import torch
import logging
import traceback
import time # get_detailed_gpu_utilization에서 사용
import subprocess # get_detailed_gpu_utilization에서 사용
import psutil # cleanup_excel_processes에서 사용

# logger 정의 추가
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """GPU 사용 가능성 및 현재 디바이스 정보를 확인하고 로깅하는 함수"""
    try:
        logger.info("=" * 60)
        logger.info("🔍 GPU 및 디바이스 정보 확인")
        logger.info("=" * 60)
        
        # CUDA 사용 가능성 확인
        cuda_available = torch.cuda.is_available()
        logger.info(f"🔧 CUDA 사용 가능: {cuda_available}")
        
        if cuda_available:
            # GPU 개수 및 정보
            gpu_count = torch.cuda.device_count()
            logger.info(f"🎮 사용 가능한 GPU 개수: {gpu_count}")
            
            # 각 GPU 정보 출력
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory = gpu_props.total_memory / 1024**3  # GB
                    
                    # 추가 정보 수집 (안전한 방법)
                    compute_capability = f"{getattr(gpu_props, 'major', 0)}.{getattr(gpu_props, 'minor', 0)}"
                    
                    logger.info(f"  📱 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB, Compute {compute_capability})")
                    
                    # 멀티프로세서 개수 (존재하는 경우)
                    if hasattr(gpu_props, 'multiprocessor_count'):
                        mp_count = gpu_props.multiprocessor_count
                        logger.info(f"    🔧 멀티프로세서: {mp_count}개")
                    elif hasattr(gpu_props, 'multi_processor_count'):
                        mp_count = gpu_props.multi_processor_count
                        logger.info(f"    🔧 멀티프로세서: {mp_count}개")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ GPU {i} 정보 수집 실패: {str(e)}")
                    logger.info(f"  📱 GPU {i}: 정보 확인 불가")
            
            # 현재 GPU 디바이스
            current_device = torch.cuda.current_device()
            current_gpu_name = torch.cuda.get_device_name(current_device)
            logger.info(f"🎯 현재 사용 중인 GPU: {current_device} ({current_gpu_name})")
            
                    # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            cached = torch.cuda.memory_reserved(current_device) / 1024**3
            logger.info(f"💾 GPU 메모리 사용량: {allocated:.2f}GB (할당) / {cached:.2f}GB (캐시)")
            
            # 간단한 GPU 테스트 수행
            try:
                logger.info("🧪 GPU 기능 테스트 시작...")
                test_tensor = torch.randn(1000, 1000, device=current_device)
                test_result = torch.matmul(test_tensor, test_tensor.T)
                
                # 테스트 후 메모리 사용량 재확인
                allocated_after = torch.cuda.memory_allocated(current_device) / 1024**3
                cached_after = torch.cuda.memory_reserved(current_device) / 1024**3
                logger.info(f"✅ GPU 테스트 완료! 테스트 후 메모리: {allocated_after:.2f}GB (할당) / {cached_after:.2f}GB (캐시)")
                
                # 테스트 텐서 정리
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
                # 정리 후 메모리 상태
                allocated_final = torch.cuda.memory_allocated(current_device) / 1024**3
                cached_final = torch.cuda.memory_reserved(current_device) / 1024**3
                logger.info(f"🧹 메모리 정리 후: {allocated_final:.2f}GB (할당) / {cached_final:.2f}GB (캐시)")
                
            except Exception as e:
                logger.error(f"❌ GPU 테스트 실패: {str(e)}")
        
        # 사용할 디바이스 결정
        device = torch.device('cuda' if cuda_available else 'cpu')
        logger.info(f"⚡ 모델 학습/예측에 사용할 디바이스: {device}")
        
        # PyTorch 버전 정보
        logger.info(f"🔢 PyTorch 버전: {torch.__version__}")
        
        # CUDNN 정보 (CUDA 사용 가능한 경우)
        if cuda_available:
            try:
                logger.info(f"🔧 cuDNN 버전: {torch.backends.cudnn.version()}")
                logger.info(f"🔧 cuDNN 활성화: {torch.backends.cudnn.enabled}")
            except Exception as e:
                logger.warning(f"⚠️ cuDNN 정보 확인 실패: {str(e)}")
                
            # GPU 속성 디버깅 정보 (첫 번째 GPU만)
            if gpu_count > 0:
                try:
                    props = torch.cuda.get_device_properties(0)
                    available_attrs = [attr for attr in dir(props) if not attr.startswith('_')]
                    logger.info(f"🔍 사용 가능한 GPU 속성들: {available_attrs}")
                except Exception as e:
                    logger.warning(f"⚠️ GPU 속성 확인 실패: {str(e)}")
        
        logger.info("=" * 60)
        
        return device, cuda_available
        
    except Exception as e:
        logger.error(f"❌ GPU 정보 확인 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return torch.device('cpu'), False

def get_detailed_gpu_utilization():
    """nvidia-smi를 사용하여 상세한 GPU 활용률을 확인하는 함수"""
    try:
        import subprocess
        
        # 기본 활용률 정보
        basic_result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        # 상세 활용률 정보 (Encoder, Decoder 등)
        detailed_result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        # 실행 중인 프로세스 정보
        process_result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        gpu_stats = []
        
        if basic_result.returncode == 0 and basic_result.stdout.strip():
            basic_lines = basic_result.stdout.strip().split('\n')
            detailed_lines = detailed_result.stdout.strip().split('\n') if detailed_result.returncode == 0 else []
            
            for i, line in enumerate(basic_lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_util = parts[0].strip()
                    mem_util = parts[1].strip()
                    temp = parts[2].strip()
                    power_draw = parts[3].strip() if len(parts) > 3 else 'N/A'
                    power_limit = parts[4].strip() if len(parts) > 4 else 'N/A'
                    
                    # 상세 정보 추가
                    encoder_util = 'N/A'
                    decoder_util = 'N/A'
                    
                    if i < len(detailed_lines):
                        detailed_parts = detailed_lines[i].split(', ')
                        if len(detailed_parts) >= 4:
                            encoder_util = detailed_parts[2].strip()
                            decoder_util = detailed_parts[3].strip()
                    
                    gpu_stat = {
                        'gpu_id': i,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': mem_util,
                        'encoder_utilization': encoder_util,
                        'decoder_utilization': decoder_util,
                        'temperature': temp,
                        'power_draw': power_draw,
                        'power_limit': power_limit,
                        'measurement_method': 'nvidia-smi',
                        'timestamp': time.time()
                    }
                    
                    gpu_stats.append(gpu_stat)
        
        # 실행 중인 프로세스 정보 추가
        if process_result.returncode == 0 and process_result.stdout.strip():
            process_lines = process_result.stdout.strip().split('\n')
            compute_processes = []
            for line in process_lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    compute_processes.append({
                        'pid': parts[0].strip(),
                        'name': parts[1].strip(),
                        'gpu_memory_mb': parts[2].strip()
                    })
            
            # 첫 번째 GPU에 프로세스 정보 추가
            if gpu_stats:
                gpu_stats[0]['compute_processes'] = compute_processes
        
        return gpu_stats
        
    except Exception as e:
        logger.warning(f"⚠️ 상세 GPU 활용률 확인 실패: {str(e)}")
        return None

def get_gpu_utilization():
    """nvidia-smi를 사용하여 GPU 활용률을 확인하는 함수 (기존 호환성 유지)"""
    detailed_stats = get_detailed_gpu_utilization()
    if detailed_stats:
        # 기존 형식으로 변환
        return [{
            'gpu_id': stat['gpu_id'],
            'gpu_utilization': stat['gpu_utilization'],
            'memory_utilization': stat['memory_utilization'],
            'temperature': stat['temperature'],
            'power_draw': stat['power_draw'],
            'power_limit': stat['power_limit']
        } for stat in detailed_stats]
    return None

def compare_gpu_monitoring_methods():
    """다양한 GPU 모니터링 방법을 비교하는 함수"""
    comparison_results = {
        'nvidia_smi': None,
        'torch_cuda': None,
        'monitoring_notes': []
    }
    
    try:
        # nvidia-smi 결과
        nvidia_stats = get_detailed_gpu_utilization()
        if nvidia_stats:
            comparison_results['nvidia_smi'] = nvidia_stats[0]  # 첫 번째 GPU
            comparison_results['monitoring_notes'].append(
                "nvidia-smi: CUDA 연산 활용률 측정 (ML/AI 작업에 정확)"
            )
        
        # PyTorch CUDA 정보
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            cached = torch.cuda.memory_reserved(device_id) / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            comparison_results['torch_cuda'] = {
                'allocated_memory_gb': round(allocated, 3),
                'cached_memory_gb': round(cached, 3),
                'total_memory_gb': round(total, 1),
                'memory_usage_percent': round((allocated / total) * 100, 2)
            }
            comparison_results['monitoring_notes'].append(
                "PyTorch CUDA: 실제 PyTorch 텐서 메모리 사용량"
            )
        
        comparison_results['monitoring_notes'].extend([
            "Windows 작업 관리자: 주로 3D 그래픽 엔진 활용률 (CUDA와 다름)",
            "nvidia-smi GPU 활용률: CUDA 연산 활용률 (ML/AI 작업)",
            "nvidia-smi Encoder/Decoder: 비디오 인코딩/디코딩 활용률",
            "측정 시점에 따라 순간적인 변화가 클 수 있음"
        ])
        
    except Exception as e:
        comparison_results['error'] = str(e)
    
    return comparison_results

def log_device_usage(device, context=""):
    """특정 상황에서의 디바이스 사용 정보를 로깅하는 함수"""
    try:
        context_str = f"[{context}] " if context else ""
        logger.info(f"🎯 {context_str}사용 중인 디바이스: {device}")
        
        if device.type == 'cuda' and torch.cuda.is_available():
            device_id = device.index if device.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            cached = torch.cuda.memory_reserved(device_id) / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            logger.info(f"💾 {context_str}GPU 메모리: {allocated:.3f}GB 사용 / {total:.1f}GB 전체 (캐시: {cached:.3f}GB)")
            
            # 메모리 사용률 계산 및 상태 표시
            usage_percentage = (allocated / total) * 100
            cache_percentage = (cached / total) * 100
            
            if allocated > 0.001:  # 1MB 이상 사용 중인 경우
                logger.info(f"📊 {context_str}메모리 사용률: {usage_percentage:.2f}% (캐시: {cache_percentage:.2f}%)")
                
                if usage_percentage > 80:
                    logger.warning(f"⚠️ {context_str}GPU 메모리 사용률이 높습니다: {usage_percentage:.1f}%")
                elif usage_percentage > 50:
                    logger.info(f"📈 {context_str}GPU 메모리 사용률: {usage_percentage:.1f}% (정상)")
            else:
                logger.info(f"💭 {context_str}현재 GPU 메모리 사용량 없음 (대기 상태)")
            
            # GPU 활용률 확인 (상세)
            detailed_stats = get_detailed_gpu_utilization()
            if detailed_stats and len(detailed_stats) > device_id:
                stat = detailed_stats[device_id]
                logger.info(f"⚡ {context_str}CUDA 활용률: {stat['gpu_utilization']}% (메모리: {stat['memory_utilization']}%)")
                logger.info(f"🎬 {context_str}Encoder: {stat['encoder_utilization']}%, Decoder: {stat['decoder_utilization']}%")
                logger.info(f"🌡️ {context_str}GPU 온도: {stat['temperature']}°C, 전력: {stat['power_draw']}/{stat['power_limit']}W")
                
                # 실행 중인 프로세스 정보
                if 'compute_processes' in stat and stat['compute_processes']:
                    process_count = len(stat['compute_processes'])
                    logger.info(f"🔄 {context_str}CUDA 프로세스: {process_count}개")
                    for proc in stat['compute_processes'][:3]:  # 최대 3개까지만 표시
                        logger.info(f"    📱 PID {proc['pid']}: {proc['name']} ({proc['gpu_memory_mb']}MB)")
                
                # 낮은 활용률 분석 및 설명
                try:
                    gpu_util_num = float(stat['gpu_utilization'])
                    if gpu_util_num < 10:
                        logger.warning(f"⚠️ {context_str}CUDA 활용률이 매우 낮습니다: {gpu_util_num}%")
                        logger.info(f"💡 {context_str}참고: 작업 관리자의 GPU는 3D 그래픽을, nvidia-smi는 CUDA 연산을 측정합니다")
                        logger.info(f"💡 {context_str}ML/AI 작업에서는 nvidia-smi의 CUDA 활용률이 정확합니다")
                    elif gpu_util_num < 30:
                        logger.info(f"📉 {context_str}CUDA 활용률이 낮습니다: {gpu_util_num}% - 배치 크기 증가 고려")
                    else:
                        logger.info(f"✅ {context_str}CUDA 활용률이 양호합니다: {gpu_util_num}%")
                except:
                    pass
                
            # GPU 활성 프로세스 수 확인 (가능한 경우)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    active_processes = len(result.stdout.strip().split('\n'))
                    logger.info(f"🔄 {context_str}GPU에서 실행 중인 프로세스: {active_processes}개")
            except:
                pass  # nvidia-smi가 없거나 실행 실패 시 무시
                
        elif device.type == 'cpu':
            logger.info(f"🖥️ {context_str}CPU 모드로 실행 중")
            
    except Exception as e:
        logger.error(f"❌ 디바이스 사용 정보 로깅 중 오류: {str(e)}")
