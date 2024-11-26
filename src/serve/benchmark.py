import base64
import concurrent.futures
import time
import numpy as np
import requests
import torch
import timm
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import psutil
try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
import hydra
from omegaconf import DictConfig
import logging

# Disable timm logs
logging.getLogger("timm").setLevel(logging.WARNING)

# Constants
SERVER_URL = "http://localhost:8000/predict"
TEST_IMAGE_URL = 'data:image/jpeg;base64,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'

def get_baseline_throughput(batch_size, num_iterations, model_name, input_size, accelerator, num_classes):
    """Calculate baseline model throughput without API overhead"""
    device = accelerator if accelerator != 'auto' else ("cuda" if torch.cuda.is_available() else "cpu")
    # Create model and move to device
    model = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Create random input data with configurable input size
    x = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    throughputs = []
    
    # Warm-up run
    with torch.no_grad():
        model(x)
    
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # Each request processes batch_size images
        reqs_per_sec = (batch_size)/(t1-t0)  # Requests per second
        throughputs.append(reqs_per_sec)
    
    return np.mean(throughputs)  # reqs/sec

def prepare_test_payload():
    """Prepare a test image payload"""
    img_data = urlopen(TEST_IMAGE_URL).read()
    return base64.b64encode(img_data).decode('utf-8')

def send_request(payload):
    """Send a single request and measure response time"""
    start_time = time.time()
    response = requests.post(SERVER_URL, json={"image": payload})
    end_time = time.time()
    return end_time - start_time, response.status_code

def get_system_metrics():
    """Get current GPU and CPU usage"""
    metrics = {"cpu_usage": psutil.cpu_percent(0.1)}
    if GPU_AVAILABLE:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            metrics["gpu_usage"] = sum([gpu.utilization for gpu in gpu_stats.gpus])
        except Exception:
            metrics["gpu_usage"] = -1
    else:
        metrics["gpu_usage"] = -1
    return metrics

def benchmark_api(num_requests=100, concurrency_level=10, server_url=None):
    """Benchmark the API server"""
    global SERVER_URL
    SERVER_URL = server_url or SERVER_URL
    payload = prepare_test_payload()
    system_metrics = []
    
    start_benchmark_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        response_times = []
        status_codes = []
        
        # Collect system metrics during the benchmark
        while any(not f.done() for f in futures):
            system_metrics.append(get_system_metrics())
            time.sleep(0.1)
        
        for future in futures:
            response_time, status_code = future.result()
            response_times.append(response_time)
            status_codes.append(status_code)
    
    end_benchmark_time = time.time()
    total_benchmark_time = end_benchmark_time - start_benchmark_time
    
    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics])
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE else -1
    
    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": np.mean(response_times) * 1000,  # Convert to ms
        "success_rate": (status_codes.count(200) / num_requests) * 100,
        "requests_per_second": num_requests / total_benchmark_time,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu
    }

@hydra.main(version_base=None, config_path="../../configs", config_name="benchmark")
def run_benchmarks(cfg: DictConfig):
    """Run comprehensive benchmarks and create plots"""
    # Test different batch sizes for baseline throughput
    batch_sizes = cfg.batch_sizes
    baseline_throughput = []
    
    print("Running baseline throughput tests...")
    for batch_size in batch_sizes:
        reqs_per_sec = get_baseline_throughput(
            batch_size=batch_size,
            num_iterations=cfg.num_iterations,
            model_name=cfg.model_name,
            input_size=cfg.input_size,
            accelerator=cfg.accelerator,
            num_classes=cfg.num_classes
        )
        baseline_throughput.append(reqs_per_sec)
        print(f"Batch size {batch_size}: {reqs_per_sec:.2f} reqs/sec")
    
    # Test different concurrency levels for API
    concurrency_levels = cfg.api.concurrency_levels
    api_throughput = []
    cpu_usage = []
    gpu_usage = []
    
    print("\nRunning API benchmarks...")
    for concurrency in concurrency_levels:
        metrics = benchmark_api(
            num_requests=cfg.api.num_requests,
            concurrency_level=concurrency,
            server_url=cfg.api.server_url
        )
        api_throughput.append(metrics["requests_per_second"])
        cpu_usage.append(metrics["avg_cpu_usage"])
        gpu_usage.append(metrics["avg_gpu_usage"])
        print(f"Concurrency {concurrency}: {metrics['requests_per_second']:.2f} reqs/sec, "
              f"CPU: {metrics['avg_cpu_usage']:.1f}%, GPU: {metrics['avg_gpu_usage']:.1f}%")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Throughput comparison
    plt.subplot(1, 3, 1)
    plt.plot(batch_sizes, baseline_throughput, 'b-', label='Baseline Model')
    plt.plot(concurrency_levels, api_throughput, 'r-', label='API Server')
    plt.xlabel('Batch Size / Concurrency Level')
    plt.ylabel('Throughput (requests/second)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True)
    
    # CPU Usage
    plt.subplot(1, 3, 2)
    plt.plot(concurrency_levels, cpu_usage, 'g-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage')
    plt.grid(True)
    
    # GPU Usage
    plt.subplot(1, 3, 3)
    plt.plot(concurrency_levels, gpu_usage, 'm-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    run_benchmarks() 