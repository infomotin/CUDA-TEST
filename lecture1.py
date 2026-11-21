import time
import torch

def lecture1_memory_workflow():
    print("=" * 70)
    print(" LECTURE 1 SLIDE CODE: Host-Device Workflow & Performance Profiling")
    print("=" * 70)

    # 1. Device Identification & Specs (Target: GTX 1650)
    if not torch.cuda.is_available():
        raise SystemExit("Error: CUDA is not available on this system!")
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    
    print(f"[INFO] Target Hardware Detected: {gpu_name}")
    print(f"[INFO] Available VRAM: {total_vram:.2f} GB\n")

    # 2. Data Initialization on Host (CPU RAM)
    # Let's create 20 Million elements for heavy parallel load
    N = 20_000_000
    print(f"[INFO] Allocating Host Tensor of size: {N:,} elements (Float32)...")
    h_data = torch.linspace(1.0, 50.0, N, dtype=torch.float32)

    # --- CPU BENCHMARK ---
    t_cpu_start = time.perf_counter()
    h_cpu_result = torch.sin(h_data) + torch.cos(h_data)
    t_cpu_duration = (time.perf_counter() - t_cpu_start) * 1000
    print(f"[BENCHMARK] CPU Execution Time: {t_cpu_duration:.2f} ms")

    # --- GPU WORKFLOW: HOST TO DEVICE (H2D) ---
    t_h2d_start = time.perf_counter()
    d_data = h_data.to(device)  # Transfering RAM -> VRAM
    torch.cuda.synchronize()    # Ensuring transfer completion
    t_h2d_duration = (time.perf_counter() - t_h2d_start) * 1000
    print(f"[WORKFLOW] H2D Transfer Time: {t_h2d_duration:.2f} ms")

    # --- GPU BENCHMARK (CUDA KERNEL EXECUTION) ---
    # Warm-up run to avoid JIT compilation overhead overhead
    _ = torch.sin(d_data) + torch.cos(d_data)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    d_gpu_result = torch.sin(d_data) + torch.cos(d_data)
    end_event.record()
    torch.cuda.synchronize()

    t_gpu_duration = start_event.elapsed_time(end_event)
    print(f"[BENCHMARK] GPU Kernel Execution Time: {t_gpu_duration:.2f} ms")

    # --- GPU WORKFLOW: DEVICE TO HOST (D2H) ---
    t_d2h_start = time.perf_counter()
    h_final_output = d_gpu_result.cpu()  # Transfering VRAM -> RAM
    torch.cuda.synchronize()
    t_d2h_duration = (time.perf_counter() - t_d2h_start) * 1000
    print(f"[WORKFLOW] D2H Transfer Time: {t_d2h_duration:.2f} ms")

    # Validation
    is_correct = torch.allclose(h_cpu_result, h_final_output, atol=1e-5)
    print(f"\n[VERIFICATION] Numerical Accuracy Check (CPU vs GPU): {is_correct}")
    print("=" * 70)

if __name__ == "__main__":
    lecture1_memory_workflow()