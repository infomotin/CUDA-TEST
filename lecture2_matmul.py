import sys
import time
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def lecture2_matrix_benchmark():
    print("=" * 72)
    print(" LECTURE 2 SLIDE CODE: Matrix Multiplication & VRAM Profiling")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA ডিভাইস পাওয়া যায়নি!")

    device = torch.device("cuda:0")
    print(f"[DEVICE] {torch.cuda.get_device_name(0)}")

    # দুটি বর্গাকার ম্যাট্রিক্স: A (2048x2048) এবং B (2048x2048)
    # মোট গুণফল ম্যাট্রিক্স C হবে 2048x2048 (~4.19M এলিমেন্ট)
    N = 4096  # 4096x4096 ম্যাট্রিক্সের জন্য
    print(f"[MATRIX] A({N}x{N}) @ B({N}x{N}) in float32\n")

    # 1. Host (CPU) ম্যাট্রিক্স তৈরি
    a_cpu = torch.randn((N, N), dtype=torch.float32)
    b_cpu = torch.randn((N, N), dtype=torch.float32)

    # --- CPU Benchmark ---
    t0 = time.perf_counter()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"[CPU] MatMul Time: {cpu_ms:.2f} ms")

    # 2. VRAM ক্লিয়ার ও প্রাথমিক মেমোরি অবস্থা দেখা
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # --- H2D Transfer ---
    t_h2d = time.perf_counter()
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t_h2d) * 1000

    # --- GPU Warm-up ---
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    # --- GPU Timed Execution (CUDA Events) ---
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    end_evt.record()
    torch.cuda.synchronize()

    gpu_ms = start_evt.elapsed_time(end_evt)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # --- D2H Transfer ---
    t_d2h = time.perf_counter()
    c_from_gpu = c_gpu.cpu()
    torch.cuda.synchronize()
    d2h_ms = (time.perf_counter() - t_d2h) * 1000

    # নির্ভুলতা যাচাই
    is_close = torch.allclose(c_cpu, c_from_gpu, rtol=1e-3, atol=1e-3)
    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0.0

    print(f"[GPU] H2D Transfer Time: {h2d_ms:.2f} ms")
    print(f"[GPU] MatMul Kernel Time: {gpu_ms:.2f} ms")
    print(f"[GPU] D2H Transfer Time: {d2h_ms:.2f} ms")
    print(f"[GPU] Peak VRAM Allocated: {peak_vram_mb:.2f} MB")
    print(f"[SPEEDUP] Compute Speedup (CPU/GPU): {speedup:.2f}x")
    print(f"[VERIFY] Numerical Match: {is_close}")
    print("=" * 72)


if __name__ == "__main__":
    lecture2_matrix_benchmark()