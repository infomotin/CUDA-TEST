import sys
import time
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def lecture4_precision_and_reduction():
    print("=" * 72)
    print(" LECTURE 4 SLIDE CODE: Broadcasting, Reduction & FP32 vs FP16")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA ডিভাইস পাওয়া যায়নি!")

    device = torch.device("cuda:0")
    print(f"[DEVICE] {torch.cuda.get_device_name(0)}")

    M, K, N = 3072, 3072, 2048
    print(f"[WORKLOAD] A({M}x{K}) @ B({K}x{N}) + bias({N},) -> Reduction\n")

    # --- 1. FP32 Execution ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    a32 = torch.randn((M, K), device=device, dtype=torch.float32)
    b32 = torch.randn((K, N), device=device, dtype=torch.float32)
    bias32 = torch.randn((N,), device=device, dtype=torch.float32)

    # Warm-up
    _ = (a32 @ b32 + bias32).mean()
    torch.cuda.synchronize()

    start_32 = torch.cuda.Event(enable_timing=True)
    end_32 = torch.cuda.Event(enable_timing=True)

    start_32.record()
    out32 = a32 @ b32 + bias32   # MatMul + Broadcasting add
    red32 = out32.mean()         # Reduction to scalar
    end_32.record()
    torch.cuda.synchronize()

    time_32 = start_32.elapsed_time(end_32)
    peak_32_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del a32, b32, bias32, out32
    torch.cuda.empty_cache()

    # --- 2. FP16 Execution ---
    torch.cuda.reset_peak_memory_stats(device)

    a16 = torch.randn((M, K), device=device, dtype=torch.float16)
    b16 = torch.randn((K, N), device=device, dtype=torch.float16)
    bias16 = torch.randn((N,), device=device, dtype=torch.float16)

    # Warm-up
    _ = (a16 @ b16 + bias16).float().mean()
    torch.cuda.synchronize()

    start_16 = torch.cuda.Event(enable_timing=True)
    end_16 = torch.cuda.Event(enable_timing=True)

    start_16.record()
    out16 = a16 @ b16 + bias16
    red16 = out16.float().mean()  # Accumulation in float32 for better numerical stability
    end_16.record()
    torch.cuda.synchronize()

    time_16 = start_16.elapsed_time(end_16)
    peak_16_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"[FP32] Time: {time_32:.2f} ms | Peak VRAM: {peak_32_mb:.2f} MB | Mean: {red32.item():.6f}")
    print(f"[FP16] Time: {time_16:.2f} ms | Peak VRAM: {peak_16_mb:.2f} MB | Mean: {red16.item():.6f}")
    print(f"[VRAM SAVED] {peak_32_mb - peak_16_mb:.2f} MB saved with FP16")
    print("=" * 72)


if __name__ == "__main__":
    lecture4_precision_and_reduction()