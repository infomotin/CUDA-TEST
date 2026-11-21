import sys
import time
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def lecture3_pinned_and_streams():
    print("=" * 72)
    print(" LECTURE 3 SLIDE CODE: Pinned Memory, Streams & VRAM Management")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA ডিভাইস পাওয়া যায়নি!")

    device = torch.device("cuda:0")
    print(f"[DEVICE] {torch.cuda.get_device_name(0)}")

    N = 2048

    # 1. Standard (Pageable) vs Pinned Host Tensor
    a_std = torch.randn((N, N), dtype=torch.float32)
    b_std = torch.randn((N, N), dtype=torch.float32)

    a_pin = a_std.pin_memory()
    b_pin = b_std.pin_memory()

    # Warm-up
    _ = torch.matmul(a_pin.to(device, non_blocking=True), b_pin.to(device, non_blocking=True))
    torch.cuda.synchronize()

    # 2. Measure Pageable H2D
    t0 = time.perf_counter()
    _a1 = a_std.to(device)
    _b1 = b_std.to(device)
    torch.cuda.synchronize()
    std_h2d_ms = (time.perf_counter() - t0) * 1000
    del _a1, _b1
    torch.cuda.empty_cache()

    # 3. Measure Pinned non_blocking H2D
    t1 = time.perf_counter()
    a_gpu = a_pin.to(device, non_blocking=True)
    b_gpu = b_pin.to(device, non_blocking=True)
    torch.cuda.synchronize()
    pin_h2d_ms = (time.perf_counter() - t1) * 1000

    # 4. Custom CUDA Stream Execution
    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)

    torch.cuda.reset_peak_memory_stats(device)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    with torch.cuda.stream(s1):
        res1 = torch.matmul(a_gpu, b_gpu)

    with torch.cuda.stream(s2):
        res2 = torch.matmul(b_gpu, a_gpu)

    # মূল ডিফল্ট স্ট্রিমকে s1 এবং s2 শেষ হওয়ার জন্য অপেক্ষা করানো
    torch.cuda.current_stream(device).wait_stream(s1)
    torch.cuda.current_stream(device).wait_stream(s2)
    end_evt.record()
    torch.cuda.synchronize()

    dual_stream_ms = start_evt.elapsed_time(end_evt)
    alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"[TRANSFER] Pageable H2D (2x 2048x2048): {std_h2d_ms:.2f} ms")
    print(f"[TRANSFER] Pinned H2D (2x 2048x2048):   {pin_h2d_ms:.2f} ms")
    print(f"[STREAMS]  Two Parallel-Submitted MatMuls: {dual_stream_ms:.2f} ms")
    print(f"[VRAM] Current Allocated: {alloc_mb:.2f} MB | Reserved: {reserved_mb:.2f} MB | Peak: {peak_mb:.2f} MB")

    # ক্লিনআপ
    del a_gpu, b_gpu, res1, res2
    torch.cuda.empty_cache()
    print(f"[VRAM AFTER CLEANUP] Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    print("=" * 72)


if __name__ == "__main__":
    lecture3_pinned_and_streams()