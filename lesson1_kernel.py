import time
import torch

# কনসোলে UTF-8 এনকোডিং নিশ্চিত করা
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("         PYTORCH GPU WAVE & MATH BENCHMARK")
print("=" * 60)

# ডিভাইস সেটআপ
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"ব্যবহৃত ডিভাইস: {device} ({torch.cuda.get_device_name(0)})")

N = 10_000_000  # ১ কোটি ডেটা পয়েন্ট
print(f"ডেটা সাইজ: {N:,} এলিমেন্ট")

# ১. CPU-তে ডেটা তৈরি ও হিসাব
h_in = torch.linspace(1.0, 100.0, N, dtype=torch.float32)

t0 = time.perf_counter()
h_out_cpu = (h_in * h_in) + (h_in * 0.5) - 1.25
cpu_time = (time.perf_counter() - t0) * 1000
print(f"CPU সময়: {cpu_time:.2f} ms")

# ২. GPU-তে ডেটা পাঠানো এবং প্যারালাল হিসাব (CUDA Acceleration)
h_in_gpu = h_in.to(device)

# ওয়ার্ম-আপ
_ = (h_in_gpu * h_in_gpu) + (h_in_gpu * 0.5) - 1.25
torch.cuda.synchronize()

# আসল সময় পরিমাপ
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
h_out_gpu = (h_in_gpu * h_in_gpu) + (h_in_gpu * 0.5) - 1.25
end_event.record()
torch.cuda.synchronize()

gpu_time_ms = start_event.elapsed_time(end_event)
print(f"GPU (CUDA) সময়: {gpu_time_ms:.2f} ms")

# রেজাল্ট চেক
print(f"রেজাল্ট সঠিক কি না: {torch.allclose(h_out_cpu, h_out_gpu.cpu())}")
print("=" * 60)