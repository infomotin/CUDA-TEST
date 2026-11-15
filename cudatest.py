import sys
import time
import torch

# কনসোলে UTF-8 এনকোডিং নিশ্চিত করা
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("           CUDA GPGPU DIAGNOSTIC & BENCHMARK")
print("=" * 60)

# PyTorch ও পাইথন তথ্য
print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"PyTorch CUDA Build: {torch.version.cuda}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available    : {cuda_available}")
print("-" * 60)

if not cuda_available:
    print("\n[!] সমস্যা সনাক্তকরণ (DIAGNOSIS):")
    print("আপনার ডিভাইসে NVIDIA GeForce GTX 1650 রয়েছে যা অবশ্যই CUDA সাপোর্ট করে।")
    if torch.version.cuda is None:
        print("কারণ: আপনি PyTorch-এর CPU-only ভার্সন ব্যবহার করছেন (কোনো CUDA বিল্ড নেই)।")
    else:
        print(f"কারণ: PyTorch CUDA {torch.version.cuda} বিল্ড পেলেও ড্রাইভারের সাথে সংযুক্ত হতে পারছে না।")

    print("\n[+] সমাধান (SOLUTION):")
    print("১. আপনার বর্তমান PyTorch আনইনস্টল করুন:")
    print("   pip uninstall torch torchvision torchaudio -y")
    print("\n২. CUDA 12.4 বা 12.1 সাপোর্টেড PyTorch ইনস্টল করুন (টার্মিনালে চালান):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("\n৩. আপনার NVIDIA Driver ইনস্টল/আপডেট করা না থাকলে GeForce Experience থেকে ড্রাইভার আপডেট করুন।")
    print("=" * 60)
    sys.exit(0)

# জিপিইউ বিবরণ
device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"GPU Model         : {gpu_name}")
print(f"Compute Capability: {capability[0]}.{capability[1]}")
print(f"Total VRAM        : {total_mem_gb:.2f} GB")
print("-" * 60)

# GPGPU পারফরম্যান্স টেস্ট (Matrix Multiplication: CPU vs GPU)
MATRIX_SIZE = 4000  # GTX 1650 (4GB)-এর জন্য নিরাপদ সাইজ
print(f"\n[১] CPU vs GPU গতি পরীক্ষা (Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE}):")

# CPU টেস্ট
print("-> CPU-তে হিসাব চলছে...")
x_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32)
y_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32)

start_cpu = time.perf_counter()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.perf_counter() - start_cpu
print(f"   CPU সময় নিয়েছে : {cpu_time:.3f} সেকেন্ড ({cpu_time * 1000:.1f} ms)")

# GPU টেস্ট (GPGPU)
print("-> জিপিইউ (CUDA)-তে হিসাব চলছে...")
x_gpu = x_cpu.to(device)
y_gpu = y_cpu.to(device)

# ওয়ার্মআপ (GPU Warm-up)
_ = torch.matmul(x_gpu, y_gpu)
torch.cuda.synchronize()

# আসল পরিমাপ (Event Timing)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
z_gpu = torch.matmul(x_gpu, y_gpu)
end_event.record()
torch.cuda.synchronize()

gpu_time_ms = start_event.elapsed_time(end_event)
gpu_time_sec = gpu_time_ms / 1000.0

print(f"   GPU সময় নিয়েছে : {gpu_time_sec:.4f} সেকেন্ড ({gpu_time_ms:.1f} ms)")

# ফলাফল ও স্পিডআপ
speedup = cpu_time / gpu_time_sec if gpu_time_sec > 0 else 0
print("-" * 60)
print(f"ফলাফল: GPU (GTX 1650) CPU-এর চেয়ে {speedup:.1f}x গুণ দ্রুত হিসাব সম্পন্ন করেছে!")

# মেমোরি ব্যবহারের তথ্য
allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
print(f"VRAM Memory Used  : {allocated_mb:.1f} MB (Reserved: {reserved_mb:.1f} MB)")
print("=" * 60)
print("অভিনন্দন! আপনার জিপিইউ-তে CUDA GPGPU সফলভাবে কাজ করছে।")
