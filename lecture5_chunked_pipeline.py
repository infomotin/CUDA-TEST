import sys
import time
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_chunked_matmul(a_host: torch.Tensor, b_gpu: torch.Tensor, device: torch.device, chunk_rows: int = 2048):
    """
    বড় ম্যাট্রিক্স A (Host)-কে ছোট ছোট row-chunk-এ ভাগ করে GPU-তে থাকা B-এর সাথে গুণ করে।
    """
    total_rows = a_host.shape[0]
    out_cols = b_gpu.shape[1]
    result_host = torch.empty((total_rows, out_cols), dtype=torch.float32)

    start_row = 0
    while start_row < total_rows:
        end_row = min(start_row + chunk_rows, total_rows)
        try:
            # শুধু বর্তমান chunk-টুকু GPU-তে পাঠানো হচ্ছে
            a_chunk_gpu = a_host[start_row:end_row].pin_memory().to(device, non_blocking=True)
            c_chunk_gpu = torch.matmul(a_chunk_gpu, b_gpu)
            result_host[start_row:end_row] = c_chunk_gpu.cpu()

            del a_chunk_gpu, c_chunk_gpu
            start_row = end_row

        except torch.cuda.OutOfMemoryError:
            # VRAM ভরে গেলে ক্যাশ পরিষ্কার করে chunk_rows অর্ধেক করে পুনরায় চেষ্টা
            torch.cuda.empty_cache()
            if chunk_rows <= 64:
                raise RuntimeError("খুব ছোট chunk সাইজেও VRAM অপর্যাপ্ত! ডেটা বা হার্ডওয়্যার সীমা পরীক্ষা করুন।")
            chunk_rows //= 2
            print(f"[WARN] CUDA OOM detected! Reducing chunk_rows to {chunk_rows} and retrying...")

    return result_host


def lecture5_capstone_pipeline():
    print("=" * 72)
    print(" LECTURE 5 SLIDE CODE: Chunked GPU Pipeline & OOM Recovery")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA ডিভাইস পাওয়া যায়নি!")

    device = torch.device("cuda:0")
    print(f"[DEVICE] {torch.cuda.get_device_name(0)}")

    # বড় ইনপুট ম্যাট্রিক্স A (12288 x 4096) এবং B (4096 x 2048)
    M, K, N = 12288, 4096, 2048
    chunk_size = 3072

    print(f"[DATASET] A({M}x{K}) @ B({K}x{N}) | Initial Chunk Rows: {chunk_size}\n")

    a_host = torch.randn((M, K), dtype=torch.float32)
    b_host = torch.randn((K, N), dtype=torch.float32)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    b_gpu = b_host.to(device)
    output_host = run_chunked_matmul(a_host, b_gpu, device, chunk_rows=chunk_size)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # প্রথম chunk-এর নির্ভুলতা যাচাই (CPU vs Chunked GPU)
    ref_first_chunk = torch.matmul(a_host[:chunk_size], b_host)
    is_accurate = torch.allclose(ref_first_chunk, output_host[:chunk_size], rtol=1e-3, atol=1e-3)

    print(f"[PIPELINE] Total Chunked Execution Time: {total_ms:.2f} ms")
    print(f"[VRAM] Peak VRAM Allocated: {peak_vram_mb:.2f} MB")
    print(f"[VERIFY] First Chunk Numerical Match: {is_accurate}")

    del b_gpu, output_host
    torch.cuda.empty_cache()
    print("=" * 72)


if __name__ == "__main__":
    lecture5_capstone_pipeline()