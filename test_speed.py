import time
import torch

print("-> Checking CUDA Device...")
if not torch.cuda.is_available():
    print("CUDA not available!")
    exit(1)

device = "cuda"
print(f"-> GPU: {torch.cuda.get_device_name(0)}")
print(f"-> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test raw matrix multiply speed (Teraflops check)
print("\n-> Testing GPU Compute & Tensor Cores (GEMM)...")
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)

# Warmup
for _ in range(5):
    c = torch.matmul(a, b)
torch.cuda.synchronize()

t0 = time.time()
iterations = 50
for _ in range(iterations):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - t0
tflops = (2 * 4096**3 * iterations) / (elapsed * 1e12)

print(f"-> 50 Matrix Multiplications finished in {elapsed:.3f}s")
print(f"-> GPU Compute Speed: {tflops:.2f} TFLOPS")

if tflops > 10:
    print("-> GPU is 100% HEALTHY and running at full speed!\n")
else:
    print("-> GPU is throttled in low power state!\n")
