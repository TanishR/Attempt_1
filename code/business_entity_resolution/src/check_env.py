import sys
import psutil

print("Environment Check:")
print(f"Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"Torch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: Available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: Unavailable")
except Exception as e:
    print(f"Torch: Error - {e}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except Exception as e:
    print(f"Transformers: Error - {e}")

try:
    import sentence_transformers
    print(f"Sentence-Transformers: {sentence_transformers.__version__}")
except Exception as e:
    print(f"Sentence-Transformers: Error - {e}")

try:
    import lightgbm
    print(f"LightGBM: {lightgbm.__version__}")
except Exception as e:
    print(f"LightGBM: Error - {e.__class__.__name__} (On Mac, you may need brew install libomp)")

try:
    import rapidfuzz
    print(f"RapidFuzz: {rapidfuzz.__version__}")
except Exception as e:
    print(f"RapidFuzz: Error - {e}")

try:
    import anyascii
    print(f"AnyAscii: {anyascii.__version__}")
except Exception as e:
    print(f"AnyAscii: Error - {e}")

print(f"CPU count: {psutil.cpu_count(logical=True)}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
