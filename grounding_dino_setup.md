# 🧠 GroundingDINO Setup Guide (CUDA 12.8 + RTX 3090)

This guide walks through a clean, reproducible setup for GroundingDINO.

---

## 1️⃣ Check System CUDA + GPU

```bash
nvcc --version
nvidia-smi
```

Expected:
```
CUDA Version: 12.8
GPU: RTX 3090
```

---

## 2️⃣ Create and Activate Conda Environment

```bash
conda create -p ~/miniconda3/envs/groundingdino python=3.10 -y
conda activate groundingdino
```

---

## 3️⃣ Verify Python + Pip

```bash
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel ninja
```

---

## 4️⃣ Install PyTorch Nightly (CUDA 12.8)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 5️⃣ Confirm Torch CUDA Link

```bash
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

---

## 6️⃣ Setup Project Structure

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git ground-dino
cd GroundingDINO
```

(Optional: specific commit)
```bash
git checkout 57535c5a79791cb76e36fdb64975271354f10251
```

---

## 7️⃣ Build CUDA Extension (RTX 3090 = SM 8.6)

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

rm -rf build groundingdino.egg-info groundingdino/_C*.so
python setup.py build_ext --inplace
```

---

## 8️⃣ Install GroundingDINO in Editable Mode

```bash
pip install -e . --no-build-isolation
```

---

## 9️⃣ Verify CUDA Extension

```bash
python - <<'PY'
import torch
from groundingdino.models.GroundingDINO import ms_deform_attn
print("✅ GroundingDINO extension loaded on CUDA", torch.version.cuda)
PY
```


---


**Your GroundingDINO + SAM environment is now ready.**

