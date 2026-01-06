# Quick Start Guide

This is a condensed guide to get you running the pipeline quickly.

## Prerequisites

1. **Install Python 3.8+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download GTZAN Dataset** and place in `data/raw_gtzan/`

## Run Full Pipeline (Step-by-Step)

### 1. Generate Clips (5-10 min)
```bash
python src/preprocess.py
```
✅ Creates 3000 clips in `data/clips_10s/` and `data/metadata.csv`

### 2. Extract Audio Features (10-15 min)
```bash
python src/features.py --audio_only
```
✅ Creates `data/features/mfcc_stats.npy` and `data/features/logmel.npy`

### 3. Run Baselines (5-10 min)
```bash
python src/baselines.py --visualize
```
✅ Creates results in `results/metrics/` and `results/figures/`

### 4. Train EASY Model - MLP-VAE (5-10 min)
```bash
python src/train.py --mode vae_mlp --epochs 50 --device cpu
```
✅ EASY task complete! Results in `results/`

### 5. Transcribe Audio (OPTIONAL - for MEDIUM/HARD)

**On Colab (15-30 min with GPU):**
```bash
python src/transcribe.py --model_size tiny --device cuda --compute_type float16
```

**On Local CPU (2-4 hours - slow!):**
```bash
python src/transcribe.py --model_size tiny --device cpu --compute_type int8 --max_clips 100
```

### 6. Extract Text Features (1-2 min)
```bash
python src/features.py --text_only
```

### 7. Train MEDIUM Model - Conv-VAE + Multimodal (15-20 min)
```bash
python src/train.py --mode vae_conv --epochs 50 --device cpu --multimodal
```
✅ MEDIUM task complete!

### 8. Train HARD Model - CVAE (10-15 min)
```bash
python src/train.py --mode cvae --epochs 50 --device cpu
```
✅ HARD task complete!

## View Results

```bash
# Open Jupyter notebooks
jupyter notebook notebooks/01_results_and_plots.ipynb
```

Or check:
- `results/metrics/metrics.csv` - All experiment results
- `results/figures/` - All visualizations

## Minimal Test Run (Fast)

Want to test the pipeline quickly? Use a small subset:

```bash
# 1. Preprocess (full - needed)
python src/preprocess.py

# 2. Features (full - needed)
python src/features.py --audio_only

# 3. Quick baseline test
python src/baselines.py --methods direct_kmeans

# 4. Quick VAE test with fewer epochs
python src/train.py --mode vae_mlp --epochs 10 --device cpu

# 5. Test transcription on 50 clips
python src/transcribe.py --max_clips 50
```

## Troubleshooting

### "GTZAN not found"
Place GTZAN files in: `data/raw_gtzan/genres_original/<genre>/*.wav`

### "faster-whisper not installed"
```bash
pip install faster-whisper
```

### "CUDA out of memory"
Use smaller batch size: `--batch_size 16`

## Expected Timeline

| Task | CPU Time | GPU Time |
|------|----------|----------|
| Preprocessing | 5-10 min | - |
| Audio Features | 10-15 min | - |
| Baselines | 5-10 min | - |
| MLP-VAE Training | 5-10 min | 1-2 min |
| **Transcription** | **2-4 hours** | **15-30 min** |
| Text Features | 1-2 min | - |
| Conv-VAE Training | 15-20 min | 3-5 min |
| CVAE Training | 10-15 min | 2-3 min |
| **Total (CPU)** | **3-4 hours** | - |
| **Total (GPU)** | - | **30-60 min** |

**Recommendation:** Do transcription on Google Colab with GPU, everything else can run locally.

## What Gets Saved

```
data/
  clips_10s/          → 3000 WAV files (~1.5GB)
  features/           → Feature arrays (~500MB)
  metadata.csv        → Clip info + splits + lyrics

results/
  metrics/
    metrics.csv       → All experiment results
    latents_*.npy     → Latent representations
    confusion_*.csv   → Confusion matrices
  figures/
    tsne_*.png        → t-SNE visualizations
    umap_*.png        → UMAP visualizations
    recon_*.png       → Reconstructions
  checkpoints/
    *.pt              → Trained models
```

## Next Steps

After running the pipeline:

1. Open `notebooks/01_results_and_plots.ipynb` to view results
2. Compare different methods in `results/metrics/metrics.csv`
3. Analyze visualizations in `results/figures/`
4. Try different hyperparameters (latent_dim, beta, etc.)
5. Experiment with different clustering algorithms

---

For full documentation, see `README.md`.

