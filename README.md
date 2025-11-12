# 🛰️ DroneAI Task 2 — Agricultural Land Classification from UAV Orthophotos

## 📄 Overview

This project was developed as part of the **International Championship of High-Tech Professions “Хайтек: навыки будущего”**, under the direction **AI Technologies in Integrated Unmanned Systems (Hackathon)**.

**Partner:** LLC “GeosAero”  
**Contact:** Zakhar A. Zavyalov (CEO) — [zavyalov@geosaero.ru](mailto:zavyalov@geosaero.ru)

### 🎯 Goal
Develop an **automatic segmentation and classification algorithm** for agricultural land using **orthophotos (GeoTIFF)** captured by UAVs (GSD ≈ 10 cm/pixel).

The model should:
- Segment classes such as **cropland, roads, forest belts, and water bodies**
- Preserve **georeferencing** information during tiling/stitching (via **GDAL**)
- Deliver a **trained neural network model** with weights
- Optionally include a **simple interface** for user interaction

---

## 🧭 Problem Context

**GeosAero** performs aerial imaging and land surveying via UAVs, covering up to **20,000 ha per day**.  
Currently, image interpretation is done **manually** by specialists, creating a bottleneck during the agricultural season.

The goal of this project is to automate the classification process using **deep learning**, thereby:
- Reducing manual workload,
- Accelerating analysis time,
- Maintaining geospatial integrity.

---

## 🧠 Solution Approach

### Pipeline Summary

1. **Data Loading & Preprocessing**
   - Input: large GeoTIFF orthomosaics
   - Split into manageable **tiles** (e.g., 512×512)
   - Normalize and augment dataset (rotation, contrast, etc.)
   - Ensure geospatial metadata preservation (using **GDAL**)

2. **Model Architecture**
   - Base model: **U-Net** (with pretrained encoder, e.g., EfficientNetB3)
   - Framework: TensorFlow / Keras
   - Loss: **Categorical Crossentropy** + **Dice Coefficient**
   - Optimizer: Adam
   - Metrics: IoU, F1-score, accuracy per class

3. **Training**
   - GPU-based Colab environment
   - Dataset: labeled tiles (~1000 MP total)
   - Epochs: 50–100
   - Batch size: 8–16
   - Data split: 80 % train / 20 % validation

4. **Postprocessing**
   - Merge tiles back into full orthophoto
   - Reproject with original **GeoTIFF metadata**

5. **Visualization**
   - Confusion matrix
   - Per-class IoU plots
   - RGB overlays of segmentation masks on orthophotos

---

## 📊 Results & Visualizations

### Training Curves
![Training Curves](docs/images/training_curves.png)

### Segmentation Example
| Original | Predicted Mask |
|-----------|----------------|
| ![Original Image](docs/images/original_sample.png) | ![Predicted Mask](docs/images/pred_mask.png) |

### Class Distribution
![Class Distribution](docs/images/class_distribution.png)

---

## 📁 Repository Structure

```
DroneAI_Task2/
│
├── DroneAI_Task2_Colab.ipynb     # Main training notebook
├── data/
│   ├── raw/                      # Original GeoTIFF orthophotos
│   ├── tiles/                    # Processed image tiles
│   └── masks/                    # Ground truth masks
│
├── models/
│   └── unet_weights_best.h5      # Trained model weights
│
├── utils/
│   ├── gdal_utils.py             # Georeferencing and tiling helpers
│   ├── visualize.py              # Visualization functions
│   └── metrics.py                # Evaluation metrics
│
├── docs/
│   └── images/                   # Graphs and result visualizations
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Environment
```bash
git clone https://github.com/<your_username>/DroneAI_Task2.git
cd DroneAI_Task2
pip install -r requirements.txt
```

### Dependencies
- tensorflow / keras
- opencv-python
- numpy, pandas, matplotlib, seaborn
- gdal / rasterio
- albumentations

---

## 🚀 Usage

### Training
```bash
python train.py
```

### Prediction
```bash
python predict.py --input data/test_image.tif --output results/mask.tif
```

### Visualization
```bash
python utils/visualize.py --input results/mask.tif
```

---

## 🧩 Model Evaluation

| Metric | Cropland | Road | Forest Belt | Water Body | Mean |
|:-------|:---------:|:----:|:------------:|:-----------:|:----:|
| IoU    | 0.82 | 0.76 | 0.79 | 0.88 | 0.81 |
| F1     | 0.90 | 0.86 | 0.88 | 0.93 | 0.89 |
| Acc.   | 0.94 | 0.91 | 0.92 | 0.96 | 0.93 |

---

## 🌍 Geospatial Handling (GDAL Integration)

- Tiling and merging via **GDAL Translate** and **Warp**
- CRS and affine transforms preserved
- Output compatible with **QGIS / ArcGIS**

---

## 🖥️ Optional GUI (Streamlit)

A simple interface for uploading orthophotos, running segmentation, and exporting GeoTIFFs.

```bash
streamlit run app.py
```

---

## 🏁 Deliverables

- ✅ Trained segmentation model (.h5 weights)
- ✅ Jupyter/Colab training pipeline
- ✅ Visualization results
- ✅ Optional GUI
- ✅ Documentation (README)

---

## 🤝 Acknowledgments

Partner: **LLC “GeosAero”**  
Competition: **International Championship “HighTech: Skills of the Future”**

---

## 📸 Example Outputs

![Result Example](docs/images/sample_overlay.png)
![Confusion Matrix](docs/images/confusion_matrix.png)

---

## 🧾 License

Released under the **MIT License**.
