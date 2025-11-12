# 🛰️ DroneAI Task 2 — Agricultural Land Classification from UAV Orthophotos

## 📄 Overview

This project was developed as part of the **International Championship of High-Tech Professions “Хайтек: навыки будущего”**, under the direction **AI Technologies in Integrated Unmanned Systems (Hackathon)**.

**Team Name:** ADPOLY AI Drone 

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
<img width="808" height="367" alt="image" src="https://github.com/user-attachments/assets/0f5f1897-0f0b-4590-8be6-2521a8a5fc4f" />


<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/e4562b71-6ed5-4a98-8af1-9e3491f85f45" />


### Segmentation Example
| Tile RGB | Label Mask | Overlay | 
|-----------|----------------|----------------|
| <img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/3cd6013f-d54c-4a6f-b90b-6bfbdd9f43f7" />
 | <img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/64b35eba-a17c-42b8-b83f-9aa7f9e629a7" />
 |<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/e4a7af84-f3a1-419e-b484-70fc86576ca5" />|


### Class Distribution
![Class Distribution](docs/images/class_distribution.png)

---

## ⚙️ Installation & Setup

### Environment
```bash
We used GoogleColab
```

### Dependencies
- tensorflow / keras
- opencv-python
- numpy, pandas, matplotlib, seaborn
- gdal / rasterio
- albumentations

---

## 🚀 Usage
Just run the below in the google colab
```bash
DroneAI_Task2_Colab.ipynb 
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

