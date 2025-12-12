# Frequency-Residual Defect Saliency: Unsupervised Inspection for Metal SSD Cases

This project implements an unsupervised computer vision pipeline for detecting defects (scratches, dents) on metal SSD casings. It improves upon the classic **Spectral Residual (SR)** saliency model by fusing it with **Oriented Gradient maps** and introducing a **Robust Statistical Thresholding** mechanism.

The system is designed for industrial quality assurance, capable of distinguishing between "clean" panels with heavy texture and "defective" panels with faint scratches, without requiring large labeled training datasets.

## 📌 Overview

Standard saliency detection methods (like Hou & Zhang, CVPR 2007) often fail on metal surfaces because they mistake reflections, texture noise, and screw holes for defects. This project solves those issues by:
1.  **Fusion:** Combining Frequency-domain anomalies (SR) with Spatial-domain directional features (Oriented Gradients).
2.  **Robust Thresholding:** Replacing rank-based percentile thresholding with `Mean + k*StdDev` to handle clean images correctly (0 False Alarms).
3.  **Smart Masking:** Automatically ignoring non-ROI features like corners (screw holes) and borders.
4.  **Confidence Scoring:** Calculating a Contrast-to-Noise Ratio (CNR) to classify parts as *Clean*, *Defect*, or *Not Sure* (for Human-in-the-loop workflows).

## 🛠️ Prerequisites

The code relies on standard Python computer vision libraries.

**Environment:**
* Python 3.8+
* OS: Linux

## 🚀 Usage

1.  **Prepare Data:**
    * Create a folder named `ssd_frames` in the project root.
    * Place the raw SSD images (`.jpg` or `.png`) inside.

2.  **Run the Inspection:**
    ```bash
    python final_project_main.py
    ```

3.  **View Results:**
    * **Terminal:** Displays Real-time HR (Hit Rate), FAR (False Alarm Rate), and CNR (Confidence Score).
    * **Visual Output:** Results (Heatmaps + Detections) are saved to the `./ssd_results` directory.

## ⚙️ Hyperparameters

The method is tuned for brushed metal surfaces. Key parameters in the main script:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **`K_STD`** | `3.5` | **Sensitivity Threshold.** Sets the cut-off at `Mean + 3.5 * StdDev`. Lower (3.0) is more sensitive; Higher (4.0) reduces false alarms. |
| **`AREA_MIN`** | `60` | **Size Filter.** Blobs smaller than 60 pixels are treated as noise and removed. |
| **`sal_weight`** | `0.6` | Weight for the Spectral Residual (Frequency) map. |
| **`grad_weight`** | `0.4` | Weight for the Oriented Gradient (Scratch) map. |
| **`ENABLE_MASK`** | `True` | Toggles the masking of corners (15%) and borders (5%) to ignore screw holes. |

## 📊 Experiment Results

We compared this **Improved Method** against the **Baseline (Hou & Zhang, 2007)** on a dataset of real industrial SSD panels.

### 1. Quantitative Performance
| Method | False Alarm Rate (FAR) | Hit Rate (HR) | Status on Clean Images |
| :--- | :--- | :--- | :--- |
| **Baseline (2007)** | High (~4.0% - 9.0%) | Unreliable | **Fails** (Detects noise as defect) |
| **Improved (Ours)** | **Near Zero (< 0.1%)** | **High** | **Pass** (0 detections on clean panels) |


### 2. Confidence Classification
The system computes a **Contrast-to-Noise Ratio (CNR)** to determine the confidence of a detection, enabling robot decision-making for Human-Robot Collaboration task:
* **CNR < 1.5:** Status **CLEAN**
* **1.5 < CNR < 6.0:** Status **NOT SURE** (Ask Human help for inspection)
* **CNR > 6.0:** Status **DEFECT DETECTED**

## 📄 References
1.  X. Hou and L. Zhang, "Saliency Detection: A Spectral Residual Approach," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2007.