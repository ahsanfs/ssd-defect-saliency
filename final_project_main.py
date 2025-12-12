import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 1. UTILITIES & METRICS
# ==========================================

def create_combined_mask(shape, border_pct=0.05, corner_pct=0.15):
    """
    Creates a mask that zeros out borders (edge noise) and corners (screw holes).
    """
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    
    # 1. Mask Borders
    b_h, b_w = int(h * border_pct), int(w * border_pct)
    mask[0:b_h, :] = 0; mask[h-b_h:h, :] = 0
    mask[:, 0:b_w] = 0; mask[:, w-b_w:w] = 0

    # 2. Mask Corners
    c_h, c_w = int(h * corner_pct), int(w * corner_pct)
    mask[0:c_h, 0:c_w] = 0;       mask[0:c_h, w-c_w:w] = 0
    mask[h-c_h:h, 0:c_w] = 0;     mask[h-c_h:h, w-c_w:w] = 0
    
    return mask

def load_ssd_image(path):
    img = cv2.imread(path)
    if img is None: return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def compute_hr_far(pred_mask, gt_mask):
    """
    Computes Hit Rate (HR) and False Alarm Rate (FAR).
    """
    if gt_mask is None: return 0.0, 0.0
    
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred = (pred_mask > 0)
    gt = (gt_mask > 0)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    neg = (~gt).sum() 

    hr = tp / (tp + fn + 1e-8)
    far = fp / (neg + 1e-8)
    return hr, far

def compute_confidence_score(saliency_map, prediction_mask):
    """
    Computes Contrast-to-Noise Ratio (CNR).
    """
    if np.sum(prediction_mask) == 0: return 0.0
    defect_pixels = saliency_map[prediction_mask > 0]
    bg_pixels = saliency_map[prediction_mask == 0]
    if len(bg_pixels) == 0: return 0.0
    
    # Avoid division by zero
    std_bg = np.std(bg_pixels)
    if std_bg < 1e-6: std_bg = 1e-6
        
    return (np.mean(defect_pixels) - np.mean(bg_pixels)) / std_bg

def classify_defect(cnr):
    """
    Classifies panel status based on Confidence Score.
    Returns: Status String, Color Tuple (R,G,B) for visualization.
    """
    if cnr < 1.5:
        return "CLEAN", (0, 255, 0) # Green
    elif cnr < 6.0:
        return "NOT SURE", (0, 255, 255) # Yellow
    else:
        return "SCRATCH DETECTED", (0, 0, 255) # Red

# ==========================================
# 2. BASELINE METHOD (Hou & Zhang 2007)
# ==========================================

def paper_baseline_method(gray, use_mask=False):
    """
    Strict implementation: Downsample to 64px -> SR -> Upsample.
    """
    target_size = 64
    I = gray.astype(np.float32) / 255.0
    h, w = I.shape

    # 1. Downsample
    scale = target_size / float(min(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    I_resized = cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding to ensure at least 64x64
    if new_h < target_size or new_w < target_size:
        pad_y = max(0, target_size - new_h)
        pad_x = max(0, target_size - new_w)
        I_resized = cv2.copyMakeBorder(I_resized, pad_y//2, pad_y-pad_y//2, pad_x//2, pad_x-pad_x//2, cv2.BORDER_REFLECT_101)

    # Center crop
    curr_h, curr_w = I_resized.shape
    y0, x0 = (curr_h - target_size) // 2, (curr_w - target_size) // 2
    I_crop = I_resized[y0:y0 + target_size, x0:x0 + target_size]

    # 2. Spectral Residual
    dft = np.fft.fft2(I_crop)
    log_amp = np.log(np.abs(dft) + 1e-8)
    spectral_residual = log_amp - cv2.blur(log_amp, (3, 3))
    
    # 3. Saliency Map
    saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * np.angle(dft)))) ** 2
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=8, sigmaY=8)

    # 4. Upsample & Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency_full = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)

    # 5. Apply Mask (Optional)
    if use_mask:
        c_mask = create_combined_mask(saliency_full.shape, border_pct=0.05, corner_pct=0.15)
        saliency_full = saliency_full * c_mask

    # 6. Threshold
    mask = (saliency_full > np.mean(saliency_full) * 3).astype(np.uint8)
    return saliency_full, mask

# ==========================================
# 3. IMPROVED METHOD (SR + Grad + Robust Threshold)
# ==========================================

def spectral_residual_map_improved(gray):
    I = cv2.bilateralFilter(gray.astype(np.float32)/255.0, 5, 10, 10)
    F = np.fft.fft2(I)
    mag = np.log(np.abs(F) + 1e-8)
    # Larger kernel (9x9) for better background suppression
    S = np.abs(np.fft.ifft2(np.exp((mag - cv2.blur(mag, (9, 9))) + 1j * np.angle(F))))
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    return S

def oriented_gradient_map(gray):
    I = gray.astype(np.float32) / 255.0
    dx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
    G = np.zeros_like(I)
    for t in np.linspace(0, np.pi, 8, endpoint=False):
        resp = cv2.GaussianBlur(np.abs(np.cos(t)*dx + np.sin(t)*dy), (0,0), 1.0)
        G = np.maximum(G, resp)
    return (G - G.min()) / (G.max() - G.min() + 1e-8)

def defect_segmentation_improved(gray, sal_weight=0.6, grad_weight=0.4, 
                                 k_std=3.5, area_min=60, use_mask=False):
    """
    Uses Statistical Thresholding (Mean + k*Std) to be robust on clean images.
    """
    # 1. Fuse Maps
    S = spectral_residual_map_improved(gray)
    G = oriented_gradient_map(gray)
    M = sal_weight * S + grad_weight * G
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)

    # 2. Masking & Valid Pixel Extraction
    if use_mask:
        c_mask = create_combined_mask(M.shape, border_pct=0.05, corner_pct=0.15)
        M = M * c_mask
        valid_pixels = M[c_mask > 0]
    else:
        valid_pixels = M.flatten()

    # 3. Robust Thresholding
    threshold = np.mean(valid_pixels) + (k_std * np.std(valid_pixels))
    if threshold > 1.0: threshold = 1.0
    
    mask = (M > threshold).astype(np.uint8)

    # 4. Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. Size Filter
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            cleaned[labels == i] = 1

    return M, cleaned

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    input_folder = "./ssd_frames"
    output_folder = "./ssd_results"
    
    # --- CONFIG ---
    K_STD = 3.5          # Robust sensitivity
    AREA_MIN = 60        # Minimum size to be considered a scratch
    ENABLE_MASK = True   # Ignore borders/corners

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    image_files = sorted([f for f in glob.glob(os.path.join(input_folder, "*.*")) 
                          if "_mask" not in f and f.lower().endswith(('.jpg', '.png'))])

    print(f"{'Filename':<20} | {'Baseline':<30} | {'Improved':<35} | {'Decision':<20}")
    print("-" * 115)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]

        rgb_img, gray = load_ssd_image(img_path)
        if rgb_img is None: continue

        # GT Load
        gt_mask = None
        has_gt = False
        for m in [f"{base_name}_mask.png", f"{base_name}_mask.jpg"]:
            full_m = os.path.join(input_folder, m)
            if os.path.exists(full_m):
                gt_mask = cv2.resize(cv2.imread(full_m, 0), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                has_gt = True

        # Run Algos
        base_heat, base_mask = paper_baseline_method(gray, use_mask=ENABLE_MASK)
        imp_heat, imp_mask = defect_segmentation_improved(gray, k_std=K_STD, area_min=AREA_MIN, use_mask=ENABLE_MASK)

        # Metrics
        cnr_score = compute_confidence_score(imp_heat, imp_mask)
        status, status_color = classify_defect(cnr_score)
        
        base_metric = "N/A"
        imp_metric_str = f"CNR:{cnr_score:.2f}"
        
        if gt_mask is not None:
            hr_b, far_b = compute_hr_far(base_mask, gt_mask)
            hr_i, far_i = compute_hr_far(imp_mask, gt_mask)
            base_metric = f"HR:{hr_b:.2f}/FAR:{far_b:.3f}"
            imp_metric_str = f"HR:{hr_i:.2f}/FAR:{far_i:.3f} | CNR:{cnr_score:.2f}"

        print(f"{filename:<20} | {base_metric:<30} | {imp_metric_str:<35} | {status:<20}")

        # ==========================================
        # 5. Visualization (Your requested 2x4 Grid)
        # ==========================================
        plt.figure(figsize=(15, 8))
        
        # 1. Input Image
        plt.subplot(2, 4, 1)
        plt.title(f"Input: {filename}\nStatus: {status}", fontsize=10, color='blue')
        plt.imshow(gray, cmap='gray')
        plt.axis('off')
        
        # 2. Baseline Heatmap
        plt.subplot(2, 4, 2)
        plt.title("Baseline Heatmap", fontsize=10)
        plt.imshow(base_heat, cmap='inferno')
        plt.axis('off')
        
        # 3. Improved Heatmap
        plt.subplot(2, 4, 3)
        plt.title(f"Imp Heatmap\nCNR: {cnr_score:.2f}", fontsize=10)
        plt.imshow(imp_heat, cmap='inferno')
        plt.axis('off')
        
        # 4. Improved Detections (Red Contours on RGB)
        vis_imp = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_BGR2RGB)
        cnts, _ = cv2.findContours(imp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_imp, cnts, -1, (255, 0, 0), 2)
        plt.subplot(2, 4, 4)
        plt.title("Improved Detections", fontsize=10)
        plt.imshow(vis_imp)
        plt.axis('off')

        # 5. Ground Truth
        plt.subplot(2, 4, 5)
        plt.title("Ground Truth", fontsize=10)
        if has_gt: 
            plt.imshow(gt_mask, cmap='gray')
        else: 
            plt.imshow(np.zeros_like(gray), cmap='gray')
            plt.text(gray.shape[1]//4, gray.shape[0]//2, "No GT", color='white', fontsize=12)
        plt.axis('off')

        # 6. Baseline Mask
        plt.subplot(2, 4, 6)
        plt.title(f"Baseline Mask\n{base_metric}", fontsize=10)
        plt.imshow(base_mask, cmap='gray')
        plt.axis('off')
        
        # 7. Improved Mask
        plt.subplot(2, 4, 7)
        plt.title(f"Improved Mask\n{imp_metric_str}", fontsize=9)
        plt.imshow(imp_mask, cmap='gray')
        plt.axis('off')
        
        # 8. Empty Slot or Logo (Optional)
        plt.subplot(2, 4, 8)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"result_{base_name}.png"))
        plt.close()

    print("-" * 115)
    print(f"Done! Results saved in {output_folder}")