import cv2
import numpy as np
import os
import time

# module‐level, once at import time
KSIZE, SIGMA, LAMBDA, GAMMA, PSI = 31, 4.0, 10.0, 0.5, 0
KERNEL1 = cv2.getGaborKernel((KSIZE, KSIZE), SIGMA, np.pi/4,  LAMBDA, GAMMA, PSI, ktype=cv2.CV_32F)
KERNEL2 = cv2.getGaborKernel((KSIZE, KSIZE), SIGMA, np.pi/2,  LAMBDA, GAMMA, PSI, ktype=cv2.CV_32F)
MORPH_KERNEL = np.ones((5,5), np.uint8)
KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

import cv2
import numpy as np

# Precompute the morphological kernel once
MORPH_KERNEL = np.ones((5, 5), np.uint8)

import numpy as np
import cv2

# module‐level cache
_GLOBAL_SAMPLE_IDX = None

def hue_kmeans_sample(hue, sample_frac=0.1, k=2, random_state=42):
    global _GLOBAL_SAMPLE_IDX

    H, W = hue.shape
    N = H * W
    pixels = hue.reshape(-1,1).astype(np.float32)

    # On first call, compute & cache the indices
    if _GLOBAL_SAMPLE_IDX is None:
        sample_size = max(int(N * sample_frac), k)
        rng = np.random.default_rng(random_state)
        # fast approximate sampling with mask
        mask = rng.random(N) < sample_frac
        idx = np.nonzero(mask)[0]
        if idx.shape[0] < k:
            extra = rng.choice(N, k - idx.shape[0], replace=False)
            idx = np.concatenate([idx, extra])
        _GLOBAL_SAMPLE_IDX = idx  # cache for later

    # reuse the same idx every time
    sample = pixels[_GLOBAL_SAMPLE_IDX]

    # run kmeans on `sample`
    _, _, centers = cv2.kmeans(
        sample, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.flatten().astype(np.float32)

    # assign every pixel to nearest center…
    pixels_flat = pixels.flatten()
    delta = np.abs(pixels_flat[:, None] - centers[None, :])
    dists = np.minimum(delta, 180.0 - delta)
    labels_flat = np.argmin(dists, axis=1).astype(np.uint8)
    labels_full = labels_flat.reshape(H, W)

    return labels_full, centers.reshape(-1,1)


def process_image(image_path):
    flag = False
    # Read image from file
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load {image_path}")
        return None, flag
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]  # brightness channel

    # Use two orientations to capture different texture directions
    response1 = cv2.filter2D(value, cv2.CV_8UC3, KERNEL1)
    response2 = cv2.filter2D(value, cv2.CV_8UC3, KERNEL2)
    combined_texture = np.maximum(response1, response2)

    # Use Otsu thresholding to create a binary texture mask
    _, texture_mask = cv2.threshold(combined_texture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ----- Anomaly Detection using k-means on Hue -----

    labels_full, centers = hue_kmeans_sample(hue, sample_frac=0.1)

    # If you want a binary “abnormal” mask (255 where not the dominant cluster):
    # find which center is dominant (closest to overall mean hue, or largest cluster in sample)
    dominant = np.argmin(np.abs(centers.flatten() - np.mean(hue)))
    mask_abnormal = np.where(labels_full == dominant, 0, 255).astype(np.uint8)

    # Create a brightness mask to filter out very dark areas
    # _, mask_brightness = cv2.threshold(value, 50, 255, cv2.THRESH_BINARY)
    mean_val = np.mean(value)
    _, mask_brightness = cv2.threshold(value, int(mean_val * 0.45), 255, cv2.THRESH_BINARY)
    anomaly_mask = cv2.bitwise_and(mask_abnormal, mask_brightness)

    #combined_mask = cv2.bitwise_and(mask_abnormal, mask_brightness)

    # ----- Combine Anomaly and Texture Masks -----
    combined_mask = cv2.bitwise_and(anomaly_mask, texture_mask)
    #combined_mask = cv2.bitwise_and(mask_abnormal, texture_mask)

    # ----- Create Road Mask -----
    # Roads often appear with low saturation. Adjust thresholds as needed.
    road_mask = cv2.inRange(hsv, (0, 0, 50), (179, 50, 255))
    # Optionally, smooth the road mask to reduce noise:
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    non_road_mask = cv2.bitwise_not(road_mask)
    refined_mask = cv2.bitwise_and(combined_mask, non_road_mask)


    # ----- Further Clean Up with Morphological Operations -----
    kernel = np.ones((5, 5), np.uint8)
    #clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_DILATE, kernel)

    # ----- Find and Filter Contours -----
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if not flag:
                flag = [(x,y)]
            else:
                flag.append((x,y))

    return image, flag

# Define your input and output folders
input_folder = r"inputfolder"
output_folder = r"outputfolder"


# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        processed_image, flag = process_image(input_path)
        end_time = time.time()  # End timing after processing
                
        if processed_image is not None:
            cv2.imwrite(output_path, processed_image)
            print(f"Processed and saved: {output_path}, Found Something: {flag}")
            
