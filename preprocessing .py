import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import io, color, measure
from skimage import filters
from scipy.ndimage import center_of_mass
import numpy as np
import cv2 as cv

# ==================== Bagian 1: Preprocessing dan Skeletonisasi ====================
image_path = "p04-lineimg15.png"
image = io.imread(image_path)

if image.shape[2] == 4:
    image = image[:, :, :3]

# Grayscale asli
gray_float = color.rgb2gray(image)

# Inversi hanya untuk thresholding
gray_inverted = 1 - gray_float
gray_uint8 = (gray_inverted * 255).astype(np.uint8)

# Threshold (agar huruf jadi putih/foreground)
_, binary = cv.threshold(gray_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Labeling komponen
labeled_image, num_labels = measure.label(binary, return_num=True, connectivity=2)
props = measure.regionprops(labeled_image)

main_components = np.zeros_like(binary, dtype=bool)
dot_components = np.zeros_like(binary, dtype=bool)

for prop in props:
    if prop.area > 50:
        main_components[labeled_image == prop.label] = True
    elif prop.area > 5:
        min_distance = np.min([
            np.linalg.norm(np.array(prop.centroid) - np.array(main_prop.centroid))
            for main_prop in props if main_prop.area > 45
        ])
        if min_distance <= 27:
            dot_components[labeled_image == prop.label] = True

cleaned_binary = main_components | dot_components

# Skeletonisasi
skeleton_default = skeletonize(cleaned_binary)  # Zhang-Suen
skeleton_lee = skeletonize(cleaned_binary, method='lee')  # Lee 94

# Visualisasi hasil
outputs = {
    "Grayscale Image": gray_float,
    "Thresholding Image": binary,
    "Cleaned Thresholding Image": cleaned_binary,
    "Skeletonized (Zhang-Suen)": skeleton_default,
    "Skeletonized (Lee 94)": skeleton_lee
}

for title, img in outputs.items():
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==================== Bagian 2: Deteksi Titik Penting ====================
def find_endpoints(skel):
    endpoints = []
    intersections = []
    turns = []

    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j]:
                neighborhood = skel[i-1:i+2, j-1:j+2]
                num_neighbors = np.sum(neighborhood) - 1

                if num_neighbors == 1:
                    endpoints.append((i, j))
                elif num_neighbors >= 3:
                    intersections.append((i, j))
                else:
                    horizontal = skel[i, j-1] + skel[i, j+1]
                    vertical = skel[i-1, j] + skel[i+1, j]
                    if horizontal == 1 and vertical == 1:
                        turns.append((i, j))

    return np.array(endpoints), np.array(intersections), np.array(turns)

# Gunakan skeleton Zhang-Suen (atau ganti ke skeleton_lee jika ingin)
skeleton_used = skeleton_default
labeled_letters, num_features = measure.label(cleaned_binary, return_num=True, connectivity=2)

letter_features = {}
for region_label in range(1, num_features + 1):
    single_letter = (labeled_letters == region_label)
    letter_skeleton = skeleton_used * single_letter

    endpoints, intersections, turns = find_endpoints(letter_skeleton)
    centroid = center_of_mass(single_letter)

    start = None
    end = None
    if len(endpoints) >= 1:
        endpoints_sorted = sorted(endpoints, key=lambda p: (-p[1], p[0]))
        start = tuple(endpoints_sorted[0])
    if len(endpoints) >= 2:
        for ep in endpoints_sorted[1:]:
            end = tuple(ep)
            break

    letter_features[region_label] = {
        "start": start,
        "end": end,
        "intersections": intersections,
        "turns": turns,
        "centroid": centroid
    }

# Visualisasi titik-titik penting
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(skeleton_used, cmap=plt.cm.gray)

for letter, features in letter_features.items():
    if features["start"] is not None:
        sy, sx = features["start"]
        ax.plot(sx, sy, marker='o', markersize=5, color='blue', label="Awal Huruf" if 'Awal Huruf' not in ax.get_legend_handles_labels()[1] else "")
    if features["end"] is not None:
        ey, ex = features["end"]
        ax.plot(ex, ey, marker='o', markersize=5, color='yellow', label="Akhir Huruf" if 'Akhir Huruf' not in ax.get_legend_handles_labels()[1] else "")
    if len(features["intersections"]) > 0:
        for iy, ix in features["intersections"]:
            ax.plot(ix, iy, marker='o', markersize=3, color='green', label="Persimpangan" if 'Persimpangan' not in ax.get_legend_handles_labels()[1] else "")
    if len(features["turns"]) > 0:
        for ty, tx in features["turns"]:
            ax.plot(tx, ty, marker='o', markersize=3, color='cyan', label="Belokan 90°" if 'Belokan 90°' not in ax.get_legend_handles_labels()[1] else "")
    if features["centroid"] is not None:
        cy, cx = features["centroid"]
        ax.plot(cx, cy, marker='o', markersize=5, color='red', label="Centroid Huruf" if 'Centroid Huruf' not in ax.get_legend_handles_labels()[1] else "")

ax.legend()
plt.title("Titik-Titik Penting pada Skeleton")
plt.axis("off")
plt.show()
