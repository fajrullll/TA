# ==================== IMPORT ====================
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import io, color, measure
from scipy.ndimage import center_of_mass
import numpy as np
import cv2 as cv
from skimage.segmentation import relabel_sequential
from scipy.spatial.distance import euclidean
import networkx as nx
from scipy.spatial import ConvexHull
import os
from skimage.measure import regionprops

# ==================== FUNGSI BANTUAN ====================

def hapus_noise(binary, min_area=5):
    """Menghapus komponen kecil dari citra biner."""
    labeled, num = measure.label(binary, return_num=True, connectivity=2)
    props = measure.regionprops(labeled)
    clean_binary = np.zeros_like(binary, dtype=bool)
    for prop in props:
        if prop.area >= min_area:
            clean_binary[labeled == prop.label] = True
    return clean_binary

def gabungkan_diakritik_dan_mainstroke(binary, area_main=50, area_dot=0, jarak_maks=60, rasio_alif=2.5):
    labeled_image, _ = measure.label(binary, return_num=True, connectivity=2)
    props = measure.regionprops(labeled_image)

    # ambil area terbesar sebagai acuan
    max_area = max([p.area for p in props]) if props else 0
    area_main_dyn = max(area_main, int(0.12 * max_area))  # 12% dari komponen terbesar

    main_infos = {}
    dot_mask = np.zeros_like(binary, dtype=np.uint8)
    label_map = labeled_image.copy()

    for prop in props:
        area = prop.area
        bbox = prop.bbox
        height = bbox[2] - bbox[0]
        width  = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 1e-5)

        if area > area_main_dyn or (area > 10 and aspect_ratio > 3.5):
            main_infos[prop.label] = {"centroid": prop.centroid, "bbox": bbox}

    for prop in props:
        label = prop.label
        area  = prop.area
        if label in main_infos or area <= area_dot:
            continue



        bbox = prop.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 1e-5)

        if area > area_dot and aspect_ratio > rasio_alif:
            continue

        centroid = prop.centroid
        if not main_infos:
            continue
        nearest_main = min(main_infos.items(), key=lambda item: euclidean(centroid, item[1]["centroid"]))[0]
        dist = euclidean(centroid, main_infos[nearest_main]["centroid"])
        if dist < jarak_maks:
            cy, _ = map(int, centroid)
            main_cy = int(main_infos[nearest_main]["centroid"][0])
            dot_mask[labeled_image == label] = 1 if cy < main_cy else 2
            label_map[labeled_image == label] = nearest_main

    labeled_image, _, _ = relabel_sequential(label_map)
    cleaned_binary = labeled_image > 0
    return labeled_image, cleaned_binary, dot_mask

def find_endpoints(skel):
    endpoints, intersections, turns = [], [], []
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

def skeleton_to_graph(skel):
    G = nx.Graph()
    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if skel[ni, nj]:
                            G.add_edge((i, j), (ni, nj))
    return G

def deteksi_loop(letter_skeleton):
    G = skeleton_to_graph(letter_skeleton)
    all_cycles = nx.cycle_basis(G)
    valid_cycles = []
    for cycle in all_cycles:
        if len(cycle) >= 2:
            cycle_array = np.array(cycle)
            try:
                hull = ConvexHull(cycle_array)
                area = hull.volume
            except:
                area = 0
            if area > 1.5:
                valid_cycles.append((cycle, area))
    return valid_cycles

def tsp_skeleton_traversal(points, prefer_lower=True, dist_threshold=3):
    if len(points) == 0:
        return []
    points = points.tolist()
    points.sort(key=lambda p: (-p[0], p[1]))
    start_point = points.pop(0)
    path = [start_point]
    segments = []
    current_segment = [start_point]
    while points:
        last = path[-1]
        distances = [euclidean(last, p) for p in points]
        idx_min = np.argmin(distances)
        next_point = points.pop(idx_min)
        dist = euclidean(last, next_point)
        if dist > dist_threshold:
            segments.append(current_segment)
            current_segment = [next_point]
        else:
            current_segment.append(next_point)
        path.append(next_point)
    segments.append(current_segment)
    return segments

def cektianglam(skel, start_yx, min_height=12, max_width_spread=20):
    """
    Tidak ada perubahan logika internal, hanya memastikan parameter min_height 
    bisa diatur secara dinamis oleh fungsi pemanggil.
    """
    sy, sx = start_yx
    h, w = skel.shape

    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = sy + dy, sx + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                neighbors.append((ny, nx))

    for (ny, nx) in neighbors:
        if ny > sy: continue # Hanya cari ke atas

        stack = [(ny, nx)]
        visited = set([(ny, nx), (sy, sx)]) 

        branch_min_y, branch_max_y = ny, ny
        branch_min_x, branch_max_x = nx, nx
        
        is_valid_branch = True

        while stack:
            cy, cx = stack.pop()
            branch_min_y = min(branch_min_y, cy)
            branch_max_y = max(branch_max_y, cy)
            branch_min_x = min(branch_min_x, cx)
            branch_max_x = max(branch_max_x, cx)

            if (branch_max_x - branch_min_x) > max_width_spread:
                is_valid_branch = False
                break 

            for ddy in [-1, 0, 1]:
                for ddx in [-1, 0, 1]:
                    if ddy == 0 and ddx == 0: continue
                    nny, nnx = cy + ddy, cx + ddx
                    if 0 <= nny < h and 0 <= nnx < w:
                        if skel[nny, nnx] and (nny, nnx) not in visited:
                            if nny < sy + 5: 
                                visited.add((nny, nnx))
                                stack.append((nny, nnx))
        
        if not is_valid_branch: continue

        b_height = sy - branch_min_y 
        b_width = branch_max_x - branch_min_x
        
        # LOGIKA UTAMA: Menggunakan min_height yang dilempar dari argumen
        if b_height >= min_height:
            ratio = b_height / (b_width + 1)
            if ratio > 1.2 or b_height > 20:
                return True 

    return False

# Update fungsi ceknun agar lebih ketat (menghindari gigi Sin dianggap Nun)
def ceknun(skel, start_yx, min_depth=5, min_width=10): # Naikkan min_width jadi 10
    sy, sx = start_yx
    h, w = skel.shape
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = sy + dy, sx + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                neighbors.append((ny, nx))
    
    for (ny, nx) in neighbors:
        stack = [(ny, nx)]
        visited = set([(ny, nx), (sy, sx)])
        branch_min_y, branch_max_y = ny, ny
        branch_min_x, branch_max_x = nx, nx
        
        while stack:
            cy, cx = stack.pop()
            branch_min_y = min(branch_min_y, cy)
            branch_max_y = max(branch_max_y, cy)
            branch_min_x = min(branch_min_x, cx)
            branch_max_x = max(branch_max_x, cx)
            
            if (branch_max_x - branch_min_x) > 40: break # Safety break
            
            for ddy in [-1, 0, 1]:
                for ddx in [-1, 0, 1]:
                    if ddy == 0 and ddx == 0: continue
                    nny, nnx = cy + ddy, cx + ddx
                    if 0 <= nny < h and 0 <= nnx < w:
                        if skel[nny, nnx] and (nny, nnx) not in visited:
                             visited.add((nny, nnx)); stack.append((nny, nnx))
        
        depth_down = branch_max_y - sy
        width_branch = branch_max_x - branch_min_x
        
        # PERBAIKAN: Tambah syarat width harus signifikan dibanding depth
        # Agar lembah gigi yang curam tidak dianggap mangkok
        if depth_down >= min_depth and width_branch >= min_width:
             if width_branch > (depth_down * 0.5): # Rasio diperketat (0.4 -> 0.5)
                 return True
    return False

def hitung_garis_potong(component_features, binary_image, labeled_letters, skeleton_used_ignored, min_gap=5):
    """
    UPDATE V5 (FIXED): Menambahkan return_num=True pada measure.label.
    """
    all_cols = []
    threshold_gigi = 35  
    
    for label in range(1, np.max(labeled_letters) + 1):
        mask = (labeled_letters == label)
        if np.sum(mask) < 5: continue

        # 1. ISOLASI MAIN BODY (FIXED LINE)
        # Menambahkan return_num=True agar mengembalikan 2 nilai
        labeled_sub, num_sub = measure.label(mask, return_num=True, connectivity=2)
        
        if num_sub == 0: continue
        props = measure.regionprops(labeled_sub)
        main_prop = max(props, key=lambda x: x.area) 
        main_mask = (labeled_sub == main_prop.label)
        
        # Skeletonize hanya Main Body
        letter_skeleton = skeletonize(main_mask)
        
        # Hitung fitur ulang berdasarkan main body
        _, intersections, _ = find_endpoints(letter_skeleton)
        loops = deteksi_loop(letter_skeleton)
        
        raw_intersections = sorted([ix for iy, ix in intersections])
        intersections_map = {}
        for iy, ix in intersections:
            if ix not in intersections_map: intersections_map[ix] = []
            intersections_map[ix].append(iy)
        
        if raw_intersections:
            groups = []
            current_group = [raw_intersections[0]]
            for i in range(1, len(raw_intersections)):
                if raw_intersections[i] - raw_intersections[i-1] < threshold_gigi:
                    current_group.append(raw_intersections[i])
                else:
                    groups.append(current_group); current_group = [raw_intersections[i]]
            groups.append(current_group)
            
            for grp in groups:
                group_cuts = []
                has_special_feature = False
                
                # Scanning
                for ix in grp:
                    ys = intersections_map.get(ix, [])
                    for y_check in ys:
                        if cektianglam(letter_skeleton, (y_check, ix), min_height=20):
                            has_special_feature = True
                        if ceknun(letter_skeleton, (y_check, ix)):
                            has_special_feature = True
                    if has_special_feature: break
                
                if has_special_feature:
                    for ix in grp:
                        ys = intersections_map.get(ix, [])
                        is_tiang, is_nun = False, False
                        for y_check in ys:
                            if not is_tiang and cektianglam(letter_skeleton, (y_check, ix), min_height=20):
                                is_tiang = True
                            if not is_nun and ceknun(letter_skeleton, (y_check, ix)):
                                is_nun = True
                        
                        if is_tiang: group_cuts.append(ix + 2); group_cuts.append(ix - 2)
                        if is_nun: group_cuts.append(ix + 2)
                    all_cols.extend(group_cuts)
                else:
                    target_ix = grp[0] 
                    skip_cut = False
                    if loops:
                        loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                        if loop_xs and min([abs(target_ix - lx) for lx in loop_xs]) < 5: skip_cut = True
                    if not skip_cut: all_cols.append(target_ix + 2)

    all_cols = sorted(set(all_cols))
    merged_cols = []
    if all_cols:
        merged_cols = [all_cols[0]]
        for col in all_cols[1:]:
            if col - merged_cols[-1] < min_gap:
                merged_cols[-1] = (merged_cols[-1] + col) // 2
            else:
                merged_cols.append(col)
                
    return [0] + merged_cols + [binary_image.shape[1] - 1]

def potong_persimpangan(labeled_letters, skeleton_used_ignored):
    """
    UPDATE V5 (FIXED): Menambahkan return_num=True pada measure.label.
    """
    hasil_potongan = []
    num_features = np.max(labeled_letters)
    threshold_gigi = 35  
    min_gap = 5 

    for region_label in range(1, num_features + 1):
        mask = (labeled_letters == region_label)
        if np.sum(mask) < 5: continue

        # 1. ISOLASI: Pisahkan Badan (Main) dan Titik (Diacritics)
        # FIXED LINE: Tambahkan return_num=True
        labeled_sub, num_sub = measure.label(mask, return_num=True, connectivity=2)
        
        if num_sub == 0: continue
        
        props = measure.regionprops(labeled_sub)
        main_prop = max(props, key=lambda x: x.area)
        main_mask = (labeled_sub == main_prop.label)
        
        diacritics = [p for p in props if p.label != main_prop.label]

        # 2. PROSES CUTTING (Hanya pada Main Body)
        letter_skeleton = skeletonize(main_mask)
        coords = np.argwhere(main_mask)
        _, intersections, _ = find_endpoints(letter_skeleton)
        loops = deteksi_loop(letter_skeleton)
        x_coords = coords[:, 1]
        
        raw_intersections = sorted([ix for iy, ix in intersections])
        intersections_map = {}
        for iy, ix in intersections:
            if ix not in intersections_map: intersections_map[ix] = []
            intersections_map[ix].append(iy)

        cut_points = []
        if raw_intersections:
            groups = []
            current_group = [raw_intersections[0]]
            for i in range(1, len(raw_intersections)):
                if raw_intersections[i] - raw_intersections[i-1] < threshold_gigi:
                    current_group.append(raw_intersections[i])
                else:
                    groups.append(current_group); current_group = [raw_intersections[i]]
            groups.append(current_group)
            
            for grp in groups:
                group_cuts = []
                has_special_feature = False
                for ix in grp:
                    ys = intersections_map.get(ix, [])
                    for y_check in ys:
                        if cektianglam(letter_skeleton, (y_check, ix), min_height=20):
                            has_special_feature = True
                        if ceknun(letter_skeleton, (y_check, ix)):
                            has_special_feature = True
                    if has_special_feature: break
                
                if has_special_feature:
                    for ix in grp:
                        ys = intersections_map.get(ix, [])
                        is_tiang, is_nun = False, False
                        for y_check in ys:
                            if not is_tiang and cektianglam(letter_skeleton, (y_check, ix), min_height=20):
                                is_tiang = True
                            if not is_nun and ceknun(letter_skeleton, (y_check, ix)):
                                is_nun = True
                        
                        if is_tiang: group_cuts.append(ix + 2); group_cuts.append(ix - 2)
                        if is_nun: group_cuts.append(ix + 2)
                    cut_points.extend(group_cuts)
                else:
                    target_ix = grp[0] 
                    skip_cut = False
                    if loops:
                        loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                        if loop_xs and min([abs(target_ix - lx) for lx in loop_xs]) < 5: skip_cut = True
                    if not skip_cut: cut_points.append(target_ix + 2)

        # 3. MERGING CUTS
        cuts = []
        if cut_points:
            minc = x_coords.min(); maxc = x_coords.max()
            valid_cuts = sorted(list(set([int(np.clip(c, minc, maxc)) for c in cut_points])))
            if valid_cuts:
                merged_cuts = [valid_cuts[0]]
                for c in valid_cuts[1:]:
                    if c - merged_cuts[-1] < min_gap:
                        merged_cuts[-1] = (merged_cuts[-1] + c) // 2
                    else:
                        merged_cuts.append(c)
                valid_cuts = merged_cuts
            cuts = sorted(set([minc] + valid_cuts + [maxc + 1]))
        
        # 4. MEMBUAT SEGMEN
        segments = [] 
        if not cuts:
             segments.append(main_mask)
        else:
            for i in range(len(cuts) - 1):
                seg_mask = np.zeros_like(main_mask)
                seg_mask[:, cuts[i]:cuts[i + 1]] = main_mask[:, cuts[i]:cuts[i + 1]]
                if np.sum(seg_mask) > 20: 
                    segments.append(seg_mask)

        # 5. ASSIGN DIAKRITIK
        for dot in diacritics:
            dot_center = np.array(dot.centroid)
            best_idx = -1
            min_dist = float('inf')
            
            dot_mask = (labeled_sub == dot.label)
            
            for idx, seg in enumerate(segments):
                seg_pixels = np.argwhere(seg)
                if len(seg_pixels) == 0: continue
                
                dists = np.linalg.norm(seg_pixels - dot_center, axis=1)
                min_d = np.min(dists)
                
                if min_d < min_dist:
                    min_dist = min_d
                    best_idx = idx
            
            if best_idx != -1:
                segments[best_idx] = np.logical_or(segments[best_idx], dot_mask)

        hasil_potongan.extend(segments)

    return hasil_potongan

def tampilkan_garis_potong_skeleton_dan_biner(component_features, cleaned_binary, skeleton_used, labeled_letters):
    # UPDATE: Pass skeleton_used ke fungsi hitung
    merged_cols = hitung_garis_potong(component_features, cleaned_binary, labeled_letters, skeleton_used)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Skeleton
    ax[0].imshow(skeleton_used, cmap='gray')
    for col in merged_cols:
        ax[0].axvline(x=col, color='red', linestyle='--', linewidth=1.5)
    ax[0].set_title("Perpotongan pada Skeleton (Sinkron)")
    ax[0].axis('off')

    # Plot Biner
    ax[1].imshow(cleaned_binary, cmap='gray')
    for col in merged_cols:
        ax[1].axvline(x=col, color='red', linestyle='--', linewidth=1.5)
    ax[1].set_title("Perpotongan pada Biner (Sinkron)")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================
# Ganti path ini sesuai lokasi file Anda
image_path = r"E:\progres\salin.png"

image = io.imread(image_path)
if image.shape[2] == 4:
    image = image[:, :, :3]
gray = 1 - color.rgb2gray(image)
gray_uint8 = (gray * 255).astype(np.uint8)
_, binary = cv.threshold(gray_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

binary = hapus_noise(binary, min_area=5)
labeled_letters, cleaned_binary, dot_mask = gabungkan_diakritik_dan_mainstroke(binary)
skeleton_used = skeletonize(cleaned_binary)

component_features = {}
for region_label in range(1, np.max(labeled_letters) + 1):
    mask = (labeled_letters == region_label)
    if np.sum(mask) < 5:
        continue
    letter_skeleton = skeleton_used * mask
    endpoints, intersections, _ = find_endpoints(letter_skeleton)
    loops = deteksi_loop(letter_skeleton)
    component_features[region_label] = {
        "intersections": intersections,
        "has_loop": len(loops) > 0,
        "cycles": [cycle for cycle, _ in loops]
    }

# --- VISUALISASI UTAMA ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(skeleton_used, cmap='gray')

for label, props in component_features.items():
    mask = (labeled_letters == label)
    skel = skeletonize(mask)
    centroid = center_of_mass(mask)
    endpoints, intersections, _ = find_endpoints(skel)
    loops = props["cycles"]

    if centroid is not None:
        cy, cx = centroid
        ax.plot(cx, cy, 'o', color='red', label='Centroid' if label == 1 else "")
    if len(endpoints) > 0:
        sy, sx = endpoints[0]
        ax.plot(sx, sy, 'o', color='blue', label='Start' if label == 1 else "")
    if len(endpoints) > 1:
        ey, ex = endpoints[-1]
        ax.plot(ex, ey, 'o', color='yellow', label='End' if label == 1 else "")
    for iy, ix in props["intersections"]:
        ax.plot(ix, iy, 'o', color='green', markersize=3, label='Intersection' if label == 1 else "")
    for cycle in loops:
        cycle = np.array(cycle)
        ax.plot(cycle[:, 1], cycle[:, 0], 'r-', linewidth=2, label="Loop" if label == 1 else "")
    
    points = np.argwhere(skel)
    tsp_segments = tsp_skeleton_traversal(points)
    for segment in tsp_segments:
        segment = np.array(segment)
        if len(segment) >= 2:
            ax.scatter(segment[:, 1], segment[:, 0], color='blue', s=10, label='TSP Node' if label == 1 else "")

for yx in np.argwhere(dot_mask == 1):
    ax.plot(yx[1], yx[0], 'o', color='magenta', label='Diakritik Atas')
for yx in np.argwhere(dot_mask == 2):
    ax.plot(yx[1], yx[0], 'o', color='orange', label='Diakritik Bawah')

# === BAGIAN INI DIUPDATE ===
# Tambahkan skeleton_used ke dalam argumen
final_cols = hitung_garis_potong(component_features, cleaned_binary, labeled_letters, skeleton_used)
# ===========================

for col in final_cols:
    ax.axvline(x=col, color='red', linestyle='--', linewidth=1.5)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')
ax.set_title("Visualisasi Perpotongan")
ax.axis('off')
plt.show()

tampilkan_garis_potong_skeleton_dan_biner(component_features, cleaned_binary, skeleton_used, labeled_letters)

# --- PROSES PEMOTONGAN ---
hasil_potongan = potong_persimpangan(labeled_letters, skeleton_used)
sorted_hasil_potongan = sorted(hasil_potongan, key=lambda mask: np.max(np.argwhere(mask)[:, 1]), reverse=True)

output_folder = "hasil_potongan"
os.makedirs(output_folder, exist_ok=True)

for idx, region_mask in enumerate(sorted_hasil_potongan):
    output_image = (region_mask * 255).astype(np.uint8)
    output_filename = os.path.join(output_folder, f'potongan_huruf_{idx + 1}.png')
    cv.imwrite(output_filename, output_image)

print(f"Hasil potongan huruf disimpan di folder: {output_folder}")

# --- EVALUASI ---
jumlah_GT = 21   
jumlah_DT = len(sorted_hasil_potongan)
TP = 17         
FP = 3            
FN = 2           

accuracy = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

print("\n===== Hasil Evaluasi Segmentasi Huruf =====")
print(f"GT (Ground Truth) : {jumlah_GT}")
print(f"DT (Detected)     : {jumlah_DT}")
print(f"TP (Benar)        : {TP}")
print(f"FP (Salah)        : {FP}")
print(f"FN (Terlewat)     : {FN}")
print(f"Accuracy          : {accuracy:.3f}")
print(f"Precision         : {precision:.3f}")
print(f"Recall            : {recall:.3f}")
print(f"F1 Score          : {f1_score:.3f}")

# --- VISUALISASI DETAIL PER HURUF ---
print("\nMenampilkan visualisasi Biner vs Skeleton+TSP per huruf...")
for idx, region_mask in enumerate(sorted_hasil_potongan):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Kiri: Biner
    ax[0].imshow(region_mask, cmap='gray')
    ax[0].set_title(f'Huruf Potong ke-{idx+1} (Biner)')
    ax[0].axis('off')

    # Kanan: Skeleton + TSP + Fitur
    letter_skeleton = skeletonize(region_mask)
    endpoints, intersections, turns = find_endpoints(letter_skeleton)
    centroid = center_of_mass(region_mask)
    
    ax[1].imshow(letter_skeleton, cmap='gray')
    
    # Plot fitur di kanan
    if len(endpoints) > 0:
        sy, sx = endpoints[0]
        ax[1].plot(sx, sy, 'o', color='blue', label="Start")
    if len(endpoints) > 1:
        ey, ex = endpoints[-1]
        ax[1].plot(ex, ey, 'o', color='yellow', label="End")
    for iy, ix in intersections:
        ax[1].plot(ix, iy, 'o', color='green', markersize=4, label="Intersection")
    for ty, tx in turns:
        ax[1].plot(tx, ty, 'o', color='cyan', markersize=3, label="Turn")
    if centroid is not None:
        cy, cx = centroid
        ax[1].plot(cx, cy, 'o', color='red', label="Centroid")
        
    loops = deteksi_loop(letter_skeleton)
    for cycle, _ in loops:
        cycle_array = np.array(cycle)
        ax[1].plot(cycle_array[:, 1], cycle_array[:, 0], 'r-', linewidth=2, label="Loop")
        
    # Plot TSP
    points = np.argwhere(letter_skeleton)
    tsp_segments = tsp_skeleton_traversal(points)
    for segment in tsp_segments:
        segment = np.array(segment)
        ax[1].scatter(segment[:, 1], segment[:, 0], color='blue', s=5, zorder=5)
        for i in range(len(segment) - 1):
            p1, p2 = segment[i], segment[i + 1]
            ax[1].plot([p1[1], p2[1]], [p1[0], p2[0]], color='blue', linewidth=0.8, alpha=0.6, zorder=4)
            
    ax[1].set_title(f'Huruf Potong ke-{idx+1} (Skeleton + TSP)')
    ax[1].axis('off')
    
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()

# --- VISUALISASI TOTAL JARAK ---
fig, axes = plt.subplots(1, len(sorted_hasil_potongan), figsize=(3 * len(sorted_hasil_potongan), 4))
if len(sorted_hasil_potongan) == 1:
    axes = [axes]
    
jarak_per_huruf = []
for idx, region_mask in enumerate(sorted_hasil_potongan):
    skel = skeletonize(region_mask)
    points = np.argwhere(skel)
    tsp_segments = tsp_skeleton_traversal(points)
    total_distance = 0
    ax = axes[idx] if len(sorted_hasil_potongan) > 1 else axes[0]
    
    ax.imshow(skel, cmap='gray')
    for segment in tsp_segments:
        for i in range(len(segment) - 1):
            p1, p2 = segment[i], segment[i + 1]
            total_distance += euclidean(p1, p2)
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color='blue', linewidth=1)
            
    ax.set_title(f'Huruf {idx+1}\nJarak: {total_distance:.2f}')
    ax.axis('off')

plt.suptitle("Struktur Huruf + Jalur TSP + Jarak Total")
plt.tight_layout()
plt.show()