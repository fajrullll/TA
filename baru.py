# ==================== IMPORT ====================
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
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

# ==================== FUNGSI BANTUAN DASAR ====================

def hapus_noise(binary, min_area=5):
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

    max_area = max([p.area for p in props]) if props else 0
    area_main_dyn = max(area_main, int(0.12 * max_area))

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
        if label in main_infos or area <= area_dot: continue

        bbox = prop.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 1e-5)

        if area > area_dot and aspect_ratio > rasio_alif: continue

        centroid = prop.centroid
        if not main_infos: continue
        nearest_main = min(main_infos.items(), key=lambda item: euclidean(centroid, item[1]["centroid"]))[0]
        dist = euclidean(centroid, main_infos[nearest_main]["centroid"])
        if dist < jarak_maks:
            cy, _ = map(int, centroid)
            main_cy = int(main_infos[nearest_main]["centroid"][0])
            # 1 = Titik Atas, 2 = Titik Bawah
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
                if num_neighbors == 1: endpoints.append((i, j))
                elif num_neighbors >= 3: intersections.append((i, j))
                else:
                    horizontal = skel[i, j-1] + skel[i, j+1]
                    vertical = skel[i-1, j] + skel[i+1, j]
                    if horizontal == 1 and vertical == 1: turns.append((i, j))
    return np.array(endpoints), np.array(intersections), np.array(turns)

def skeleton_to_graph(skel):
    G = nx.Graph()
    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0: continue
                        ni, nj = i + di, j + dj
                        if skel[ni, nj]: G.add_edge((i, j), (ni, nj))
    return G

def deteksi_loop(letter_skeleton):
    G = skeleton_to_graph(letter_skeleton)
    all_cycles = nx.cycle_basis(G)
    valid_cycles = []
    for cycle in all_cycles:
        if len(cycle) >= 2:
            cycle_array = np.array(cycle)
            try: area = ConvexHull(cycle_array).volume
            except: area = 0
            if area > 1.5: valid_cycles.append((cycle, area))
    return valid_cycles

def tsp_skeleton_traversal(points, prefer_lower=True, dist_threshold=3):
    if len(points) == 0: return []
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

def cektiang(skel, start_yx, min_height=15, max_width_spread=20, debug=False):
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
        if ny > sy: continue
        stack = [(ny, nx)]
        visited = set([(ny, nx), (sy, sx)]) 
        visited_list_branch = [(ny, nx)] 
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
                                visited_list_branch.append((nny, nnx))
                                stack.append((nny, nnx))
        
        if not is_valid_branch: continue
        b_height = sy - branch_min_y 
        b_width = branch_max_x - branch_min_x
        if b_height < min_height: continue 
        ratio = b_height / (b_width + 1)
        if ratio > 1.2 or b_height > 20:
            if debug: return True, visited_list_branch, (branch_min_x, branch_min_y, b_width, b_height)
            return True
    if debug: return False, [], (0, 0, 0, 0)
    return False

# ==================== LOGIKA JURI (TERMASUK RADAR DIAKRITIK) ====================

def dominasi_diakritik(min_x, max_x, dot_mask):
    """
    RADAR DIAKRITIK: Menembakkan radar lurus sejajar dengan lebar potongan huruf.
    Jika mendeteksi kode 1 = Titik Atas (Magenta). Kode 2 = Titik Bawah (Oranye).
    """
    # Toleransi lebar radar 10 piksel ke kiri & kanan agar tidak meleset
    x_start = max(0, min_x - 10)
    x_end = min(dot_mask.shape[1], max_x + 10)
    
    # Ambil irisan mask titik sejalur dengan huruf ini
    slice_dot = dot_mask[:, x_start:x_end]
    
    jumlah_atas = np.sum(slice_dot == 1)
    jumlah_bawah = np.sum(slice_dot == 2)
    
    # Putuskan mana yang lebih dominan
    if jumlah_atas > 0 and jumlah_atas > jumlah_bawah:
        return "ATAS"
    elif jumlah_bawah > 0 and jumlah_bawah > jumlah_atas:
        return "BAWAH"
    
    return "TIDAK_ADA"

def analisis_ujung_atas(seg_mask):
    skel = skeletonize(seg_mask)
    endpoints, _, _ = find_endpoints(skel)
    if len(endpoints) == 0: return 0
    coords = np.argwhere(seg_mask)
    min_y = coords[:, 0].min()
    max_y = coords[:, 0].max()
    batas_atas = min_y + (max_y - min_y) * 0.50 
    top_endpoints = [ep for ep in endpoints if ep[0] < batas_atas]
    return len(top_endpoints)

def cek_lam(h_seg, w_seg):
    return h_seg >= 45 and h_seg > w_seg * 1.2

def cek_sin(seg_mask, h_seg):
    if h_seg >= 45: return False
    gigi_atas = analisis_ujung_atas(seg_mask)
    return gigi_atas >= 3

def cek_nun(seg_mask, h_seg, w_seg, posisi_titik):
    if h_seg >= 45: return False
    
    # ATURAN EMAS: Punya titik di atas? SAH ITU NUN!
    if posisi_titik == "ATAS": return True
    
    # Kalau tidak ada titik, gunakan tebakan bentuk fisik (Lebar dan Ujung 2)
    if w_seg < 35: return False 
    gigi_atas = analisis_ujung_atas(seg_mask)
    return gigi_atas == 2 or (gigi_atas == 0 and w_seg > 20)

def cek_ya(seg_mask, h_seg, w_seg, posisi_titik):
    if h_seg >= 45: return False
    
    # ATURAN EMAS: Punya titik di bawah? SAH ITU YA!
    if posisi_titik == "BAWAH": return True
    
    # Kalau tidak ada titik, gunakan tebakan fisik (Sempit atau Ujung <= 1)
    if w_seg < 35: return True 
    gigi_atas = analisis_ujung_atas(seg_mask)
    return gigi_atas <= 1


# ==================== LOGIKA PEMOTONGAN & MAGNET DOT ====================

def hitung_garis_potong_sinkron(component_features, labeled_letters, skeleton_used, min_gap=3):
    all_cols = []
    for label, feat in component_features.items():
        mask = (labeled_letters == label)
        if np.sum(mask) < 5: continue
        letter_skeleton = skeleton_used * mask
        loops = deteksi_loop(letter_skeleton)
        
        intersections = sorted([ix for iy, ix in feat["intersections"]])
        if not intersections: continue
            
        intersections_map = {}
        for iy, ix in feat["intersections"]:
            if ix not in intersections_map: intersections_map[ix] = []
            intersections_map[ix].append(iy)

        first_ix = intersections[0]
        skip_start = False
        if loops:
            loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
            if loop_xs and abs(first_ix - loop_xs[0]) < 8: skip_start = True
        if not skip_start: all_cols.append(first_ix + 2)

        for i in range(len(intersections) - 1):
            curr_x = intersections[i]
            next_x = intersections[i+1]
            gap = next_x - curr_x
            
            is_curr_tiang = False
            ys = intersections_map.get(curr_x, [])
            for y_check in ys:
                if cektiang(letter_skeleton, (y_check, curr_x)): 
                    is_curr_tiang = True
                    break
            
            if is_curr_tiang:
                all_cols.append((curr_x + next_x) // 2)
            elif gap < 20:
                continue 
            else:
                skip_cut = False
                if loops:
                    loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                    if loop_xs:
                        dist = min([abs(next_x - lx) for lx in loop_xs])
                        if dist < 8: skip_cut = True
                if not skip_cut:
                    all_cols.append(curr_x + 2) 

    all_cols = sorted(set(all_cols))
    merged_cols = []
    if all_cols:
        merged_cols = [all_cols[0]]
        for col in all_cols[1:]:
            if col - merged_cols[-1] < min_gap:
                merged_cols[-1] = (merged_cols[-1] + col) // 2
            else:
                merged_cols.append(col)

    final_cols_vis = [0] + merged_cols + [skeleton_used.shape[1] - 1]
    return final_cols_vis

def potong_persimpangan(labeled_letters, skeleton_used, dot_mask):
    hasil_potongan = []
    num_features = np.max(labeled_letters)
    labeled_dots, num_dots = measure.label(dot_mask > 0, return_num=True, connectivity=2)
    dots_props = measure.regionprops(labeled_dots)

    for region_label in range(1, num_features + 1):
        mask_group = (labeled_letters == region_label)
        if np.sum(mask_group) < 5: continue

        letter_skeleton = skeleton_used * mask_group
        coords = np.argwhere(mask_group)
        if len(coords) == 0: continue
        
        x_coords = coords[:, 1]
        
        _, raw_intersections, _ = find_endpoints(letter_skeleton)
        loops = deteksi_loop(letter_skeleton)
        
        intersections = sorted([ix for iy, ix in raw_intersections])
        cut_points = []
        if intersections:
            intersections_map = {}
            for iy, ix in raw_intersections:
                if ix not in intersections_map: intersections_map[ix] = []
                intersections_map[ix].append(iy)

            first_ix = intersections[0]
            skip_start = False
            if loops:
                loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                if loop_xs and abs(first_ix - loop_xs[0]) < 8: skip_start = True
            if not skip_start: cut_points.append(first_ix + 2)

            for i in range(len(intersections) - 1):
                curr_x = intersections[i]
                next_x = intersections[i+1]
                gap = next_x - curr_x
                
                is_curr_tiang = False
                ys = intersections_map.get(curr_x, [])
                for y_check in ys:
                    if cektiang(letter_skeleton, (y_check, curr_x)): 
                        is_curr_tiang = True
                        break
                
                if is_curr_tiang:
                    cut_points.append((curr_x + next_x) // 2)
                elif gap < 20:
                    continue 
                else:
                    skip_cut = False
                    if loops:
                        loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                        if loop_xs:
                            dist = min([abs(next_x - lx) for lx in loop_xs])
                            if dist < 8: skip_cut = True
                    if not skip_cut:
                        cut_points.append(curr_x + 2)

        cuts = []
        if cut_points:
            minc, maxc = x_coords.min(), x_coords.max()
            valid_cuts = [int(np.clip(c, minc, maxc)) for c in cut_points]
            cuts = sorted(set([minc] + valid_cuts + [maxc + 1]))
        else:
            cuts = [x_coords.min(), x_coords.max() + 1]

        clean_segments = [] 
        segment_metadata = [] 

        for i in range(len(cuts) - 1):
            start_col, end_col = cuts[i], cuts[i + 1]
            segment_width = end_col - start_col
            segment_mask = np.zeros_like(mask_group)
            segment_mask[:, start_col:end_col] = mask_group[:, start_col:end_col]
            segment_mask = segment_mask & (dot_mask == 0)
            
            if np.sum(segment_mask) > 30 and segment_width > 5:
                seg_crds = np.argwhere(segment_mask)
                if len(seg_crds) > 0:
                    min_y_seg, min_x_seg = seg_crds.min(axis=0)
                    max_y_seg, max_x_seg = seg_crds.max(axis=0)
                    h_seg = max_y_seg - min_y_seg
                    w_seg = max_x_seg - min_x_seg
                    
                    # Tembakkan Radar Titik!
                    posisi_titik = dominasi_diakritik(min_x_seg, max_x_seg, dot_mask)
                else:
                    h_seg = 999; w_seg = 999
                    posisi_titik = "TIDAK_ADA"
                
                tipe_huruf = "UNKNOWN"
                if cek_lam(h_seg, w_seg):
                    tipe_huruf = "LAM"
                elif cek_sin(segment_mask, h_seg):
                    tipe_huruf = "SIN"
                elif cek_nun(segment_mask, h_seg, w_seg, posisi_titik):
                    tipe_huruf = "NUN"
                elif cek_ya(segment_mask, h_seg, w_seg, posisi_titik):
                    tipe_huruf = "YA"
                
                clean_segments.append(segment_mask)
                segment_metadata.append({'tipe_huruf': tipe_huruf})

        if not clean_segments: 
            hasil_potongan.append(mask_group)
            continue

        for dot_prop in dots_props:
            dy, dx = dot_prop.centroid
            grp_min_r, grp_min_c, grp_max_r, grp_max_c = measure.regionprops(mask_group.astype(int))[0].bbox
            if not (grp_min_r - 50 <= dy <= grp_max_r + 200 and grp_min_c - 50 <= dx <= grp_max_c + 50):
                continue

            best_candidate = None
            best_score = float('inf') 

            for idx, seg_mask in enumerate(clean_segments):
                seg_cy, seg_cx = center_of_mass(seg_mask)
                tipe_huruf = segment_metadata[idx].get('tipe_huruf', 'UNKNOWN')
                
                dist_x_centroid = abs(dx - seg_cx)
                dist_y = abs(dy - seg_cy)
                
                if tipe_huruf == "YA":
                    if dist_x_centroid < 25: final_score = -10000 + dist_x_centroid 
                    else: final_score = dist_x_centroid + (dist_y * 0.8)
                elif tipe_huruf == "NUN":
                    if dy < seg_cy and dist_x_centroid < 35: final_score = -10000 + dist_x_centroid
                    else: final_score = dist_x_centroid + (dist_y * 0.8)
                else:
                    final_score = dist_x_centroid + (dist_y * 0.8)

                if final_score < best_score:
                    best_score = final_score
                    best_candidate = idx

            if best_candidate is not None:
                seg_mask_winner = clean_segments[best_candidate]
                win_cy, win_cx = center_of_mass(seg_mask_winner)
                dist_real = euclidean((dy, dx), (win_cy, win_cx))
                is_win_via_magnet = (best_score < -5000)
                
                if is_win_via_magnet or dist_real < 180:
                    dot_indices = (labeled_dots == dot_prop.label)
                    clean_segments[best_candidate] = np.logical_or(clean_segments[best_candidate], dot_indices)

        hasil_potongan.extend(clean_segments)
    return hasil_potongan

# ==================== MAIN EXECUTION ====================
                               
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
    if np.sum(mask) < 5: continue
    letter_skeleton = skeleton_used * mask
    endpoints, intersections, _ = find_endpoints(letter_skeleton)
    loops = deteksi_loop(letter_skeleton)
    component_features[region_label] = {
        "intersections": intersections,
        "has_loop": len(loops) > 0,
        "cycles": [cycle for cycle, _ in loops]
    }

# =============================================================================
# 1. JALANKAN PROSES PEMOTONGAN DULU
# =============================================================================
hasil_potongan = potong_persimpangan(labeled_letters, skeleton_used, dot_mask)

# Urutkan potongan dari Kanan ke Kiri
sorted_hasil_potongan = sorted(hasil_potongan, key=lambda mask: np.max(np.argwhere(mask)[:, 1]), reverse=True)

# =============================================================================
# 2. VISUALISASI UTAMA (SINKRON DENGAN HASIL POTONGAN)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 12)) 
ax.imshow(skeleton_used, cmap='gray')

# --- A. Plot Fitur Dasar ---
for label, props in component_features.items():
    mask = (labeled_letters == label)
    skel = skeletonize(mask)
    for iy, ix in props["intersections"]:
        ax.plot(ix, iy, 'o', color='green', markersize=4, label='Intersection' if label == 1 else "")
    loops = props["cycles"]
    for cycle in loops:
        cycle = np.array(cycle)
        ax.plot(cycle[:, 1], cycle[:, 0], 'r-', linewidth=2, label="Loop" if label == 1 else "")
    _, endpoints, _ = find_endpoints(skel)
    if len(endpoints) > 0:
        ax.plot(endpoints[0][1], endpoints[0][0], 'o', color='blue') 
        ax.plot(endpoints[-1][1], endpoints[-1][0], 'o', color='yellow')

for yx in np.argwhere(dot_mask == 1):
    ax.plot(yx[1], yx[0], 'o', color='magenta', label='Diakritik Atas')
for yx in np.argwhere(dot_mask == 2):
    ax.plot(yx[1], yx[0], 'o', color='orange', label='Diakritik Bawah')

final_cols = hitung_garis_potong_sinkron(component_features, labeled_letters, skeleton_used)
for col in final_cols:
    ax.axvline(x=col, color='red', linestyle='--', linewidth=2, label='Garis Potong')

# --- B. PLOT KOTAK DETEKSI BERDASARKAN RADAR DIAKRITIK ---
for idx, region_mask in enumerate(sorted_hasil_potongan):
    badan_huruf = region_mask & (dot_mask == 0)
    seg_crds = np.argwhere(badan_huruf)
    if len(seg_crds) == 0: continue
    
    min_y_seg, min_x_seg = seg_crds.min(axis=0)
    max_y_seg, max_x_seg = seg_crds.max(axis=0)
    h_seg = max_y_seg - min_y_seg
    w_seg = max_x_seg - min_x_seg
    
    # Tembakkan Radar Titik di Visualisasi
    posisi_titik = dominasi_diakritik(min_x_seg, max_x_seg, dot_mask)
    
    skel_part = skeletonize(badan_huruf)
    py, px = np.where(skel_part)

    tipe_huruf = "UNKNOWN"
    if cek_lam(h_seg, w_seg):
        tipe_huruf = "LAM"
    elif cek_sin(badan_huruf, h_seg):
        tipe_huruf = "SIN"
    elif cek_nun(badan_huruf, h_seg, w_seg, posisi_titik):
        tipe_huruf = "NUN"
    elif cek_ya(badan_huruf, h_seg, w_seg, posisi_titik):
        tipe_huruf = "YA"

    if tipe_huruf == "YA":
        rect = patches.Rectangle((min_x_seg, min_y_seg), w_seg, h_seg, linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(rect)
        ax.text(min_x_seg, min_y_seg-10, f"YA' (H{idx+1})", color='orange', fontsize=10, fontweight='bold')
        ax.scatter(px, py, color='yellow', s=20, alpha=1.0, label='Area Gigi (Ya)' if idx == 0 else "")
    elif tipe_huruf == "SIN":
        rect = patches.Rectangle((min_x_seg, min_y_seg), w_seg, h_seg, linewidth=2, edgecolor='magenta', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(min_x_seg, min_y_seg-10, f"SIN (H{idx+1})", color='magenta', fontsize=10, fontweight='bold')
    elif tipe_huruf == "NUN":
        rect = patches.Rectangle((min_x_seg, min_y_seg), w_seg, h_seg, linewidth=2, edgecolor='deepskyblue', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(min_x_seg, min_y_seg-10, f"NUN (H{idx+1})", color='deepskyblue', fontsize=10, fontweight='bold')
    elif tipe_huruf == "LAM":
        rect = patches.Rectangle((min_x_seg, min_y_seg), w_seg, h_seg, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(min_x_seg, min_y_seg-10, f"LAM (H{idx+1})", color='lime', fontsize=10, fontweight='bold')
        ax.scatter(px, py, color='cyan', s=20, alpha=1.0, label='Area Tiang (Lam)' if idx == 0 else "")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_title("Visualisasi Utama Sinkron: Klasifikasi Super (Mendeteksi Diakritik)")
ax.axis('off')
plt.tight_layout()
plt.show()

# =============================================================================
# 3. PENYIMPANAN DAN EVALUASI 
# =============================================================================

output_folder = "hasil_potongan"
os.makedirs(output_folder, exist_ok=True)

for idx, region_mask in enumerate(sorted_hasil_potongan):
    output_image = (region_mask * 255).astype(np.uint8)
    output_filename = os.path.join(output_folder, f'potongan_huruf_{idx + 1}.png')
    cv.imwrite(output_filename, output_image)

print(f"Hasil potongan huruf disimpan di folder: {output_folder}")

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
    
    ax[0].imshow(region_mask, cmap='gray')
    ax[0].set_title(f'Huruf Potong ke-{idx+1} (Biner)')
    ax[0].axis('off')

    letter_skeleton = skeletonize(region_mask)
    endpoints, intersections, turns = find_endpoints(letter_skeleton)
    centroid = center_of_mass(region_mask)
    
    ax[1].imshow(letter_skeleton, cmap='gray')
    
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