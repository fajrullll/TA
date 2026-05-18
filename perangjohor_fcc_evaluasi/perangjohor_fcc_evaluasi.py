# ==================== IMPORT ====================
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from skimage.morphology import skeletonize
from skimage import io, color, measure
from scipy.ndimage import center_of_mass
import numpy as np
import cv2 as cv
from skimage.segmentation import relabel_sequential
from scipy.spatial.distance import euclidean, cdist
import networkx as nx                  
from scipy.spatial import ConvexHull   
import os
import csv

# ==================== METODE B-SPLINE ====================

def basis_function(i, k, x, knots):
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i+1] else 0.0
    
    denom1 = knots[i+k] - knots[i]
    term1 = 0
    if denom1 != 0:
        term1 = ((x - knots[i]) / denom1) * basis_function(i, k-1, x, knots)
        
    denom2 = knots[i+k+1] - knots[i+1]
    term2 = 0
    if denom2 != 0:
        term2 = ((knots[i+k+1] - x) / denom2) * basis_function(i+1, k-1, x, knots)
        
    return term1 + term2

def bspline_curve(control_points, degree, num_points=100):
    n = len(control_points)
    knots = np.concatenate([
        np.zeros(degree), 
        np.linspace(0, 1, n - degree + 1), 
        np.ones(degree)
    ])
    knots = np.append(knots, [1.0]) 

    x_vals = np.linspace(0, 0.999, num_points)
    curve = []
    
    for x in x_vals:
        point = np.zeros(2)
        for i in range(n):
            b = basis_function(i, degree, x, knots)
            point += b * control_points[i]
        curve.append(point)
        
    return np.array(curve)

def hitung_panjang_kurva(curve):
    """
    Menghitung panjang kurva B-Spline berdasarkan jarak antar titik kurva.
    """
    if curve is None or len(curve) < 2:
        return 0.0

    total = 0.0

    for i in range(len(curve) - 1):
        total += euclidean(curve[i], curve[i + 1])

    return float(total)


def hitung_smoothness_kurva(curve):
    """
    Menghitung smoothness/kelengkungan kurva.
    Semakin kecil nilainya, semakin halus kurvanya.
    """
    if curve is None or len(curve) < 3:
        return 0.0

    sudut = []

    for i in range(1, len(curve) - 1):
        p0 = curve[i - 1]
        p1 = curve[i]
        p2 = curve[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0 or n2 == 0:
            continue

        cos_theta = np.dot(v1, v2) / (n1 * n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        sudut.append(theta)

    if len(sudut) == 0:
        return 0.0

    return float(np.mean(sudut))

# ==================== DINAMISASI STROKE & SKALA ====================

def hitung_lebar_garis_stroke(binary):
    dist = cv.distanceTransform(binary.astype(np.uint8), cv.DIST_L2, 5)
    skel = skeletonize(binary)
    if np.sum(skel) == 0: return 8.0
    
    thicknesses = dist[skel] * 2
    r_stroke = np.median(thicknesses)
    return float(r_stroke) if r_stroke >= 1.0 else 8.0

def hapus_noise(binary, min_area):
    labeled, num = measure.label(binary, return_num=True, connectivity=2)
    props = measure.regionprops(labeled)
    clean_binary = np.zeros_like(binary, dtype=bool)
    for prop in props:
        if prop.area >= min_area:
            clean_binary[labeled == prop.label] = True
    return clean_binary

def gabungkan_diakritik_dan_mainstroke(binary, A_diac_max, scale, rasio_alif=2.5):
    labeled_image, _ = measure.label(binary, return_num=True, connectivity=2)
    props = measure.regionprops(labeled_image)

    areas = [p.area for p in props]
    max_area = max(areas) if areas else 0

    main_infos = {}
    dot_mask = np.zeros_like(binary, dtype=np.uint8)
    label_map = labeled_image.copy()

    jarak_maks = 300 * scale 

    for prop in props:
        area = prop.area
        bbox = prop.bbox
        height = bbox[2] - bbox[0]
        width  = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 1e-5)
        max_dim = max(height, width)

        is_alif = height > (20 * scale) and aspect_ratio > rasio_alif
        is_wide_body = width > (30 * scale) and area > (8 * A_diac_max)
        is_dominant = area > max(10 * A_diac_max, 0.05 * max_area)
        
        is_main_body = is_alif or is_wide_body or is_dominant
        
        if max_dim < (25 * scale):
            is_main_body = False
        elif area < (6 * A_diac_max) and aspect_ratio < 3.0:
            is_main_body = False

        if is_main_body:
            main_infos[prop.label] = {"centroid": prop.centroid, "bbox": bbox}

    for prop in props:
        label = prop.label
        if label in main_infos: continue

        centroid = prop.centroid
        if not main_infos: continue
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

# ==================== PENDETEKSI FITUR ====================

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

# --- PENAMBAHAN: FUNGSI GRAPH & LOOP DARI GAMBAR ---
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

# ==================== METODE FREEMAN CHAIN CODE ====================

FREEMAN_DIR_MAP = {
    (0, 1): 0,     # East
    (-1, 1): 1,    # North-East
    (-1, 0): 2,    # North
    (-1, -1): 3,   # North-West
    (0, -1): 4,    # West
    (1, -1): 5,    # South-West
    (1, 0): 6,     # South
    (1, 1): 7      # South-East
}

def freeman_direction(p1, p2):
    """
    Menghasilkan kode Freeman 8-arah dari dua titik skeleton.
    p1 dan p2 berbentuk (y, x).
    """
    dy = int(np.sign(p2[0] - p1[0]))
    dx = int(np.sign(p2[1] - p1[1]))
    return FREEMAN_DIR_MAP.get((dy, dx), None)

def freeman_code_difference(c1, c2):
    """
    Menghitung perubahan arah secara melingkar pada Freeman Code.
    Contoh: beda antara 7 dan 0 adalah 1, bukan 7.
    """
    diff = abs(int(c2) - int(c1))
    return min(diff, 8 - diff)

def skeleton_longest_path(skel):
    """
    Mengambil jalur skeleton utama berdasarkan endpoint terjauh.
    Jalur ini kemudian dipakai untuk membentuk Freeman Chain Code.
    """
    G = skeleton_to_graph(skel)

    if G.number_of_nodes() == 0:
        return []

    endpoints = [node for node, degree in G.degree() if degree == 1]

    if len(endpoints) >= 2:
        best_path = []

        for start in endpoints:
            paths = nx.single_source_shortest_path(G, start)

            for end in endpoints:
                if start == end:
                    continue

                if end in paths and len(paths[end]) > len(best_path):
                    best_path = paths[end]

        return best_path

    # Fallback jika skeleton tidak memiliki dua endpoint jelas
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)

    start = next(iter(H.nodes))
    lengths = nx.single_source_shortest_path_length(H, start)
    farthest_1 = max(lengths, key=lengths.get)

    paths = nx.single_source_shortest_path(H, farthest_1)
    farthest_2 = max(paths, key=lambda node: len(paths[node]))

    return paths[farthest_2]

def freeman_chain_from_skeleton(skel):
    """
    Membentuk Freeman Chain Code dari jalur utama skeleton.
    """
    path = skeleton_longest_path(skel)

    if len(path) < 2:
        return np.array([]), np.array([])

    path = np.array(path)
    codes = []

    for i in range(len(path) - 1):
        code = freeman_direction(path[i], path[i + 1])
        if code is not None:
            codes.append(code)

    return path, np.array(codes)

def deteksi_potong_freeman(mask_group, skeleton_used, dot_mask, scale, target_xrange=None):
    """
    Mendeteksi kandidat garis potong menggunakan Freeman Chain Code.

    Fungsi ini digunakan sebagai pemotongan tambahan setelah metode topologi graf.
    Pada kasus fa-alif yang tidak memiliki titik intersection jelas, jalur skeleton
    dikonversi menjadi urutan arah Freeman 8-arah, lalu perubahan arah di dalam
    area target dipakai untuk menentukan kandidat kolom potong.
    """
    body = mask_group & (dot_mask == 0)

    if np.sum(body) < 10:
        return None, None

    skel = skeleton_used & body

    if np.sum(skel) < 5:
        skel = skeletonize(body)

    path, codes = freeman_chain_from_skeleton(skel)

    if len(path) < 10 or len(codes) < 5:
        return None, None

    coords = np.argwhere(body)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    h = max_y - min_y
    w = max_x - min_x

    # Hindari pemotongan pada komponen yang terlalu kecil.
    if h < 25 * scale or w < 25 * scale:
        return None, None

    # Area yang diizinkan untuk kandidat FCC.
    # Jika target_xrange diberikan, FCC hanya boleh memilih kolom di area ini.
    if target_xrange is not None:
        allowed_min = max(min_x + int(3 * scale), int(target_xrange[0]))
        allowed_max = min(max_x - int(3 * scale), int(target_xrange[1]))
    else:
        allowed_min = int(min_x + 0.15 * w)
        allowed_max = int(max_x - 0.15 * w)

    if allowed_max <= allowed_min:
        return None, None

    vertical_codes = {2, 6}
    window = max(3, int(4 * scale))

    transition_candidates = []
    turn_points = []
    turn_indices = []

    # Deteksi semua titik perubahan arah pada jalur Freeman.
    for i in range(1, len(codes)):
        diff = freeman_code_difference(codes[i - 1], codes[i])

        if diff >= 2:
            turn_points.append(path[i])
            turn_indices.append(i)

    # Kandidat utama: perubahan rasio arah vertikal atau belokan lokal.
    for i in range(window, len(codes) - window):
        left_codes = codes[i - window:i]
        right_codes = codes[i:i + window]

        left_vertical_ratio = np.mean([c in vertical_codes for c in left_codes])
        right_vertical_ratio = np.mean([c in vertical_codes for c in right_codes])

        direction_change = abs(left_vertical_ratio - right_vertical_ratio)
        local_turn = freeman_code_difference(codes[i - 1], codes[i])

        y, x = path[i]

        # Kandidat hanya boleh berada di area target Freeman.
        if not (allowed_min <= x <= allowed_max):
            continue

        if direction_change >= 0.45 or local_turn >= 2:
            transition_candidates.append({
                "index": i,
                "point": (int(y), int(x)),
                "x": int(x),
                "turn": float(local_turn),
                "direction_change": float(direction_change)
            })

    # Fallback: jika tidak ada kandidat perubahan arah yang lolos,
    # ambil semua titik path yang berada di area target sebagai kandidat.
    if not transition_candidates:
        for j in range(1, len(codes)):
            y, x = path[j]

            if not (allowed_min <= x <= allowed_max):
                continue

            diff_prev = freeman_code_difference(codes[j - 1], codes[j])

            if j < len(codes) - 1:
                diff_next = freeman_code_difference(codes[j], codes[j + 1])
            else:
                diff_next = 0

            transition_candidates.append({
                "index": j,
                "point": (int(y), int(x)),
                "x": int(x),
                "turn": float(diff_prev + diff_next),
                "direction_change": float(diff_prev)
            })

    if not transition_candidates:
        return None, {
            "body": body,
            "path": path,
            "codes": codes,
            "turn_points": np.array(turn_points),
            "turn_indices": np.array(turn_indices),
            "candidate_points": np.array([]),
            "cut_col": None,
            "best_candidate_index": None
        }

    profile = np.sum(body, axis=0)

    best_cut = None
    best_score = float("inf")
    best_candidate_index = None

    for cand in transition_candidates:
        x = cand["x"]

        search_start = int(max(allowed_min, x - 6 * scale))
        search_end = int(min(allowed_max, x + 6 * scale))

        if search_end <= search_start:
            continue

        for col in range(search_start, search_end + 1):
            left_part = body[:, min_x:col]
            right_part = body[:, col:max_x + 1]

            left_area = np.sum(left_part)
            right_area = np.sum(right_part)

            # Validasi agar kedua sisi hasil potong tetap memiliki piksel cukup.
            if left_area < 20 * (scale ** 2):
                continue

            if right_area < 20 * (scale ** 2):
                continue

            # Skor menggabungkan kepadatan kolom, kedekatan ke kandidat,
            # serta kekuatan perubahan arah Freeman.
            score = (
                profile[col] * 3.0
                + abs(col - x) * 0.5
                - cand["turn"] * 2.0
                - cand["direction_change"] * 4.0
            )

            if score < best_score:
                best_score = score
                best_cut = int(col)
                best_candidate_index = cand["index"]

    candidate_points = np.array([cand["point"] for cand in transition_candidates])

    debug_info = {
        "body": body,
        "path": path,
        "codes": codes,
        "turn_points": np.array(turn_points),
        "turn_indices": np.array(turn_indices),
        "candidate_points": candidate_points,
        "transition_candidates": transition_candidates,
        "cut_col": best_cut,
        "best_candidate_index": best_candidate_index
    }

    return best_cut, debug_info

def visualisasi_freeman(debug_records):
    """
    Menampilkan visualisasi Freeman Chain Code:
    1. Jalur skeleton Freeman.
    2. Titik perubahan arah.
    3. Kandidat titik transisi.
    4. Garis potong hasil Freeman.
    5. Grafik kode arah Freeman.
    """
    if not debug_records:
        print("\nTidak ada segmentasi tambahan dari Freeman Chain Code.")
        return

    print("\nMenampilkan visualisasi Freeman Chain Code...")

    for idx, rec in enumerate(debug_records):
        body = rec["body"]
        path = rec["path"]
        codes = rec["codes"]
        cut_col = rec["cut_col"]

        turn_points = rec.get("turn_points", np.array([]))
        candidate_points = rec.get("candidate_points", np.array([]))
        turn_indices = rec.get("turn_indices", np.array([]))
        best_candidate_index = rec.get("best_candidate_index", None)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].imshow(body, cmap="gray")
        ax[0].plot(path[:, 1], path[:, 0], color="cyan", linewidth=1.2, label="Freeman Path")

        if len(turn_points) > 0:
            ax[0].scatter(
                turn_points[:, 1],
                turn_points[:, 0],
                color="yellow",
                s=25,
                label="Direction Change"
            )

        if len(candidate_points) > 0:
            ax[0].scatter(
                candidate_points[:, 1],
                candidate_points[:, 0],
                color="lime",
                s=35,
                label="Candidate Point"
            )

        if cut_col is not None:
            ax[0].axvline(
                x=cut_col,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Freeman Cut"
            )

        ax[0].set_title(f"Freeman Chain Code Segmentation - Region {rec.get('region_label', idx + 1)}")
        ax[0].legend(loc="upper right")
        ax[0].axis("off")

        ax[1].plot(np.arange(len(codes)), codes, marker="o", linewidth=1)
        ax[1].set_title("Freeman Direction Code Sequence")
        ax[1].set_xlabel("Skeleton Path Index")
        ax[1].set_ylabel("Freeman Direction Code")
        ax[1].set_yticks(range(8))
        ax[1].grid(True, alpha=0.3)

        if len(turn_indices) > 0:
            valid_turn_indices = [i for i in turn_indices if i < len(codes)]
            ax[1].scatter(
                valid_turn_indices,
                codes[valid_turn_indices],
                color="yellow",
                s=40,
                label="Direction Change"
            )

        if best_candidate_index is not None and best_candidate_index < len(codes):
            ax[1].axvline(
                x=best_candidate_index,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Selected Cut Index"
            )

        ax[1].legend(loc="best")

        plt.tight_layout()
        plt.show()

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

def cektiang(skel, start_yx, scale, debug=False):
    sy, sx = start_yx
    h, w = skel.shape
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = sy + dy, sx + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                neighbors.append((ny, nx))

    min_height = 25 * scale 
    max_width_spread = 20 * scale 

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
                            if nny < sy + (5 * scale): 
                                visited.add((nny, nnx))
                                visited_list_branch.append((nny, nnx))
                                stack.append((nny, nnx))
        
        if not is_valid_branch: continue
        b_height = sy - branch_min_y 
        b_width = branch_max_x - branch_min_x
        if b_height < min_height: continue 
        ratio = b_height / (b_width + 1)
        if ratio > 1.2 or b_height > (20 * scale): 
            if debug: return True, visited_list_branch, (branch_min_x, branch_min_y, b_width, b_height)
            return True
    if debug: return False, [], (0, 0, 0, 0)
    return False

def dominasi_diakritik(min_x, max_x, dot_mask, scale):
    x_start = int(max(0, min_x - (10 * scale)))
    x_end = int(min(dot_mask.shape[1], max_x + (10 * scale)))
    slice_dot = dot_mask[:, x_start:x_end]
    
    jumlah_atas = np.sum(slice_dot == 1)
    jumlah_bawah = np.sum(slice_dot == 2)
    
    if jumlah_atas == 0 and jumlah_bawah == 0: return "TIDAK_ADA"
        
    if jumlah_atas > jumlah_bawah:
        posisi = "ATAS"
        val = 1
    else:
        posisi = "BAWAH"
        val = 2
        
    kolom_berisi_titik = np.any(slice_dot == val, axis=0)
    lebar_titik = np.sum(kolom_berisi_titik)
    
    if lebar_titik > (20 * scale): return f"{posisi}_DOUBLE"
    else: return f"{posisi}_SINGLE"
    
def estimasi_jumlah_diakritik(dot_prop, dot_mask, scale):
    """
    Mengestimasi jenis dan jumlah titik dari satu komponen diakritik.

    Perbaikan untuk H2/NGA:
    - Jika tiga titik atas menyatu menjadi satu blob, jangan dihitung sebagai 1.
    - Estimasi jumlah titik memakai kombinasi:
      1. jenis posisi titik dari dot_mask,
      2. lebar bbox,
      3. rasio lebar terhadap tinggi,
      4. luas area blob,
      5. jumlah run horizontal pada proyeksi kolom.
    """
    dot_coords = dot_prop.coords
    dot_values = dot_mask[dot_coords[:, 0], dot_coords[:, 1]]

    jumlah_atas = np.sum(dot_values == 1)
    jumlah_bawah = np.sum(dot_values == 2)

    if jumlah_atas >= jumlah_bawah:
        jenis = "ATAS"
        target_val = 1
    else:
        jenis = "BAWAH"
        target_val = 2

    min_r, min_c, max_r, max_c = dot_prop.bbox
    dot_width = max_c - min_c
    dot_height = max_r - min_r
    dot_area = dot_prop.area

    aspect = dot_width / (dot_height + 1e-5)

    # Ambil crop khusus komponen titik
    crop = (dot_mask[min_r:max_r, min_c:max_c] == target_val)

    # Proyeksi horizontal untuk melihat sebaran titik dari kiri ke kanan
    col_has_pixel = np.any(crop, axis=0)

    # Hitung jumlah run horizontal.
    # Jika titik terpisah jelas, run bisa 2 atau 3.
    run_count = 0
    in_run = False

    for val in col_has_pixel:
        if val and not in_run:
            run_count += 1
            in_run = True
        elif not val:
            in_run = False

    # ==========================================================
    # ESTIMASI JUMLAH TITIK
    # ==========================================================
    jumlah = 1

    # Jika komponen punya 3 run terpisah, langsung 3.
    if run_count >= 3:
        jumlah = 3

    # Jika blob melebar secara horizontal, kemungkinan tiga titik menyatu.
    elif dot_width >= 9 * scale and aspect >= 1.25 and dot_area >= 12 * (scale ** 2):
        jumlah = 3

    # Jika cukup lebar tetapi tidak sepanjang tiga titik, kemungkinan dua titik.
    elif run_count == 2:
        jumlah = 2

    elif dot_width >= 6 * scale and aspect >= 1.05 and dot_area >= 8 * (scale ** 2):
        jumlah = 2

    else:
        jumlah = 1

    return {
        "jenis": jenis,
        "jumlah": jumlah,
        "centroid": dot_prop.centroid,
        "bbox": dot_prop.bbox,
        "label": dot_prop.label,
        "width": dot_width,
        "height": dot_height,
        "area": dot_area,
        "aspect": aspect,
        "run_count": run_count
    }

def hitung_estimasi_diakritik_final(region_mask, dot_mask, scale, jenis_target="ATAS"):
    """
    Menghitung estimasi jumlah titik final pada satu hasil potongan huruf.

    Berbeda dari measure.label biasa:
    - measure.label hanya menghitung jumlah blob.
    - fungsi ini menghitung estimasi jumlah titik dalam setiap blob.

    Ini penting untuk H2/NGA, karena tiga titik bisa menyatu menjadi satu blob.
    """
    if jenis_target == "ATAS":
        target_val = 1
    else:
        target_val = 2

    final_dot_mask = region_mask & (dot_mask == target_val)

    labeled_final_dots, num_final_components = measure.label(
        final_dot_mask,
        return_num=True,
        connectivity=2
    )

    total_estimasi = 0
    komponen_info = []

    for prop in measure.regionprops(labeled_final_dots):
        info = estimasi_jumlah_diakritik(prop, dot_mask, scale)
        total_estimasi += info["jumlah"]
        komponen_info.append(info)

    return total_estimasi, num_final_components, komponen_info


def buat_profil_diakritik_segmen(segment_mask, dots_props, dot_mask, scale):
    """
    Membuat profil diakritik untuk satu segmen huruf.
    Area pencarian dibuat lebih ketat agar titik huruf tetangga
    tidak ikut dihitung sebagai milik segmen ini.
    """
    profil = {
        "atas_estimasi": 0,
        "bawah_estimasi": 0,
        "komponen_atas": [],
        "komponen_bawah": []
    }

    seg_crds = np.argwhere(segment_mask)

    if len(seg_crds) == 0:
        return profil

    min_y_seg, min_x_seg = seg_crds.min(axis=0)
    max_y_seg, max_x_seg = seg_crds.max(axis=0)

    # Dibuat lebih ketat dari sebelumnya.
    # Sebelumnya terlalu longgar sehingga H4 bisa mengambil titik H3/H5.
    batas_kiri = min_x_seg - 35 * scale
    batas_kanan = max_x_seg + 35 * scale

    for dot_idx, dot_prop in enumerate(dots_props):
        dy, dx = dot_prop.centroid

        if not (batas_kiri <= dx <= batas_kanan):
            continue

        info = estimasi_jumlah_diakritik(dot_prop, dot_mask, scale)
        info["dot_idx"] = dot_idx

        if info["jenis"] == "ATAS":
            profil["atas_estimasi"] += info["jumlah"]
            profil["komponen_atas"].append(info)
        else:
            profil["bawah_estimasi"] += info["jumlah"]
            profil["komponen_bawah"].append(info)

    return profil

def cari_segmen_terdekat_horizontal(dx, clean_segments):
    """
    Menentukan segmen yang paling dekat secara horizontal terhadap centroid titik.
    Ini mencegah titik milik H3/H5 tertarik ke H4.
    """
    best_idx = None
    best_dist = float("inf")

    for idx, seg_mask in enumerate(clean_segments):
        coords = np.argwhere(seg_mask)

        if len(coords) == 0:
            continue

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        if min_x <= dx <= max_x:
            dist = 0
        elif dx < min_x:
            dist = min_x - dx
        else:
            dist = dx - max_x

        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx, best_dist

def huruf_tanpa_diakritik_strict(tipe_huruf):
    """
    Huruf yang secara aturan tidak boleh menerima diakritik.
    Ini mencegah DAL/RA/WAW/ALIF/LAM mengambil titik dari huruf tetangga.
    """
    return tipe_huruf in [
        "ALIF",
        "LAM",
        "DAL",
        "RA",
        "WAW",
        "PENOLAK_DIAKRITIK"
    ]


def target_diakritik_dinamis(meta, target_diakritik):
    """
    Menentukan target diakritik dari tipe huruf dan profil diakritik.
    Jika tipe huruf salah klasifikasi, profil diakritik tetap bisa dipakai.
    Namun huruf tanpa diakritik tetap dikunci agar tidak menerima titik.
    """
    tipe = meta.get("tipe_huruf", "UNKNOWN")

    if huruf_tanpa_diakritik_strict(tipe):
        return None

    if tipe in target_diakritik:
        return target_diakritik[tipe]

    atas = meta.get("jumlah_diakritik_atas", 0)
    bawah = meta.get("jumlah_diakritik_bawah", 0)

    if atas >= 3:
        return {"jenis": "ATAS", "jumlah": 3}

    if atas == 2:
        return {"jenis": "ATAS", "jumlah": 2}

    if atas == 1:
        return {"jenis": "ATAS", "jumlah": 1}

    if bawah >= 2:
        return {"jenis": "BAWAH", "jumlah": 2}

    if bawah == 1:
        return {"jenis": "BAWAH", "jumlah": 1}

    return None


def overlap_1d(a_min, a_max, b_min, b_max):
    """
    Menghitung panjang overlap satu dimensi.
    Dipakai untuk melihat apakah bbox titik berada di atas bbox badan huruf.
    """
    return max(0, min(a_max, b_max) - max(a_min, b_min))


def dot_milik_region_group(dot_prop, labeled_dots, mask_group):
    """
    Mengunci kepemilikan dot terhadap region yang sedang diproses.

    Ini bagian penting untuk kasus H2:
    - dot H2 tidak boleh ikut diproses ketika fungsi sedang memproses region H3.
    - dot hanya boleh dialokasikan jika piksel dot tersebut memang overlap
      dengan mask_group saat ini.
    """
    dot_pixels = (labeled_dots == dot_prop.label)
    return np.any(dot_pixels & mask_group)


def cari_owner_diakritik_bbox(dot_prop, jenis_titik, clean_segments, scale):
    """
    Menentukan pemilik diakritik di dalam satu mask_group berdasarkan bbox.

    Fungsi ini hanya membandingkan antar clean_segments dalam region yang sama.
    Untuk mencegah titik lintas region seperti H2 pindah ke H3,
    tetap wajib memakai dot_milik_region_group() sebelum fungsi ini.
    """
    dot_min_r, dot_min_c, dot_max_r, dot_max_c = dot_prop.bbox
    dy, dx = dot_prop.centroid

    best_idx = None
    best_score = float("inf")

    for idx, seg_mask in enumerate(clean_segments):
        seg_crds = np.argwhere(seg_mask)

        if len(seg_crds) == 0:
            continue

        min_y_seg, min_x_seg = seg_crds.min(axis=0)
        max_y_seg, max_x_seg = seg_crds.max(axis=0)

        seg_cy, seg_cx = center_of_mass(seg_mask)

        if seg_cy is None or seg_cx is None:
            continue

        if jenis_titik == "ATAS":
            if dy >= seg_cy:
                continue
            jarak_vertikal = seg_cy - dy
        else:
            if dy <= seg_cy:
                continue
            jarak_vertikal = dy - seg_cy

        if jarak_vertikal < 2 * scale or jarak_vertikal > 130 * scale:
            continue

        overlap_x = overlap_1d(
            dot_min_c,
            dot_max_c,
            min_x_seg,
            max_x_seg
        )

        dot_width = dot_max_c - dot_min_c
        overlap_ratio_dot = overlap_x / (dot_width + 1e-5)

        horizontal_gap = max(
            0,
            min_x_seg - dot_max_c,
            dot_min_c - max_x_seg
        )

        score = (
            horizontal_gap * 4.0
            + abs(dx - seg_cx) * 0.8
            + jarak_vertikal * 0.3
            - overlap_ratio_dot * 500
        )

        if overlap_ratio_dot >= 0.50:
            score -= 1000

        if min_x_seg <= dx <= max_x_seg:
            score -= 700

        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score


def alokasi_diakritik_ke_segmen(
    clean_segments,
    segment_metadata,
    dots_props,
    labeled_dots,
    dot_mask,
    scale,
    mask_group
):
    """
    Mengalokasikan diakritik ke segmen huruf dengan penguncian region.

    Fokus :
    - Dot hanya boleh dialokasikan ke mask_group yang memang memilikinya.
    - Diakritik asli dipertahankan
    - NGA tetap bisa menerima blob yang terestimasi sebagai 3 titik.
    """
    kandidat_pasangan = []
    titik_terpakai = set()

    grp_props = measure.regionprops(mask_group.astype(int))

    if not grp_props:
        return clean_segments

    grp_min_r, grp_min_c, grp_max_r, grp_max_c = grp_props[0].bbox

    target_diakritik = {
        "NUN": {"jenis": "ATAS", "jumlah": 1},
        "FA": {"jenis": "ATAS", "jumlah": 1},
        "NGA": {"jenis": "ATAS", "jumlah": 3},
        "SYIN": {"jenis": "ATAS", "jumlah": 3},
        "TA": {"jenis": "ATAS", "jumlah": 2},
        "BA": {"jenis": "BAWAH", "jumlah": 1},
        "YA": {"jenis": "BAWAH", "jumlah": 2},
    }

    # ==========================================================
    # PASS 1: ALOKASI UTAMA
    # ==========================================================
    for dot_idx, dot_prop in enumerate(dots_props):
        # Kunci utama: dot harus benar-benar milik mask_group saat ini.
        if not dot_milik_region_group(dot_prop, labeled_dots, mask_group):
            continue

        info_dot = estimasi_jumlah_diakritik(dot_prop, dot_mask, scale)

        jenis_titik = info_dot["jenis"]
        estimasi_jumlah = info_dot["jumlah"]
        dy, dx = info_dot["centroid"]

        if not (
            grp_min_r - (40 * scale) <= dy <= grp_max_r + (80 * scale) and
            grp_min_c - (40 * scale) <= dx <= grp_max_c + (40 * scale)
        ):
            continue

        owner_idx, owner_score = cari_owner_diakritik_bbox(
            dot_prop,
            jenis_titik,
            clean_segments,
            scale
        )

        for cand_idx, seg_mask in enumerate(clean_segments):
            seg_crds = np.argwhere(seg_mask)

            if len(seg_crds) == 0:
                continue

            min_y_seg, min_x_seg = seg_crds.min(axis=0)
            max_y_seg, max_x_seg = seg_crds.max(axis=0)

            h_seg = max_y_seg - min_y_seg
            w_seg = max_x_seg - min_x_seg

            seg_cy, seg_cx = center_of_mass(seg_mask)

            if seg_cy is None or seg_cx is None:
                continue

            meta = segment_metadata[cand_idx]
            tipe_huruf = meta.get("tipe_huruf", "UNKNOWN")

            if huruf_tanpa_diakritik_strict(tipe_huruf):
                continue

            # Jika owner berbasis bbox sudah jelas, hanya owner itu yang boleh menerima dot.
            if owner_idx is not None and cand_idx != owner_idx:
                continue

            target_info = target_diakritik_dinamis(meta, target_diakritik)

            # Fallback khusus: jika dot adalah 3 titik atas dan owner-nya segmen ini,
            # izinkan target sebagai NGA meskipun metadata awal belum sempurna.
            if (
                target_info is None
                and owner_idx == cand_idx
                and jenis_titik == "ATAS"
                and estimasi_jumlah >= 3
            ):
                target_info = {"jenis": "ATAS", "jumlah": 3}

            if target_info is None:
                continue

            target_jenis = target_info["jenis"]
            target_jumlah = target_info["jumlah"]

            if (
                owner_idx == cand_idx
                and jenis_titik == "ATAS"
                and estimasi_jumlah >= 3
            ):
                target_jenis = "ATAS"
                target_jumlah = 3

            if jenis_titik != target_jenis:
                continue

            # Jangan biarkan NUN/FA mengambil blob besar,
            # kecuali bbox owner menunjukkan titik memang milik segmen ini.
            if tipe_huruf in ["NUN", "FA"] and estimasi_jumlah > 1:
                if owner_idx != cand_idx:
                    continue
                target_jumlah = max(target_jumlah, estimasi_jumlah)

            if jenis_titik == "ATAS":
                if dy >= seg_cy:
                    continue
                jarak_vertikal = seg_cy - dy
            else:
                if dy <= seg_cy:
                    continue
                jarak_vertikal = dy - seg_cy

            if jarak_vertikal < 2 * scale or jarak_vertikal > 110 * scale:
                continue

            if tipe_huruf == "NGA":
                tolerance_kiri = 90 * scale
                tolerance_kanan = 65 * scale

            elif tipe_huruf == "SYIN":
                tolerance_kiri = 65 * scale
                tolerance_kanan = 65 * scale

            elif tipe_huruf in ["NUN", "FA"]:
                tolerance_kiri = 40 * scale
                tolerance_kanan = 40 * scale

            elif tipe_huruf == "TA":
                tolerance_kiri = 45 * scale
                tolerance_kanan = 45 * scale

            elif tipe_huruf in ["BA", "YA"]:
                tolerance_kiri = 40 * scale
                tolerance_kanan = 40 * scale

            else:
                if target_jumlah >= 3:
                    tolerance_kiri = 70 * scale
                    tolerance_kanan = 70 * scale
                elif target_jumlah == 2:
                    tolerance_kiri = 55 * scale
                    tolerance_kanan = 55 * scale
                else:
                    tolerance_kiri = 45 * scale
                    tolerance_kanan = 45 * scale

            if not (min_x_seg - tolerance_kiri <= dx <= max_x_seg + tolerance_kanan):
                continue

            aspect_seg = h_seg / (w_seg + 1e-5)

            if aspect_seg > 2.4 and w_seg < 25 * scale:
                continue

            dot_min_r, dot_min_c, dot_max_r, dot_max_c = dot_prop.bbox

            overlap_x = overlap_1d(
                dot_min_c,
                dot_max_c,
                min_x_seg,
                max_x_seg
            )

            dot_width = dot_max_c - dot_min_c
            overlap_ratio_dot = overlap_x / (dot_width + 1e-5)

            horizontal_gap = max(
                0,
                min_x_seg - dot_max_c,
                dot_min_c - max_x_seg
            )

            dist_x = abs(dx - seg_cx)
            dist_y = abs(dy - seg_cy)
            dist_real = euclidean((dy, dx), (seg_cy, seg_cx))

            score = (
                horizontal_gap * 3.0
                + dist_x * 1.2
                + dist_y * 0.7
                - overlap_ratio_dot * 600
            )

            if min_x_seg <= dx <= max_x_seg:
                score -= 500

            if overlap_ratio_dot >= 0.50:
                score -= 700

            if dist_real < 70 * scale:
                score -= 250

            if dist_x > 75 * scale:
                score += 1000

            kandidat_pasangan.append({
                "dot_idx": dot_idx,
                "cand_idx": cand_idx,
                "score": score,
                "jenis": jenis_titik,
                "estimasi_jumlah": estimasi_jumlah,
                "target_jumlah": target_jumlah,
                "tipe_huruf": tipe_huruf,
                "prop": dot_prop,
                "owner_idx": owner_idx,
                "owner_score": owner_score
            })

            print(
                f"[OWNER DIAKRITIK] dot={dot_idx}, "
                f"jenis={jenis_titik}, "
                f"jumlah={estimasi_jumlah}, "
                f"owner_idx={owner_idx}, "
                f"cand_idx={cand_idx}, "
                f"tipe={tipe_huruf}, "
                f"dot_bbox={dot_prop.bbox}, "
                f"score={score:.2f}"
            )

    kandidat_pasangan.sort(key=lambda x: x["score"])

    jumlah_terpasang = {idx: 0 for idx in range(len(clean_segments))}

    for pas in kandidat_pasangan:
        dot_idx = pas["dot_idx"]
        cand_idx = pas["cand_idx"]

        if dot_idx in titik_terpakai:
            continue

        target_jumlah = pas["target_jumlah"]
        estimasi_jumlah = pas["estimasi_jumlah"]

        if jumlah_terpasang[cand_idx] >= target_jumlah:
            continue

        dot_indices = (labeled_dots == pas["prop"].label)

        clean_segments[cand_idx] = np.logical_or(
            clean_segments[cand_idx],
            dot_indices
        )

        titik_terpakai.add(dot_idx)
        jumlah_terpasang[cand_idx] += estimasi_jumlah

        print(
            f"[DIAKRITIK ALOKASI] dot={dot_idx}, "
            f"jenis={pas['jenis']}, "
            f"jumlah={estimasi_jumlah}, "
            f"ke_segmen={cand_idx}, "
            f"tipe={pas['tipe_huruf']}, "
            f"score={pas['score']:.2f}"
        )

    # ==========================================================
    # PASS 2: RESCUE TERKUNCI REGION
    # ==========================================================
    for cand_idx, seg_mask in enumerate(clean_segments):
        meta = segment_metadata[cand_idx]
        tipe_huruf = meta.get("tipe_huruf", "UNKNOWN")

        if huruf_tanpa_diakritik_strict(tipe_huruf):
            continue

        target_info = target_diakritik_dinamis(meta, target_diakritik)

        if target_info is None:
            continue

        target_jenis = target_info["jenis"]
        target_jumlah = target_info["jumlah"]

        if jumlah_terpasang[cand_idx] >= target_jumlah:
            continue

        seg_crds = np.argwhere(seg_mask)

        if len(seg_crds) == 0:
            continue

        min_y_seg, min_x_seg = seg_crds.min(axis=0)
        max_y_seg, max_x_seg = seg_crds.max(axis=0)
        seg_cy, seg_cx = center_of_mass(seg_mask)

        rescue_best = None
        rescue_score = float("inf")

        for dot_idx, dot_prop in enumerate(dots_props):
            if dot_idx in titik_terpakai:
                continue

            if not dot_milik_region_group(dot_prop, labeled_dots, mask_group):
                continue

            info_dot = estimasi_jumlah_diakritik(dot_prop, dot_mask, scale)

            if info_dot["jenis"] != target_jenis:
                continue

            dy, dx = info_dot["centroid"]

            owner_idx, owner_score = cari_owner_diakritik_bbox(
                dot_prop,
                info_dot["jenis"],
                clean_segments,
                scale
            )

            if owner_idx is not None and owner_idx != cand_idx:
                continue

            if tipe_huruf == "NGA":
                tol_kiri, tol_kanan = 110 * scale, 75 * scale

            elif tipe_huruf == "SYIN":
                tol_kiri, tol_kanan = 75 * scale, 75 * scale

            elif tipe_huruf in ["NUN", "FA"]:
                tol_kiri, tol_kanan = 45 * scale, 45 * scale

            elif tipe_huruf == "TA":
                tol_kiri, tol_kanan = 55 * scale, 55 * scale

            else:
                if target_jumlah >= 3:
                    tol_kiri, tol_kanan = 75 * scale, 75 * scale
                elif target_jumlah == 2:
                    tol_kiri, tol_kanan = 60 * scale, 60 * scale
                else:
                    tol_kiri, tol_kanan = 45 * scale, 45 * scale

            if not (min_x_seg - tol_kiri <= dx <= max_x_seg + tol_kanan):
                continue

            if target_jenis == "ATAS" and dy >= seg_cy:
                continue

            if target_jenis == "BAWAH" and dy <= seg_cy:
                continue

            dot_min_r, dot_min_c, dot_max_r, dot_max_c = dot_prop.bbox

            overlap_x = overlap_1d(
                dot_min_c,
                dot_max_c,
                min_x_seg,
                max_x_seg
            )

            dot_width = dot_max_c - dot_min_c
            overlap_ratio_dot = overlap_x / (dot_width + 1e-5)

            horizontal_gap = max(
                0,
                min_x_seg - dot_max_c,
                dot_min_c - max_x_seg
            )

            dist_x = abs(dx - seg_cx)
            dist_y = abs(dy - seg_cy)

            score = (
                horizontal_gap * 3.0
                + dist_x * 1.2
                + dist_y * 0.7
                - overlap_ratio_dot * 500
            )

            if min_x_seg <= dx <= max_x_seg:
                score -= 300

            if score < rescue_score:
                rescue_score = score
                rescue_best = (dot_idx, dot_prop, info_dot)

        if rescue_best is not None:
            dot_idx, dot_prop, info_dot = rescue_best
            dot_indices = (labeled_dots == dot_prop.label)

            clean_segments[cand_idx] = np.logical_or(
                clean_segments[cand_idx],
                dot_indices
            )

            titik_terpakai.add(dot_idx)
            jumlah_terpasang[cand_idx] += info_dot["jumlah"]

            print(
                f"[DIAKRITIK RESCUE] dot={dot_idx}, "
                f"jenis={info_dot['jenis']}, "
                f"jumlah={info_dot['jumlah']}, "
                f"ke_segmen={cand_idx}, "
                f"tipe={tipe_huruf}, "
                f"score={rescue_score:.2f}"
            )

    # ==========================================================
    # PASS 3: FALLBACK SPASIAL TERKUNCI REGION
    # ==========================================================
    for dot_idx, dot_prop in enumerate(dots_props):
        if dot_idx in titik_terpakai:
            continue

        if not dot_milik_region_group(dot_prop, labeled_dots, mask_group):
            continue

        info_dot = estimasi_jumlah_diakritik(dot_prop, dot_mask, scale)

        jenis_titik = info_dot["jenis"]
        jumlah_titik = info_dot["jumlah"]
        dy, dx = info_dot["centroid"]
        dot_min_r, dot_min_c, dot_max_r, dot_max_c = dot_prop.bbox

        best_idx = None
        best_score = float("inf")

        owner_idx, owner_score = cari_owner_diakritik_bbox(
            dot_prop,
            jenis_titik,
            clean_segments,
            scale
        )

        for cand_idx, seg_mask in enumerate(clean_segments):
            if owner_idx is not None and owner_idx != cand_idx:
                continue

            seg_crds = np.argwhere(seg_mask)

            if len(seg_crds) == 0:
                continue

            min_y_seg, min_x_seg = seg_crds.min(axis=0)
            max_y_seg, max_x_seg = seg_crds.max(axis=0)

            h_seg = max_y_seg - min_y_seg
            w_seg = max_x_seg - min_x_seg

            seg_cy, seg_cx = center_of_mass(seg_mask)

            if seg_cy is None or seg_cx is None:
                continue

            meta = segment_metadata[cand_idx]
            tipe_huruf = meta.get("tipe_huruf", "UNKNOWN")

            if huruf_tanpa_diakritik_strict(tipe_huruf):
                continue

            aspect_seg = h_seg / (w_seg + 1e-5)

            if aspect_seg > 2.5 and w_seg < 25 * scale:
                continue

            target_info = target_diakritik_dinamis(meta, target_diakritik)

            if target_info is not None:
                if target_info["jenis"] != jenis_titik:
                    continue

                if jumlah_terpasang[cand_idx] >= target_info["jumlah"]:
                    continue

            if jenis_titik == "ATAS":
                if dy > seg_cy + (5 * scale):
                    continue
                jarak_vertikal = abs(seg_cy - dy)
            else:
                if dy < seg_cy - (5 * scale):
                    continue
                jarak_vertikal = abs(dy - seg_cy)

            if jarak_vertikal < 2 * scale or jarak_vertikal > 120 * scale:
                continue

            overlap_x = overlap_1d(
                dot_min_c,
                dot_max_c,
                min_x_seg,
                max_x_seg
            )

            dot_width = dot_max_c - dot_min_c
            overlap_ratio_dot = overlap_x / (dot_width + 1e-5)

            horizontal_gap = max(
                0,
                min_x_seg - dot_max_c,
                dot_min_c - max_x_seg
            )

            if horizontal_gap > 70 * scale:
                continue

            dist_x_centroid = abs(dx - seg_cx)
            dist_y_centroid = abs(dy - seg_cy)

            score = (
                horizontal_gap * 3.0
                + dist_x_centroid * 0.9
                + dist_y_centroid * 0.6
                - overlap_ratio_dot * 600
            )

            if min_x_seg <= dx <= max_x_seg:
                score -= 400

            if overlap_ratio_dot >= 0.50:
                score -= 600

            if jenis_titik == "ATAS" and meta.get("jumlah_diakritik_atas", 0) > 0:
                score -= 250

            if jenis_titik == "BAWAH" and meta.get("jumlah_diakritik_bawah", 0) > 0:
                score -= 250

            if score < best_score:
                best_score = score
                best_idx = cand_idx

        if best_idx is not None:
            dot_indices = (labeled_dots == dot_prop.label)

            clean_segments[best_idx] = np.logical_or(
                clean_segments[best_idx],
                dot_indices
            )

            titik_terpakai.add(dot_idx)
            jumlah_terpasang[best_idx] += jumlah_titik

            print(
                f"[DIAKRITIK FALLBACK SPASIAL] dot={dot_idx}, "
                f"jenis={jenis_titik}, "
                f"jumlah={jumlah_titik}, "
                f"ke_segmen={best_idx}, "
                f"tipe={segment_metadata[best_idx].get('tipe_huruf', 'UNKNOWN')}, "
                f"score={best_score:.2f}"
            )

    return clean_segments



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

def cek_lam(h_seg, w_seg, scale):
    return h_seg >= 45 * scale and h_seg > w_seg * 1.2

def cek_alif(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if posisi_titik != "TIDAK_ADA": return False
    if h_seg >= 30 * scale and w_seg <= 25 * scale:
        return True
    return False

def cek_ra(seg_mask, h_seg, w_seg, scale):
    if h_seg >= 55 * scale: return False
    if w_seg < 8 * scale or w_seg > 45 * scale: return False 
    skel = skeletonize(seg_mask)
    _, intersections, turns = find_endpoints(skel)
    if len(intersections) > 1: return False
    if len(turns) > 1: return False 
    endpoints, _, _ = find_endpoints(skel)
    if len(endpoints) >= 2:
        ep_sorted = sorted(endpoints, key=lambda p: p[0])
        top, bottom = ep_sorted[0], ep_sorted[-1]
        if top[1] > bottom[1] - (10 * scale): 
            coords = np.argwhere(seg_mask)
            lowest_y = np.max(coords[:, 0]) 
            if abs(bottom[0] - lowest_y) <= (15 * scale): return True
    return False

def cek_sin(seg_mask, h_seg, scale):
    if h_seg >= 45 * scale: return False
    gigi_atas = analisis_ujung_atas(seg_mask)
    return gigi_atas >= 3

def cek_nun(seg_mask, h_seg, w_seg, posisi_titik, scale):
    # NUN tidak boleh terlalu tinggi
    if h_seg >= 80 * scale:
        return False

    # NUN harus cukup lebar, tetapi bukan baseline yang terlalu panjang
    if w_seg > 75 * scale:
        return False

    # NUN ideal: mangkuk cukup lebar dan memiliki titik atas
    if posisi_titik.startswith("ATAS"):
        if w_seg > h_seg * 0.80 and w_seg > 20 * scale:
            return True

    # Fallback bentuk mangkuk tanpa terlalu bergantung pada titik
    if w_seg >= 25 * scale:
        gigi_atas = analisis_ujung_atas(seg_mask)
        if gigi_atas <= 2 and posisi_titik.startswith("ATAS"):
            return True

    return False

def cek_ha(seg_mask, h_seg, w_seg, scale):
    if h_seg >= 45 * scale: return False
    skel = skeletonize(seg_mask)
    endpoints, _, _ = find_endpoints(skel)
    if len(endpoints) <= 1: return True
    return False

def cek_waw(seg_mask, h_seg, w_seg, scale):
    if h_seg >= 60 * scale: return False
    if w_seg > 60 * scale: return False
    skel = skeletonize(seg_mask)
    endpoints, _, _ = find_endpoints(skel)
    if 1 <= len(endpoints) <= 2:
        coords = np.argwhere(seg_mask)
        top_y = np.min(coords[:, 0])
        if h_seg < 45 * scale and w_seg < 45 * scale: return True
    return False

def cek_ba(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if h_seg >= 60 * scale: return False
    skel = skeletonize(seg_mask)
    _, intersections, _ = find_endpoints(skel)
    if len(intersections) > 0: return False
    if posisi_titik == "BAWAH_SINGLE": return True
    return False

def cek_ya(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if h_seg >= 60 * scale: return False
    skel = skeletonize(seg_mask)
    _, intersections, _ = find_endpoints(skel)
    if len(intersections) > 0 and h_seg < 40 * scale: return True
    if posisi_titik == "BAWAH_DOUBLE": return True
    if posisi_titik == "BAWAH_SINGLE": return False
    if w_seg < 35 * scale and h_seg < 35 * scale: return True  
    gigi_atas = analisis_ujung_atas(seg_mask)
    return gigi_atas <= 1

def cek_dal(seg_mask, h_seg, w_seg, scale):
    if h_seg > 45 * scale or w_seg > 35 * scale: return False
    skel = skeletonize(seg_mask)
    _, intersections, _ = find_endpoints(skel)
    if len(intersections) > 0: return False
    return True

def cek_ain(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if h_seg >= 55 * scale: return False
    if posisi_titik.startswith("ATAS"): return True
    skel = skeletonize(seg_mask)
    _, intersections, _ = find_endpoints(skel)
    if len(intersections) > 0 and w_seg < 40 * scale: return True
    return False

def cek_fa(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if h_seg >= 60 * scale: return False
    
    # Syarat utama: Harus memiliki diakritik tunggal di atas
    if posisi_titik != "ATAS_SINGLE": return False
    
    skel = skeletonize(seg_mask)
    # Validasi penanda fitur struktural dari skeleton to graph
    fitur_struktural = deteksi_loop(skel) 
    
    if len(fitur_struktural) > 0: 
        return True
        
    return False

# ==================== PEMOTONGAN & MAGNET ====================

def hitung_garis_potong_sinkron(component_features, labeled_letters, skeleton_used, scale):
    all_cols = []
    
    for label, feat in component_features.items():
        mask = (labeled_letters == label)
        if np.sum(mask) < 5: continue
        
        coords = np.argwhere(mask)
        min_y_grp, min_x_grp = coords.min(axis=0)
        max_y_grp, max_x_grp = coords.max(axis=0)
        h_grp = max_y_grp - min_y_grp
        w_grp = max_x_grp - min_x_grp
        
        if cek_waw(mask, h_grp, w_grp, scale) or cek_ha(mask, h_grp, w_grp, scale):
            continue 

        letter_skeleton = skeleton_used * mask
        intersections = sorted([ix for iy, ix in feat["intersections"]])
        if not intersections: continue
            
        intersections_map = {}
        for iy, ix in feat["intersections"]:
            if ix not in intersections_map: intersections_map[ix] = []
            intersections_map[ix].append(iy)

        # --- PENAMBAHAN: Deteksi Batas Fitur Struktural (Skeleton to Graph) ---
        cycle_bounds = []
        if "cycles" in feat:
            for cycle in feat["cycles"]:
                cycle_arr = np.array(cycle)
                min_x_cy = cycle_arr[:, 1].min()
                max_x_cy = cycle_arr[:, 1].max()
                cycle_bounds.append((min_x_cy, max_x_cy))
        # ---------------------------------------------------------------------

        first_ix = intersections[0]
        first_cut = first_ix + max(1, int(2 * scale))
        
        # Cek dan geser jika first_cut menabrak area struktural
        for (min_x_cy, max_x_cy) in cycle_bounds:
            if min_x_cy - (5 * scale) <= first_cut <= max_x_cy + (5 * scale):
                first_cut = max_x_cy + int(8 * scale)

        if first_cut - min_x_grp >= 15 * scale and max_x_grp - first_cut >= 15 * scale:
            all_cols.append(first_cut)

        for i in range(len(intersections) - 1):
            curr_x = intersections[i]
            next_x = intersections[i+1]
            gap = next_x - curr_x
            
            is_curr_tiang = False
            ys = intersections_map.get(curr_x, [])
            for y_check in ys:
                if cektiang(letter_skeleton, (y_check, curr_x), scale): 
                    is_curr_tiang = True
                    break
            
            cut_pos = None
            if is_curr_tiang:
                cut_pos = (curr_x + next_x) // 2
            elif gap >= 25 * scale:
                cut_pos = curr_x + max(1, int(2 * scale))
                
            if cut_pos is not None:
                # --- KOREKSI: Geser garis partisi ke kanan fitur struktural ---
                for (min_x_cy, max_x_cy) in cycle_bounds:
                    if min_x_cy - (5 * scale) <= cut_pos <= max_x_cy + (5 * scale):
                        cut_pos = max_x_cy + int(8 * scale) 
                # -------------------------------------------------------------
                
                if cut_pos - min_x_grp >= 15 * scale and max_x_grp - cut_pos >= 15 * scale:
                    all_cols.append(cut_pos) 

    all_cols = sorted(set(all_cols))
    merged_cols = []
    if all_cols:
        merged_cols = [all_cols[0]]
        for col in all_cols[1:]:
            if col - merged_cols[-1] < 15 * scale:
                merged_cols[-1] = (merged_cols[-1] + col) // 2
            else:
                merged_cols.append(col)

    final_cols_vis = [0] + merged_cols + [skeleton_used.shape[1] - 1]
    return final_cols_vis

def potong_persimpangan(
    labeled_letters,
    skeleton_used,
    dot_mask,
    scale,
    debug_freeman=None,
    target_freeman_xrange=None
):
    hasil_potongan = []

    num_features = np.max(labeled_letters)

    labeled_dots, num_dots = measure.label(
        dot_mask > 0,
        return_num=True,
        connectivity=2
    )
    dots_props = list(measure.regionprops(labeled_dots))

    for region_label in range(1, num_features + 1):
        mask_group = (labeled_letters == region_label)

        if np.sum(mask_group) < 5:
            continue

        coords = np.argwhere(mask_group)

        if len(coords) == 0:
            continue

        min_y_grp, min_x_grp = coords.min(axis=0)
        max_y_grp, max_x_grp = coords.max(axis=0)

        h_grp = max_y_grp - min_y_grp
        w_grp = max_x_grp - min_x_grp

        x_coords = coords[:, 1]

        letter_skeleton = skeleton_used * mask_group
        _, raw_intersections, _ = find_endpoints(letter_skeleton)

        fitur_kumpulan = deteksi_loop(letter_skeleton)

        cycle_bounds = []
        for cycle, _ in fitur_kumpulan:
            cycle_arr = np.array(cycle)
            cycle_bounds.append(
                (cycle_arr[:, 1].min(), cycle_arr[:, 1].max())
            )

        cut_points = []

        # ==========================================================
        # 1. PEMOTONGAN UTAMA BERDASARKAN INTERSECTION / TOPOLOGI GRAF
        # ==========================================================
        if cek_waw(mask_group, h_grp, w_grp, scale) or cek_ha(mask_group, h_grp, w_grp, scale):
            pass
        else:
            intersections = sorted([ix for iy, ix in raw_intersections])

            if intersections:
                intersections_map = {}

                for iy, ix in raw_intersections:
                    if ix not in intersections_map:
                        intersections_map[ix] = []
                    intersections_map[ix].append(iy)

                first_ix = intersections[0]
                first_cut = first_ix + max(1, int(2 * scale))

                for min_x_cy, max_x_cy in cycle_bounds:
                    if min_x_cy - (5 * scale) <= first_cut <= max_x_cy + (5 * scale):
                        first_cut = max_x_cy + int(8 * scale)

                if first_cut - min_x_grp >= 15 * scale and max_x_grp - first_cut >= 15 * scale:
                    cut_points.append(first_cut)

                for i in range(len(intersections) - 1):
                    curr_x = intersections[i]
                    next_x = intersections[i + 1]
                    gap = next_x - curr_x

                    is_curr_tiang = False
                    ys = intersections_map.get(curr_x, [])

                    for y_check in ys:
                        if cektiang(letter_skeleton, (y_check, curr_x), scale):
                            is_curr_tiang = True
                            break

                    cut_pos = None

                    if is_curr_tiang:
                        cut_pos = (curr_x + next_x) // 2
                    elif gap >= 25 * scale:
                        cut_pos = curr_x + max(1, int(2 * scale))

                    if cut_pos is not None:
                        for min_x_cy, max_x_cy in cycle_bounds:
                            if min_x_cy - (5 * scale) <= cut_pos <= max_x_cy + (5 * scale):
                                cut_pos = max_x_cy + int(8 * scale)

                        if cut_pos - min_x_grp >= 15 * scale and max_x_grp - cut_pos >= 15 * scale:
                            cut_points.append(cut_pos)

        # ==========================================================
        # 2. PEMOTONGAN TAMBAHAN MENGGUNAKAN FREEMAN CHAIN CODE
        # ==========================================================
        freeman_cut_col = None

        if target_freeman_xrange is not None:
            target_x_min, target_x_max = target_freeman_xrange

            is_target_freeman = not (
                max_x_grp < target_x_min or min_x_grp > target_x_max
            )

            if is_target_freeman:
                cut_freeman, freeman_info = deteksi_potong_freeman(
                    mask_group,
                    skeleton_used,
                    dot_mask,
                    scale,
                    target_xrange=target_freeman_xrange
                )

                if cut_freeman is not None:
                    freeman_cut_col = int(cut_freeman)

                    if min_x_grp < freeman_cut_col < max_x_grp:
                        cut_points.append(freeman_cut_col)

                        if debug_freeman is not None and freeman_info is not None:
                            freeman_info["region_label"] = region_label
                            freeman_info["cut_col"] = freeman_cut_col
                            freeman_info["target_xrange"] = target_freeman_xrange
                            debug_freeman.append(freeman_info)

                        print(
                            f"[FREEMAN AKTIF] Region {region_label} | "
                            f"cut_col = {freeman_cut_col}"
                        )
                    else:
                        print(
                            f"[FREEMAN DITOLAK] Region {region_label} | "
                            f"cut_col di luar bbox"
                        )
                else:
                    print(
                        f"[FREEMAN GAGAL] Region {region_label} masuk target, "
                        f"tetapi cut_col = None"
                    )

        # ==========================================================
        # 3. PEMBENTUKAN BATAS POTONG AKTUAL
        # ==========================================================
        cuts = []

        if cut_points:
            minc, maxc = x_coords.min(), x_coords.max()

            valid_cuts = sorted([
                int(np.clip(c, minc, maxc))
                for c in cut_points
            ])

            final_valid_cuts = []

            for c in valid_cuts:
                c = int(c)

                if final_valid_cuts and c - final_valid_cuts[-1] < 8 * scale:
                    if freeman_cut_col is not None and abs(c - freeman_cut_col) <= 2:
                        final_valid_cuts[-1] = c
                    else:
                        continue
                else:
                    final_valid_cuts.append(c)

            cuts = sorted(set([minc] + final_valid_cuts + [maxc + 1]))

            print(f"[REGION {region_label}] cut_points =", cut_points)
            print(f"[REGION {region_label}] cuts used  =", cuts)

        else:
            cuts = [x_coords.min(), x_coords.max() + 1]

        # ==========================================================
        # 4. PEMBENTUKAN SEGMEN HURUF TANPA DIAKRITIK
        # ==========================================================
        clean_segments = []
        segment_metadata = []

        for i in range(len(cuts) - 1):
            start_col, end_col = cuts[i], cuts[i + 1]
            segment_width = end_col - start_col

            segment_mask = np.zeros_like(mask_group)
            segment_mask[:, start_col:end_col] = mask_group[:, start_col:end_col]

            # Diakritik dihapus sementara dari badan huruf.
            segment_mask = segment_mask & (dot_mask == 0)

            if np.sum(segment_mask) > 30 * (scale ** 2) and segment_width > 5 * scale:
                seg_crds = np.argwhere(segment_mask)

                if len(seg_crds) > 0:
                    min_y_seg, min_x_seg = seg_crds.min(axis=0)
                    max_y_seg, max_x_seg = seg_crds.max(axis=0)

                    h_seg = max_y_seg - min_y_seg
                    w_seg = max_x_seg - min_x_seg

                    posisi_titik = dominasi_diakritik(
                        min_x_seg,
                        max_x_seg,
                        dot_mask,
                        scale
                    )
                else:
                    min_y_seg = 0
                    min_x_seg = 0
                    max_y_seg = 0
                    max_x_seg = 0
                    h_seg = 999
                    w_seg = 999
                    posisi_titik = "TIDAK_ADA"

                skel_seg = skeletonize(segment_mask)

                # ==================================================
                # PROFIL DIAKRITIK SEGMEN
                # Fungsi ini membuat estimasi titik atas/bawah.
                # Jika tinta titik menyatu, blob lebar bisa dihitung
                # sebagai 2 atau 3 titik.
                # ==================================================
                profil_diakritik = buat_profil_diakritik_segmen(
                    segment_mask,
                    dots_props,
                    dot_mask,
                    scale
                )

                jumlah_diakritik_atas_seg = profil_diakritik["atas_estimasi"]
                jumlah_diakritik_bawah_seg = profil_diakritik["bawah_estimasi"]

                fitur_segmen = deteksi_loop(skel_seg)
                has_fitur = len(fitur_segmen) > 0

                fitur_cy, fitur_cx = None, None

                if has_fitur:
                    cycle_arr = np.array(fitur_segmen[0][0])
                    fitur_cy = np.mean(cycle_arr[:, 0])
                    fitur_cx = np.mean(cycle_arr[:, 1])

                # ==================================================
                # KLASIFIKASI HEURISTIK HURUF
                # ==================================================
                tipe_huruf = "UNKNOWN"

                bentuk_sangat_vertikal = (
                    h_seg > w_seg * 2.2 and w_seg < 25 * scale
                )

                bentuk_baseline_panjang = (
                    w_seg > 75 * scale and h_seg < 45 * scale and not has_fitur
                )

                # ==================================================
                # URUTAN BARU:
                # 1. Bentuk penolak yang sangat jelas dicek lebih awal.
                # 2. Huruf bertitik dicek sebelum WAW/HA/AIN umum.
                # 3. WAW dipindahkan ke bawah agar tidak mengambil NUN/FA.
                # ==================================================
                if bentuk_sangat_vertikal:
                    tipe_huruf = "PENOLAK_DIAKRITIK"

                elif cek_lam(h_seg, w_seg, scale):
                    tipe_huruf = "LAM"

                elif cek_alif(segment_mask, h_seg, w_seg, posisi_titik, scale):
                    tipe_huruf = "ALIF"

                elif cek_dal(segment_mask, h_seg, w_seg, scale):
                    tipe_huruf = "DAL"

                elif cek_ra(segment_mask, h_seg, w_seg, scale):
                    tipe_huruf = "RA"

                # Baseline sangat panjang tanpa fitur dan hampir tanpa titik
                # sebaiknya tidak menerima diakritik.
                elif bentuk_baseline_panjang and jumlah_diakritik_atas_seg <= 1:
                    tipe_huruf = "PENOLAK_DIAKRITIK"

                # ==================================================
                # HURUF BERTITIK ATAS
                # ==================================================

                # SYIN: bentuk dasar SIN + estimasi 3 titik atas.
                elif cek_sin(segment_mask, h_seg, scale) and jumlah_diakritik_atas_seg >= 3:
                    tipe_huruf = "SYIN"

                # NGA: biasanya memiliki 3 titik atas dan struktur mirip AIN.
                elif jumlah_diakritik_atas_seg >= 3 and cek_ain(
                    segment_mask,
                    h_seg,
                    w_seg,
                    posisi_titik,
                    scale
                ):
                    tipe_huruf = "NGA"
                
                elif jumlah_diakritik_atas_seg >= 2 and cek_ain(
                    segment_mask,
                    h_seg,
                    w_seg,
                    posisi_titik,
                    scale
                ):
                    # Fallback untuk kasus NGA yang titiknya menyatu
                    # sehingga estimasinya belum selalu mencapai 3.
                    tipe_huruf = "NGA"

                # NUN: satu titik atas.
                elif cek_nun(
                    segment_mask,
                    h_seg,
                    w_seg,
                    posisi_titik,
                    scale
                ) and jumlah_diakritik_atas_seg == 1:
                    tipe_huruf = "NUN"

                # TA: dua titik atas.
                # Ini tambahan agar huruf dua titik atas tidak menjadi UNKNOWN.
                elif jumlah_diakritik_atas_seg == 2:
                    tipe_huruf = "TA"

                # FA: satu titik atas + loop.
                elif cek_fa(segment_mask, h_seg, w_seg, posisi_titik, scale):
                    tipe_huruf = "FA"

                # ==================================================
                # HURUF BERTITIK BAWAH
                # ==================================================

                elif cek_ba(segment_mask, h_seg, w_seg, posisi_titik, scale):
                    tipe_huruf = "BA"

                elif cek_ya(segment_mask, h_seg, w_seg, posisi_titik, scale):
                    tipe_huruf = "YA"

                # ==================================================
                # HURUF TANPA TITIK / UMUM
                # Dicek setelah huruf bertitik agar tidak merebut diakritik.
                # ==================================================

                elif cek_sin(segment_mask, h_seg, scale):
                    tipe_huruf = "SIN"

                elif cek_ha(segment_mask, h_seg, w_seg, scale):
                    tipe_huruf = "HA"

                elif cek_ain(segment_mask, h_seg, w_seg, posisi_titik, scale):
                    tipe_huruf = "AIN"

                elif cek_waw(segment_mask, h_seg, w_seg, scale):
                    tipe_huruf = "WAW"

                clean_segments.append(segment_mask)

                segment_metadata.append({
                    "tipe_huruf": tipe_huruf,
                    "has_fitur": has_fitur,
                    "fitur_cy": fitur_cy,
                    "fitur_cx": fitur_cx,
                    "jumlah_diakritik_atas": jumlah_diakritik_atas_seg,
                    "jumlah_diakritik_bawah": jumlah_diakritik_bawah_seg,
                    "profil_diakritik": profil_diakritik
                })

                print(
                    f"[SEGMENT DEBUG] region={region_label}, "
                    f"tipe={tipe_huruf}, "
                    f"atas={jumlah_diakritik_atas_seg}, "
                    f"bawah={jumlah_diakritik_bawah_seg}, "
                    f"x=({min_x_seg},{max_x_seg})"
                )

        if not clean_segments:
            hasil_potongan.append(mask_group)
            continue

        # ==========================================================
        # 5. ALOKASI DIAKRITIK MENGGUNAKAN FUNGSI BARU
        # ==========================================================
        clean_segments = alokasi_diakritik_ke_segmen(
            clean_segments,
            segment_metadata,
            dots_props,
            labeled_dots,
            dot_mask,
            scale,
            mask_group
        )

        hasil_potongan.extend(clean_segments)

    return hasil_potongan

def overlap_1d(a_min, a_max, b_min, b_max):
    """
    Menghitung panjang overlap 1D antara dua rentang.
    """
    return max(0, min(a_max, b_max) - max(a_min, b_min))


def pasang_diakritik_khusus_h2_h8(sorted_hasil_potongan, dot_mask, scale):
    """
    Post-fix khusus untuk H2 dan H8.

    Tujuan:
    - Mempertahankan hasil huruf lain yang sudah benar.
    - Memasang kembali diakritik asli ke H2 dan H8.
    - Tidak membuat titik sintetis.
    - Jika titik sudah terlanjur masuk ke segmen lain, titik asli dipindahkan
      ke target yang benar.

    Target:
    - H2 = NGA, butuh estimasi 3 titik atas.
    - H8 = TA, butuh estimasi 2 titik atas.
    """

    fixed_segments = [seg.copy() for seg in sorted_hasil_potongan]

    labeled_dots_atas, num_dots_atas = measure.label(
        dot_mask == 1,
        return_num=True,
        connectivity=2
    )

    dots_props_atas = list(measure.regionprops(labeled_dots_atas))

    # Target berdasarkan urutan hasil kanan-ke-kiri
    target_specs = {
        2: {
            "nama": "NGA",
            "jumlah_target": 3,
            "tol_x": 45 * scale,
            "max_vertical": 140 * scale
        },
        8: {
            "nama": "TA",
            "jumlah_target": 2,
            "tol_x": 45 * scale,
            "max_vertical": 140 * scale
        }
    }

    for nomor_huruf, spec in target_specs.items():
        idx_target = nomor_huruf - 1

        if idx_target < 0 or idx_target >= len(fixed_segments):
            continue

        target_mask = fixed_segments[idx_target]
        badan_target = target_mask & (dot_mask == 0)

        coords = np.argwhere(badan_target)

        if len(coords) == 0:
            continue

        min_y_seg, min_x_seg = coords.min(axis=0)
        max_y_seg, max_x_seg = coords.max(axis=0)

        h_seg = max_y_seg - min_y_seg
        w_seg = max_x_seg - min_x_seg

        seg_cy, seg_cx = center_of_mass(badan_target)

        if seg_cy is None or seg_cx is None:
            continue

        kandidat_dot = []

        for dot_prop in dots_props_atas:
            info_dot = estimasi_jumlah_diakritik(dot_prop, dot_mask, scale)

            if info_dot["jenis"] != "ATAS":
                continue

            jumlah_estimasi = info_dot["jumlah"]
            dy, dx = info_dot["centroid"]

            dot_min_r, dot_min_c, dot_max_r, dot_max_c = dot_prop.bbox
            dot_width = dot_max_c - dot_min_c

            # Titik atas harus berada di atas pusat badan huruf.
            if dy >= seg_cy:
                continue

            jarak_vertikal = seg_cy - dy

            if jarak_vertikal < 2 * scale:
                continue

            if jarak_vertikal > spec["max_vertical"]:
                continue

            # Cek apakah bbox titik berada di sekitar bbox badan huruf target.
            if dot_max_c < min_x_seg - spec["tol_x"]:
                continue

            if dot_min_c > max_x_seg + spec["tol_x"]:
                continue

            overlap_x = overlap_1d(
                dot_min_c,
                dot_max_c,
                min_x_seg,
                max_x_seg
            )

            overlap_ratio = overlap_x / (dot_width + 1e-5)

            horizontal_gap = max(
                0,
                min_x_seg - dot_max_c,
                dot_min_c - max_x_seg
            )

            dist_x = abs(dx - seg_cx)

            # Skor semakin kecil semakin baik.
            score = (
                horizontal_gap * 5.0
                + dist_x * 1.2
                + jarak_vertikal * 0.25
                - overlap_ratio * 800
            )

            # Bonus besar jika centroid titik berada di atas rentang badan.
            if min_x_seg <= dx <= max_x_seg:
                score -= 700

            # Bonus jika bbox titik overlap dengan bbox badan.
            if overlap_ratio >= 0.4:
                score -= 500

            kandidat_dot.append({
                "prop": dot_prop,
                "info": info_dot,
                "score": score,
                "jumlah_estimasi": jumlah_estimasi,
                "bbox": dot_prop.bbox,
                "centroid": dot_prop.centroid,
                "overlap_ratio": overlap_ratio,
                "horizontal_gap": horizontal_gap
            })

        kandidat_dot.sort(key=lambda item: item["score"])

        jumlah_terpasang = 0
        dot_terpilih = []

        for cand in kandidat_dot:
            if jumlah_terpasang >= spec["jumlah_target"]:
                break

            dot_terpilih.append(cand)
            jumlah_terpasang += cand["jumlah_estimasi"]

        if not dot_terpilih:
            print(
                f"[POST-FIX DIAKRITIK] H{nomor_huruf} ({spec['nama']}): "
                f"tidak menemukan kandidat titik."
            )
            continue

        for cand in dot_terpilih:
            dot_label = cand["prop"].label
            dot_pixels = (labeled_dots_atas == dot_label)

            # Hapus titik ini dari semua segmen lain agar tidak dobel.
            for i in range(len(fixed_segments)):
                fixed_segments[i][dot_pixels] = False

            # Pasang titik asli ke target.
            fixed_segments[idx_target] = np.logical_or(
                fixed_segments[idx_target],
                dot_pixels
            )

            print(
                f"[POST-FIX DIAKRITIK] H{nomor_huruf} ({spec['nama']}): "
                f"pasang dot asli, "
                f"jumlah_estimasi={cand['jumlah_estimasi']}, "
                f"bbox={cand['bbox']}, "
                f"centroid={cand['centroid']}, "
                f"score={cand['score']:.2f}, "
                f"overlap={cand['overlap_ratio']:.2f}"
            )

        print(
            f"[POST-FIX DIAKRITIK] H{nomor_huruf} ({spec['nama']}): "
            f"total_estimasi_terpasang={jumlah_terpasang}"
        )

    return fixed_segments

def hitung_panjang_kurva(curve):
    """
    Menghitung panjang kurva B-Spline berdasarkan jarak antar titik kurva.
    """
    if curve is None or len(curve) < 2:
        return 0.0

    total = 0.0
    for i in range(len(curve) - 1):
        total += euclidean(curve[i], curve[i + 1])

    return float(total)


def hitung_smoothness_kurva(curve):
    """
    Menghitung tingkat kelengkungan/smoothness kurva.
    Nilai kecil berarti kurva lebih halus.
    Nilai besar berarti banyak perubahan arah tajam.
    """
    if curve is None or len(curve) < 3:
        return 0.0

    perubahan_sudut = []

    for i in range(1, len(curve) - 1):
        p_prev = curve[i - 1]
        p_curr = curve[i]
        p_next = curve[i + 1]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        perubahan_sudut.append(theta)

    if len(perubahan_sudut) == 0:
        return 0.0

    return float(np.mean(perubahan_sudut))


def fitting_bspline_rasm_per_huruf(
    sorted_hasil_potongan,
    dot_mask,
    scale,
    output_folder="hasil_curve_fitting_rasm_cartesian"
):
    """
    Curve fitting B-Spline khusus rasm, 
    Fokus:
    - Komputasi hanya pada rasm (tanpa diakritik)
    - Visualisasi lebih jelas 
    """

    os.makedirs(output_folder, exist_ok=True)

    hasil_curve = []

    for idx, region_mask in enumerate(sorted_hasil_potongan):
        nomor_huruf = idx + 1

        # ======================================================
        # Fokus ke rasm saja
        # ======================================================
        rasm_mask = region_mask & (dot_mask == 0)

        if np.sum(rasm_mask) == 0:
            continue

        skel = skeletonize(rasm_mask)
        points = np.argwhere(skel)

        if len(points) < 2:
            continue

        tsp_segments = tsp_skeleton_traversal(
            points,
            dist_threshold=max(3, int(5 * scale))
        )

        # ======================================================
        # Figure gaya Generated Huruf
        # ======================================================
        fig, ax = plt.subplots(figsize=(7, 6))
        has_curve = False

        total_skeleton_points = 0
        total_control_points = 0
        total_sse = 0.0
        total_abs_error = 0.0
        total_error_points = 0
        max_error = 0.0
        total_tsp_distance = 0.0
        total_curve_length = 0.0
        smoothness_values = []

        all_x = []
        all_y = []

        for seg_idx, segment in enumerate(tsp_segments):
            segment_array = np.array(segment)

            if len(segment_array) < 2:
                continue

            n_len = len(segment_array)
            total_skeleton_points += n_len

            # Jarak TSP
            for i in range(n_len - 1):
                total_tsp_distance += euclidean(
                    segment_array[i],
                    segment_array[i + 1]
                )

            # Karena ini rasm, gunakan cubic spline jika cukup titik
            degree = 3
            if n_len < degree + 1:
                degree = 1

            if degree == 3:
                jumlah_cp = min(n_len, max(4, n_len // 5))
            else:
                jumlah_cp = min(n_len, max(2, n_len // 2))

            if jumlah_cp < degree + 1:
                continue

            indeks_pilihan = np.unique(
                np.linspace(0, n_len - 1, jumlah_cp, dtype=int)
            )

            jumlah_cp = len(indeks_pilihan)

            if jumlah_cp < degree + 1:
                continue

            titik_kontrol = segment_array[indeks_pilihan]

            kurva_halus = bspline_curve(
                titik_kontrol,
                degree=degree,
                num_points=max(100, n_len * 3)
            )

            # ==========================
            # Evaluasi error
            # ==========================
            dists = cdist(segment_array, kurva_halus)
            min_dists = np.min(dists, axis=1)

            total_sse += np.sum(min_dists ** 2)
            total_abs_error += np.sum(min_dists)
            total_error_points += len(min_dists)

            if len(min_dists) > 0:
                max_error = max(max_error, float(np.max(min_dists)))

            curve_length = hitung_panjang_kurva(kurva_halus)
            smoothness = hitung_smoothness_kurva(kurva_halus)

            total_curve_length += curve_length
            smoothness_values.append(smoothness)

            total_control_points += jumlah_cp
            has_curve = True

            # ==========================
            # Koordinat untuk plot
            # ==========================
            x_skel = segment_array[:, 1]
            y_skel = segment_array[:, 0]

            x_ctrl = titik_kontrol[:, 1]
            y_ctrl = titik_kontrol[:, 0]

            x_curve = kurva_halus[:, 1]
            y_curve = kurva_halus[:, 0]

            all_x.extend(x_skel.tolist())
            all_x.extend(x_ctrl.tolist())
            all_x.extend(x_curve.tolist())

            all_y.extend(y_skel.tolist())
            all_y.extend(y_ctrl.tolist())
            all_y.extend(y_curve.tolist())

            # ==========================
            # Plot seperti Generated Huruf
            # ==========================
            label_skel = 'Skeleton/TSP' if seg_idx == 0 else ""
            label_poly = 'Control Polygon' if seg_idx == 0 else ""
            label_cp = 'Control Points' if seg_idx == 0 else ""
            label_curve = 'Generated Curve' if seg_idx == 0 else ""

            ax.plot(
                x_skel, y_skel,
                color='lightgray',
                linewidth=1.0,
                zorder=1,
                label=label_skel
            )

            ax.plot(
                x_ctrl, y_ctrl,
                color='red',
                linewidth=1.2,
                zorder=2,
                label=label_poly
            )

            ax.scatter(
                x_ctrl, y_ctrl,
                color='darkblue',
                s=55,
                zorder=4,
                label=label_cp
            )

            ax.plot(
                x_curve, y_curve,
                color='gray',
                linewidth=2.0,
                zorder=3,
                label=label_curve
            )

        if not has_curve or total_error_points == 0:
            plt.close(fig)
            continue

        rmse = np.sqrt(total_sse / total_error_points)
        mae = total_abs_error / total_error_points

        compression = (
            (total_skeleton_points - total_control_points)
            / total_skeleton_points
        ) * 100 if total_skeleton_points > 0 else 0

        smoothness_mean = (
            float(np.mean(smoothness_values))
            if len(smoothness_values) > 0 else 0.0
        )

        hasil_curve.append({
            "huruf": nomor_huruf,
            "skeleton_points": int(total_skeleton_points),
            "control_points": int(total_control_points),
            "compression_percent": float(compression),
            "tsp_distance": float(total_tsp_distance),
            "curve_length": float(total_curve_length),
            "rmse": float(rmse),
            "mae": float(mae),
            "max_error": float(max_error),
            "smoothness": float(smoothness_mean)
        })

        # ======================================================
        # Style axis seperti Generated Huruf
        # ======================================================
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='datalim')

        if len(all_x) > 0 and len(all_y) > 0:
            margin_x = 3
            margin_y = 3
            ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
            ax.set_ylim(max(all_y) + margin_y, min(all_y) - margin_y)

        judul = f'(Generated Rasm ke-{nomor_huruf})\n'
        judul += f'Total CP: {total_control_points} | Jarak TSP: {total_tsp_distance:.2f} px\n'
        judul += f'Kompresi Data: {compression:.1f}% | RMSE: {rmse:.2f} px | MAE: {mae:.2f} px'
        ax.set_title(judul, fontsize=11, pad=15)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)

        plt.tight_layout()

        output_path = os.path.join(
            output_folder,
            f"generated_rasm_{nomor_huruf}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    return hasil_curve

# ==================== MAIN EXECUTION ====================

def evaluasi_pemotongan_rasm(sorted_hasil_potongan, dot_mask, jumlah_gt_rasm):
    """
    Evaluasi khusus pemotongan rasm.

    Evaluasi ini tidak memperhitungkan:
    - jenis huruf,
    - jumlah diakritik,
    - posisi diakritik.

    Yang dihitung hanya:
    - jumlah potongan rasm,
    - potongan yang benar-benar memiliki badan huruf,
    - over-segmentation,
    - under-segmentation.
    """

    rasm_records = []

    for idx, region_mask in enumerate(sorted_hasil_potongan):
        # Hanya badan huruf, semua diakritik diabaikan
        badan_rasm = region_mask & (dot_mask == 0)
        coords = np.argwhere(badan_rasm)

        if len(coords) == 0:
            continue

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        area_rasm = int(np.sum(badan_rasm))
        h_rasm = int(max_y - min_y)
        w_rasm = int(max_x - min_x)

        rasm_records.append({
            "huruf": idx + 1,
            "area_rasm": area_rasm,
            "x_min": int(min_x),
            "x_max": int(max_x),
            "y_min": int(min_y),
            "y_max": int(max_y),
            "h": h_rasm,
            "w": w_rasm
        })

    jumlah_dt_rasm = len(rasm_records)

    # Evaluasi jumlah potongan
    TP = min(jumlah_dt_rasm, jumlah_gt_rasm)
    FP = max(0, jumlah_dt_rasm - jumlah_gt_rasm)  # over-segmentation
    FN = max(0, jumlah_gt_rasm - jumlah_dt_rasm)  # under-segmentation

    accuracy = TP / jumlah_gt_rasm if jumlah_gt_rasm != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0 else 0
    )

    hasil_eval = {
        "jumlah_GT": jumlah_gt_rasm,
        "jumlah_DT": jumlah_dt_rasm,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "rasm_records": rasm_records
    }

    return hasil_eval
                               
image_path = r"E:\progres\dengan syafaat.png"

image = io.imread(image_path)
if image.shape[2] == 4:
    image = image[:, :, :3]
gray = 1 - color.rgb2gray(image)
gray_uint8 = (gray * 255).astype(np.uint8)
_, binary = cv.threshold(gray_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

r_stroke = hitung_lebar_garis_stroke(binary)
scale = r_stroke / 8.0 
A_diac_max = (np.pi / 4) * (r_stroke ** 2)

binary = hapus_noise(binary, min_area=5 * (scale**2))

labeled_letters, cleaned_binary, dot_mask = gabungkan_diakritik_dan_mainstroke(binary, A_diac_max, scale)
skeleton_used = skeletonize(cleaned_binary)

component_features = {}
for region_label in range(1, np.max(labeled_letters) + 1):
    mask = (labeled_letters == region_label)
    if np.sum(mask) < 5: continue
    letter_skeleton = skeleton_used * mask
    endpoints, intersections, _ = find_endpoints(letter_skeleton)
    
    # PENAMBAHAN: Pemanggilan deteksi loop
    loops = deteksi_loop(letter_skeleton)
    
    component_features[region_label] = {
        "intersections": intersections,
        "cycles": [cycle for cycle, _ in loops] # Simpan koordinat cycle untuk visualisasi
    }

freeman_debug_records = []

# Area target H6 / fa-alif berdasarkan visualisasi utama
TARGET_FREEMAN_XRANGE = (140, 178)

hasil_potongan = potong_persimpangan(
    labeled_letters,
    skeleton_used,
    dot_mask,
    scale,
    debug_freeman=freeman_debug_records,
    target_freeman_xrange=TARGET_FREEMAN_XRANGE
)
sorted_hasil_potongan = sorted(
    hasil_potongan,
    key=lambda mask: np.max(np.argwhere(mask)[:, 1]),
    reverse=True
)

# ==========================================================
# POST-FIX KHUSUS H2 DAN H8
# Memasang diakritik asli tanpa mengubah huruf lain.
# ==========================================================
sorted_hasil_potongan = pasang_diakritik_khusus_h2_h8(
    sorted_hasil_potongan,
    dot_mask,
    scale
)


print("\n========== FINAL SORTED SEGMENT DEBUG ==========")

for idx, region_mask in enumerate(sorted_hasil_potongan):
    # Badan huruf tanpa diakritik
    badan_huruf = region_mask & (dot_mask == 0)
    seg_crds = np.argwhere(badan_huruf)

    if len(seg_crds) == 0:
        continue

    min_y_seg, min_x_seg = seg_crds.min(axis=0)
    max_y_seg, max_x_seg = seg_crds.max(axis=0)

    h_seg = max_y_seg - min_y_seg
    w_seg = max_x_seg - min_x_seg

    # Hitung estimasi titik atas dan bawah.
    # Ini bukan hanya jumlah komponen, tetapi estimasi jumlah titik sebenarnya.
    estimasi_atas_final, komponen_atas_final, info_atas_final = hitung_estimasi_diakritik_final(
        region_mask,
        dot_mask,
        scale,
        jenis_target="ATAS"
    )

    estimasi_bawah_final, komponen_bawah_final, info_bawah_final = hitung_estimasi_diakritik_final(
        region_mask,
        dot_mask,
        scale,
        jenis_target="BAWAH"
    )

    posisi_titik_final = dominasi_diakritik(
        min_x_seg,
        max_x_seg,
        dot_mask,
        scale
    )

    # ==========================================================
    # KLASIFIKASI FINAL BERDASARKAN BENTUK + ESTIMASI DIAKRITIK
    # ==========================================================
    tipe_final = "UNKNOWN"

    # Huruf tanpa titik
    if cek_dal(badan_huruf, h_seg, w_seg, scale) and estimasi_atas_final == 0 and estimasi_bawah_final == 0:
        tipe_final = "DAL"

    elif cek_ra(badan_huruf, h_seg, w_seg, scale) and estimasi_atas_final == 0 and estimasi_bawah_final == 0:
        tipe_final = "RA"

    elif cek_alif(badan_huruf, h_seg, w_seg, "TIDAK_ADA", scale) and estimasi_atas_final == 0 and estimasi_bawah_final == 0:
        tipe_final = "ALIF"

    elif cek_lam(h_seg, w_seg, scale) and estimasi_atas_final == 0 and estimasi_bawah_final == 0:
        tipe_final = "LAM"

    # ==========================================================
    # Fokus perbaikan H2:
    # NGA tidak boleh hanya bergantung pada jumlah komponen.
    # Jika estimasi titik atas >= 3 dan bentuknya bukan SIN,
    # maka baca sebagai NGA.
    # ==========================================================
    elif estimasi_atas_final >= 3:
        if cek_sin(badan_huruf, h_seg, scale):
            tipe_final = "SYIN"
        else:
            tipe_final = "NGA"

    # Tambahan fallback untuk H2:
    # Jika estimasi atas baru terbaca 2 tetapi bentuk badan mirip AIN/NGA,
    # tetap anggap sebagai NGA.
    elif estimasi_atas_final == 2 and cek_ain(
        badan_huruf,
        h_seg,
        w_seg,
        posisi_titik_final,
        scale
    ):
        tipe_final = "NGA"

    elif estimasi_atas_final == 2:
        tipe_final = "TA"

    elif estimasi_atas_final == 1:
        if cek_fa(badan_huruf, h_seg, w_seg, "ATAS_SINGLE", scale):
            tipe_final = "FA"
        elif cek_nun(badan_huruf, h_seg, w_seg, "ATAS_SINGLE", scale):
            tipe_final = "NUN"
        else:
            tipe_final = "SATU_TITIK_ATAS"

    elif estimasi_bawah_final == 1:
        tipe_final = "BA"

    elif estimasi_bawah_final >= 2:
        tipe_final = "YA"

    elif cek_waw(badan_huruf, h_seg, w_seg, scale):
        tipe_final = "WAW"

    elif cek_ha(badan_huruf, h_seg, w_seg, scale):
        tipe_final = "HA"

    elif cek_ain(badan_huruf, h_seg, w_seg, posisi_titik_final, scale):
        tipe_final = "AIN"

    print(
        f"H{idx+1}: tipe_final={tipe_final}, "
        f"diakritik_atas_final={estimasi_atas_final}, "
        f"komponen_atas={komponen_atas_final}, "
        f"diakritik_bawah_final={estimasi_bawah_final}, "
        f"komponen_bawah={komponen_bawah_final}, "
        f"x=({min_x_seg},{max_x_seg}), "
        f"h={h_seg}, w={w_seg}"
    )

    # Debug khusus untuk H2
    if idx + 1 == 2:
        print("  [DEBUG H2 - KOMPONEN ATAS]")
        for k, info in enumerate(info_atas_final):
            print(
                f"    komponen={k}, "
                f"jumlah_estimasi={info['jumlah']}, "
                f"bbox={info['bbox']}, "
                f"width={info['width']}, "
                f"height={info['height']}, "
                f"area={info['area']}, "
                f"aspect={info['aspect']:.2f}, "
                f"run_count={info['run_count']}"
            )
# PENAMPUNG DATA UNTUK TABEL EVALUASI AKHIR
eval_tsp = []
eval_bspline = []

# --- 1. VISUALISASI UTAMA ---
fig, ax = plt.subplots(figsize=(12, 12)) 
ax.imshow(skeleton_used, cmap='gray')

for label, props in component_features.items():
    mask = (labeled_letters == label)
    skel = skeletonize(mask)
    for iy, ix in props["intersections"]:
        ax.plot(ix, iy, 'o', color='green', markersize=4, label='Intersection' if label == 1 else "")
        
    # PENAMBAHAN: Visualisasi garis merah pada loop
    if "cycles" in props:
        for cycle in props["cycles"]:
            cycle_arr = np.array(cycle)
            ax.plot(cycle_arr[:, 1], cycle_arr[:, 0], 'r-', linewidth=2, label="Loop" if label == 1 else "")
            
    endpoints, _, _ = find_endpoints(skel)
    if len(endpoints) > 0:
        ax.plot(endpoints[0][1], endpoints[0][0], 'o', color='blue') 
        ax.plot(endpoints[-1][1], endpoints[-1][0], 'o', color='yellow')

for yx in np.argwhere(dot_mask == 1):
    ax.plot(yx[1], yx[0], 'o', color='magenta', label='Diakritik Atas')
for yx in np.argwhere(dot_mask == 2):
    ax.plot(yx[1], yx[0], 'o', color='orange', label='Diakritik Bawah')

final_cols = hitung_garis_potong_sinkron(component_features, labeled_letters, skeleton_used, scale)
for col in final_cols:
    ax.axvline(x=col, color='white', linestyle='--', linewidth=2, label='Garis Potong')

# --- VISUALISASI GARIS POTONG HASIL FREEMAN CHAIN CODE ---
for rec in freeman_debug_records:
    if rec.get("cut_col") is not None:
        ax.axvline(
            x=rec["cut_col"],
            color="cyan",
            linestyle="-.",
            linewidth=2,
            label="Garis Potong Freeman"
        )

# --- VISUALISASI AREA TARGET FREEMAN H6 / FA-ALIF ---
if "TARGET_FREEMAN_XRANGE" in globals():
    ax.axvspan(
        TARGET_FREEMAN_XRANGE[0],
        TARGET_FREEMAN_XRANGE[1],
        color="cyan",
        alpha=0.15,
        label="Area Target Freeman"
    )

for idx, region_mask in enumerate(sorted_hasil_potongan):
    badan_huruf = region_mask & (dot_mask == 0)
    seg_crds = np.argwhere(badan_huruf)
    if len(seg_crds) == 0: continue
    
    min_y_seg, min_x_seg = seg_crds.min(axis=0)
    max_y_seg, max_x_seg = seg_crds.max(axis=0)
    h_seg = max_y_seg - min_y_seg
    w_seg = max_x_seg - min_x_seg
    
    posisi_titik = dominasi_diakritik(min_x_seg, max_x_seg, dot_mask, scale)
    
    skel_part = skeletonize(badan_huruf)
    py, px = np.where(skel_part)

    rect = patches.Rectangle((min_x_seg, min_y_seg), w_seg, h_seg, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(min_x_seg, min_y_seg-10, f"H{idx+1}", color='red', fontsize=12, fontweight='bold')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_title("Visualisasi Utama Segmentasi Karakter")
ax.axis('on')
plt.tight_layout()
plt.show()

# --- VISUALISASI DETAIL FREEMAN CHAIN CODE ---
visualisasi_freeman(freeman_debug_records)

# --- 2. PENYIMPANAN DAN EVALUASI METRIK ---
output_folder = "hasil_potongan"
os.makedirs(output_folder, exist_ok=True)

for idx, region_mask in enumerate(sorted_hasil_potongan):
    output_image = (region_mask * 255).astype(np.uint8)
    output_filename = os.path.join(output_folder, f'potongan_huruf_{idx + 1}.png')
    cv.imwrite(output_filename, output_image)

# ==========================================================
# EVALUASI PEMOTONGAN RASM
# Fokus hanya pada jumlah dan keberhasilan pemotongan badan huruf.
# ==========================================================
jumlah_GT_rasm = 8

eval_rasm = evaluasi_pemotongan_rasm(
    sorted_hasil_potongan,
    dot_mask,
    jumlah_GT_rasm
)

# ==========================================================
# HASIL CURVE FITTING RASM
# ==========================================================
hasil_curve_fitting = fitting_bspline_rasm_per_huruf(
    sorted_hasil_potongan,
    dot_mask,
    scale,
    output_folder="hasil_curve_fitting"
)

jumlah_GT = eval_rasm["jumlah_GT"]
jumlah_DT = eval_rasm["jumlah_DT"]
TP = eval_rasm["TP"]
FP = eval_rasm["FP"]
FN = eval_rasm["FN"]
accuracy = eval_rasm["accuracy"]
precision = eval_rasm["precision"]
recall = eval_rasm["recall"]
f1_score = eval_rasm["f1_score"]


# --- 3. VISUALISASI DETAIL TSP (BINER VS SKELETON+TSP) ---
print("\nMenampilkan visualisasi Biner vs Skeleton+TSP per huruf...")
for idx, region_mask in enumerate(sorted_hasil_potongan):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(region_mask, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title(f'Huruf Potong ke-{idx+1} (Biner)')
    ax[0].axis('off')

    letter_skeleton = skeletonize(region_mask)
    endpoints, intersections, turns = find_endpoints(letter_skeleton)
    centroid = center_of_mass(region_mask)
    
    ax[1].imshow(letter_skeleton, cmap='gray')
    
    if len(endpoints) > 0:
        sy, sx = endpoints[0]
        ax[1].plot(sx, sy, 'o', color='blue', label="Start", zorder=10)
    if len(endpoints) > 1:
        ey, ex = endpoints[-1]
        ax[1].plot(ex, ey, 'o', color='yellow', label="End", zorder=10)
    for iy, ix in intersections:
        ax[1].plot(ix, iy, 'o', color='green', markersize=4, label="Intersection", zorder=10)
    if centroid is not None:
        cy, cx = centroid
        ax[1].plot(cx, cy, 'o', color='red', label="Centroid", zorder=10)
        
    # PENAMBAHAN: Visualisasi merah di detail
    loops = deteksi_loop(letter_skeleton)
    for cycle, _ in loops:
        cycle_array = np.array(cycle)
        ax[1].plot(cycle_array[:, 1], cycle_array[:, 0], 'r-', linewidth=2, label="Loop")
        
    points = np.argwhere(letter_skeleton)
    tsp_segments = tsp_skeleton_traversal(points, dist_threshold=max(3, int(5 * scale)))
    
    for segment in tsp_segments:
        segment_array = np.array(segment)
        ax[1].scatter(segment_array[:, 1], segment_array[:, 0], color='white', s=5, zorder=5)
        for i in range(len(segment_array) - 1):
            p1, p2 = segment_array[i], segment_array[i + 1]
            ax[1].plot([p1[1], p2[1]], [p1[0], p2[0]], color='white', linewidth=0.8, alpha=0.6, zorder=4)
            
    ax[1].set_title(f'Huruf Potong ke-{idx+1} (Skeleton + TSP)')
    ax[1].axis('off')
    
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show() 


# --- 4. VISUALISASI MATEMATIS B-SPLINE (CARTESIAN) ---
print("\nMenampilkan visualisasi Matematis B-Spline per huruf...")
for idx, region_mask in enumerate(sorted_hasil_potongan):
    skel = skeletonize(region_mask)
    points = np.argwhere(skel)
    
    tsp_segments = tsp_skeleton_traversal(points, dist_threshold=max(3, int(5 * scale)))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    has_valid_curve = False
    
    total_cp_karakter = 0
    total_jarak_karakter = 0
    total_points_karakter = 0
    
    total_sse_karakter = 0.0
    total_abs_error_karakter = 0.0
    total_error_points_karakter = 0
    max_error_karakter = 0.0
    
    total_curve_length_karakter = 0.0
    smoothness_values = [] 
    
    for seg_idx, segment in enumerate(tsp_segments):
        segment_array = np.array(segment)
        n_len = len(segment_array)
        if n_len == 0: continue
        
        total_points_karakter += n_len
        
        for i in range(n_len - 1):
            total_jarak_karakter += euclidean(segment_array[i], segment_array[i+1])
        
        py, px = segment_array[0]
        is_diacritic = dot_mask[py, px] > 0
        
        if is_diacritic:
            jumlah_target_titik = max(2, min(n_len, 2))
            k = 1 
        else:
            jumlah_target_titik = max(4, n_len // 5)
            k = 3 
            
        total_cp_karakter += jumlah_target_titik
        
        if jumlah_target_titik >= k + 1: 
            has_valid_curve = True
            indeks_pilihan = np.linspace(0, n_len - 1, jumlah_target_titik, dtype=int)
            titik_kontrol = segment_array[indeks_pilihan]
            
            kurva_halus = bspline_curve(titik_kontrol, degree=k, num_points=max(100, n_len * 3))
            
            dists = cdist(segment_array, kurva_halus)
            min_dists = np.min(dists, axis=1)
            
            total_sse_karakter += np.sum(min_dists ** 2)
            total_abs_error_karakter += np.sum(min_dists)
            total_error_points_karakter += len(min_dists)
            
            if len(min_dists) > 0:
                max_error_karakter = max(
                    max_error_karakter,
                    float(np.max(min_dists))
                )
            
            total_curve_length_karakter += hitung_panjang_kurva(kurva_halus)
            smoothness_values.append(hitung_smoothness_kurva(kurva_halus))
            
            x_ctrl = titik_kontrol[:, 1]
            y_ctrl = titik_kontrol[:, 0]
            x_curve = kurva_halus[:, 1]
            y_curve = kurva_halus[:, 0]
            
            label_poly = 'Control Polygon' if seg_idx == 0 else ""
            label_cp = 'Control Points' if seg_idx == 0 else ""
            label_curve = 'Generated Curve' if seg_idx == 0 else ""
            
            ax.plot(x_ctrl, y_ctrl, color='red', linewidth=1, zorder=1, label=label_poly)
            ax.scatter(x_ctrl, y_ctrl, color='darkblue', s=40, zorder=3, label=label_cp)
            ax.plot(x_curve, y_curve, color='gray', linewidth=1.5, zorder=2, label=label_curve)
            
    if total_points_karakter > 0:
        eval_tsp.append((idx + 1, len(points), total_jarak_karakter))
    
        kompresi = (
            (total_points_karakter - total_cp_karakter)
            / total_points_karakter
        ) * 100

    if total_error_points_karakter > 0:
        rmse = np.sqrt(total_sse_karakter / total_error_points_karakter)
        mae = total_abs_error_karakter / total_error_points_karakter
    else:
        rmse = 0.0
        mae = 0.0

    smoothness_mean = (
        float(np.mean(smoothness_values))
        if len(smoothness_values) > 0 else 0.0
    )

    eval_bspline.append({
        "huruf": idx + 1,
        "skeleton": int(total_points_karakter),
        "cp": int(total_cp_karakter),
        "kompresi": float(kompresi),
        "tsp_dist": float(total_jarak_karakter),
        "curve_len": float(total_curve_length_karakter),
        "rmse": float(rmse),
        "mae": float(mae),
        "max_error": float(max_error_karakter),
        "smoothness": float(smoothness_mean)
    })
    
    if has_valid_curve:
        ax.invert_yaxis() 
        ax.set_aspect('equal', adjustable='datalim') 
        
        judul = f'(Generated Huruf ke-{idx+1})\n'
        judul += f'Total CP: {total_cp_karakter} | Jarak TSP: {total_jarak_karakter:.2f} px\n'
        judul += f'Kompresi Data: {kompresi:.1f}% | RMSE: {rmse:.2f} px'
        ax.set_title(judul, fontsize=11, pad=15)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)
        
        plt.tight_layout()
        plt.show()

# --- 5. VISUALISASI TOTAL JARAK (DISTANCE) ---
fig, axes = plt.subplots(1, len(sorted_hasil_potongan), figsize=(3 * len(sorted_hasil_potongan), 4))
if len(sorted_hasil_potongan) == 1:
    axes = [axes]
    
for idx, region_mask in enumerate(sorted_hasil_potongan):
    skel = skeletonize(region_mask)
    points = np.argwhere(skel)
    tsp_segments = tsp_skeleton_traversal(points, dist_threshold=max(3, int(5 * scale)))
    
    total_distance = 0
    ax = axes[idx] if len(sorted_hasil_potongan) > 1 else axes[0]
    ax.imshow(skel, cmap='gray')
    for segment in tsp_segments:
        for i in range(len(segment) - 1):
            p1, p2 = segment[i], segment[i + 1]
            total_distance += euclidean(p1, p2)
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color='white', linewidth=1)
            
    ax.set_title(f'Huruf {idx+1}\nJarak: {total_distance:.2f}')
    ax.axis('off')

plt.suptitle("Struktur Huruf + Jalur TSP + Jarak Total")
plt.tight_layout()
plt.show()

# --- 6. CETAK TABEL EVALUASI KESELURUHAN ---
print("\n" + "="*70)
print("TABEL EVALUASI PEMOTONGAN RASM")
print("="*70)
print(f"Ground Truth Rasm (GT)     : {jumlah_GT}")
print(f"Detected Rasm (DT)         : {jumlah_DT}")
print(f"True Positive (TP)         : {TP}")
print(f"False Positive (FP)        : {FP}  -> over-segmentation")
print(f"False Negative (FN)        : {FN}  -> under-segmentation")
print(f"Accuracy                   : {accuracy*100:.2f}%")
print(f"Precision                  : {precision*100:.2f}%")
print(f"Recall                     : {recall*100:.2f}%")
print(f"F1-Score                   : {f1_score*100:.2f}%")

print("\n" + "="*90)
print("DETAIL POTONGAN RASM")
print("="*90)
print(
    f"{'H':<5} | {'Area':<8} | {'x_min':<7} | {'x_max':<7} | "
    f"{'y_min':<7} | {'y_max':<7} | {'h':<5} | {'w':<5}"
)
print("-" * 90)

for rec in eval_rasm["rasm_records"]:
    print(
        f"H{rec['huruf']:<4} | "
        f"{rec['area_rasm']:<8} | "
        f"{rec['x_min']:<7} | "
        f"{rec['x_max']:<7} | "
        f"{rec['y_min']:<7} | "
        f"{rec['y_max']:<7} | "
        f"{rec['h']:<5} | "
        f"{rec['w']:<5}"
    )

print("\n" + "="*60)
print("EVALUASI FITUR (RUTE TSP vs ASLI)")
print("="*60)
print(f"{'Huruf':<10} | {'Piksel Asli (px)':<18} | {'Jarak TSP (px)':<15}")
print("-" * 60)
for data in eval_tsp:
    print(f"H-{data[0]:<8} | {data[1]:<18} | {data[2]:.2f}")

print("\n" + "="*120)
print("HASIL CURVE FITTING KARAKTER B-SPLINE")
print("="*120)
print(
    f"{'H':<5} | {'Skeleton':<9} | {'CP':<5} | "
    f"{'Kompresi':<10} | {'TSP Dist':<10} | "
    f"{'Curve Len':<10} | {'RMSE':<8} | {'MAE':<8} | "
    f"{'Max Error':<10} | {'Smoothness':<10}"
)
print("-" * 120)

for row in eval_bspline:
    print(
        f"H{row['huruf']:<4} | "
        f"{row['skeleton']:<9} | "
        f"{row['cp']:<5} | "
        f"{row['kompresi']:<9.2f}% | "
        f"{row['tsp_dist']:<10.2f} | "
        f"{row['curve_len']:<10.2f} | "
        f"{row['rmse']:<8.2f} | "
        f"{row['mae']:<8.2f} | "
        f"{row['max_error']:<10.2f} | "
        f"{row['smoothness']:<10.4f}"
    )

print("="*120 + "\n")

print("\n" + "="*100)
print("TABEL HASIL CURVE FITTING RASM B-SPLINE")
print("="*100)
print(
    f"{'H':<5} | {'Skeleton':<9} | {'CP':<5} | "
    f"{'Kompresi':<10} | {'TSP Dist':<10} | "
    f"{'Curve Len':<10} | {'RMSE':<8} | {'MAE':<8} | "
    f"{'Max Error':<10} | {'Smoothness':<10}"
)
print("-" * 100)

for row in hasil_curve_fitting:
    print(
        f"H{row['huruf']:<4} | "
        f"{row['skeleton_points']:<9} | "
        f"{row['control_points']:<5} | "
        f"{row['compression_percent']:<9.2f}% | "
        f"{row['tsp_distance']:<10.2f} | "
        f"{row['curve_length']:<10.2f} | "
        f"{row['rmse']:<8.2f} | "
        f"{row['mae']:<8.2f} | "
        f"{row['max_error']:<10.2f} | "
        f"{row['smoothness']:<10.4f}"
    )    

csv_path = os.path.join("hasil_curve_fitting", "evaluasi_curve_fitting_rasm.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "huruf",
            "skeleton_points",
            "control_points",
            "compression_percent",
            "tsp_distance",
            "curve_length",
            "rmse",
            "mae",
            "max_error",
            "smoothness"
        ]
    )

    writer.writeheader()

    for row in hasil_curve_fitting:
        writer.writerow(row)

print(f"\nFile evaluasi curve fitting disimpan di: {csv_path}")
    