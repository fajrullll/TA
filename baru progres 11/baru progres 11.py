# ================================================================================
# IMPORT LIBRARY
# ================================================================================

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

def figsize_dinamis(data, px_per_inch=45, min_w=8, max_w=20, min_h=3.5, max_h=9):
    """
    Membuat ukuran figure Matplotlib mengikuti ukuran data input.
    h, w diambil dari shape citra/mask sehingga visualisasi tidak dipaksa kotak.
    """
    h, w = data.shape[:2]
    aspect = w / max(h, 1)

    fig_w = np.clip(w / px_per_inch, min_w, max_w)
    fig_h = fig_w / max(aspect, 1e-5)
    fig_h = np.clip(fig_h, min_h, max_h)

    return float(fig_w), float(fig_h)


# ================================================================================
# PRA-PEMROSESAN CITRA DAN DINAMISASI SKALA
# ================================================================================

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


# ================================================================================
# EKSTRAKSI FITUR SKELETON DAN TOPOLOGI GRAF
# ================================================================================

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


# ================================================================================
# FREEMAN CHAIN CODE (FCC) SEBAGAI METODE TAMBAHAN PEMOTONGAN
# ================================================================================

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

def balik_arah_freeman_code(codes):
    """
    Membalik arah pembacaan Freeman Chain Code.

    Jika jalur skeleton dibaca berlawanan dengan arah goresan penulisan,
    maka urutan kode dibalik dan setiap arah ditambah 4 modulo 8.

    Rumus:
        kode_balik = (kode_asli + 4) % 8

    Karena urutan path juga dibalik, maka digunakan:
        codes_tulis = (codes[::-1] + 4) % 8
    """
    if codes is None or len(codes) == 0:
        return np.array([])

    return (codes[::-1] + 4) % 8

def balik_xaxis_freeman_code(codes):
    """
    Membalik kode Freeman terhadap sumbu-x citra.

    Ini digunakan untuk interpretasi arah kanan-kiri,
    terutama karena tulisan Arab/Jawi dibaca dari kanan ke kiri.

    Mapping arah:
        0(E)  -> 4(W)
        1(NE) -> 3(NW)
        2(N)  -> 2(N)
        3(NW) -> 1(NE)
        4(W)  -> 0(E)
        5(SW) -> 7(SE)
        6(S)  -> 6(S)
        7(SE) -> 5(SW)
    """
    if codes is None or len(codes) == 0:
        return np.array([])

    return (4 - codes) % 8

def normalisasi_freeman_arah_tulis(codes, mode="reverse_direction"):
    """
    Normalisasi kode Freeman untuk kebutuhan visualisasi arah goresan.

    mode:
    - "original"          : kode asli tanpa perubahan
    - "reverse_direction" : urutan dibalik + setiap kode ditambah 4 modulo 8
    - "flip_x"            : kode direfleksikan terhadap sumbu-x
    """
    if codes is None or len(codes) == 0:
        return np.array([])

    if mode == "original":
        return codes

    if mode == "reverse_direction":
        return balik_arah_freeman_code(codes)

    if mode == "flip_x":
        return balik_xaxis_freeman_code(codes)

    return codes

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


# ================================================================================
# VISUALISASI FREEMAN CHAIN CODE
# ================================================================================

def visualisasi_freeman(debug_records, label_step=4, target_xrange_visual=None):
    """
    Visualisasi Freeman Chain Code setelah normalisasi arah goresan.

    Tampilan tetap seperti sebelumnya:
    - background hitam
    - skeleton putih
    - path FCC cyan
    - angka FCC kuning
    - candidate point merah
    - selected candidate ungu
    - garis potong Freeman merah putus-putus

    Perubahan:
    - path FCC, angka FCC, candidate point, selected candidate, dan cut line
      hanya ditampilkan pada area TARGET_FREEMAN_XRANGE.
    - Perhitungan FCC tetap tidak diubah.
    """

    if not debug_records:
        print("\nTidak ada segmentasi tambahan dari Freeman Chain Code.")
        return

    if target_xrange_visual is None:
        if "TARGET_FREEMAN_XRANGE" in globals():
            target_xrange_visual = TARGET_FREEMAN_XRANGE

    print("\nMenampilkan visualisasi Freeman Chain Code setelah normalisasi arah goresan...")

    teks_keterangan = (
        "Keterangan Arah Freeman Chain Code (FCC)\n"
        "bergerak kanan        : 0\n"
        "bergerak kanan-atas   : 1\n"
        "bergerak atas         : 2\n"
        "bergerak kiri-atas    : 3\n"
        "bergerak kiri         : 4\n"
        "bergerak kiri-bawah   : 5\n"
        "bergerak bawah        : 6\n"
        "bergerak kanan-bawah  : 7\n\n"
    )

    def pecah_index_kontinu(indices):
        """
        Memecah index menjadi beberapa kelompok kontinu.
        Tujuannya agar path tidak tersambung loncat saat difilter.
        """
        if indices is None or len(indices) == 0:
            return []

        groups = []
        current = [indices[0]]

        for i in indices[1:]:
            if i == current[-1] + 1:
                current.append(i)
            else:
                groups.append(np.array(current))
                current = [i]

        groups.append(np.array(current))
        return groups

    for idx, rec in enumerate(debug_records):
        body = rec["body"]
        path = rec["path"]
        codes = rec["codes"]
        cut_col = rec["cut_col"]

        candidate_points = rec.get("candidate_points", np.array([]))
        best_candidate_index = rec.get("best_candidate_index", None)

        if path is None or len(path) < 2 or codes is None or len(codes) == 0:
            continue

        # ======================================================
        # DATA ASLI
        # ======================================================
        path_asli = np.array(path)
        codes_asli = np.array(codes)

        # ======================================================
        # DATA SETELAH NORMALISASI ARAH GORESAN
        # path dibalik, kode +4 modulo 8
        # ======================================================
        path_arah_tulis = path_asli[::-1].copy()
        codes_arah_tulis = normalisasi_freeman_arah_tulis(
            codes_asli,
            mode="reverse_direction"
        )

        # ======================================================
        # FULL SKELETON DARI BODY
        # supaya tampilan tetap hitam-putih seperti sebelumnya
        # ======================================================
        body_bool = body > 0
        full_skeleton = skeletonize(body_bool)

        skeleton_canvas = np.zeros_like(body, dtype=np.uint8)
        skeleton_canvas[full_skeleton] = 1

        # ======================================================
        # FILTER PATH BERDASARKAN TARGET_FREEMAN_XRANGE
        # hanya untuk visualisasi, bukan untuk perhitungan
        # ======================================================
        if target_xrange_visual is not None:
            tx_min, tx_max = target_xrange_visual

            mask_visual = (
                (path_arah_tulis[:, 1] >= tx_min) &
                (path_arah_tulis[:, 1] <= tx_max)
            )

            visual_indices = np.where(mask_visual)[0]
        else:
            visual_indices = np.arange(len(path_arah_tulis))

        visual_groups = pecah_index_kontinu(visual_indices)

        # ======================================================
        # PILIH INDEX ANGKA YANG DITAMPILKAN
        # ======================================================
        indeks_label = set()

        if len(visual_indices) >= 2:
            indeks_label.add(int(visual_indices[0]))

            last_code_idx = min(
                int(visual_indices[-1]),
                len(codes_arah_tulis) - 1
            )
            indeks_label.add(last_code_idx)

            for pos in range(0, len(visual_indices), label_step):
                real_idx = int(visual_indices[pos])

                if real_idx < len(codes_arah_tulis):
                    indeks_label.add(real_idx)

        indeks_label = sorted(indeks_label)

        # ======================================================
        # FIGURE
        # ======================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("black")

        # Background skeleton penuh
        ax.imshow(
            skeleton_canvas,
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=1.0
        )

        # ======================================================
        # OPSIONAL: tampilkan area target Freeman sebagai latar
        # ======================================================
        if target_xrange_visual is not None:
            tx_min, tx_max = target_xrange_visual
            ax.axvspan(
                tx_min,
                tx_max,
                color="cyan",
                alpha=0.10,
                zorder=1,
                label="_nolegend_"
            )

        # ======================================================
        # FREEMAN PATH ARAH GORESAN
        # hanya yang masuk target_xrange_visual
        # ======================================================
        sudah_label_path = False

        for group in visual_groups:
            if len(group) < 2:
                continue

            path_seg = path_arah_tulis[group]

            ax.plot(
                path_seg[:, 1],
                path_seg[:, 0],
                color="cyan",
                linewidth=2.0,
                label="Freeman Path Arah Goresan" if not sudah_label_path else "_nolegend_",
                zorder=3
            )

            sudah_label_path = True

        # ======================================================
        # START dan END pada path visual
        # ======================================================
        if len(visual_indices) >= 2:
            start_idx = int(visual_indices[0])
            end_idx = int(visual_indices[-1])

            ax.scatter(
                path_arah_tulis[start_idx, 1],
                path_arah_tulis[start_idx, 0],
                color="blue",
                s=55,
                zorder=5,
                label="Start Path"
            )

            ax.scatter(
                path_arah_tulis[end_idx, 1],
                path_arah_tulis[end_idx, 0],
                color="orange",
                s=55,
                zorder=5,
                label="End Path"
            )

        # ======================================================
        # ANGKA FREEMAN SETELAH NORMALISASI
        # hanya pada area target visual
        # ======================================================
        for k, i in enumerate(indeks_label):
            if i < 0 or i >= len(codes_arah_tulis):
                continue

            if i >= len(path_arah_tulis):
                continue

            y, x = path_arah_tulis[i]
            code = int(codes_arah_tulis[i])

            if k % 4 == 0:
                dx, dy = 0.45, -0.45
            elif k % 4 == 1:
                dx, dy = 0.45, 0.45
            elif k % 4 == 2:
                dx, dy = -0.45, -0.45
            else:
                dx, dy = -0.45, 0.45

            ax.text(
                x + dx,
                y + dy,
                str(code),
                color="yellow",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.06",
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.75
                ),
                zorder=6
            )

        # ======================================================
        # FILTER DAN TAMPILKAN CANDIDATE POINT
        # ======================================================
        candidate_points_visual = np.array([])

        if candidate_points is not None and len(candidate_points) > 0:
            candidate_points = np.array(candidate_points)

            if target_xrange_visual is not None:
                tx_min, tx_max = target_xrange_visual

                mask_candidate = (
                    (candidate_points[:, 1] >= tx_min) &
                    (candidate_points[:, 1] <= tx_max)
                )

                candidate_points_visual = candidate_points[mask_candidate]
            else:
                candidate_points_visual = candidate_points

        if candidate_points_visual is not None and len(candidate_points_visual) > 0:
            ax.scatter(
                candidate_points_visual[:, 1],
                candidate_points_visual[:, 0],
                color="red",
                s=45,
                marker="x",
                linewidths=1.5,
                zorder=7,
                label="Candidate Point Potong"
            )

        # ======================================================
        # SELECTED CANDIDATE
        # tetap dari FCC asli, tetapi hanya tampil jika masuk target
        # ======================================================
        if best_candidate_index is not None:
            if 0 <= best_candidate_index < len(path_asli):
                y_best, x_best = path_asli[best_candidate_index]

                tampilkan_selected = True

                if target_xrange_visual is not None:
                    tx_min, tx_max = target_xrange_visual
                    tampilkan_selected = (tx_min <= x_best <= tx_max)

                if tampilkan_selected:
                    ax.scatter(
                        x_best,
                        y_best,
                        color="magenta",
                        s=80,
                        marker="o",
                        edgecolor="white",
                        linewidths=1.0,
                        zorder=8,
                        label="Selected Candidate"
                    )

        # ======================================================
        # GARIS POTONG FREEMAN
        # hanya tampil jika masuk target
        # ======================================================
        if cut_col is not None:
            tampilkan_cut = True

            if target_xrange_visual is not None:
                tx_min, tx_max = target_xrange_visual
                tampilkan_cut = (tx_min <= cut_col <= tx_max)

            if tampilkan_cut:
                ax.axvline(
                    x=cut_col,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Freeman Cut",
                    zorder=4
                )

        # ======================================================
        # TEXTBOX KETERANGAN
        # ======================================================
        ax.text(
            0.02,
            0.98,
            teks_keterangan,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            color="black",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="black",
                alpha=0.95
            ),
            zorder=20
        )

        ax.set_title(
            f"Freeman Chain Code Setelah Normalisasi Arah Goresan - Region {rec.get('region_label', idx + 1)}",
            fontsize=12
        )

        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.9
        )

        ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

        # ======================================================
        # DEBUG CONSOLE
        # ======================================================
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus
        pass  # debug print dihapus

def overlay_fcc_pada_visualisasi_utama(
    ax,
    freeman_debug_records,
    label_step=8,
    target_xrange_visual=None
):
    """
    Menambahkan overlay Freeman Chain Code pada visualisasi utama.

    Konsep yang digunakan:
    - Kandidat titik potong tetap dihitung dari FCC asli
      di fungsi deteksi_potong_freeman().
    - Visualisasi angka FCC ditampilkan setelah normalisasi arah goresan:
        path dibalik + code = (code + 4) mod 8
    - Pada visualisasi utama, Freeman Path hanya ditampilkan
      pada area TARGET_FREEMAN_XRANGE agar tidak terlihat seolah-olah
      H4 ikut diproses oleh FCC.

    Overlay yang ditampilkan:
    - Freeman Path Arah Goresan, hanya di area target Freeman
    - Angka FCC setelah normalisasi arah goresan
    - Candidate Point Potong
    - Selected Candidate
    - Garis Potong Freeman
    """

    if not freeman_debug_records:
        return

    # ==========================================================
    # Ambil target_xrange untuk visualisasi
    # Prioritas:
    # 1. parameter target_xrange_visual
    # 2. variable global TARGET_FREEMAN_XRANGE
    # ==========================================================
    if target_xrange_visual is None:
        if "TARGET_FREEMAN_XRANGE" in globals():
            target_xrange_visual = TARGET_FREEMAN_XRANGE

    # Supaya label legend tidak muncul berulang-ulang
    sudah_label_path = False
    sudah_label_start = False
    sudah_label_end = False
    sudah_label_candidate = False
    sudah_label_selected = False
    sudah_label_cut = False

    def pecah_index_kontinu(indices):
        """
        Memecah index menjadi beberapa kelompok kontinu.

        Contoh:
        [3, 4, 5, 10, 11] menjadi:
        [[3, 4, 5], [10, 11]]

        Ini penting agar garis path tidak menyambung melompat
        jika path keluar-masuk area target Freeman.
        """
        if indices is None or len(indices) == 0:
            return []

        groups = []
        current = [indices[0]]

        for i in indices[1:]:
            if i == current[-1] + 1:
                current.append(i)
            else:
                groups.append(np.array(current))
                current = [i]

        groups.append(np.array(current))
        return groups

    for idx, rec in enumerate(freeman_debug_records):
        path = rec.get("path", None)
        codes = rec.get("codes", None)
        cut_col = rec.get("cut_col", None)

        candidate_points = rec.get("candidate_points", np.array([]))
        best_candidate_index = rec.get("best_candidate_index", None)

        if path is None or codes is None:
            continue

        if len(path) < 2 or len(codes) == 0:
            continue

        path_asli = np.array(path)
        codes_asli = np.array(codes)

        # ======================================================
        # Normalisasi arah goresan hanya untuk visualisasi
        # Catatan:
        # Kandidat titik potong tetap dihitung dari path dan codes asli
        # di fungsi deteksi_potong_freeman().
        # ======================================================
        path_arah_tulis = path_asli[::-1].copy()
        codes_arah_tulis = normalisasi_freeman_arah_tulis(
            codes_asli,
            mode="reverse_direction"
        )

        # ======================================================
        # FILTER VISUALISASI BERDASARKAN TARGET_FREEMAN_XRANGE
        # Ini hanya membatasi tampilan path FCC pada visualisasi utama.
        # Perhitungan FCC tetap tidak diubah.
        # ======================================================
        if target_xrange_visual is not None:
            tx_min, tx_max = target_xrange_visual

            mask_visual = (
                (path_arah_tulis[:, 1] >= tx_min) &
                (path_arah_tulis[:, 1] <= tx_max)
            )

            visual_indices = np.where(mask_visual)[0]
        else:
            visual_indices = np.arange(len(path_arah_tulis))

        if len(visual_indices) < 2:
            # Tidak ada bagian path yang berada dalam target visual
            # tetapi candidate/cut tetap bisa divisualkan jika ada.
            visual_groups = []
        else:
            visual_groups = pecah_index_kontinu(visual_indices)

        # ======================================================
        # GAMBAR FREEMAN PATH ARAH GORESAN
        # hanya bagian yang masuk target_xrange_visual
        # ======================================================
        for g in visual_groups:
            if len(g) < 2:
                continue

            path_seg = path_arah_tulis[g]

            ax.plot(
                path_seg[:, 1],
                path_seg[:, 0],
                color="cyan",
                linewidth=2.0,
                label="Freeman Path Arah Goresan" if not sudah_label_path else "_nolegend_",
                zorder=20
            )

            sudah_label_path = True

        # ======================================================
        # START dan END FCC pada bagian visual target
        # ======================================================
        if len(visual_indices) >= 2:
            start_idx = visual_indices[0]
            end_idx = visual_indices[-1]

            ax.scatter(
                path_arah_tulis[start_idx, 1],
                path_arah_tulis[start_idx, 0],
                color="blue",
                s=45,
                zorder=25,
                label="Start FCC" if not sudah_label_start else "_nolegend_"
            )
            sudah_label_start = True

            ax.scatter(
                path_arah_tulis[end_idx, 1],
                path_arah_tulis[end_idx, 0],
                color="orange",
                s=45,
                zorder=25,
                label="End FCC" if not sudah_label_end else "_nolegend_"
            )
            sudah_label_end = True

        # ======================================================
        # ANGKA FCC PADA AREA TARGET
        # Dibuat jarang agar visualisasi utama tidak terlalu ramai.
        # ======================================================
        if len(visual_indices) >= 2:
            indeks_label = set()

            # Awal dan akhir visual
            indeks_label.add(int(visual_indices[0]))

            # Untuk kode, indeks maksimal adalah len(codes_arah_tulis)-1
            last_code_idx = min(int(visual_indices[-1]), len(codes_arah_tulis) - 1)
            indeks_label.add(last_code_idx)

            # Sampling berdasarkan label_step
            for pos in range(0, len(visual_indices), label_step):
                real_idx = int(visual_indices[pos])

                if real_idx < len(codes_arah_tulis):
                    indeks_label.add(real_idx)

            indeks_label = sorted(indeks_label)

            for k, i in enumerate(indeks_label):
                if i < 0 or i >= len(codes_arah_tulis):
                    continue

                if i >= len(path_arah_tulis):
                    continue

                y, x = path_arah_tulis[i]
                code = int(codes_arah_tulis[i])

                # Offset bergantian agar angka tidak menutupi skeleton
                if k % 4 == 0:
                    dx, dy = 0.45, -0.45
                elif k % 4 == 1:
                    dx, dy = 0.45, 0.45
                elif k % 4 == 2:
                    dx, dy = -0.45, -0.45
                else:
                    dx, dy = -0.45, 0.45

                ax.text(
                    x + dx,
                    y + dy,
                    str(code),
                    color="yellow",
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.06",
                        facecolor="black",
                        edgecolor="none",
                        alpha=0.75
                    ),
                    zorder=30
                )

        # ======================================================
        # FILTER CANDIDATE POINT AGAR HANYA YANG DI AREA TARGET
        # ======================================================
        candidate_points_visual = np.array([])

        if candidate_points is not None and len(candidate_points) > 0:
            candidate_points = np.array(candidate_points)

            if target_xrange_visual is not None:
                tx_min, tx_max = target_xrange_visual

                mask_candidate = (
                    (candidate_points[:, 1] >= tx_min) &
                    (candidate_points[:, 1] <= tx_max)
                )

                candidate_points_visual = candidate_points[mask_candidate]
            else:
                candidate_points_visual = candidate_points

        # ======================================================
        # TAMPILKAN CANDIDATE POINT POTONG
        # ======================================================
        if candidate_points_visual is not None and len(candidate_points_visual) > 0:
            ax.scatter(
                candidate_points_visual[:, 1],
                candidate_points_visual[:, 0],
                color="red",
                s=45,
                marker="x",
                linewidths=1.5,
                zorder=28,
                label="Candidate Point Potong" if not sudah_label_candidate else "_nolegend_"
            )

            sudah_label_candidate = True

        # ======================================================
        # SELECTED CANDIDATE
        # Tetap memakai best_candidate_index dari FCC asli,
        # tetapi hanya ditampilkan jika berada di area target visual.
        # ======================================================
        if best_candidate_index is not None:
            if 0 <= best_candidate_index < len(path_asli):
                y_best, x_best = path_asli[best_candidate_index]

                tampilkan_selected = True

                if target_xrange_visual is not None:
                    tx_min, tx_max = target_xrange_visual
                    tampilkan_selected = (tx_min <= x_best <= tx_max)

                if tampilkan_selected:
                    ax.scatter(
                        x_best,
                        y_best,
                        color="magenta",
                        s=80,
                        marker="o",
                        edgecolor="white",
                        linewidths=1.0,
                        zorder=29,
                        label="Selected Candidate" if not sudah_label_selected else "_nolegend_"
                    )

                    sudah_label_selected = True

        # ======================================================
        # GARIS POTONG FREEMAN
        # Tetap ditampilkan jika berada dalam area target visual.
        # ======================================================
        if cut_col is not None:
            tampilkan_cut = True

            if target_xrange_visual is not None:
                tx_min, tx_max = target_xrange_visual
                tampilkan_cut = (tx_min <= cut_col <= tx_max)

            if tampilkan_cut:
                ax.axvline(
                    x=cut_col,
                    color="cyan",
                    linestyle="-.",
                    linewidth=2.2,
                    label="Garis Potong Freeman" if not sudah_label_cut else "_nolegend_",
                    zorder=18
                )

                sudah_label_cut = True


# ================================================================================
# TRAVERSAL SKELETON BERBASIS TSP
# ================================================================================

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


# ================================================================================
# ANALISIS MORFOLOGI DAN KLASIFIKASI HEURISTIK HURUF
# ================================================================================

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

def analisis_ujung_atas_sin(seg_mask, skel, scale):
    endpoints, _, _ = find_endpoints(skel)

    if len(endpoints) == 0:
        return np.array([])

    coords = np.argwhere(seg_mask)

    if len(coords) == 0:
        return np.array([])

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    h = max_y - min_y
    batas_atas = min_y + (h * 0.55)

    top_endpoints = np.array([
        ep for ep in endpoints
        if ep[0] <= batas_atas
    ])

    return top_endpoints


def deteksi_zona_sin(body_rasm, skeleton_rasm, scale):
    coords = np.argwhere(body_rasm)

    if len(coords) == 0:
        return [], {
            "top_endpoints": np.array([]),
            "status": "TIDAK_ADA_BODY"
        }

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    w = max_x - min_x

    top_endpoints = analisis_ujung_atas_sin(
        body_rasm,
        skeleton_rasm,
        scale
    )

    sin_zones = []

    if len(top_endpoints) < 3:
        return sin_zones, {
            "top_endpoints": top_endpoints,
            "status": "TOP_ENDPOINT_KURANG_DARI_3"
        }

    top_sorted = sorted(
        top_endpoints,
        key=lambda p: p[1],
        reverse=True
    )

    max_gap_gigi = max(10, int(35 * scale))
    margin_x = max(5, int(10 * scale))

    groups = []
    current_group = [top_sorted[0]]

    for p in top_sorted[1:]:
        prev = current_group[-1]
        gap = abs(prev[1] - p[1])

        if gap <= max_gap_gigi:
            current_group.append(p)
        else:
            if len(current_group) >= 3:
                groups.append(current_group)

            current_group = [p]

    if len(current_group) >= 3:
        groups.append(current_group)

    for group in groups:
        group_arr = np.array(group)

        x_min = int(np.min(group_arr[:, 1]) - margin_x)
        x_max = int(np.max(group_arr[:, 1]) + margin_x)

        x_min = max(x_min, min_x)
        x_max = min(x_max, max_x)

        span = x_max - x_min

        if span >= 10 * scale and span <= max(120 * scale, 0.65 * w):
            sin_zones.append((x_min, x_max))

    status = "SIN_TERDETEKSI" if sin_zones else "POLA_GIGI_TIDAK_VALID"

    return sin_zones, {
        "top_endpoints": top_endpoints,
        "groups": groups,
        "status": status
    }


def kelompokkan_intersection_berdasarkan_x(raw_intersections, scale):
    if raw_intersections is None or len(raw_intersections) == 0:
        return []

    x_tolerance = max(2, int(3 * scale))

    points = sorted(
        raw_intersections,
        key=lambda p: p[1]
    )

    groups = []
    current = [points[0]]

    for p in points[1:]:
        if abs(p[1] - current[-1][1]) <= x_tolerance:
            current.append(p)
        else:
            arr = np.array(current)
            groups.append({
                "points": arr,
                "x": int(np.round(np.mean(arr[:, 1]))),
                "y": int(np.round(np.mean(arr[:, 0])))
            })

            current = [p]

    arr = np.array(current)
    groups.append({
        "points": arr,
        "x": int(np.round(np.mean(arr[:, 1]))),
        "y": int(np.round(np.mean(arr[:, 0])))
    })

    return groups


def berada_di_zona_sin(x, sin_zones):
    for x_min, x_max in sin_zones:
        if x_min <= x <= x_max:
            return True

    return False

def deteksi_kandidat_ain_nga_group(mask_group, body_group, dot_mask, skeleton_body, scale):
    """
    Deteksi awal keluarga bentuk AIN/NGA sebelum pemotongan.

    AIN dan NGA memiliki body yang mirip.
    Perbedaan utama:
    - NGA punya estimasi 3 titik atas
    - AIN tidak punya titik atas
    """

    atas_estimasi, num_blob_atas, info_atas = hitung_estimasi_diakritik_final(
        mask_group,
        dot_mask,
        scale,
        jenis_target="ATAS"
    )

    body_coords = np.argwhere(body_group)

    if len(body_coords) == 0:
        return False, {
            "status": "BODY_KOSONG",
            "atas_estimasi": atas_estimasi
        }

    min_y, min_x = body_coords.min(axis=0)
    max_y, max_x = body_coords.max(axis=0)

    h_body = max_y - min_y
    w_body = max_x - min_x
    aspect_body = w_body / (h_body + 1e-5)

    _, intersections, _ = find_endpoints(skeleton_body)
    loops = deteksi_loop(skeleton_body)

    punya_struktur_ain_nga = (
        len(intersections) > 0 or len(loops) > 0
    )

    # Kandidat NGA: body cukup kompak + punya 3 titik atas
    is_nga_like = (
        atas_estimasi >= 3 and
        h_body <= 65 * scale and
        w_body >= 12 * scale and
        w_body <= 95 * scale and
        aspect_body >= 0.35
    )

    # Kandidat AIN: body kompak + tanpa titik atas + punya struktur loop/intersection
    is_ain_like = (
        atas_estimasi == 0 and
        punya_struktur_ain_nga and
        h_body <= 60 * scale and
        w_body >= 10 * scale and
        w_body <= 55 * scale and
        aspect_body >= 0.30
    )

    is_ain_nga_like = is_nga_like or is_ain_like

    if is_nga_like:
        status = "NGA_LIKE"
    elif is_ain_like:
        status = "AIN_LIKE"
    else:
        status = "BUKAN_AIN_NGA"

    return is_ain_nga_like, {
        "status": status,
        "atas_estimasi": atas_estimasi,
        "num_blob_atas": num_blob_atas,
        "h_body": h_body,
        "w_body": w_body,
        "aspect_body": aspect_body,
        "jumlah_intersection": len(intersections),
        "jumlah_loop": len(loops)
    }

def deteksi_kandidat_sin_syin_group(body_group, skeleton_body, scale):
    """
    Deteksi awal keluarga SIN/SYIN sebelum pemotongan.

    Prinsip baru:
    - Struktur lebih penting daripada ukuran.
    - SIN/SYIN valid jika punya 2-3 gigi dan 3 intersection utama.
    - Titik potong adalah intersection nomor 3 dari kanan.
    """

    body_coords = np.argwhere(body_group)

    if len(body_coords) == 0:
        return False, {
            "status": "BODY_KOSONG",
            "cut_x": None,
            "cut_y": None
        }

    min_y, min_x = body_coords.min(axis=0)
    max_y, max_x = body_coords.max(axis=0)

    h_body = max_y - min_y
    w_body = max_x - min_x
    aspect_body = w_body / (h_body + 1e-5)

    endpoints, intersections, _ = find_endpoints(skeleton_body)

    intersection_groups = kelompokkan_intersection_berdasarkan_x(
        intersections,
        scale
    )

    jumlah_intersection_group = len(intersection_groups)

    if len(endpoints) == 0:
        return False, {
            "status": "TIDAK_ADA_ENDPOINT",
            "cut_x": None,
            "cut_y": None,
            "h_body": h_body,
            "w_body": w_body,
            "aspect_body": aspect_body,
            "jumlah_gigi_atas": 0,
            "jumlah_intersection_group": jumlah_intersection_group,
            "jumlah_intersection_raw": len(intersections)
        }

    # Dibuat longgar agar gigi tulisan tangan tetap masuk.
    batas_atas = min_y + (h_body * 0.80)

    top_endpoints = np.array([
        ep for ep in endpoints
        if ep[0] <= batas_atas
    ])

    # Kelompokkan endpoint atas agar noise skeleton tidak dihitung sebagai gigi baru.
    gigi_groups = []

    if len(top_endpoints) > 0:
        titik_gigi = sorted(top_endpoints, key=lambda p: p[1])
        gap_gigi = max(4, int(8 * scale))

        current = [titik_gigi[0]]

        for p in titik_gigi[1:]:
            if abs(p[1] - current[-1][1]) <= gap_gigi:
                current.append(p)
            else:
                gigi_groups.append(current)
                current = [p]

        gigi_groups.append(current)

    jumlah_gigi_atas = len(gigi_groups)

    # ==========================================================
    # ATURAN STRUKTUR UTAMA
    # ==========================================================
    struktur_sin_syin_ok = (
        jumlah_gigi_atas >= 2
        and jumlah_gigi_atas <= 3
        and jumlah_intersection_group == 3
    )

    # ==========================================================
    # FILTER DIMENSI LONGGAR
    # Jangan terlalu bergantung pada scale, karena scale pada data ini
    # membuat karakter valid seperti Region 3 tertolak.
    # ==========================================================
    dimensi_ok = (
        h_body <= max(70 * scale, 55)
        and w_body >= max(18 * scale, 18)
        and w_body <= max(140 * scale, 140)
        and aspect_body >= 0.40
    )

    is_sin_syin_like = struktur_sin_syin_ok and dimensi_ok

    cut_x = None
    cut_y = None

    if is_sin_syin_like:
        groups_kanan_ke_kiri = sorted(
            intersection_groups,
            key=lambda g: g["x"],
            reverse=True
        )

        # Intersection nomor 3 dari kanan.
        target_group = groups_kanan_ke_kiri[2]

        cut_x = int(target_group["x"] + max(1, int(2 * scale)))
        cut_y = int(target_group["y"])

    return is_sin_syin_like, {
        "status": "SIN_SYIN_LIKE" if is_sin_syin_like else "BUKAN_SIN_SYIN",
        "cut_x": cut_x,
        "cut_y": cut_y,
        "h_body": h_body,
        "w_body": w_body,
        "aspect_body": aspect_body,
        "jumlah_gigi_atas": jumlah_gigi_atas,
        "jumlah_intersection_group": jumlah_intersection_group,
        "jumlah_intersection_raw": len(intersections),
        "struktur_sin_syin_ok": struktur_sin_syin_ok,
        "dimensi_ok": dimensi_ok,
        "intersection_groups": intersection_groups
    }
def validasi_cut_points_berdasarkan_luas(body_rasm, cut_points, min_x, max_x, scale):
    cut_points = sorted(list(set([
        int(c) for c in cut_points
        if min_x < int(c) < max_x
    ])))

    if not cut_points:
        return []

    min_area_segment = max(15, int(25 * (scale ** 2)))

    cleaned_cut_points = cut_points.copy()
    berubah = True

    while berubah and cleaned_cut_points:
        berubah = False

        batas = [min_x] + cleaned_cut_points + [max_x + 1]

        for i in range(len(batas) - 1):
            x_start = batas[i]
            x_end = batas[i + 1]

            area = np.sum(body_rasm[:, x_start:x_end])

            if area < min_area_segment:
                if i == 0:
                    cleaned_cut_points.pop(0)
                elif i == len(batas) - 2:
                    cleaned_cut_points.pop(-1)
                else:
                    lebar_kiri = batas[i] - batas[i - 1]
                    lebar_kanan = batas[i + 1] - batas[i]

                    if lebar_kiri <= lebar_kanan:
                        cleaned_cut_points.pop(i - 1)
                    else:
                        cleaned_cut_points.pop(i)

                berubah = True
                break

    return cleaned_cut_points

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

def cek_sin_syin_final(seg_mask, scale):
    """
    Deteksi bentuk badan SIN/SYIN pada tahap klasifikasi final.

    Aturan:
    - SIN  = bentuk SIN/SYIN + tanpa diakritik atas.
    - SYIN = bentuk SIN/SYIN + 1 sampai 3 diakritik atas.
    - Bentuk badan memiliki 2-3 gigi/end point atas.
    - Untuk final classification dibuat lebih fleksibel,
      karena setelah pemotongan, jumlah intersection bisa berkurang.
    """

    body = seg_mask > 0
    coords = np.argwhere(body)

    if len(coords) == 0:
        return False, {
            "status": "BODY_KOSONG",
            "gigi": 0,
            "intersection_group": 0
        }

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    h_body = max_y - min_y
    w_body = max_x - min_x
    aspect_body = w_body / (h_body + 1e-5)

    skel = skeletonize(body)
    endpoints, intersections, _ = find_endpoints(skel)

    intersection_groups = kelompokkan_intersection_berdasarkan_x(
        intersections,
        scale
    )

    # Ambil endpoint bagian atas.
    # Dibuat longgar karena gigi tulisan tangan kadang turun.
    batas_atas = min_y + (h_body * 0.80)

    top_endpoints = np.array([
        ep for ep in endpoints
        if ep[0] <= batas_atas
    ])

    # Kelompokkan endpoint berdasarkan x agar noise tidak dihitung gigi baru.
    gigi_groups = []

    if len(top_endpoints) > 0:
        titik_gigi = sorted(top_endpoints, key=lambda p: p[1])
        gap_gigi = max(4, int(8 * scale))

        current = [titik_gigi[0]]

        for p in titik_gigi[1:]:
            if abs(p[1] - current[-1][1]) <= gap_gigi:
                current.append(p)
            else:
                gigi_groups.append(current)
                current = [p]

        gigi_groups.append(current)

    jumlah_gigi = len(gigi_groups)
    jumlah_intersection_group = len(intersection_groups)

    # Filter ukuran dibuat berdasarkan pixel minimum juga,
    # karena scale pada data ini bisa membuat threshold terlalu kecil/besar.
    dimensi_ok = (
        h_body <= max(70 * scale, 55)
        and w_body >= max(18 * scale, 28)
        and w_body <= max(140 * scale, 140)
        and aspect_body >= 0.40
    )

    # Bentuk SIN/SYIN final:
    # - 2 sampai 3 gigi
    # - punya minimal 1 intersection group, atau gigi tepat 3
    #   karena setelah dipotong intersection kadang berkurang.
    struktur_ok = (
        jumlah_gigi >= 2
        and jumlah_gigi <= 3
        and (
            jumlah_intersection_group >= 1
            or jumlah_gigi == 3
        )
    )

    is_sin_syin = dimensi_ok and struktur_ok

    return is_sin_syin, {
        "status": "SIN_SYIN_FINAL" if is_sin_syin else "BUKAN_SIN_SYIN_FINAL",
        "gigi": jumlah_gigi,
        "intersection_group": jumlah_intersection_group,
        "h": h_body,
        "w": w_body,
        "aspect": aspect_body,
        "dimensi_ok": dimensi_ok,
        "struktur_ok": struktur_ok
    }

def cek_ain_final(seg_mask, h_seg, w_seg, scale):
    """
    Deteksi AIN final.
    AIN pada data ini cenderung melebar dengan ekor panjang,
    tanpa diakritik.
    """

    if h_seg > max(55 * scale, 50):
        return False

    if w_seg < max(60 * scale, 65):
        return False

    aspect = w_seg / (h_seg + 1e-5)

    if aspect < 2.0:
        return False

    skel = skeletonize(seg_mask > 0)
    endpoints, intersections, _ = find_endpoints(skel)

    return len(endpoints) >= 1


def cek_ra_final(seg_mask, h_seg, w_seg, scale):
    """
    Deteksi RA final.
    RA umumnya tanpa titik, kecil-sedang, terbuka, dan tidak memiliki loop kuat.
    """

    if h_seg > max(50 * scale, 45):
        return False

    if w_seg < max(20 * scale, 25) or w_seg > max(65 * scale, 65):
        return False

    skel = skeletonize(seg_mask > 0)
    _, intersections, _ = find_endpoints(skel)
    loops = deteksi_loop(skel)

    if len(loops) > 0:
        return False

    if len(intersections) > 1:
        return False

    return True


def cek_waw_final(seg_mask, h_seg, w_seg, scale, sin_syin_info=None):
    """
    Deteksi WAW final.
    WAW harus berbentuk curl/loop atau lengkung membulat.
    Bentuk bergigi seperti SIN tidak boleh masuk WAW.
    """

    if sin_syin_info is not None:
        gigi = sin_syin_info.get("gigi", 0)
        struktur_ok = sin_syin_info.get("struktur_ok", False)

        # Kalau punya pola gigi SIN/SYIN, jangan klasifikasikan sebagai WAW
        if struktur_ok and gigi >= 2:
            return False

    if h_seg > max(55 * scale, 50):
        return False

    if w_seg < max(25 * scale, 28) or w_seg > max(70 * scale, 70):
        return False

    skel = skeletonize(seg_mask > 0)
    endpoints, intersections, _ = find_endpoints(skel)
    loops = deteksi_loop(skel)

    aspect = w_seg / (h_seg + 1e-5)

    # WAW kuat: punya loop/curl
    if len(loops) > 0:
        return True

    # Fallback WAW harus lebih ketat:
    # jangan hanya intersection >= 1, karena SIN juga punya intersection.
    if (
        0.75 <= aspect <= 1.8
        and len(intersections) >= 1
        and len(endpoints) <= 2
    ):
        return True

    return False
def cek_nun(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if not posisi_titik.startswith("ATAS"):
        return False

    # Lebih aman: gabungkan threshold berbasis scale dan pixel absolut
    max_h_nun = max(80 * scale, 45)
    max_w_nun = max(75 * scale, 55)

    if h_seg >= max_h_nun:
        return False

    if w_seg > max_w_nun:
        return False

    # NUN ideal: mangkuk cukup lebar dan punya satu titik atas
    if w_seg > h_seg * 0.80 and w_seg > max(20 * scale, 20):
        return True

    # Fallback bentuk mangkuk
    if w_seg >= max(25 * scale, 25):
        gigi_atas = analisis_ujung_atas(seg_mask)
        if gigi_atas <= 2:
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


# ================================================================================
# ESTIMASI DAN ALOKASI DIAKRITIK KE SEGMEN HURUF
# ================================================================================

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
    # Dibuat lebih hati-hati agar titik TA tidak salah menjadi 3.
    # ==========================================================
    jumlah = 1
    
    # Jika benar-benar terlihat 3 run terpisah, langsung 3.
    if run_count >= 3:
        jumlah = 3
    
    # Jika terlihat 2 run terpisah, langsung 2.
    elif run_count == 2:
        jumlah = 2
    
    # Blob melebar boleh dianggap 3 hanya jika sangat horizontal.
    # Ini menjaga NGA tetap bisa 3, tetapi TA tidak mudah naik menjadi 3.
    elif (
        dot_width >= max(12 * scale, 12)
        and aspect >= 1.65
        and dot_area >= max(12 * (scale ** 2), 40)
    ):
        jumlah = 3
    
    # Blob cukup melebar tetapi tidak sangat horizontal: anggap 2.
    elif (
        dot_width >= max(7 * scale, 7)
        and aspect >= 1.05
        and dot_area >= max(8 * (scale ** 2), 20)
    ):
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

            pass  # debug print dihapus

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

        pass  # debug print dihapus

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

            pass  # debug print dihapus

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

            pass  # debug print dihapus

    return clean_segments


# ================================================================================
# SEGMENTASI DAN PEMOTONGAN KARAKTER
# ================================================================================

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
    target_freeman_xrange=None,
    debug_cut_records=None
):
    hasil_potongan = []

    num_features = np.max(labeled_letters)

    labeled_dots, num_dots = measure.label(
        dot_mask > 0,
        return_num=True,
        connectivity=2
    )

    dots_props = list(measure.regionprops(labeled_dots))

    # Debug sementara untuk melihat kenapa SIN/SYIN masuk atau tidak.
    # Setelah stabil, boleh ubah menjadi False.
    DEBUG_SIN_SYIN = False

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

        # ==========================================================
        # BODY RASM TANPA DIAKRITIK
        # Ini penting supaya titik/diakritik tidak ikut membentuk
        # intersection palsu.
        # ==========================================================
        body_group = mask_group & (dot_mask == 0)
        body_coords = np.argwhere(body_group)

        if len(body_coords) == 0:
            continue

        min_y_body, min_x_body = body_coords.min(axis=0)
        max_y_body, max_x_body = body_coords.max(axis=0)

        h_body = max_y_body - min_y_body
        w_body = max_x_body - min_x_body

        x_coords = body_coords[:, 1]

        letter_skeleton = skeleton_used & body_group

        if np.sum(letter_skeleton) < 5:
            letter_skeleton = skeletonize(body_group)

        _, raw_intersections, _ = find_endpoints(letter_skeleton)

        fitur_kumpulan = deteksi_loop(letter_skeleton)

        cycle_bounds = []

        for cycle, _ in fitur_kumpulan:
            cycle_arr = np.array(cycle)
            cycle_bounds.append(
                (cycle_arr[:, 1].min(), cycle_arr[:, 1].max())
            )

        # ==========================================================
        # DETEKSI AIN / NGA
        # AIN dan NGA diproteksi dari pemotongan internal.
        # ==========================================================
        atas_estimasi_group, num_blob_atas_group, info_atas_group = hitung_estimasi_diakritik_final(
            mask_group,
            dot_mask,
            scale,
            jenis_target="ATAS"
        )

        aspect_body = w_body / (h_body + 1e-5)

        jumlah_intersection_group = len(raw_intersections) if raw_intersections is not None else 0
        jumlah_loop_group = len(fitur_kumpulan)

        punya_struktur_ain_nga = (
            jumlah_intersection_group > 0 or jumlah_loop_group > 0
        )

        is_nga_like_group = (
            atas_estimasi_group >= 3
            and h_body <= 70 * scale
            and w_body >= 10 * scale
            and w_body <= 110 * scale
            and aspect_body >= 0.25
        )

        is_ain_like_group = (
            atas_estimasi_group == 0
            and punya_struktur_ain_nga
            and h_body <= 65 * scale
            and w_body >= 8 * scale
            and w_body <= 70 * scale
            and aspect_body >= 0.25
        )

        is_ain_nga_like_group = is_nga_like_group or is_ain_like_group

        if is_nga_like_group:
            status_ain_nga = "NGA_LIKE"
        elif is_ain_like_group:
            status_ain_nga = "AIN_LIKE"
        else:
            status_ain_nga = "BUKAN_AIN_NGA"

        # ==========================================================
        # DETEKSI SIN / SYIN
        # WAJIB dicek sebelum AIN/NGA di blok pemotongan.
        # SIN tidak punya diakritik sehingga bisa salah masuk AIN_LIKE.
        # ==========================================================
        is_sin_syin_like_group, sin_syin_info = deteksi_kandidat_sin_syin_group(
            body_group,
            letter_skeleton,
            scale
        )

        if DEBUG_SIN_SYIN:
            pass  # debug print dihapus

        cut_points = []

        # ==========================================================
        # 1. PEMOTONGAN UTAMA BERDASARKAN INTERSECTION / TOPOLOGI
        # ==========================================================

        # ----------------------------------------------------------
        # PRIORITAS 1: SIN / SYIN
        # Untuk SIN/SYIN, tidak semua intersection dipotong.
        # Hanya intersection nomor 3 dari kanan yang menjadi cut.
        # ----------------------------------------------------------
        if is_sin_syin_like_group:
            cut_pos = sin_syin_info.get("cut_x")
            cut_y = sin_syin_info.get("cut_y")

            if cut_pos is not None:
                cut_pos = int(cut_pos)

                if (
                    cut_pos - min_x_body >= 10 * scale
                    and max_x_body - cut_pos >= 10 * scale
                ):
                    cut_points.append(cut_pos)

                    pass  # debug print dihapus

                else:
                    pass  # debug print dihapus

            else:
                pass  # debug print dihapus

        # ----------------------------------------------------------
        # PRIORITAS 2: WAW / HA / AIN / NGA
        # Untuk karakter ini, intersection internal tidak dipakai
        # sebagai titik potong.
        # ----------------------------------------------------------
        elif (
            cek_waw(body_group, h_body, w_body, scale)
            or cek_ha(body_group, h_body, w_body, scale)
            or is_ain_nga_like_group
        ):
            if is_ain_nga_like_group:
                pass  # debug print dihapus

            pass

        # ----------------------------------------------------------
        # PRIORITAS 3: PEMOTONGAN INTERSECTION UMUM
        # Untuk selain SIN/SYIN, AIN/NGA, WAW, HA.
        # ----------------------------------------------------------
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

                if (
                    first_cut - min_x_body >= 15 * scale
                    and max_x_body - first_cut >= 15 * scale
                ):
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

                        if (
                            cut_pos - min_x_body >= 15 * scale
                            and max_x_body - cut_pos >= 15 * scale
                        ):
                            cut_points.append(cut_pos)

        # ==========================================================
        # 2. PEMOTONGAN TAMBAHAN MENGGUNAKAN FREEMAN CHAIN CODE
        # ==========================================================
        freeman_cut_col = None

        if target_freeman_xrange is not None:
            target_x_min, target_x_max = target_freeman_xrange

            is_target_freeman = not (
                max_x_body < target_x_min or min_x_body > target_x_max
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

                    if min_x_body < freeman_cut_col < max_x_body:
                        cut_points.append(freeman_cut_col)

                        if debug_freeman is not None and freeman_info is not None:
                            freeman_info["region_label"] = region_label
                            freeman_info["cut_col"] = freeman_cut_col
                            freeman_info["target_xrange"] = target_freeman_xrange
                            debug_freeman.append(freeman_info)

                        pass  # debug print dihapus

                    else:
                        pass  # debug print dihapus

                else:
                    pass  # debug print dihapus

        # ==========================================================
        # 3. PEMBENTUKAN BATAS POTONG AKTUAL
        # ==========================================================
        cuts = []
        final_valid_cuts = []

        if cut_points:
            minc, maxc = x_coords.min(), x_coords.max()

            cut_points = validasi_cut_points_berdasarkan_luas(
                body_rasm=body_group,
                cut_points=cut_points,
                min_x=minc,
                max_x=maxc,
                scale=scale
            )

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

            pass  # debug print dihapus
            pass  # debug print dihapus

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

                elif bentuk_baseline_panjang and jumlah_diakritik_atas_seg <= 1:
                    tipe_huruf = "PENOLAK_DIAKRITIK"

                # ==================================================
                # KELUARGA SIN / SYIN
                # SYIN boleh 1-3 diakritik atas sesuai catatan Anda.
                # ==================================================
                elif cek_sin(segment_mask, h_seg, scale) and jumlah_diakritik_atas_seg >= 1:
                    tipe_huruf = "SYIN"

                elif cek_sin(segment_mask, h_seg, scale):
                    tipe_huruf = "SIN"

                # ==================================================
                # HURUF BERTITIK ATAS
                # ==================================================
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
                    tipe_huruf = "NGA"

                elif cek_nun(
                    segment_mask,
                    h_seg,
                    w_seg,
                    posisi_titik,
                    scale
                ) and jumlah_diakritik_atas_seg == 1:
                    tipe_huruf = "NUN"

                elif jumlah_diakritik_atas_seg == 2:
                    tipe_huruf = "TA"

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
                # ==================================================
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

                pass  # debug print dihapus

        if not clean_segments:
            hasil_potongan.append(mask_group)
            continue

        # ==========================================================
        # 5. ALOKASI DIAKRITIK
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

        # ==========================================================
        # 6. DEBUG VISUAL GARIS POTONG AKTUAL
        # Hanya mencatat cut yang benar-benar menghasilkan dua segmen.
        # ==========================================================
        if debug_cut_records is not None and len(clean_segments) >= 2 and final_valid_cuts:
            accepted_bboxes = []

            for seg in clean_segments:
                body_seg = seg & (dot_mask == 0)
                seg_coords_dbg = np.argwhere(body_seg)

                if len(seg_coords_dbg) == 0:
                    continue

                sy_min, sx_min = seg_coords_dbg.min(axis=0)
                sy_max, sx_max = seg_coords_dbg.max(axis=0)

                accepted_bboxes.append(
                    (
                        int(sx_min),
                        int(sx_max),
                        int(sy_min),
                        int(sy_max)
                    )
                )

            raw_arr = (
                np.array(raw_intersections)
                if raw_intersections is not None and len(raw_intersections) > 0
                else np.array([])
            )

            for c in final_valid_cuts:
                c = int(c)

                ada_kiri = any(
                    xmax < c
                    for xmin, xmax, ymin, ymax in accepted_bboxes
                )

                ada_kanan = any(
                    xmin >= c
                    for xmin, xmax, ymin, ymax in accepted_bboxes
                )

                if not (ada_kiri and ada_kanan):
                    continue

                sumber_cut = "INTERSECTION"

                if freeman_cut_col is not None and abs(c - freeman_cut_col) <= max(2, int(2 * scale)):
                    sumber_cut = "FREEMAN"

                intersection_y = None
                intersection_x = None

                if raw_arr.size > 0:
                    dists = np.abs(raw_arr[:, 1] - c)
                    idx_best = int(np.argmin(dists))

                    if dists[idx_best] <= max(8, int(6 * scale)):
                        intersection_y = int(raw_arr[idx_best][0])
                        intersection_x = int(raw_arr[idx_best][1])

                if intersection_y is None:
                    local_band = letter_skeleton[
                        :,
                        max(0, c - int(2 * scale)):min(
                            letter_skeleton.shape[1],
                            c + int(2 * scale) + 1
                        )
                    ]

                    local_coords = np.argwhere(local_band)

                    if len(local_coords) > 0:
                        intersection_y = int(np.median(local_coords[:, 0]))
                    else:
                        intersection_y = int((min_y_grp + max_y_grp) / 2)

                debug_cut_records.append({
                    "region_label": int(region_label),
                    "cut_col": int(c),
                    "source": sumber_cut,
                    "min_y": int(min_y_body),
                    "max_y": int(max_y_body),
                    "min_x": int(min_x_body),
                    "max_x": int(max_x_body),
                    "intersection_y": intersection_y,
                    "intersection_x": intersection_x,
                })

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
            pass  # debug print dihapus
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

            pass  # debug print dihapus

        pass  # debug print dihapus

    return fixed_segments


# ================================================================================
# B-SPLINE CURVE FITTING DAN METRIK REKONSTRUKSI
# ================================================================================

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


# ================================================================================
# EVALUASI PEMOTONGAN RASM
# ================================================================================

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


# ================================================================================
# MAIN EXECUTION: ALUR PRA-PEMROSESAN → SEGMENTASI → VISUALISASI → EVALUASI
# ================================================================================

image_path = r"E:\progres\p01v2-line1.png"

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
cut_debug_records = []

# Area target H6 / fa-alif berdasarkan visualisasi utama
TARGET_FREEMAN_XRANGE = (290, 320)

hasil_potongan = potong_persimpangan(
    labeled_letters,
    skeleton_used,
    dot_mask,
    scale,
    debug_freeman=freeman_debug_records,
    target_freeman_xrange=TARGET_FREEMAN_XRANGE,
    debug_cut_records=cut_debug_records
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
    
    is_sin_syin_final, sin_syin_final_info = cek_sin_syin_final(
        badan_huruf,
        scale
    )
    
    gigi_final = sin_syin_final_info.get("gigi", 0)
    intersection_final = sin_syin_final_info.get("intersection_group", 0)
    dimensi_sin_ok = sin_syin_final_info.get("dimensi_ok", False)
    
    syin_kuat = (
        is_sin_syin_final
        and dimensi_sin_ok
        and gigi_final >= 3
        and intersection_final >= 2
    )
    
    sin_kuat = (
        estimasi_atas_final == 0
        and estimasi_bawah_final == 0
        and dimensi_sin_ok
        and h_seg <= max(45 * scale, 40)
        and w_seg <= max(65 * scale, 65)
        and (
            gigi_final >= 3
            or (
                gigi_final >= 2
                and intersection_final >= 1
            )
        )
    )
    
    # ==========================================================
    # 1. HURUF TANPA DIAKRITIK
    # ==========================================================
    if estimasi_atas_final == 0 and estimasi_bawah_final == 0:

        if cek_dal(badan_huruf, h_seg, w_seg, scale):
            tipe_final = "DAL"
    
        elif cek_alif(badan_huruf, h_seg, w_seg, "TIDAK_ADA", scale):
            tipe_final = "ALIF"
    
        elif cek_lam(h_seg, w_seg, scale):
            tipe_final = "LAM"
    
        elif cek_ain_final(badan_huruf, h_seg, w_seg, scale):
            tipe_final = "AIN"
    
        # SIN harus dicek sebelum WAW
        elif sin_kuat:
            tipe_final = "SIN"
    
        elif cek_waw_final(badan_huruf, h_seg, w_seg, scale):
            tipe_final = "WAW"
    
        elif cek_ra_final(badan_huruf, h_seg, w_seg, scale):
            tipe_final = "RA"
    
    # ==========================================================
    # 2. HURUF DENGAN 1 TITIK ATAS
    # Urutan penting: FA dulu, lalu SYIN kuat, lalu NUN.
    # ==========================================================
    elif estimasi_atas_final == 1 and estimasi_bawah_final == 0:
    
        if cek_fa(badan_huruf, h_seg, w_seg, "ATAS_SINGLE", scale):
            tipe_final = "FA"
    
        elif syin_kuat:
            tipe_final = "SYIN"
    
        elif cek_nun(badan_huruf, h_seg, w_seg, "ATAS_SINGLE", scale):
            tipe_final = "NUN"
    
        else:
            tipe_final = "SATU_TITIK_ATAS"
    
    # ==========================================================
    # 3. HURUF DENGAN 2 TITIK ATAS
    # ==========================================================
    elif estimasi_atas_final == 2 and estimasi_bawah_final == 0:
        tipe_final = "TA"
    
    # ==========================================================
    # 4. HURUF DENGAN 3 TITIK ATAS
    # ==========================================================
    elif estimasi_atas_final >= 3 and estimasi_bawah_final == 0:
    
        if syin_kuat:
            tipe_final = "SYIN"
    
        elif cek_ain(
            badan_huruf,
            h_seg,
            w_seg,
            posisi_titik_final,
            scale
        ):
            tipe_final = "NGA"
    
        else:
            tipe_final = "TIGA_TITIK_ATAS"
    
    # ==========================================================
    # 5. HURUF BERTITIK BAWAH
    # ==========================================================
    elif estimasi_bawah_final == 1:
        tipe_final = "BA"
    
    elif estimasi_bawah_final >= 2:
        tipe_final = "YA"
        
    print(
    f"H{idx+1}: "
    f"tipe_final={tipe_final}, "
    f"atas={estimasi_atas_final}, "
    f"bawah={estimasi_bawah_final}, "
    f"x=({min_x_seg},{max_x_seg}), "
    f"h={h_seg}, "
    f"w={w_seg}"
)
    
    # PENAMPUNG DATA UNTUK TABEL EVALUASI AKHIR
    eval_tsp = []
    eval_bspline = []

# --- 1. VISUALISASI UTAMA ---
fig, ax = plt.subplots(
    figsize=figsize_dinamis(
        skeleton_used,
        px_per_inch=45,
        min_w=10,
        max_w=22,
        min_h=4,
        max_h=9
    ),
    dpi=120
)
ax.imshow(skeleton_used, cmap='gray')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, skeleton_used.shape[1])
ax.set_ylim(skeleton_used.shape[0], 0)

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

# ==========================================================
# GARIS POTONG AKTUAL / SINKRON
# Menampilkan garis potong putih seperti visual yang Anda inginkan:
# - memakai kolom cut yang BENAR-BENAR dipakai saat pemotongan,
# - garis digambar full-height pada canvas utama,
# - cut Freeman tetap tidak digambar putih karena sudah punya garis cyan sendiri.
# ==========================================================
TAMPILKAN_GARIS_POTONG_AKTUAL = True

if TAMPILKAN_GARIS_POTONG_AKTUAL:
    sudah_label_garis_potong = False
    kolom_sudah_digambar = []
    toleransi_merge_kolom = max(1, int(2 * scale))

    for rec in cut_debug_records:
        # Freeman sudah divisualisasikan sendiri dengan garis cyan.
        if rec.get("source") == "FREEMAN":
            continue

        col = int(rec["cut_col"])
        cy = rec.get("intersection_y")
        region_label_cut = rec.get("region_label")

        # Pastikan cut benar-benar terkait badan huruf/rasm.
        body_vis = (labeled_letters == region_label_cut) & (dot_mask == 0)
        band = max(1, int(2 * scale))
        x1_band = max(0, col - band)
        x2_band = min(skeleton_used.shape[1], col + band + 1)
        local_body = body_vis[:, x1_band:x2_band]
        body_y = np.where(np.any(local_body, axis=1))[0]

        # Jika di sekitar kolom ini tidak ada badan huruf, abaikan.
        if len(body_y) == 0:
            continue

        # Hindari menggambar garis ganda pada kolom yang sangat berdekatan.
        if any(abs(col - c_prev) <= toleransi_merge_kolom for c_prev in kolom_sudah_digambar):
            continue

        ax.axvline(
            x=col,
            color="white",
            linestyle="--",
            linewidth=1.5,
            alpha=0.95,
            zorder=10,
            label="Garis Potong" if not sudah_label_garis_potong else "_nolegend_"
        )

        # Tampilkan lingkaran putih kecil pada titik intersection/cut referensi,
        # selama titik itu bukan diakritik.
        if cy is not None:
            cy = int(cy)
            if 0 <= cy < dot_mask.shape[0] and 0 <= col < dot_mask.shape[1]:
                if dot_mask[cy, col] == 0:
                    ax.scatter(
                        col,
                        cy,
                        s=34,
                        facecolors="none",
                        edgecolors="white",
                        linewidths=1.1,
                        zorder=19
                    )

        kolom_sudah_digambar.append(col)
        sudah_label_garis_potong = True


# --- VISUALISASI AREA TARGET FREEMAN H6 / FA-ALIF ---
if "TARGET_FREEMAN_XRANGE" in globals():
    ax.axvspan(
        TARGET_FREEMAN_XRANGE[0],
        TARGET_FREEMAN_XRANGE[1],
        color="cyan",
        alpha=0.15,
        label="Area Target Freeman"
    )

# --- VISUALISASI OVERLAY FREEMAN CHAIN CODE PADA VISUALISASI UTAMA ---
overlay_fcc_pada_visualisasi_utama(
    ax,
    freeman_debug_records,
    label_step=5
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

urutan_legend = [
    "Intersection",
    "Loop",
    "Diakritik Atas",
    "Diakritik Bawah",
    "Garis Potong",
    "Area Target Freeman",
    "Freeman Path Arah Goresan",
    "Candidate Point Potong",
    "Selected Candidate",
    "Garis Potong Freeman"
]

handles_final = []
labels_final = []

for label in urutan_legend:
    if label in by_label:
        handles_final.append(by_label[label])
        labels_final.append(label)

ax.legend(
    handles_final,
    labels_final,
    loc='upper right',
    fontsize=9,
    framealpha=0.85
)

ax.set_title("Visualisasi Utama Segmentasi Karakter")
ax.axis('off')
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
jumlah_GT_rasm = 12

eval_rasm = evaluasi_pemotongan_rasm(
    sorted_hasil_potongan,
    dot_mask,
    jumlah_GT_rasm
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
    fig_w, fig_h = figsize_dinamis(
        region_mask,
        px_per_inch=45,
        min_w=5,
        max_w=9,
        min_h=2.8,
        max_h=5
    )
    fig, ax = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h), dpi=120)
    
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
    
    fig, ax = plt.subplots(
        figsize=figsize_dinamis(
            region_mask,
            px_per_inch=45,
            min_w=5,
            max_w=9,
            min_h=3,
            max_h=5
        ),
        dpi=120
    )
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
jumlah_potongan_visual = max(1, len(sorted_hasil_potongan))
fig, axes = plt.subplots(
    1,
    jumlah_potongan_visual,
    figsize=(min(3 * jumlah_potongan_visual, 24), 4),
    dpi=120
)
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
