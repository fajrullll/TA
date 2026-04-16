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

# ==================== METODE B-SPLINE ====================

def basis_function(i, k, x, knots):
    """Menghitung fungsi basis N_{i,k}(x) secara rekursif"""
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
    """Membangun kurva dari titik kontrol dan derajat"""
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

    max_area = max([p.area for p in props]) if props else 0
    area_main_dyn = max(A_diac_max, 0.12 * max_area)

    main_infos = {}
    dot_mask = np.zeros_like(binary, dtype=np.uint8)
    label_map = labeled_image.copy()

    jarak_maks = 60 * scale 

    for prop in props:
        area = prop.area
        bbox = prop.bbox
        height = bbox[2] - bbox[0]
        width  = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 1e-5)

        if area > area_main_dyn or (area > 10 * (scale**2) and aspect_ratio > 3.5):
            main_infos[prop.label] = {"centroid": prop.centroid, "bbox": bbox}

    for prop in props:
        label = prop.label
        area  = prop.area
        if label in main_infos: continue

        bbox = prop.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = height / (width + 1e-5)

        if aspect_ratio > rasio_alif: continue

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

    min_height = 15 * scale 
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

def cek_lam(h_seg, w_seg, scale):
    return h_seg >= 45 * scale and h_seg > w_seg * 1.2

def cek_alif(seg_mask, h_seg, w_seg, posisi_titik, scale):
    if posisi_titik != "TIDAK_ADA": return False
    if h_seg >= 30 * scale and w_seg <= 25 * scale:
        skel = skeletonize(seg_mask)
        loops = deteksi_loop(skel)
        if len(loops) == 0: return True
    return False

def cek_ra(seg_mask, h_seg, w_seg, scale):
    if h_seg >= 55 * scale: return False
    if w_seg < 8 * scale or w_seg > 45 * scale: return False 
    skel = skeletonize(seg_mask)
    loops = deteksi_loop(skel)
    if len(loops) > 0: return False
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
    if h_seg >= 55 * scale: return False
    if posisi_titik.startswith("ATAS"): return True
    if w_seg < 35 * scale: return False 
    gigi_atas = analisis_ujung_atas(seg_mask)
    return gigi_atas == 2 or (gigi_atas == 0 and w_seg > 20 * scale)

def cek_ha(seg_mask, h_seg, w_seg, scale):
    if h_seg >= 45 * scale: return False
    skel = skeletonize(seg_mask)
    loops = deteksi_loop(skel)
    if len(loops) > 0:
        endpoints, _, _ = find_endpoints(skel)
        if len(endpoints) <= 1: return True
    return False

def cek_waw(seg_mask, h_seg, w_seg, scale):
    if h_seg >= 60 * scale: return False
    if w_seg > 60 * scale: return False
    skel = skeletonize(seg_mask)
    loops = deteksi_loop(skel)
    if len(loops) == 1:
        endpoints, _, _ = find_endpoints(skel)
        if 1 <= len(endpoints) <= 2:
            cycle_arr = np.array(loops[0][0])
            loop_min_y = np.min(cycle_arr[:, 0])
            coords = np.argwhere(seg_mask)
            top_y = np.min(coords[:, 0])
            if loop_min_y - top_y < 20 * scale: return True
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

# ==================== PEMOTONGAN & MAGNET ====================

def hitung_garis_potong_sinkron(component_features, labeled_letters, skeleton_used, scale):
    all_cols = []
    min_gap = 3 * scale
    
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
            if loop_xs and abs(first_ix - loop_xs[0]) < 5 * scale: skip_start = True
        if not skip_start: all_cols.append(first_ix + max(1, int(2 * scale)))

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
            
            if is_curr_tiang:
                all_cols.append((curr_x + next_x) // 2)
            elif gap < 20 * scale:
                continue 
            else:
                skip_cut = False
                if loops:
                    loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                    if loop_xs:
                        dist = min([abs(next_x - lx) for lx in loop_xs])
                        if dist < 5 * scale: skip_cut = True
                if not skip_cut:
                    all_cols.append(curr_x + max(1, int(2 * scale))) 

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

def potong_persimpangan(labeled_letters, skeleton_used, dot_mask, scale):
    hasil_potongan = []
    num_features = np.max(labeled_letters)
    labeled_dots, num_dots = measure.label(dot_mask > 0, return_num=True, connectivity=2)
    dots_props = measure.regionprops(labeled_dots)

    for region_label in range(1, num_features + 1):
        mask_group = (labeled_letters == region_label)
        if np.sum(mask_group) < 5: continue

        coords = np.argwhere(mask_group)
        if len(coords) == 0: continue
        
        min_y_grp, min_x_grp = coords.min(axis=0)
        max_y_grp, max_x_grp = coords.max(axis=0)
        h_grp = max_y_grp - min_y_grp
        w_grp = max_x_grp - min_x_grp
        
        x_coords = coords[:, 1]
        
        letter_skeleton = skeleton_used * mask_group
        _, raw_intersections, _ = find_endpoints(letter_skeleton)
        loops = deteksi_loop(letter_skeleton)
        
        cut_points = []
        
        if cek_waw(mask_group, h_grp, w_grp, scale) or cek_ha(mask_group, h_grp, w_grp, scale):
            pass 
        else:
            intersections = sorted([ix for iy, ix in raw_intersections])
            if intersections:
                intersections_map = {}
                for iy, ix in raw_intersections:
                    if ix not in intersections_map: intersections_map[ix] = []
                    intersections_map[ix].append(iy)

                first_ix = intersections[0]
                skip_start = False
                if loops:
                    loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                    if loop_xs and abs(first_ix - loop_xs[0]) < 5 * scale: skip_start = True
                if not skip_start: cut_points.append(first_ix + max(1, int(2 * scale)))

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
                    
                    if is_curr_tiang:
                        cut_points.append((curr_x + next_x) // 2)
                    elif gap < 20 * scale:
                        continue 
                    else:
                        skip_cut = False
                        if loops:
                            loop_xs = [np.mean(np.array(cycle), axis=0)[1] for cycle, _ in loops]
                            if loop_xs:
                                dist = min([abs(next_x - lx) for lx in loop_xs])
                                if dist < 5 * scale: skip_cut = True
                        if not skip_cut:
                            cut_points.append(curr_x + max(1, int(2 * scale)))

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
            
            if np.sum(segment_mask) > 30 * (scale**2) and segment_width > 5 * scale:
                seg_crds = np.argwhere(segment_mask)
                if len(seg_crds) > 0:
                    min_y_seg, min_x_seg = seg_crds.min(axis=0)
                    max_y_seg, max_x_seg = seg_crds.max(axis=0)
                    h_seg = max_y_seg - min_y_seg
                    w_seg = max_x_seg - min_x_seg
                    
                    posisi_titik = dominasi_diakritik(min_x_seg, max_x_seg, dot_mask, scale)
                else:
                    h_seg = 999; w_seg = 999
                    posisi_titik = "TIDAK_ADA"
                
                tipe_huruf = "UNKNOWN"
                if cek_lam(h_seg, w_seg, scale): tipe_huruf = "LAM"
                elif cek_alif(segment_mask, h_seg, w_seg, posisi_titik, scale): tipe_huruf = "ALIF"
                elif cek_sin(segment_mask, h_seg, scale): tipe_huruf = "SIN"
                elif cek_nun(segment_mask, h_seg, w_seg, posisi_titik, scale): tipe_huruf = "NUN"
                elif cek_ha(segment_mask, h_seg, w_seg, scale): tipe_huruf = "HA"
                elif cek_waw(segment_mask, h_seg, w_seg, scale): tipe_huruf = "WAW"
                elif cek_ra(segment_mask, h_seg, w_seg, scale): tipe_huruf = "RA"
                elif cek_ba(segment_mask, h_seg, w_seg, posisi_titik, scale): tipe_huruf = "BA"
                elif cek_ya(segment_mask, h_seg, w_seg, posisi_titik, scale): tipe_huruf = "YA"
                
                clean_segments.append(segment_mask)
                segment_metadata.append({'tipe_huruf': tipe_huruf})

        if not clean_segments: 
            hasil_potongan.append(mask_group)
            continue

        for dot_prop in dots_props:
            dy, dx = dot_prop.centroid
            grp_min_r, grp_min_c, grp_max_r, grp_max_c = measure.regionprops(mask_group.astype(int))[0].bbox
            if not (grp_min_r - (50 * scale) <= dy <= grp_max_r + (200 * scale) and grp_min_c - (50 * scale) <= dx <= grp_max_c + (50 * scale)):
                continue

            best_candidate = None
            best_score = float('inf') 

            for idx, seg_mask in enumerate(clean_segments):
                seg_cy, seg_cx = center_of_mass(seg_mask)
                tipe_huruf = segment_metadata[idx].get('tipe_huruf', 'UNKNOWN')
                
                if tipe_huruf in ["SIN", "LAM", "HA", "WAW", "ALIF"]:
                    continue 

                dist_x_centroid = abs(dx - seg_cx)
                dist_y = abs(dy - seg_cy)
                
                if tipe_huruf in ["YA", "BA"]:
                    if dy > seg_cy and dist_x_centroid < 40 * scale: 
                        final_score = -10000 + dist_x_centroid 
                    else: 
                        final_score = (dist_x_centroid * 2.0) + dist_y 
                elif tipe_huruf == "NUN":
                    if dy < seg_cy and dist_x_centroid < 40 * scale: 
                        final_score = -10000 + dist_x_centroid
                    else: 
                        final_score = (dist_x_centroid * 2.0) + dist_y
                else:
                    final_score = dist_x_centroid + (dist_y * 0.8)

                if final_score < best_score:
                    best_score = final_score
                    best_candidate = idx

            if best_candidate is not None:
                seg_mask_winner = clean_segments[best_candidate]
                win_cy, win_cx = center_of_mass(seg_mask_winner)
                dist_real = euclidean((dy, dx), (win_cy, win_cx))
                dist_x_winner = abs(dx - win_cx)
                is_win_via_magnet = (best_score < -5000)
                tipe_huruf_winner = segment_metadata[best_candidate].get('tipe_huruf', 'UNKNOWN')
                
                valid_attach = False
                
                if is_win_via_magnet:
                    valid_attach = True
                elif tipe_huruf_winner in ["YA", "BA"] and (dy < win_cy or dist_x_winner > 40 * scale):
                    valid_attach = False
                elif tipe_huruf_winner == "NUN" and (dy > win_cy or dist_x_winner > 40 * scale):
                    valid_attach = False
                elif dist_real < 90 * scale and dist_x_winner < 50 * scale:
                    valid_attach = True
                
                if valid_attach:
                    dot_indices = (labeled_dots == dot_prop.label)
                    clean_segments[best_candidate] = np.logical_or(clean_segments[best_candidate], dot_indices)

        hasil_potongan.extend(clean_segments)
    return hasil_potongan

# ==================== MAIN EXECUTION ====================
                               
image_path = r"E:\progres\berubah salin.png"

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
    loops = deteksi_loop(letter_skeleton)
    component_features[region_label] = {
        "intersections": intersections,
        "has_loop": len(loops) > 0,
        "cycles": [cycle for cycle, _ in loops]
    }

hasil_potongan = potong_persimpangan(labeled_letters, skeleton_used, dot_mask, scale)
sorted_hasil_potongan = sorted(hasil_potongan, key=lambda mask: np.max(np.argwhere(mask)[:, 1]), reverse=True)

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

final_cols = hitung_garis_potong_sinkron(component_features, labeled_letters, skeleton_used, scale)
for col in final_cols:
    ax.axvline(x=col, color='white', linestyle='--', linewidth=2, label='Garis Potong')

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

ax.set_title("Visualisasi Utama Sinkron: Segmentasi Karakter")
ax.axis('off')
plt.tight_layout()
plt.show()


# --- 2. PENYIMPANAN DAN EVALUASI METRIK ---
output_folder = "hasil_potongan"
os.makedirs(output_folder, exist_ok=True)

for idx, region_mask in enumerate(sorted_hasil_potongan):
    output_image = (region_mask * 255).astype(np.uint8)
    output_filename = os.path.join(output_folder, f'potongan_huruf_{idx + 1}.png')
    cv.imwrite(output_filename, output_image)

# Variabel Metrik Segmentasi untuk dicetak di akhir
jumlah_GT = 21   
jumlah_DT = len(sorted_hasil_potongan)
TP = 17         
FP = 3            
FN = 2           
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0


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
    plt.show() # <-- PERBAIKAN: Jendela ini tidak lagi ditutup otomatis


# --- 4. VISUALISASI MATEMATIS B-SPLINE (CARTESIAN) ---
print("\nMenampilkan visualisasi Matematis B-Spline per huruf...")
for idx, region_mask in enumerate(sorted_hasil_potongan):
    skel = skeletonize(region_mask)
    points = np.argwhere(skel)
    
    tsp_segments = tsp_skeleton_traversal(points, dist_threshold=max(3, int(5 * scale)))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    has_valid_curve = False
    
    # VARIABEL PENAMPUNG METRIK
    total_cp_karakter = 0
    total_jarak_karakter = 0
    total_points_karakter = 0
    total_sse_karakter = 0  
    
    for seg_idx, segment in enumerate(tsp_segments):
        segment_array = np.array(segment)
        n_len = len(segment_array)
        if n_len == 0: continue
        
        total_points_karakter += n_len
        
        # 1. AKUMULASI JARAK TSP
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
            
            # --- EVALUASI ERROR (RMSE) ---
            dists = cdist(segment_array, kurva_halus)
            min_dists = np.min(dists, axis=1)
            total_sse_karakter += np.sum(min_dists ** 2)
            
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
            
    # --- MENYIMPAN DATA UNTUK TABEL EVALUASI ---
    if total_points_karakter > 0:
        eval_tsp.append((idx+1, len(points), total_jarak_karakter))
        kompresi = ((total_points_karakter - total_cp_karakter) / total_points_karakter) * 100
        rmse = np.sqrt(total_sse_karakter / total_points_karakter)
        eval_bspline.append((idx+1, total_points_karakter, total_cp_karakter, kompresi, rmse))
    
    if has_valid_curve:
        ax.invert_yaxis() 
        ax.set_aspect('equal', adjustable='datalim') 
        
        # --- CETAK SEMUA METRIK DI JUDUL VISUALISASI ---
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

# --- 6. CETAK TABEL EVALUASI KESELURUHAN UNTUK SKRIPSI ---
print("\n" + "="*60)
print("TABEL 1: EVALUASI SEGMENTASI (KASHIDA)")
print("="*60)
print(f"Lebar Garis (r_stroke) : {r_stroke:.2f} px")
print(f"Faktor Skala (Scale)   : {scale:.2f}x")
print(f"Ground Truth (GT)      : {jumlah_GT}")
print(f"Detected (DT)          : {jumlah_DT}")
print(f"True Positive (TP)     : {TP}")
print(f"False Positive(FP)     : {FP}")
print(f"False Negative(FN)     : {FN}")
print(f"Akurasi                : {accuracy*100:.2f}%")
print(f"Precision              : {precision*100:.2f}%")
print(f"Recall                 : {recall*100:.2f}%")
print(f"F1-Score               : {f1_score*100:.2f}%")

print("\n" + "="*60)
print("TABEL 2: EVALUASI FITUR (RUTE TSP vs ASLI)")
print("="*60)
print(f"{'Huruf':<10} | {'Piksel Asli (px)':<18} | {'Jarak TSP (px)':<15}")
print("-" * 60)
for data in eval_tsp:
    print(f"H-{data[0]:<8} | {data[1]:<18} | {data[2]:.2f}")

print("\n" + "="*80)
print("TABEL 3: KINERJA B-SPLINE (KOMPRESI DATA & ERROR REKONSTRUKSI)")
print("="*80)
print(f"{'Huruf':<8} | {'Total Px':<12} | {'Total CP':<10} | {'Kompresi (%)':<15} | {'RMSE (px)':<10}")
print("-" * 80)
for data in eval_bspline:
    print(f"H-{data[0]:<6} | {data[1]:<12} | {data[2]:<10} | {data[3]:.1f}%{'':<14} | {data[4]:.3f}")
print("="*80 + "\n")