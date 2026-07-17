import numpy as np

def calculate_all_metrics(face_landmarks, w, h):
    """
    face_landmarks: MediaPipe face landmarks object
    w, h: image width and height
    """
    landmark_points = []
    lm_list = face_landmarks.landmark if hasattr(face_landmarks, 'landmark') else face_landmarks
    for lm in lm_list:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmark_points.append((x, y))
    landmark_points = np.array(landmark_points)

    def lm_px(idx):
        return landmark_points[idx]

    def dist_px(idx1, idx2):
        p1, p2 = lm_px(idx1), lm_px(idx2)
        return float(np.linalg.norm(p2 - p1))

    def angle_deg(vertex_idx, a_idx, b_idx):
        v, a, b = lm_px(vertex_idx), lm_px(a_idx), lm_px(b_idx)
        va, vb = a - v, b - v
        cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def angle_from_vertical(idx_top, idx_bottom):
        vec = lm_px(idx_bottom) - lm_px(idx_top)
        cos_a = np.dot(vec, [0, 1]) / (np.linalg.norm(vec) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def angle_from_horizontal(idx_a, idx_b):
        vec = lm_px(idx_b) - lm_px(idx_a)
        cos_a = np.dot(vec, [1, 0]) / (np.linalg.norm(vec) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def curvature_ratio(p_start_idx, p_mid_idx, p_end_idx):
        p1, p2, p3 = lm_px(p_start_idx), lm_px(p_mid_idx), lm_px(p_end_idx)
        chord = p3 - p1
        chord_len = np.linalg.norm(chord)
        if chord_len < 1e-6:
            return 0.0
        t = np.dot(p2 - p1, chord) / (chord_len ** 2)
        proj = p1 + t * chord
        sagitta = np.linalg.norm(p2 - proj)
        return float(sagitta / chord_len)

    def eye_aspect_ratio(p1, p2, p3, p4, p5, p6):
        p1, p2, p3, p4, p5, p6 = [lm_px(i) for i in (p1, p2, p3, p4, p5, p6)]
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        return float(vertical / (2 * horizontal + 1e-9))

    def point_line_deviation_px(point_idx, line_idx1, line_idx2):
        p, a, b = lm_px(point_idx), lm_px(line_idx1), lm_px(line_idx2)
        ab = b - a
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    # Define reference landmarks
    FOREHEAD_TOP   = 10
    GLABELLA       = 9      
    NOSE_TIP       = 4
    NOSE_BRIDGE    = 168
    SUBNASALE      = 2      
    MENTON         = 152    
    R_ZYGION       = 234    
    L_ZYGION       = 454    
    R_TEMPLE       = 127
    L_TEMPLE       = 356
    R_GONION       = 172    
    L_GONION       = 397    

    face_width_px  = dist_px(R_ZYGION, L_ZYGION)
    face_height_px = dist_px(FOREHEAD_TOP, MENTON)
    ipd_px = dist_px(33, 263)  

    mm_per_px = 63.5 / (ipd_px + 1e-9)

    metrics = {}

    # EYEBROWS
    R_BROW_INNER, R_BROW_PEAK, R_BROW_OUTER = 55, 105, 70
    L_BROW_INNER, L_BROW_PEAK, L_BROW_OUTER = 285, 334, 300
    R_EYE_TOP, R_EYE_BOTTOM = 159, 145
    L_EYE_TOP, L_EYE_BOTTOM = 386, 374

    metrics["eyebrow"] = {
        "right_brow_peak_height_mm": abs(lm_px(R_BROW_PEAK)[1] - lm_px(R_EYE_TOP)[1]) * mm_per_px,
        "left_brow_peak_height_mm": abs(lm_px(L_BROW_PEAK)[1] - lm_px(L_EYE_TOP)[1]) * mm_per_px,
        "right_brow_elevation_ratio": abs(lm_px(R_BROW_PEAK)[1] - lm_px(R_EYE_TOP)[1]) / (ipd_px + 1e-9),
        "left_brow_elevation_ratio": abs(lm_px(L_BROW_PEAK)[1] - lm_px(L_EYE_TOP)[1]) / (ipd_px + 1e-9),
        "right_brow_apex_angle_deg": angle_deg(R_BROW_PEAK, R_BROW_INNER, R_BROW_OUTER),
        "left_brow_apex_angle_deg": angle_deg(L_BROW_PEAK, L_BROW_INNER, L_BROW_OUTER),
    }

    # EYES
    metrics["eye"] = {
        "right_eye_aspect_ratio": eye_aspect_ratio(33, 160, 158, 133, 153, 144),
        "left_eye_aspect_ratio": eye_aspect_ratio(362, 385, 387, 263, 373, 380),
        "eye_spacing_ipd_over_face_width": ipd_px / (face_width_px + 1e-9),
        "right_lower_eyelid_curvature": curvature_ratio(33, R_EYE_BOTTOM, 133),
        "left_lower_eyelid_curvature": curvature_ratio(362, L_EYE_BOTTOM, 263),
    }

    # NOSE
    R_ALA, L_ALA = 129, 358
    R_BRIDGE_SIDE, L_BRIDGE_SIDE = 236, 456
    R_INNER_CANTHUS, L_INNER_CANTHUS = 133, 362

    nose_width_px = dist_px(R_ALA, L_ALA)
    nose_height_px = dist_px(NOSE_BRIDGE, SUBNASALE)
    intercanthal_px = dist_px(R_INNER_CANTHUS, L_INNER_CANTHUS)

    metrics["nose"] = {
        "nasal_width_mm": nose_width_px * mm_per_px,
        "nasal_height_mm": nose_height_px * mm_per_px,
        "nasal_aspect_ratio_width_over_height": nose_width_px / (nose_height_px + 1e-9),
        "naso_canthal_ratio_nose_width_over_intercanthal": nose_width_px / (intercanthal_px + 1e-9),
        "pyramidal_width_mm": dist_px(R_BRIDGE_SIDE, L_BRIDGE_SIDE) * mm_per_px,
    }

    # LIPS
    MOUTH_R, MOUTH_L = 61, 291
    CUPID_DIP = 0
    CUPID_PEAK_R, CUPID_PEAK_L = 37, 267
    LOWER_LIP_BOTTOM = 17

    mouth_width_px = dist_px(MOUTH_R, MOUTH_L)

    metrics["lips"] = {
        "mouth_width_mm": mouth_width_px * mm_per_px,
        "philtrum_length_mm": dist_px(SUBNASALE, CUPID_DIP) * mm_per_px,
        "cupids_bow_angle_deg": angle_deg(CUPID_DIP, CUPID_PEAK_R, CUPID_PEAK_L),
    }

    # CHEEKS
    metrics["cheeks"] = {
        "facial_width_mm": face_width_px * mm_per_px,
        "malar_width_ratio_powell": face_width_px / (face_height_px + 1e-9),
        "right_cheekbone_vertical_position_ratio": (lm_px(R_ZYGION)[1] - lm_px(FOREHEAD_TOP)[1]) / (face_height_px + 1e-9),
        "left_cheekbone_vertical_position_ratio": (lm_px(L_ZYGION)[1] - lm_px(FOREHEAD_TOP)[1]) / (face_height_px + 1e-9),
    }

    # JAW
    jaw_width_px = dist_px(R_GONION, L_GONION)
    metrics["jaw"] = {
        "frontal_jaw_rise_mm": abs(lm_px(R_GONION)[1] - lm_px(MENTON)[1]) * mm_per_px,
        "jaw_width_mm": jaw_width_px * mm_per_px,
        "right_jaw_inclination_angle_deg": angle_from_horizontal(R_GONION, MENTON),
        "left_jaw_inclination_angle_deg": angle_from_horizontal(L_GONION, MENTON),
        "face_width_mm": face_width_px * mm_per_px,
    }

    # CHIN
    CHIN_R, CHIN_L = 214, 434
    metrics["chin"] = {
        "chin_width_mm": dist_px(CHIN_R, CHIN_L) * mm_per_px,
        "chin_vertical_height_mm": dist_px(LOWER_LIP_BOTTOM, MENTON) * mm_per_px,
        "chin_midline_deviation_mm": point_line_deviation_px(MENTON, GLABELLA, NOSE_TIP) * mm_per_px,
    }

    # HAIR / FOREHEAD
    metrics["hair"] = {
        "forehead_width_mm": dist_px(R_TEMPLE, L_TEMPLE) * mm_per_px,
        "forehead_height_mm_mesh_approx": dist_px(FOREHEAD_TOP, GLABELLA) * mm_per_px,
        "right_temple_inclination_angle_deg": angle_from_vertical(R_TEMPLE, R_ZYGION),
        "left_temple_inclination_angle_deg": angle_from_vertical(L_TEMPLE, L_ZYGION),
    }

    # SMILE
    metrics["smile"] = {
        "upper_smile_arc_curvature": curvature_ratio(MOUTH_R, CUPID_DIP, MOUTH_L),
        "lower_smile_arc_curvature": curvature_ratio(MOUTH_R, LOWER_LIP_BOTTOM, MOUTH_L),
        "smile_width_mm": mouth_width_px * mm_per_px,
    }

    return metrics, mm_per_px, jaw_width_px
