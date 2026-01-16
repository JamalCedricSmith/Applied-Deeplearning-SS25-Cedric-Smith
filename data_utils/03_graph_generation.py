"""
Semantic scene-graph generator for cataract surgery clips. 
Incorporates spatial and temporal information of instruments and anatomies.
This will then be used to create the QA-pairs to finetune the multimodal model. 
This script tackles the lack of detailed scene understanding in current models 
(Kun et al. Advancing surgical VQA with scene graph knowledge).

Advantages to previous version: Deterministic, faster, more robust.
"""

import json
import numpy as np

INPUT_JSON = "datasets/cataract1k/case_objects.json"
OUTPUT_JSON = "datasets/cataract1k/graphs.json"   # ← write a single JSON array
FRAME_WIDTH, FRAME_HEIGHT = 1024, 768
NEAR_THRESHOLD_SCALAR = 0.12
GRID_COL_INDEX = {"left": 0, "center": 1, "right": 2}
GRID_ROW_INDEX = {"top": 0, "center": 1, "bottom": 2}
ANATOMY = {"iris", "pupil", "intraocular lens", "cornea"}


def compute_object_center(obj):
    """
    Derive a robust center from COCO polygon segmentation(s) by the bbox midpoint.

    Why min/max instead of a mean of vertices?
    - Polygons often have denser vertices near instrument tips, biasing a mean.

    Notes:
    - Considers *all* polygons in the 'segmentation' (COCO allows multiple).
    - Returns (center_x, center_y) as floats, or None if unavailable.
    """
    segs = obj.get("segmentation")
    if not segs:
        return None

    xs, ys = [], []
    for poly in segs:
        if not poly:
            continue
        xs.extend(poly[0::2])  # even indices are x
        ys.extend(poly[1::2])  # odd indices are y

    if not xs or not ys:
        return None

    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    x_center = min_x + (max_x - min_x) / 2.0
    y_center = min_y + (max_y - min_y) / 2.0
    return x_center, y_center


def bbox_from_segmentation(obj):
    """
    Compute [min_x, min_y, max_x, max_y] from all COCO polygons.
    Returns None if segmentation is missing/empty.
    """
    segs = obj.get("segmentation")
    if not segs:
        return None

    xs, ys = [], []
    for poly in segs:
        if not poly:
            continue
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])

    if not xs or not ys:
        return None

    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    return [min_x, min_y, max_x, max_y]


def euclidean_distance(p1, p2):
    """Plain L2 distance; used for a lightweight 'near' relation."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def place_on_grid(center_x, center_y):
    """
    Map a point to a 3×3 grid (no tight center band override).
    Middle cell is labeled 'center' (instead of 'center center').
    """
    one_third_x = FRAME_WIDTH / 3.0
    two_thirds_x = 2.0 * FRAME_WIDTH / 3.0
    one_third_y = FRAME_HEIGHT / 3.0
    two_thirds_y = 2.0 * FRAME_HEIGHT / 3.0

    # Column binning
    if center_x < one_third_x:
        col = "left"
    elif center_x < two_thirds_x:
        col = "center"
    else:
        col = "right"

    # Row binning
    if center_y < one_third_y:
        row = "top"
    elif center_y < two_thirds_y:
        row = "center"
    else:
        row = "bottom"

    # De-duplicate "center center" wording for the middle cell
    return "center" if (row == "center" and col == "center") else f"{row} {col}"


def spatial_relationships_bbox(bbox_a, bbox_b):
    """
    COCO-specific minimal spatial logic.
    Expects [x1, y1, x2, y2] with x1<=x2, y1<=y2 (COCO guarantees this).
    Always returns at least one relation.
    """
    if not bbox_a or not bbox_b:
        return []

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    # 1) containment
    a_inside_b = (ax1 >= bx1) and (ay1 >= by1) and (ax2 <= bx2) and (ay2 <= by2)
    b_inside_a = (bx1 >= ax1) and (by1 >= ay1) and (bx2 <= ax2) and (by2 <= ay2)
    if a_inside_b and not b_inside_a:
        return ["inside"]
    if b_inside_a and not a_inside_b:
        return ["surrounds"]

    # 2) centroid-based directions
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

    dx, dy = acx - bcx, acy - bcy
    rels = []

    if dx > 0:
        rels.append("right-of")
    elif dx < 0:
        rels.append("left-of")

    # note: y grows downward in image coords
    if dy > 0:
        rels.append("below")
    elif dy < 0:
        rels.append("above")

    # If exactly identical centers, force a tie-break
    if not rels:
        if abs(dx) >= abs(dy):
            rels.append("right-of" if dx >= 0 else "left-of")
        else:
            rels.append("below" if dy >= 0 else "above")

    return rels


def object_category(name):
    """
    Binary classifier based on a minimal lexicon.
    - Extensible if more anatomy terms are added upstream.
    """
    return "anatomy" if name.lower() in ANATOMY else "instrument"


def graph_id_for(case_name, timestamp_seconds_str):
    """Stable, sortable identifier: CASE@ssssss (seconds zero-padded)."""
    return f"{case_name}@{int(float(timestamp_seconds_str)):06d}"


def size_bucket_from_area(area):
    """
    Heuristic size binning (small/medium/large) from pixel area.
    - Normalized by frame size; thresholds chosen for relative prominence.
    """
    if area is None:
        return None
    frame_area = FRAME_WIDTH * FRAME_HEIGHT
    frac = float(area) / float(frame_area)
    if frac < 0.01:
        return "small"
    if frac < 0.05:
        return "medium"
    return "large"


def build_nodes(object_list):
    """
    Convert raw detections to node dicts.
    """
    nodes = []
    x_center, y_center = FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0

    for obj in object_list:
        center_point = compute_object_center(obj)
        grid_cell = place_on_grid(*center_point) if center_point else None
        dist_center = euclidean_distance(center_point, (x_center, y_center)) if center_point else None
        bbox = bbox_from_segmentation(obj)

        node = {
            "id": obj["object_name"],
            "type": object_category(obj["object_name"]),
            "centroid": [float(center_point[0]), float(center_point[1])] if center_point else None,
            "bbox": bbox,  
            "area": obj.get("area"),
            "size_bucket": size_bucket_from_area(obj.get("area")),
            "grid": grid_cell,
            "center_distance_px": dist_center,
        }
        nodes.append(node)
    return nodes


def build_edges(nodes):
    """
    Build pairwise relations:
    - Bounding-box topology (inside/surrounds/left/right/above/below)
    - Proximity ('near') via same-grid OR Euclidean threshold
    """
    edges = []

    # Threshold defined as a fraction of the frame diagonal; tune with care.
    diag = np.hypot(FRAME_WIDTH, FRAME_HEIGHT)
    near_thresh = NEAR_THRESHOLD_SCALAR * diag

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Bounding-box topology
            rels = spatial_relationships_bbox(nodes[i]["bbox"], nodes[j]["bbox"])
            for r in rels:
                edges.append({"src": nodes[i]["id"], "rel": r, "dst": nodes[j]["id"]})

            # Proximity 
            same_grid = nodes[i]["grid"] is not None and nodes[i]["grid"] == nodes[j]["grid"]
            if nodes[i]["centroid"] and nodes[j]["centroid"]:
                d = euclidean_distance(nodes[i]["centroid"], nodes[j]["centroid"])
            else:
                d = None
            if same_grid or (d is not None and d < near_thresh):
                edges.append({"src": nodes[i]["id"], "rel": "near", "dst": nodes[j]["id"]})
    return edges


def main():
    """
    Input: JSON with per-case, per-timestamp object lists + metadata. Structure:
    {
        "<case_id>": {
            "<timestamp_sec>": {
              "phase": "<string>",
              "objects": [
                  {
                    "area": <float>,                 // pixel area of the instance
                    "segmentation": [[x1,y1,...]],   // list of polygon(s), COCO style
                    "object_name": "<string>"        // e.g., "Pupil", "Katena Forceps"
                  }
              ],
              "video_filename": "<string|null>"     // cut clip covering this timestamp
            },
            ...
        }
    }

    Output: JSON with a single array of per-timestamp scene graphs + temporal diffs + key events.
    """
    with open(INPUT_JSON, "r") as f:
        cases = json.load(f)

    graphs = []  # ← collect rows to dump once as a JSON array

    for case_name, clips in cases.items():
        # Chronologically sort the clips.
        sorted_clips = sorted(clips.keys(), key=lambda s: float(s))

        # Memory of previous nodes
        prev_nodes_by_id = set()

        # Memory of previous phase
        prev_phase = None

        # Build each clip's graph in sequence
        for idx, clip in enumerate(sorted_clips):
            info = clips[clip]

            # Extract nodes for this clip
            nodes = build_nodes(info.get("objects", []))

            # Extract edges for this clip
            edges = build_edges(nodes)

            # Set-based deltas for a minimal “what changed” view.
            curr_ids = {n["id"] for n in nodes}
            entered = sorted(list(curr_ids - prev_nodes_by_id))
            exited = sorted(list(prev_nodes_by_id - curr_ids))
            persisting = sorted(list(curr_ids & prev_nodes_by_id))

            # Define List to save the key Events to
            key_events = []
            curr_phase = info.get("phase")

            # Phase boundary detection 
            if prev_phase is not None and curr_phase != prev_phase:
                key_events.append({
                    "type": "phase_change",
                    "from": prev_phase,
                    "to": curr_phase
                })

            entering_instruments = [
                n["id"] for n in nodes
                if n["id"] in entered and n["type"] == "instrument"
            ]
            if entering_instruments:
                key_events.append({
                    "type": "instrument_entered",
                    "who": entering_instruments
                })

            prev_id = graph_id_for(case_name, sorted_clips[idx - 1]) if idx > 0 else None
            next_id = graph_id_for(case_name, sorted_clips[idx + 1]) if idx < len(sorted_clips) - 1 else None

            graph_row = {
                "case": case_name,
                "graph_id": graph_id_for(case_name, clip),
                "timestamp_sec": float(clip),
                "video_filename": info.get("video_filename"),
                "phase": curr_phase,
                "nodes": nodes,
                "edges": edges,
                "temporal": {
                    "prev_graph_id": prev_id,
                    "next_graph_id": next_id
                },
                "temporal_summary": {
                    "entered": entered,
                    "exited": exited,
                    "persisting": persisting
                },
                "key_events": key_events
            }

            graphs.append(graph_row)

            # advance memory for next step for temporal information
            prev_nodes_by_id = curr_ids
            prev_phase = curr_phase

    # Write ONE JSON file (array of rows)
    with open(OUTPUT_JSON, "w") as out_file:
        json.dump(graphs, out_file, indent=2)


if __name__ == "__main__":
    main()
