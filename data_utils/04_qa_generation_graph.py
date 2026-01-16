"""
Deterministic QA generator (no LLM). Same structure as before but uses graph JSON as input.
"""

import json
from pathlib import Path

# Read the JSON array you created earlier (still backward-compatible with JSONL)
INPUT_JSON = "datasets/cataract1k/graphs.json"
# Write a single JSON array, not JSONL
OUTPUT_JSON = "datasets/cataract1k/qa_pairs_graph.json"

def load_graph_records_from_path(path: str):
    """
    Load graphs from either:
      - JSON array file (first non-space char '['), or
      - JSONL file: one JSON object per non-empty line
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = p.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)

    return [json.loads(s) for s in (ln.strip() for ln in text.splitlines()) if s]

def dedupe_preserve_order(seq):
    """De-duplicate while preserving order (hashable elements)."""
    return list(dict.fromkeys(seq))

CANONICAL_PHASE_SEQUENCE = [
    "Incision",
    "Viscoelastic",
    "Capsulorhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "Irrigation-Aspiration",
    "Capsule Polishing",
    "Lens Implantation",
    "Lens Positioning",
    "Viscoelastic-Suction",
    "Anterior-Chamber Flushing",
    "Tonifying/Antibiotics",
]

_NEXT_PHASE = {a: b for a, b in zip(CANONICAL_PHASE_SEQUENCE, CANONICAL_PHASE_SEQUENCE[1:])}

PHASE_NORMALIZE = {
    "idle": "Idle",
    "incision": "Incision",
    "viscoelastic": "Viscoelastic",
    "capsulorhexis": "Capsulorhexis",
    "hydrodissection": "Hydrodissection",
    "phacoemulsification": "Phacoemulsification",
    "irrigation-aspiration": "Irrigation-Aspiration",
    "irrigation/aspiration": "Irrigation-Aspiration",
    "capsule polishing": "Capsule Polishing",
    "lens implantation": "Lens Implantation",
    "lens positioning": "Lens Positioning",
    "viscoelastic-suction": "Viscoelastic-Suction",
    "anterior-chamber flushing": "Anterior-Chamber Flushing",
    "tonifying/antibiotics": "Tonifying/Antibiotics",
}

INSTR_NORMALIZE = {
    "slit knife": "Slit Knife",
    "incision knife": "Slit Knife",
    "slit/incision knife": "Slit Knife",
    "gauge": "Gauge",
    "spatula": "Spatula",
    "capsulorhexis cystome": "Capsulorhexis Cystotome",
    "capsulorhexis cystotome": "Capsulorhexis Cystotome",
    "phacoemulsifier tip": "Phacoemulsifier Tip",
    "phacoemulsification tip": "Phacoemulsifier Tip",
    "irrigation-aspiration": "Irrigation-Aspiration",
    "irrigation-aspiration handpiece": "Irrigation-Aspiration",
    "lens injector": "Lens Injector",
    "capsulorhexis forceps": "Capsulorhexis Forceps",
    "katena forceps": "Katena Forceps",
}

normalize_phase = lambda s: PHASE_NORMALIZE.get(s.strip().lower(), s.strip().title()) if s else ""
normalize_instrument = lambda s: INSTR_NORMALIZE.get(s.strip().lower(), s.strip()) if s else ""

def phase_after(phase_norm: str) -> str:
    """
    - Return canonical next real phase after phase_norm
    - 'Unknown' if none
    """
    return _NEXT_PHASE.get(phase_norm, "Unknown")

def infer_next_likely_phase(current_phase_norm: str, previous_phase_norm: str) -> str:
    """
    - Never return 'Idle' as the next likely. 
    - If current is Idle, try previous non-idle phase; else 'Unknown'.
    - If current is real, return canonical next (may be 'Unknown' at end)
    """
    if current_phase_norm == "Idle":
        return phase_after(previous_phase_norm) if previous_phase_norm and previous_phase_norm != "Idle" else "Unknown"
    return phase_after(current_phase_norm)

def build_relative_positions_answer_q4(edges):
    """
    question4:
    - build realational position triples
    - drop self pairs
    Allowed: left-of, right-of, above, below, overlap
    """
    allowed = {"left-of", "right-of", "above", "below", "overlap"}
    triples = [
        f"{edge['src']} {edge['rel']} {edge['dst']}"
        for edge in (edges or [])
        if edge.get("src") and edge.get("rel") and edge.get("dst")
        and edge["src"] != edge["dst"]
        and edge["rel"] in allowed
    ]
    triples = dedupe_preserve_order(triples)
    return "None" if not triples else "; ".join(triples)

def build_absolute_positions_answer_q5(nodes):
    """
    - question5: use nodes[].grid to determine absolute positions
    - dedupe by id
    - keep first occurrence
    """
    grid_by_id = {}
    for n in nodes or []:
        nid, grid = n.get("id"), n.get("grid")
        if nid and grid and nid not in grid_by_id:
            grid_by_id[nid] = grid
    return "None" if not grid_by_id else "; ".join(f"{nid} at {grid}" for nid, grid in grid_by_id.items())

def build_visible_lists(nodes):
    """
    question2, question3: split anatomy vs instruments, normalize instrument names, dedupe and return the questions.
    """
    anatomy, instruments = [], []
    for n in nodes or []:
        nid, ntype = n.get("id"), n.get("type")
        if not nid:
            continue
        if ntype == "anatomy":
            anatomy.append(nid)
        elif ntype == "instrument":
            instruments.append(normalize_instrument(nid))
    return dedupe_preserve_order(anatomy), dedupe_preserve_order(instruments)

def main():
    """
    Deterministic QA generator (no LLM).
    - Uses graph JSON as input.
    - Sequentially generates the Q1..Q9 QA pairs.
    - Writes a SINGLE JSON array to OUTPUT_JSON.
    """
    graphs = load_graph_records_from_path(INPUT_JSON)

    # Build id->phase lookup for prev label only (normalized)
    id2phase = {
        graph["graph_id"]: normalize_phase(graph.get("phase", ""))
        for graph in graphs if "graph_id" in graph
    }

    all_qa = []  # collect all records, then dump once

    for graph in graphs:
        # current phase
        phase_norm = normalize_phase(graph.get("phase", ""))
        # extract previous phase from temporal component of graph
        prev_phase_norm = id2phase.get((graph.get("temporal") or {}).get("prev_graph_id"), "") or "Unknown"

        next_likely_norm = infer_next_likely_phase(
            phase_norm,
            "" if prev_phase_norm == "Unknown" else prev_phase_norm
        )

        # video filename (ensure .mp4)
        video_fname = graph.get("video_filename", "")
        if video_fname and not video_fname.endswith(".mp4"):
            video_fname += ".mp4"

        nodes = graph.get("nodes", []) or []
        edges = graph.get("edges", []) or []
        temporal_summary = graph.get("temporal_summary", {}) or {}

        anatomy_list, instrument_list = build_visible_lists(nodes)
        instrument_set = set(instrument_list)

        rel_answer = build_relative_positions_answer_q4(edges)
        abs_answer = build_absolute_positions_answer_q5(nodes)

        # Only needs to be extracted 
        entered_raw = temporal_summary.get("entered", []) or []
        exited_raw = temporal_summary.get("exited", []) or []

        instruments_entered = dedupe_preserve_order(
            [normalize_instrument(x) for x in entered_raw if normalize_instrument(x) in instrument_set]
        )
        instruments_exited = dedupe_preserve_order(
            [normalize_instrument(x) for x in exited_raw if normalize_instrument(x)]
        )

        qa = [
            {"video_filename": video_fname},
            {"question1": "Which phase of the surgery are we currently at?",
             "answer1": f"The current phase is {phase_norm}." if phase_norm else "Unknown"},
            {"question2": "What are the names of the visible anatomical structures in the current video?",
             "answer2": "None" if not anatomy_list else "Visible anatomical structures: " + ", ".join(anatomy_list) + "."},
            {"question3": "What are the names of the visible surgical instruments in the current video?",
             "answer3": "None" if not instrument_list else "Visible surgical instruments: " + ", ".join(instrument_list) + "."},
            {"question4": "How are the relative positions of objects A vs. B (L/R/Above/Below/Overlap/None)?",
             "answer4": rel_answer},
            {"question5": "Where are the absolute positions of objects?",
             "answer5": abs_answer},
            {"question6": "What was the previous phase?",
             "answer6": prev_phase_norm},
            {"question7": "What is the next likely phase?",
             "answer7": next_likely_norm},
            {"question8": "Which instruments entered since previous step?",
             "answer8": "None" if not instruments_entered else ", ".join(instruments_entered)},
            {"question9": "Which instruments exited since previous step?",
             "answer9": "None" if not instruments_exited else ", ".join(instruments_exited)},
        ]

        all_qa.append(qa)

    # Write ONE JSON array
    Path(OUTPUT_JSON).write_text(json.dumps(all_qa, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
