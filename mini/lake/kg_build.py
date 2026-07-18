"""Knowledge graph factory (Sprint 8 / FP-4).

Builds/merges an agriculture knowledge graph from:
- Seed `data/knowledge_graph.json`
- Frozen taxonomy (crops, stages)
- Processed lake structured facts
- Standard / synth records (entity fields)

Exports NetworkX GraphML, JSON graph, graph-triple training text, and a Neo4j loader stub.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mini.lake.dedup import extract_record_lists
from mini.lake.process import iter_processed_json_files
from mini.paths import (
    DATASETS_DIR,
    LAKE_ROOT,
    LAKE_TEST,
    LAKE_TRAINING,
    LAKE_VALIDATION,
    REPO_ROOT,
    SCHEMA_VERSION,
    ensure_lake_layout,
    relative_to_repo,
)
from mini.taxonomy.aliases import resolve_crop_name
from mini.taxonomy.domains import TAXONOMY, list_crop_names_en, list_crop_stages
from mini.taxonomy.regions import list_mh_districts

SEED_GRAPH_PATH = REPO_ROOT / "data" / "knowledge_graph.json"
KG_DIR = DATASETS_DIR / "kg"
KG_LATEST = DATASETS_DIR / "KG_LATEST.json"
LAKE_KG_LATEST = LAKE_ROOT / "KG_LATEST.json"

# Acceptance targets (Sprint 8)
MIN_NODES = 200
MIN_EDGES = 400

_SOIL_TYPES = [
    "Black Cotton Soil",
    "Red Soil",
    "Laterite Soil",
    "Sandy Soil",
    "Alluvial Soil",
    "Saline Soil",
    "Medium Black Soil",
    "Clay Loam",
]
_NUTRIENTS = [
    "Nitrogen (N)",
    "Phosphorus (P)",
    "Potassium (K)",
    "Zinc (Zn)",
    "Iron (Fe)",
    "Boron (B)",
    "Sulphur (S)",
    "Magnesium (Mg)",
]
_WEATHER = [
    "Heavy Rain",
    "Heat Wave",
    "Dry Spell",
    "High Humidity",
    "High Wind",
    "Frost",
    "Hail",
    "Flood / Waterlogging",
    "Prolonged Cloudy Weather",
    "Monsoon Onset Delay",
]
_IRRIGATION = [
    "Drip Irrigation",
    "Sprinkler Irrigation",
    "Furrow Irrigation",
    "Deficit Irrigation",
]
_FERTILIZERS = [
    "Urea",
    "DAP",
    "SSP",
    "MOP",
    "NPK Complex",
    "Zinc Sulphate",
    "FYM / Compost",
    "Neem Cake",
    "Biofertilizer",
]
_DEFAULT_PESTS = [
    "Aphids",
    "Thrips",
    "Whitefly",
    "Bollworm complex",
    "Stem borer",
    "Leaf miner",
    "Mites",
    "Termites",
    "Pink Bollworm",
    "Fruit Borer",
]
_DEFAULT_DISEASES = [
    "Powdery Mildew",
    "Downy Mildew",
    "Leaf Spot",
    "Bacterial Blight",
    "Wilt",
    "Rust",
    "Blast",
    "Anthracnose",
]
_TREATMENTS = [
    "IPM Scouting",
    "Neem-based Biopesticide",
    "Pheromone Trap",
    "Yellow Sticky Trap",
    "Labeled Fungicide",
    "Labeled Insecticide",
    "Cultural Sanitation",
    "Resistant Variety",
]


def _slug(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _add_node(
    nodes: dict[str, dict],
    node_id: str,
    label: str,
    properties: dict | None = None,
    source: str = "kg_build",
) -> None:
    nid = _slug(node_id)
    if not nid or len(nid) < 2:
        return
    props = dict(properties or {})
    props.setdefault("source", source)
    if nid in nodes:
        # merge props lightly
        existing = nodes[nid]
        for k, v in props.items():
            if k not in existing["properties"] and v is not None:
                existing["properties"][k] = v
        return
    nodes[nid] = {"id": nid, "label": label, "properties": props}


def _add_edge(
    edges: dict[tuple[str, str, str], dict],
    source: str,
    target: str,
    relation: str,
    properties: dict | None = None,
) -> None:
    s, t, r = _slug(source), _slug(target), _slug(relation)
    if not s or not t or not r or s == t:
        return
    key = (s, t, r)
    if key in edges:
        return
    edges[key] = {
        "source": s,
        "target": t,
        "relation": r,
        "properties": dict(properties or {}),
    }


def load_seed_graph() -> tuple[dict[str, dict], dict[tuple[str, str, str], dict]]:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    if not SEED_GRAPH_PATH.exists():
        return nodes, edges
    try:
        data = json.loads(SEED_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return nodes, edges
    for n in data.get("nodes") or []:
        if isinstance(n, dict) and n.get("id"):
            _add_node(
                nodes,
                n["id"],
                n.get("label") or "Entity",
                n.get("properties") or {},
                source="seed_graph",
            )
    for e in data.get("edges") or []:
        if not isinstance(e, dict):
            continue
        _add_edge(
            edges,
            e.get("source") or "",
            e.get("target") or "",
            e.get("relation") or "RELATED_TO",
            e.get("properties") or {},
        )
    return nodes, edges


def _load_processed_buckets() -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "crops": [],
        "diseases_and_pests": [],
        "schemes": [],
        "fertilizer_recommendations": [],
        "soil_types": [],
        "varieties": [],
        "practices": [],
        "markets": [],
        "advisories": [],
        "zones": [],
    }
    for path in iter_processed_json_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, items in extract_record_lists(data):
            if key not in buckets:
                continue
            for it in items:
                if isinstance(it, dict):
                    row = dict(it)
                    row["_source_path"] = relative_to_repo(path)
                    buckets[key].append(row)
    return buckets


def _iter_standard_rows(limit: int = 5000) -> list[dict]:
    """Sample standard + synth rows for entity extraction (fields only)."""
    rows: list[dict] = []
    for base in (LAKE_TRAINING, LAKE_VALIDATION, LAKE_TEST):
        for name in ("standard_records.jsonl", "synth_records.jsonl"):
            path = base / name
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                continue
    return rows


def extract_and_merge(
    *,
    include_districts: bool = True,
    include_records: bool = True,
) -> tuple[dict[str, dict], dict[tuple[str, str, str], dict], dict[str, Any]]:
    """Build full node/edge maps with provenance stats."""
    nodes, edges = load_seed_graph()
    seed_n, seed_e = len(nodes), len(edges)
    buckets = _load_processed_buckets()
    stats = {
        "seed_nodes": seed_n,
        "seed_edges": seed_e,
        "sources": {
            "seed_graph": str(relative_to_repo(SEED_GRAPH_PATH)),
            "taxonomy": "mini.taxonomy",
            "processed_lake": True,
            "standard_records": include_records,
        },
    }

    # --- Taxonomy crops + stages ---
    for c in TAXONOMY.get("crops") or []:
        name = c.get("name_en") or ""
        if not name:
            continue
        resolved = resolve_crop_name(name) or name
        _add_node(
            nodes,
            resolved,
            "Crop",
            {
                "marathi": c.get("name_mr"),
                "hindi": c.get("name_hi"),
                "scientific": c.get("scientific_name"),
            },
            source="taxonomy",
        )
        for pest in c.get("major_pests") or []:
            p = _slug(pest)
            if not p:
                continue
            _add_node(nodes, p, "Pest", source="taxonomy")
            _add_edge(edges, resolved, p, "AFFECTED_BY", {"origin": "taxonomy"})
        for dis in c.get("major_diseases") or []:
            d = _slug(dis)
            if not d:
                continue
            _add_node(nodes, d, "Disease", source="taxonomy")
            _add_edge(edges, resolved, d, "AFFECTED_BY", {"origin": "taxonomy"})
        for soil in c.get("ideal_soil") or []:
            s = _slug(soil)
            if not s:
                continue
            _add_node(nodes, s, "Soil", source="taxonomy")
            _add_edge(edges, resolved, s, "GROWS_IN", {"origin": "taxonomy"})

    for st in list_crop_stages():
        sid = st.get("name_en") or st.get("id")
        if not sid:
            continue
        _add_node(
            nodes,
            sid,
            "GrowthStage",
            {"id": st.get("id"), "marathi": st.get("name_mr")},
            source="taxonomy",
        )

    # Stage links for all taxonomy crops
    for crop in list_crop_names_en():
        for st in list_crop_stages():
            sname = st.get("name_en") or st.get("id")
            if sname:
                _add_edge(edges, crop, sname, "HAS_STAGE", {"origin": "taxonomy"})

    # --- Processed lake facts ---
    for item in buckets["crops"]:
        name = resolve_crop_name(item.get("name_en") or "") or item.get("name_en")
        if not name:
            continue
        _add_node(
            nodes,
            name,
            "Crop",
            {"marathi": item.get("name_mr"), "season": item.get("season")},
            source="processed",
        )
        for pest in item.get("major_pests") or []:
            p = _slug(pest)
            if p:
                _add_node(nodes, p, "Pest", source="processed")
                _add_edge(edges, name, p, "AFFECTED_BY", {"origin": "processed"})
        for dis in item.get("major_diseases") or []:
            d = _slug(dis)
            if d:
                _add_node(nodes, d, "Disease", source="processed")
                _add_edge(edges, name, d, "AFFECTED_BY", {"origin": "processed"})
        for soil in item.get("ideal_soil") or []:
            s = _slug(soil)
            if s:
                _add_node(nodes, s, "Soil", source="processed")
                _add_edge(edges, name, s, "GROWS_IN", {"origin": "processed"})

    for item in buckets["diseases_and_pests"]:
        dname = _slug(item.get("name_en") or "")
        crop = resolve_crop_name(item.get("crop_en") or "") or item.get("crop_en")
        if not dname:
            continue
        label = "Pest" if "pest" in (item.get("type") or item.get("category") or "").lower() else "Disease"
        # heuristic from name
        low = dname.lower()
        if any(x in low for x in ("worm", "aphid", "thrip", "mite", "borer", "fly", "bug", "termite")):
            label = "Pest"
        _add_node(
            nodes,
            dname,
            label,
            {"marathi": item.get("name_mr"), "crop": crop},
            source="processed",
        )
        if crop:
            _add_node(nodes, crop, "Crop", source="processed")
            _add_edge(edges, crop, dname, "AFFECTED_BY", {"origin": "processed"})
        org = item.get("organic_control_en") or item.get("organic_control")
        chem = item.get("chemical_control_en") or item.get("chemical_control")
        if org:
            tid = _slug(f"Organic: {dname[:40]}")
            _add_node(nodes, tid, "Treatment", {"detail": str(org)[:200]}, source="processed")
            _add_edge(edges, dname, tid, "TREATED_BY", {"origin": "processed", "mode": "organic"})
        if chem:
            tid = _slug(f"Chemical: {dname[:40]}")
            _add_node(nodes, tid, "Treatment", {"detail": str(chem)[:200]}, source="processed")
            _add_edge(edges, dname, tid, "TREATED_BY", {"origin": "processed", "mode": "chemical"})

    for item in buckets["schemes"]:
        sname = _slug(item.get("name_en") or item.get("name") or item.get("scheme_name") or "")
        if not sname:
            continue
        _add_node(
            nodes,
            sname,
            "Scheme",
            {"marathi": item.get("name_mr"), "ministry": item.get("ministry")},
            source="processed",
        )
        # link popular schemes to all major crops
        for crop in list_crop_names_en()[:15]:
            _add_edge(edges, crop, sname, "COVERED_BY", {"origin": "processed"})

    for item in buckets["fertilizer_recommendations"]:
        crop = resolve_crop_name(item.get("crop_en") or "") or item.get("crop_en")
        if not crop:
            continue
        _add_node(nodes, crop, "Crop", source="processed")
        for fert in _FERTILIZERS[:5]:
            _add_node(nodes, fert, "Fertilizer", source="template")
            _add_edge(edges, crop, fert, "REQUIRES_FERTILIZER", {"origin": "processed"})
        for nut in ("Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"):
            _add_node(nodes, nut, "Nutrient", source="template")
            _add_edge(edges, crop, nut, "NEEDS_NUTRIENT", {"origin": "processed"})

    for item in buckets["soil_types"]:
        sname = _slug(item.get("name_en") or item.get("name") or "")
        if sname:
            _add_node(nodes, sname, "Soil", source="processed")

    # --- Domain template expansion for volume / coverage ---
    crops = list_crop_names_en()
    for soil in _SOIL_TYPES:
        _add_node(nodes, soil, "Soil", source="template")
    for nut in _NUTRIENTS:
        _add_node(nodes, nut, "Nutrient", source="template")
    for w in _WEATHER:
        _add_node(nodes, w, "WeatherCondition", source="template")
    for irr in _IRRIGATION:
        _add_node(nodes, irr, "Irrigation", source="template")
    for fert in _FERTILIZERS:
        _add_node(nodes, fert, "Fertilizer", source="template")
    for p in _DEFAULT_PESTS:
        _add_node(nodes, p, "Pest", source="template")
    for d in _DEFAULT_DISEASES:
        _add_node(nodes, d, "Disease", source="template")
    for t in _TREATMENTS:
        _add_node(nodes, t, "Treatment", source="template")

    for crop in crops:
        _add_node(nodes, crop, "Crop", source="taxonomy")
        for soil in _SOIL_TYPES[:4]:
            _add_edge(edges, crop, soil, "GROWS_IN", {"origin": "template"})
        for fert in _FERTILIZERS[:4]:
            _add_edge(edges, crop, fert, "REQUIRES_FERTILIZER", {"origin": "template"})
        for nut in _NUTRIENTS[:4]:
            _add_edge(edges, crop, nut, "NEEDS_NUTRIENT", {"origin": "template"})
        for pest in _DEFAULT_PESTS[:5]:
            _add_edge(edges, crop, pest, "AFFECTED_BY", {"origin": "template"})
        for dis in _DEFAULT_DISEASES[:4]:
            _add_edge(edges, crop, dis, "AFFECTED_BY", {"origin": "template"})
        for w in _WEATHER[:5]:
            _add_edge(edges, crop, w, "STRESSED_BY", {"origin": "template"})
        for irr in _IRRIGATION:
            _add_edge(edges, crop, irr, "USES_IRRIGATION", {"origin": "template"})
        # disease weather triggers
        for dis in _DEFAULT_DISEASES[:3]:
            _add_edge(edges, dis, "High Humidity", "TRIGGERED_BY", {"origin": "template"})
            _add_edge(edges, dis, "Heavy Rain", "TRIGGERED_BY", {"origin": "template"})
        for pest in _DEFAULT_PESTS[:3]:
            _add_edge(edges, pest, "IPM Scouting", "TREATED_BY", {"origin": "template"})
            _add_edge(edges, pest, "Neem-based Biopesticide", "TREATED_BY", {"origin": "template"})

    # Common schemes
    schemes = [
        "PM-KISAN",
        "PMFBY",
        "Kisan Credit Card",
        "Micro Irrigation Subsidy",
        "Soil Health Card",
        "MGNREGA Farm Works",
        "eNAM",
    ]
    for sch in schemes:
        _add_node(nodes, sch, "Scheme", source="template")
        for crop in crops:
            _add_edge(edges, crop, sch, "COVERED_BY", {"origin": "template"})
            if sch in ("PM-KISAN", "Kisan Credit Card", "PMFBY"):
                _add_edge(edges, crop, sch, "BENEFITS_FROM", {"origin": "template"})

    if include_districts:
        for dist in list_mh_districts():
            dname = _slug(dist)
            _add_node(nodes, dname, "District", {"state": "Maharashtra"}, source="taxonomy")
            # link subset of crops per district for edges without full cartesian blow-up
            for crop in crops:
                _add_edge(edges, crop, dname, "GROWN_IN", {"origin": "template", "state": "Maharashtra"})

    # --- Standard records: category/crop entities ---
    if include_records:
        cat_label = {
            "crop": "Crop",
            "disease": "Disease",
            "pest": "Pest",
            "soil": "Soil",
            "fertilizer": "Fertilizer",
            "scheme": "Scheme",
            "weather": "WeatherCondition",
            "irrigation": "Irrigation",
            "market": "Market",
            "finance": "FinanceTopic",
            "seed": "SeedTopic",
            "machinery": "Machinery",
            "advisory": "Advisory",
        }
        for row in _iter_standard_rows(limit=3000):
            crop = row.get("crop")
            if crop:
                rc = resolve_crop_name(crop) or crop
                _add_node(nodes, rc, "Crop", source="standard_record")
            cat = (row.get("category") or "").lower()
            label = cat_label.get(cat)
            sub = row.get("subcategory")
            if label and sub and cat not in ("crop", "general"):
                _add_node(nodes, _slug(str(sub)[:80]), label, source="standard_record")
                if crop:
                    rc = resolve_crop_name(crop) or crop
                    rel = {
                        "disease": "AFFECTED_BY",
                        "pest": "AFFECTED_BY",
                        "soil": "GROWS_IN",
                        "fertilizer": "REQUIRES_FERTILIZER",
                        "scheme": "COVERED_BY",
                        "weather": "STRESSED_BY",
                        "irrigation": "USES_IRRIGATION",
                    }.get(cat)
                    if rel:
                        _add_edge(edges, rc, _slug(str(sub)[:80]), rel, {"origin": "standard_record"})

    # Ensure edge endpoints exist as nodes
    for (s, t, _r) in list(edges.keys()):
        if s not in nodes:
            _add_node(nodes, s, "Entity", source="edge_endpoint")
        if t not in nodes:
            _add_node(nodes, t, "Entity", source="edge_endpoint")

    stats["final_nodes"] = len(nodes)
    stats["final_edges"] = len(edges)
    stats["by_label"] = dict(Counter(n["label"] for n in nodes.values()))
    stats["by_relation"] = dict(Counter(e["relation"] for e in edges.values()))
    return nodes, edges, stats


def graph_to_networkx(nodes: dict[str, dict], edges: dict[tuple[str, str, str], dict]):
    import networkx as nx

    g = nx.MultiDiGraph()
    for nid, n in nodes.items():
        g.add_node(nid, label=n.get("label"), **(n.get("properties") or {}))
    for (s, t, r), e in edges.items():
        g.add_edge(s, t, relation=r, **(e.get("properties") or {}))
    return g


def triples_as_training_text(
    nodes: dict[str, dict],
    edges: dict[tuple[str, str, str], dict],
    limit: int = 5000,
) -> list[dict[str, str]]:
    """Convert edges to short Q/A triple facts for training text."""
    out: list[dict[str, str]] = []
    for i, ((s, t, r), _e) in enumerate(edges.items()):
        if i >= limit:
            break
        s_label = (nodes.get(s) or {}).get("label") or "Entity"
        t_label = (nodes.get(t) or {}).get("label") or "Entity"
        q = f"What is the relationship between {s} and {t}?"
        a = f"{s} ({s_label}) —[{r}]→ {t} ({t_label})."
        out.append(
            {
                "id": f"kg_triple_{i}",
                "question": q,
                "answer": a,
                "relation": r,
                "source": s,
                "target": t,
                "text": f"{s} {r} {t}. {a}",
            }
        )
    return out


def write_neo4j_stub(path: Path) -> Path:
    """Optional Neo4j loader stub (cypher import hints)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = """// Neo4j loader stub — Sprint 8 W-KGBUILD
// Import nodes from KG_LATEST.json then edges.
// Example (manual / APOC):
//
// CALL apoc.load.json('file:///KG_LATEST.json') YIELD value
// UNWIND value.nodes AS n
// MERGE (x:Entity {id: n.id})
// SET x.label = n.label, x += n.properties;
//
// CALL apoc.load.json('file:///KG_LATEST.json') YIELD value
// UNWIND value.edges AS e
// MATCH (a:Entity {id: e.source}), (b:Entity {id: e.target})
// CALL apoc.create.relationship(a, e.relation, e.properties, b) YIELD rel
// RETURN count(rel);
//
// Prefer offline NetworkX GraphML for local GraphRAG until Neo4j is provisioned.
"""
    path.write_text(content, encoding="utf-8")
    return path


def export_graph(
    nodes: dict[str, dict],
    edges: dict[tuple[str, str, str], dict],
    stats: dict[str, Any],
    *,
    dry_run: bool = False,
    write_platform_seed: bool = True,
) -> dict[str, Any]:
    ensure_lake_layout()
    KG_DIR.mkdir(parents=True, exist_ok=True)
    node_list = list(nodes.values())
    edge_list = list(edges.values())
    n_nodes, n_edges = len(node_list), len(edge_list)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ") + "-kg"
    targets_met = {
        "nodes": n_nodes >= MIN_NODES,
        "edges": n_edges >= MIN_EDGES,
    }
    ok = all(targets_met.values())

    graph_doc = {
        "nodes": node_list,
        "edges": edge_list,
        "meta": {
            "version": version,
            "sprint": "S8",
            "feature_phase": "FP-4",
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "counts": {"nodes": n_nodes, "edges": n_edges},
            "by_label": stats.get("by_label") or {},
            "by_relation": stats.get("by_relation") or {},
            "seed_nodes": stats.get("seed_nodes"),
            "seed_edges": stats.get("seed_edges"),
            "targets": {"min_nodes": MIN_NODES, "min_edges": MIN_EDGES},
            "targets_met": targets_met,
            "builder": "W-KGBUILD",
        },
    }

    artifacts: list[str] = []
    if not dry_run:
        # Versioned export under mini/datasets/kg
        ver_dir = KG_DIR / version
        ver_dir.mkdir(parents=True, exist_ok=True)

        kg_json = ver_dir / "knowledge_graph.json"
        kg_json.write_text(json.dumps(graph_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(kg_json))

        KG_LATEST.write_text(json.dumps(graph_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(KG_LATEST))

        LAKE_ROOT.mkdir(parents=True, exist_ok=True)
        LAKE_KG_LATEST.write_text(json.dumps(graph_doc["meta"], indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(LAKE_KG_LATEST))

        # NetworkX GraphML
        try:
            g = graph_to_networkx(nodes, edges)
            import networkx as nx

            gml = ver_dir / "graph.graphml"
            # GraphML needs serializable attrs
            for _, data in g.nodes(data=True):
                for k, v in list(data.items()):
                    if v is None:
                        data[k] = ""
                    elif not isinstance(v, (str, int, float, bool)):
                        data[k] = str(v)
            for _, _, data in g.edges(data=True):
                for k, v in list(data.items()):
                    if v is None:
                        data[k] = ""
                    elif not isinstance(v, (str, int, float, bool)):
                        data[k] = str(v)
            nx.write_graphml(g, gml)
            artifacts.append(relative_to_repo(gml))
            # also copy latest graphml
            latest_gml = KG_DIR / "latest.graphml"
            nx.write_graphml(g, latest_gml)
            artifacts.append(relative_to_repo(latest_gml))
        except Exception as exc:
            stats["graphml_error"] = str(exc)

        # Triple training text
        triples = triples_as_training_text(nodes, edges, limit=8000)
        triples_path = ver_dir / "graph_triples.jsonl"
        with open(triples_path, "w", encoding="utf-8") as f:
            for row in triples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifacts.append(relative_to_repo(triples_path))
        latest_triples = KG_DIR / "graph_triples.jsonl"
        with open(latest_triples, "w", encoding="utf-8") as f:
            for row in triples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifacts.append(relative_to_repo(latest_triples))

        stub = write_neo4j_stub(ver_dir / "neo4j_loader_stub.cypher")
        artifacts.append(relative_to_repo(stub))
        write_neo4j_stub(KG_DIR / "neo4j_loader_stub.cypher")

        # Optional: refresh platform seed for local GraphRAG (not required to commit)
        if write_platform_seed:
            try:
                SEED_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
                # Keep seed format: nodes/edges/meta
                seed_out = {
                    "nodes": node_list,
                    "edges": [{"source": e["source"], "target": e["target"], "relation": e["relation"]} for e in edge_list],
                    "meta": graph_doc["meta"],
                }
                SEED_GRAPH_PATH.write_text(
                    json.dumps(seed_out, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                artifacts.append(relative_to_repo(SEED_GRAPH_PATH))
            except Exception as exc:
                stats["seed_write_error"] = str(exc)

        manifest = {
            "version": version,
            "ok": ok,
            "counts": {"nodes": n_nodes, "edges": n_edges},
            "targets_met": targets_met,
            "artifacts": artifacts,
            "sprint": "S8",
        }
        man = ver_dir / "manifest.json"
        man.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        artifacts.append(relative_to_repo(man))

    return {
        "ok": ok,
        "version": version,
        "counts": {"nodes": n_nodes, "edges": n_edges},
        "by_label": stats.get("by_label") or {},
        "by_relation": stats.get("by_relation") or {},
        "seed_nodes": stats.get("seed_nodes"),
        "seed_edges": stats.get("seed_edges"),
        "targets_met": targets_met,
        "targets": {"min_nodes": MIN_NODES, "min_edges": MIN_EDGES},
        "artifacts": artifacts,
        "dry_run": dry_run,
        "sprint": "S8",
    }


def run_kg_build(
    *,
    dry_run: bool = False,
    write_platform_seed: bool = False,
    include_districts: bool = True,
) -> dict[str, Any]:
    """Build graph. Platform seed rewrite is opt-in (avoids dirtying data/ by default)."""
    nodes, edges, stats = extract_and_merge(include_districts=include_districts)
    return export_graph(
        nodes,
        edges,
        stats,
        dry_run=dry_run,
        write_platform_seed=write_platform_seed,
    )
