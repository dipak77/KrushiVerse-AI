"""Rich live Streamlit dashboard for the autonomous factory with task & subtask progress, records, and ETAs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_status(factory_dir: str | Path = "factory") -> dict[str, Any]:
    path = Path(factory_dir) / "STATUS.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": "STATUS.json is being updated; refresh in a moment."}


def render_dashboard(factory_dir: str | Path = "factory") -> None:
    import streamlit as st

    status = load_status(factory_dir)
    if not status:
        st.warning("No STATUS.json found. Initializing monitor...")
        return

    summary = status.get("summary") or {}
    total_tasks = sum(int(v) for v in summary.values())
    completed_tasks = int(summary.get("COMPLETED", 0))
    overall_pct = (completed_tasks / total_tasks * 100.0) if total_tasks else 0.0

    st.progress(
        overall_pct / 100.0,
        text=f"Overall Factory Progress: {completed_tasks}/{total_tasks} Tasks Completed ({overall_pct:.1f}%)",
    )

    # Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Processed Records", f"{status.get('total_processed_records', 0):,}")
    m2.metric("Pending Records", f"{status.get('total_pending_records', 0):,}")
    m3.metric("Running Jobs", summary.get("RUNNING", 0))
    m4.metric("Completed Jobs", summary.get("COMPLETED", 0))
    m5.metric("Blocked Jobs", summary.get("BLOCKED", 0))

    # GPU Hardware Section
    gpu = status.get("gpu") or {}
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("⚡ GPU Acceleration: RTX 2050 4GB")
        holder = gpu.get("holder", "None (Mutex Free)")
        st.info(f"**GPU Lock State:** `{gpu.get('state', 'FREE')}` | **Active Lock Holder:** `{holder}`")
    with c2:
        sys = status.get("system") or {}
        st.subheader("💻 System Telemetry")
        st.write(f"**CPU Usage:** {sys.get('cpu_percent', 0)}% (12 Cores)")
        st.write(f"**RAM Used:** {sys.get('ram_used_percent', 0)}% of 16GB")

    # Detailed Tasks Table
    st.markdown("---")
    st.subheader("📋 Task & Subtask Live Breakdown")

    table_rows = []
    for task in status.get("tasks") or []:
        subtasks_str = " -> ".join(task.get("subtasks") or [])
        table_rows.append(
            {
                "Sprint": task.get("sprint", "S18"),
                "Task ID": task.get("id"),
                "Status": task.get("status"),
                "Progress (%)": f"{task.get('progress_pct', 0.0):.1f}%",
                "Processed Records": f"{task.get('processed_records', 0):,}",
                "Pending Records": f"{task.get('pending_records', 0):,}",
                "Total Records": f"{task.get('total_records', 0):,}",
                "ETA / Est. Duration": task.get("eta", "N/A"),
                "Subtasks": subtasks_str,
            }
        )

    try:
        st.dataframe(table_rows, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    st.caption(f"🟢 Auto-Refreshing Live Status (5s interval) | Updated At: {status.get('generated_at', 'unknown')}")


def render(factory_dir: str | Path = "factory") -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install streamlit to run the factory dashboard") from exc

    st.set_page_config(page_title="KrushiVerseAI v3 Factory Dashboard", layout="wide")
    st.title("🌾 KrushiVerseAI v3 — Autonomous Factory Live Monitor")

    # Native fragment run_every=5 for 5-second auto refresh
    if hasattr(st, "fragment"):
        @st.fragment(run_every=5)
        def live_fragment():
            render_dashboard(factory_dir)
        live_fragment()
    else:
        render_dashboard(factory_dir)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--factory-dir", default="factory")
    args, _ = parser.parse_known_args()
    render(args.factory_dir)


if __name__ == "__main__":
    main()
