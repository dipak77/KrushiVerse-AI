# KrushiVerseAI v3 autonomous factory

This package is the orchestration layer described in the v3 plan. It uses the
existing `mini` implementation as the source of data, tokenizer, knowledge-graph,
and S20 pretraining work; it does not create a second training stack.

The first executable slice is deliberately limited to the capabilities that
already exist and are v2-compatible:

```text
data_v2 ─┬─> token_v2_8k ─> pretrain_10k (one reserved GPU lane)
         └─> kg_v2
```

The remaining plan tasks are present as `BLOCKED` in the generated runtime DAG
with an explicit reason. This prevents an unattended run from executing the old
v1 SFT, quant, or deploy paths against the v2 model by mistake.

## Start safely

Initialize a local runtime DAG (it is ignored by Git):

```powershell
python -m factory.planner init
python -m factory.planner status
```

Preview ready work without launching it:

```powershell
python -m factory.planner run
```

Run the first data stage only after checking your source registry and available
disk/network access. `--execute` is intentionally required for all writes and
worker launches:

```powershell
python -m factory.planner run --auto --execute --max-cpu-workers 2
```

Run the monitor in another terminal:

```powershell
python -m factory.monitor --interval 30
```

Open the live dashboard after the monitor has written its first status file:

```powershell
streamlit run factory/dashboard.py
```

`factory/STATUS.json` contains task state, heartbeats, CPU/RAM information, and
the GPU lock holder. GPU work is not started unless CUDA is available and at
least 3GB of free VRAM is reported; the lock is held from the planner reservation
until the worker finishes or fails.

For a Windows background run that survives the terminal, use the included
`run_planner.cmd` and `run_monitor.cmd` with Windows Task Scheduler.

The scheduled `run_status_report.cmd` appends a concise workflow and pretrain
percentage line to `factory/status.log`. To watch it in a terminal:

```powershell
Get-Content factory\status.log -Wait
```
