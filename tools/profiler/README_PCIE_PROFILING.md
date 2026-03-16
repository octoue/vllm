# PCIe Bandwidth Profiling

This document describes how to profile PCIe bandwidth usage and contention for vLLM operations: KV Cache Offloading (Evict/Restore), Pipeline Parallelism (PP) transfers, and KV Prefetch.

## Disabling NVLink (Force PCIe Path)

On machines with NVLink (e.g., 8x A100), you must disable NVLink to force all GPU-to-GPU and GPU-to-host transfers over PCIe for accurate profiling.

### Setup Script

```bash
source tools/profiler/setup_pcie_env.sh
```

Or manually:

```bash
export NCCL_P2P_DISABLE=1    # Disable all NCCL P2P (including NVLink)
export NCCL_NVLS_ENABLE=0    # Disable NVLink SHARP
export NCCL_SHM_DISABLE=0    # Ensure SHM transport is available (fallback)
```

### Verification

1. Enable NCCL debug output:
   ```bash
   export NCCL_DEBUG=INFO
   ```

2. Start vLLM with pipeline parallelism (e.g., `--pipeline-parallel-size 2`).

3. Check the logs for transport selection. You should see:
   - `SHM` or `NET` transport being used
   - No `P2P` or `NVLink` in the transport chain

4. If you see `P2P` or `NVLink` in the logs, verify that `NCCL_P2P_DISABLE=1` is set before the process starts.

## Enabling PCIe Trace Collection

Set the following before starting vLLM:

```bash
export VLLM_PCIE_TRACE=1
```

This enables the PCIeTracer to record transfer events. Events are written to `pcie_events.json` when the profiler stops (see `--torch-profiler-dir` or the profiler API).

## Running with Profiling

```bash
source tools/profiler/setup_pcie_env.sh
export VLLM_PCIE_TRACE=1

vllm serve <model> \
  --gpu-profiler torch \
  --torch-profiler-dir ./profiler_output \
  --pipeline-parallel-size 2 \
  --kv-offloading-size 4
```

Then trigger profiling via API:

```bash
curl localhost:8000/start_profile
# ... send test requests ...
curl localhost:8000/stop_profile
```

## Visualizing Results

After stopping the profiler, PCIe events are saved to
`{torch_profiler_dir}/pcie_events_{rank}.json` (one file per GPU worker).

For single-GPU or to visualize one worker:

```bash
python tools/profiler/visualize_pcie_gantt.py \
  --input profiler_output/pcie_events_0.json \
  --output pcie_gantt.html
```

For multi-GPU, merge events from all ranks (events are appended):

```bash
# Example: merge rank 0 and 1
python -c "
import json
events = []
for r in [0, 1]:
    with open(f'profiler_output/pcie_events_{r}.json') as f:
        events.extend(json.load(f))
with open('pcie_events_merged.json', 'w') as f:
    json.dump(events, f, indent=2)
"
python tools/profiler/visualize_pcie_gantt.py -i pcie_events_merged.json -o pcie_gantt.html
```

Open `pcie_gantt.html` in a browser for an interactive Gantt chart of PCIe bandwidth usage.
