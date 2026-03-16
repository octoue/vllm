#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Setup environment variables to disable NVLink and force PCIe-only paths
# for PCIe bandwidth contention profiling (KV offload, PP, Prefetch).
#
# Usage: source tools/profiler/setup_pcie_env.sh
#   or:  . tools/profiler/setup_pcie_env.sh

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=0

# Optional: enable debug output to verify transport selection
# export NCCL_DEBUG=INFO

echo "PCIe profiling env: NCCL_P2P_DISABLE=1, NCCL_NVLS_ENABLE=0"
