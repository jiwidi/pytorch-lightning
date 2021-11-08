# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
from typing import Dict, List

import torch


def get_nvidia_gpu_stats(device: torch.device) -> Dict[str, float]:
    """Get GPU stats including memory, fan speed, and temperature from nvidia-smi.

    Args:
        device: GPU device for which to get stats

    Returns:
        A dictionary mapping the metrics to their values.

    Raises:
        FileNotFoundError:
            If nvidia-smi installation not found
    """
    gpu_stat_metrics = [
        ("utilization.gpu", "%"),
        ("memory.used", "MB"),
        ("memory.free", "MB"),
        ("utilization.memory", "%"),
        ("fan.speed", "%"),
        ("temperature.gpu", "°C"),
        ("temperature.memory", "°C"),
    ]
    gpu_stat_keys = [k for k, _ in gpu_stat_metrics]
    gpu_query = ",".join(gpu_stat_keys)

    gpu_id = _get_gpu_id(device.index)
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found")
    result = subprocess.run(
        [nvidia_smi_path, f"--query-gpu={gpu_query}", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.0

    s = result.stdout.strip()
    stats = [_to_float(x) for x in s.split(", ")]

    gpu_stats = {}
    for i, (x, unit) in enumerate(gpu_stat_metrics):
        gpu_stats[f"{x} ({unit})"] = stats[i]
    return gpu_stats


def _get_gpu_id(device_id: int) -> str:
    """Get the unmasked real GPU IDs."""
    # All devices if `CUDA_VISIBLE_DEVICES` unset
    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices: List[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    return cuda_visible_devices[device_id].strip()
