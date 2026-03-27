"""Hardware detection utilities for SLA validation context."""

import os
import platform

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_hardware_info() -> dict:
    """Collect hardware information relevant to LLM inference performance."""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_physical": 1,
        "cpu_count_logical": 1,
        "cpu_freq_mhz": None,
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "device": "cpu",
    }

    if _HAS_PSUTIL:
        try:
            info["cpu_count_physical"] = psutil.cpu_count(logical=False) or 1
            info["cpu_count_logical"] = psutil.cpu_count(logical=True) or 1
            vm = psutil.virtual_memory()
            info["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
            info["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"] = round(freq.current, 1)
        except Exception:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
            info["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = "Apple MPS"
            info["device"] = "mps"
    except ImportError:
        pass

    return info


def get_device() -> str:
    """Return the best available torch device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
