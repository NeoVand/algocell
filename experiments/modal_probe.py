"""Does the Modal NVIDIA driver export Vulkan at all, and does wgpu's GL (EGL) backend work? Run: modal run modal_probe.py"""

import modal

app = modal.App("algocell-vulkan-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0", "binutils")
    .pip_install("wgpu>=0.32,<0.40", "numpy>=2")
)


@app.function(image=image, gpu="H200", timeout=600)
def probe() -> dict:
    import os
    import subprocess

    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout[-2500:]

    out = {
        "vk_symbols": sh("nm -D /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep -i 'vk_icd\\|vkGetInstanceProcAddr' | head"),
        "vulkan_strings": sh("strings /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep -ci vulkan"),
        "nvidia_vk_libs": sh("ls /usr/lib/x86_64-linux-gnu | grep -i 'vulkan\\|vk\\|glvk'"),
        "egl_vendor": sh("ls -la /usr/share/glvnd/egl_vendor.d/ 2>&1; cat /usr/share/glvnd/egl_vendor.d/*nvidia* 2>&1"),
        "driver_pkg_hint": sh("ls /usr/lib/x86_64-linux-gnu | grep -c nvidia; cat /proc/driver/nvidia/version 2>&1 | head -2"),
    }
    results = {}
    for backend in ("Vulkan", "GL"):
        os.environ["WGPU_BACKEND_TYPE"] = backend
        try:
            r = subprocess.run(
                ["python", "-c", "import wgpu,json; print(json.dumps([f\"{a.info.get('device')} / {a.info.get('adapter_type')} / {a.info.get('backend_type')}\" for a in wgpu.gpu.enumerate_adapters_sync()]))"],
                capture_output=True, text=True, timeout=120, env={**os.environ, "WGPU_BACKEND_TYPE": backend},
            )
            results[backend] = (r.stdout.strip() + " " + r.stderr.strip()[-600:]).strip()
        except Exception as e:  # noqa: BLE001
            results[backend] = repr(e)
    out["wgpu_by_backend"] = results
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=1, default=str))
