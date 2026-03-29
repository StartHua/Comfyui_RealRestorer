from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Repo root (vendored diffusers is loaded lazily — see `_get_realrestorer_pipeline_cls`).
_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_DIFFUSERS_SRC = _REPO_ROOT / "diffusers" / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import comfy.model_management as mm
import folder_paths
from PIL import Image

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
if hasattr(torch, "float8_e4m3fn"):
    DTYPE_MAP["fp8_e4m3fn (transformer only)"] = "fp8_e4m3fn"

_DEVICE_STRATEGIES = frozenset({"model_cpu_offload", "sequential_offload", "full_gpu"})


def _realrestorer_accelerator_device_str() -> str:
    """
    Device string passed to diffusers offload and to manual ``cuda`` placement.

    If ComfyUI runs in CPU-only mode (``--cpu``), ``mm.get_torch_device()`` is ``cpu`` and
    ``apply_pipeline_device_strategy`` would only call ``pipe.to(\"cpu\")`` and **skip**
    ``enable_model_cpu_offload`` — nothing runs on the GPU. When CUDA is still available, use it so
    ``model_cpu_offload`` / ``sequential_offload`` / ``full_gpu`` actually target the accelerator.
    """
    dev = mm.get_torch_device()
    s = str(dev)
    if s.startswith("cuda"):
        return s
    if mm.cpu_mode() and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        out = f"cuda:{idx}"
        print(
            "[Comfyui_RealRestorer] ComfyUI is in CPU mode but CUDA is available; using %s for RealRestorer. "
            "Remove --cpu or use GPU mode so Comfy and this node agree on the device." % out,
            flush=True,
        )
        return out
    return s


def _strip_offload_and_reset_to_cpu(pipe: Any) -> None:
    """
    Remove accelerate / offload hooks and ensure weights are not left on ``meta`` or a stale device map
    before applying a new placement (e.g. sequential → ``full_gpu``, or manual per-module ``.to()``).
    """
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    if getattr(pipe, "hf_device_map", None) is not None and hasattr(pipe, "reset_device_map"):
        try:
            pipe.reset_device_map()
            return
        except Exception:
            pass
    try:
        pipe.to("cpu", silence_dtype_warnings=True)
    except Exception:
        pass


def _ensure_realrestorer_like_pipeline(pipe: Any, *, context: str) -> None:
    if pipe is None:
        raise RuntimeError(f"{context}: pipeline input is required.")
    for name in ("text_encoder", "transformer", "vae"):
        if not hasattr(pipe, name):
            raise RuntimeError(
                f"{context}: value is not a RealRestorer pipeline (missing attribute `{name}`)."
            )
    if not callable(getattr(pipe, "__call__", None)):
        raise RuntimeError(f"{context}: pipeline is not callable.")

_pipeline_cache: dict[tuple[str, str, str, str], Any] = {}
_cache_key_order: list[tuple[str, str, str, str]] = []

# Unconfigured pipeline from ``from_pretrained`` only (no offload hooks yet).
_raw_pipeline_cache: dict[tuple[str, str], Any] = {}
_raw_cache_key_order: list[tuple[str, str]] = []

_CPU_CUDA = ["cpu", "cuda"]

_RealRestorerPipeline_cls: type | None = None


def _purge_diffusers_from_sys_modules() -> None:
    """Unload `diffusers` so we can import the vendored fork (PyPI build has no RealRestorerPipeline)."""
    for key in list(sys.modules.keys()):
        if key == "diffusers" or key.startswith("diffusers."):
            del sys.modules[key]


def _get_realrestorer_pipeline_cls() -> type:
    """
    Import RealRestorerPipeline from this node's `diffusers/src`, not from site-packages.
    ComfyUI or other extensions often import PyPI diffusers first; we must evict that cache once.
    """
    global _RealRestorerPipeline_cls
    if _RealRestorerPipeline_cls is not None:
        return _RealRestorerPipeline_cls

    src = _LOCAL_DIFFUSERS_SRC.resolve()
    if not (src / "diffusers").is_dir():
        raise RuntimeError(
            f"RealRestorer patched diffusers not found at {src / 'diffusers'}. "
            "Ensure the Comfyui_RealRestorer repo includes its local `diffusers` checkout."
        )

    src_s = str(src)
    _purge_diffusers_from_sys_modules()
    if src_s in sys.path:
        sys.path.remove(src_s)
    sys.path.insert(0, src_s)

    from diffusers import RealRestorerPipeline

    _RealRestorerPipeline_cls = RealRestorerPipeline
    return RealRestorerPipeline


def _realrestorer_models_root() -> str:
    return os.path.join(folder_paths.models_dir, "RealRestorer")


# Subfolders of one diffusers pipeline (each may have config.json — not separate top-level models).
_DIFFUSERS_COMPONENT_DIRS = frozenset(
    {
        "text_encoder",
        "transformer",
        "vae",
        "scheduler",
        "tokenizer",
        "processor",
        "feature_extractor",
        "image_encoder",
        "unet",
        "safety_checker",
        "watermarker",
        "vision_encoder",
        "movq",
        "vqgan",
        "vq_model",
        "decoder",
        "encoder",
    }
)


def _is_realrestorer_pipeline_dir(path: str) -> bool:
    """True if ``path`` is one loadable pipeline (HF / RealRestorer layout)."""
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.json")):
        return True
    if os.path.isfile(os.path.join(path, "model_index.json")):
        return True
    # Split layout: transformer/, vae/, text_encoder/ — no single json at root.
    return os.path.isdir(os.path.join(path, "transformer")) and os.path.isdir(os.path.join(path, "vae"))


def _list_subfolder_bundles(root: str) -> list[tuple[str, str]]:
    """Version folders: direct children with config.json, excluding pipeline component dir names."""
    out: list[tuple[str, str]] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        if name in _DIFFUSERS_COMPONENT_DIRS:
            continue
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
            out.append((name, os.path.abspath(p)))
    return out


def resolve_realrestorer_bundle_path(model_subfolder: str | None) -> str:
    """
    Resolve ``ComfyUI/models/RealRestorer`` for ``from_pretrained``.

    HF downloads use ``transformer/``, ``vae/``, ``text_encoder/`` under one root — that is *one* bundle,
    not three choices. Optional ``model_subfolder`` selects ``models/RealRestorer/<name>`` for multiple versions.
    """
    root = _realrestorer_models_root()
    sub = (model_subfolder or "").strip()

    if sub:
        path = os.path.join(root, sub)
        if _is_realrestorer_pipeline_dir(path):
            return os.path.abspath(path)
        raise RuntimeError(
            f"Not a loadable RealRestorer directory: {path}. "
            "Expected config.json or model_index.json at root, or transformer/ + vae/ subfolders."
        )

    if not os.path.isdir(root):
        raise RuntimeError(
            f"Missing models directory: {root}. Create ComfyUI/models/RealRestorer and put the bundle there."
        )

    if _is_realrestorer_pipeline_dir(root):
        return os.path.abspath(root)

    subs = _list_subfolder_bundles(root)
    if len(subs) == 1:
        return subs[0][1]
    if len(subs) == 0:
        raise RuntimeError(
            f"No RealRestorer bundle found under {root}. "
            "Put the Hugging Face files here (e.g. transformer/, vae/, text_encoder/), or one version subfolder."
        )
    names = ", ".join(n for n, _ in subs)
    raise RuntimeError(
        f"Multiple version folders under {root}: {names}. Set model_subfolder to one of these names."
    )


def apply_pipeline_device_strategy(pipe: Any, device_str: str, strategy: str) -> None:
    """
    ``model_cpu_offload`` — diffusers default: whole components on GPU one at a time (good speed / moderate VRAM).
    ``sequential_offload`` — lowest VRAM peak, slowest (submodule streaming via accelerate).
    ``full_gpu`` — all weights on GPU (needs enough VRAM; ~24G may be tight for bf16 full stack).

    Always clears accelerate hooks first so switching e.g. sequential → full_gpu does not raise
    ``ValueError`` about ``enable_sequential_cpu_offload`` + ``.to(cuda)``.
    """
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()

    print(
        f"[Comfyui_RealRestorer] apply_pipeline_device_strategy: strategy={strategy}, device_str={device_str}",
        flush=True,
    )

    if not str(device_str).startswith("cuda"):
        print(
            f"[Comfyui_RealRestorer] WARNING: device_str={device_str} is NOT cuda! "
            "All inference will run on CPU (extremely slow). "
            "Make sure ComfyUI is NOT started with --cpu flag.",
            flush=True,
        )
        pipe.to(device_str)
        return
    s = (strategy or "model_cpu_offload").strip()
    if s not in _DEVICE_STRATEGIES:
        s = "model_cpu_offload"
    if s == "full_gpu":
        _strip_offload_and_reset_to_cpu(pipe)
        try:
            pipe.to(device_str)
            print(f"[Comfyui_RealRestorer] full_gpu: all components on {device_str}", flush=True)
        except ValueError as e:
            raise RuntimeError(
                "RealRestorer full_gpu placement failed after clearing offload hooks. "
                "Try device_strategy=model_cpu_offload or sequential_offload, or reload the pipeline. "
                f"Original error: {e}"
            ) from e
    elif s == "sequential_offload":
        pipe.enable_sequential_cpu_offload(device=device_str)
        print(f"[Comfyui_RealRestorer] sequential_offload: enabled for {device_str}", flush=True)
    else:
        pipe.enable_model_cpu_offload(device=device_str)
        print(
            f"[Comfyui_RealRestorer] model_cpu_offload: enabled for {device_str}. "
            f"_offload_device={getattr(pipe, '_offload_device', 'N/A')}, "
            f"_execution_device={getattr(pipe, '_execution_device', 'N/A')}",
            flush=True,
        )


def apply_manual_component_devices(pipe: Any, te: str, tr: str, va: str, gpu_device: str) -> None:
    """
    Place ``text_encoder`` / ``transformer`` / ``vae`` on ``cpu`` or ``cuda`` (uses Comfy's GPU id for ``cuda``).
    Removes accelerate offload hooks first. Experimental: odd combos can fail or be very slow.
    """
    _strip_offload_and_reset_to_cpu(pipe)

    def _dev(which: str) -> str:
        w = (which or "cpu").lower()
        if w == "cuda":
            return gpu_device
        return "cpu"

    pipe.text_encoder.to(_dev(te))
    pipe.transformer.to(_dev(tr))
    pipe.vae.to(_dev(va))


def _trim_raw_cache(max_entries: int = 1) -> None:
    while len(_raw_cache_key_order) > max_entries:
        old = _raw_cache_key_order.pop(0)
        p = _raw_pipeline_cache.pop(old, None)
        if p is not None:
            del p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_raw_realrestorer_pipeline(
    model_path: str, dtype: torch.dtype, force_reload: bool, RealRestorerPipeline: type
) -> Any:
    key = (model_path, str(dtype))
    if force_reload and key in _raw_pipeline_cache:
        del _raw_pipeline_cache[key]
        if key in _raw_cache_key_order:
            _raw_cache_key_order.remove(key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if key not in _raw_pipeline_cache:
        _trim_raw_cache(max_entries=1)
        
        load_dtype = torch.bfloat16 if dtype == "fp8_e4m3fn" else dtype
        pipe = RealRestorerPipeline.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            local_files_only=True,
        )
        
        if dtype == "fp8_e4m3fn":
            print("[Comfyui_RealRestorer] Casting transformer to fp8_e4m3fn to save VRAM...", flush=True)
            pipe.transformer.to(torch.float8_e4m3fn)
            
        _raw_pipeline_cache[key] = pipe
        if key not in _raw_cache_key_order:
            _raw_cache_key_order.append(key)
    return _raw_pipeline_cache[key]


def realrestorer_run_inference(
    pipe: Any,
    image: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    size_level: int,
    device_str: str,
    log_label: str,
) -> torch.Tensor:
    pipe.set_progress_bar_config(desc="RealRestorer denoise")
    neg = negative_prompt if negative_prompt is not None else ""

    # Diagnostic: show device placement for each component
    for comp_name in ("text_encoder", "transformer", "vae"):
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "parameters"):
            try:
                p = next(comp.parameters())
                has_hook = hasattr(comp, "_hf_hook")
                hook_exec_dev = None
                if has_hook:
                    for m in comp.modules():
                        if hasattr(m, "_hf_hook") and hasattr(m._hf_hook, "execution_device"):
                            hook_exec_dev = m._hf_hook.execution_device
                            break
                print(
                    f"[Comfyui_RealRestorer]   {comp_name}: params_device={p.device}, dtype={p.dtype}, "
                    f"has_hook={has_hook}, hook_exec_device={hook_exec_dev}",
                    flush=True,
                )
            except StopIteration:
                pass

    print(
        "[Comfyui_RealRestorer] Running pipeline: Qwen encode (silent) → denoise tqdm (%s steps). device=%s | %s"
        % (int(steps), device_str, log_label),
        flush=True,
    )
    out_frames: list[torch.Tensor] = []
    for b in range(image.shape[0]):
        pil_in = _tensor_hwc_to_pil(image[b])
        if image.shape[0] > 1:
            print(f"[Comfyui_RealRestorer] Frame {b + 1}/{image.shape[0]}", flush=True)
        result = pipe(
            image=pil_in,
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed) + b,
            size_level=int(size_level),
        )
        pil_out = result.images[0]
        out_frames.append(_pil_to_comfy_image(pil_out))
    print("[Comfyui_RealRestorer] Pipeline finished.", flush=True)
    return torch.stack(out_frames, dim=0)


def _tensor_hwc_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_comfy_image(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _trim_cache(max_entries: int = 1) -> None:
    while len(_cache_key_order) > max_entries:
        old = _cache_key_order.pop(0)
        pipe = _pipeline_cache.pop(old, None)
        if pipe is not None:
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RealRestorerLoadPipeline:
    """Load ``RealRestorerPipeline.from_pretrained`` only (weights under ``models/RealRestorer``). No device placement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "torch_dtype": (list(DTYPE_MAP.keys()), {"default": "bfloat16"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model_subfolder": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "image/restoration"

    def load(self, torch_dtype: str, force_reload: bool, model_subfolder: str | None = None):
        model_path = resolve_realrestorer_bundle_path(model_subfolder)
        dtype = DTYPE_MAP[torch_dtype]
        RealRestorerPipeline = _get_realrestorer_pipeline_cls()
        pipe = get_raw_realrestorer_pipeline(model_path, dtype, force_reload, RealRestorerPipeline)
        return (pipe,)


class RealRestorerApplyDevices:
    """
    Two mutually exclusive modes (do not assume presets and per-module picks combine):

    - ``use_manual_placement=False``: only ``device_strategy`` is applied (accelerate offload). The three
      ``*_device`` widgets are ignored.
    - ``use_manual_placement=True``: only ``text_encoder_device`` / ``transformer_device`` / ``vae_device``
      are applied. ``device_strategy`` is ignored.
    """

    DESCRIPTION = (
        "Preset vs manual are mutually exclusive.\n"
        "• Manual OFF: only device_strategy applies; the three cpu/cuda dropdowns are ignored.\n"
        "• Manual ON: only the three per-module devices apply; device_strategy is ignored."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ANY",),
                "use_manual_placement": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off: use device_strategy only (three module dropdowns ignored). "
                        "On: use the three dropdowns only (device_strategy ignored).",
                    },
                ),
                "device_strategy": (
                    ["model_cpu_offload", "sequential_offload", "full_gpu"],
                    {
                        "default": "model_cpu_offload",
                        "tooltip": "Used only when manual placement is OFF. Needs a GPU offload target: start ComfyUI without --cpu, "
                        "or RealRestorer falls back to pure CPU (no VRAM offload). Lowest VRAM ≈ sequential_offload.",
                    },
                ),
                "text_encoder_device": (
                    list(_CPU_CUDA),
                    {
                        "default": "cpu",
                        "tooltip": "Used only when manual placement is ON. Ignored when manual is OFF.",
                    },
                ),
                "transformer_device": (
                    list(_CPU_CUDA),
                    {
                        "default": "cuda",
                        "tooltip": "Used only when manual placement is ON. Ignored when manual is OFF.",
                    },
                ),
                "vae_device": (
                    list(_CPU_CUDA),
                    {
                        "default": "cuda",
                        "tooltip": "Used only when manual placement is ON. Ignored when manual is OFF.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "apply"
    CATEGORY = "image/restoration"

    def apply(
        self,
        pipeline: Any,
        use_manual_placement: bool,
        device_strategy: str,
        text_encoder_device: str,
        transformer_device: str,
        vae_device: str,
    ):
        _ensure_realrestorer_like_pipeline(pipeline, context="RealRestorerApplyDevices")
        device_str = _realrestorer_accelerator_device_str()
        if bool(use_manual_placement):
            apply_manual_component_devices(
                pipeline,
                text_encoder_device,
                transformer_device,
                vae_device,
                device_str,
            )
        else:
            strat = (device_strategy or "model_cpu_offload").strip()
            if strat not in _DEVICE_STRATEGIES:
                strat = "model_cpu_offload"
            apply_pipeline_device_strategy(pipeline, device_str, strat)
        return (pipeline,)


class RealRestorerRun:
    """Run inference on a configured pipeline (from Load → ApplyDevices)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ANY",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Restore the details and keep the original composition.",
                    },
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "size_level": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/restoration"

    def run(
        self,
        pipeline: Any,
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        size_level: int,
    ):
        _ensure_realrestorer_like_pipeline(pipeline, context="RealRestorerRun")
        device_str = _realrestorer_accelerator_device_str()
        out = realrestorer_run_inference(
            pipeline,
            image,
            prompt,
            negative_prompt,
            seed,
            steps,
            guidance_scale,
            size_level,
            device_str,
            log_label="chain: Load → ApplyDevices → Run",
        )
        return (out,)


class RealRestorerRestore:
    """
    Loads only from disk under ``ComfyUI/models/RealRestorer/`` (``local_files_only=True``).

    **Pipeline weights (HF bundle under ``models/RealRestorer/``)** — see also
    ``diffusers/.../pipeline_realrestorer.py`` (``model_cpu_offload_seq = text_encoder->transformer->vae``):

    - **text_encoder** — ``Qwen2_5_VLForConditionalGeneration`` (+ vision). Rough order **~12–18 GB** in bf16 if
      a multi‑billion VL checkpoint; exact size depends on the published bundle.
    - **transformer** — ``RealRestorerTransformer2DModel`` (large DiT-style stack, e.g. hidden 3072). Rough **~8–16 GB** bf16.
    - **vae** — ``RealRestorerAutoencoderKL``. Rough **~0.3–1.5 GB** bf16.
    - **processor** / **scheduler** — negligible on-disk; mostly CPU for processor.

    **If all three modules sat on GPU at once (``full_gpu``)** summed weights can approach or exceed **24 GB**;
    use ``model_cpu_offload`` (default, same idea as ``infer_realrestorer.py``) or ``sequential_offload`` to cap peaks.

    **Degradation node** (separate): MiDaS depth, blur/noise/rain/moire/reflection sub-networks — typically **sub‑GB to a few GB**
    when loaded; not the same as the restoration bundle above.

    **Chain workflow** (optional ``pipeline`` input): connect ``RealRestorer Load Pipeline`` →
    ``RealRestorer Apply Devices`` → this node **or** use ``RealRestorer Run`` instead of this node.
    When ``pipeline`` is connected, ``torch_dtype`` / ``device_strategy`` / internal load cache are skipped.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Restore the details and keep the original composition.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.05},
                ),
                "size_level": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "torch_dtype": (list(DTYPE_MAP.keys()), {"default": "bfloat16"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "pipeline": ("ANY",),
                "model_subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "device_strategy": (
                    ["model_cpu_offload", "sequential_offload", "full_gpu"],
                    {
                        "default": "model_cpu_offload",
                        "tooltip": "Offload needs a GPU target from ComfyUI (do not use --cpu). Otherwise inference stays on CPU.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "image/restoration"

    def restore(
        self,
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        size_level: int,
        torch_dtype: str,
        force_reload: bool,
        pipeline: Any | None = None,
        model_subfolder: str | None = None,
        device_strategy: str = "model_cpu_offload",
    ):
        device_str = _realrestorer_accelerator_device_str()

        if pipeline is not None:
            _ensure_realrestorer_like_pipeline(pipeline, context="RealRestorerRestore")
            out = realrestorer_run_inference(
                pipeline,
                image,
                prompt,
                negative_prompt,
                seed,
                steps,
                guidance_scale,
                size_level,
                device_str,
                log_label="external pipeline (already configured)",
            )
            return (out,)

        model_path = resolve_realrestorer_bundle_path(model_subfolder)
        dtype = DTYPE_MAP[torch_dtype]
        strat = (device_strategy or "model_cpu_offload").strip()
        if strat not in _DEVICE_STRATEGIES:
            strat = "model_cpu_offload"

        cache_key = (model_path, torch_dtype, device_str, strat)
        RealRestorerPipeline = _get_realrestorer_pipeline_cls()

        if force_reload and cache_key in _pipeline_cache:
            del _pipeline_cache[cache_key]
            if cache_key in _cache_key_order:
                _cache_key_order.remove(cache_key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cache_key not in _pipeline_cache:
            _trim_cache(max_entries=1)
            pipe = RealRestorerPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
            apply_pipeline_device_strategy(pipe, device_str, strat)
            _pipeline_cache[cache_key] = pipe
            if cache_key not in _cache_key_order:
                _cache_key_order.append(cache_key)
        else:
            pipe = _pipeline_cache[cache_key]

        out = realrestorer_run_inference(
            pipe,
            image,
            prompt,
            negative_prompt,
            seed,
            steps,
            guidance_scale,
            size_level,
            device_str,
            log_label="device_strategy=%s" % strat,
        )
        return (out,)


# --- Synthetic degradation (see ``degradation_pipeline/infer.py``) --------------------------------

_degradation_pipeline_cache: dict[tuple[str, str], Any] = {}


def _get_degradation_pipeline(device_str: str, midas_model_type: str) -> Any:
    from degradation_pipeline import DegradationPipeline

    key = (device_str, midas_model_type)
    if key not in _degradation_pipeline_cache:
        _degradation_pipeline_cache[key] = DegradationPipeline(
            device=device_str,
            midas_model_type=midas_model_type,
        )
    return _degradation_pipeline_cache[key]


def _empty_to_none(s: str) -> str | None:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    return s


class RealRestorerDegrade:
    """
    Synthetic degradations from ``DegradationPipeline`` (blur, haze, noise, rain, sr, moire, reflection).
    Mirrors ``python infer_degradation.py --image ... --degradation ... --output ...``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from degradation_pipeline import SUPPORTED_DEGRADATIONS

        return {
            "required": {
                "image": ("IMAGE",),
                "degradation": (list(SUPPORTED_DEGRADATIONS),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "midas_model_type": (
                    "STRING",
                    {"default": "DPT_Large"},
                ),
            },
            "optional": {
                "reflection_ckpt_path": ("STRING", {"default": ""}),
                "reflection_dir": ("STRING", {"default": ""}),
                "reflection_type": (
                    ["random", "focused", "defocused", "ghosting"],
                    {"default": "random"},
                ),
                "fog_texture_dir": ("STRING", {"default": ""}),
                "rain_texture_dir": ("STRING", {"default": ""}),
                "model_input_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "disable_density_averaging": ("BOOLEAN", {"default": False}),
                "disable_realesrgan_degradation": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_json")
    FUNCTION = "degrade"
    CATEGORY = "image/restoration"

    def degrade(
        self,
        image: torch.Tensor,
        degradation: str,
        seed: int,
        midas_model_type: str,
        reflection_ckpt_path: str = "",
        reflection_dir: str = "",
        reflection_type: str = "random",
        fog_texture_dir: str = "",
        rain_texture_dir: str = "",
        model_input_size: int = 512,
        disable_density_averaging: bool = False,
        disable_realesrgan_degradation: bool = False,
    ):
        device = mm.get_torch_device()
        device_str = str(device)
        pipe = _get_degradation_pipeline(device_str, midas_model_type.strip() or "DPT_Large")
        refl_t = reflection_type if reflection_type else "random"
        d_disable = bool(disable_density_averaging)
        r_disable = bool(disable_realesrgan_degradation)

        out_frames: list[torch.Tensor] = []
        metas: list[Any] = []
        for b in range(image.shape[0]):
            pil_in = _tensor_hwc_to_pil(image[b])
            result = pipe(
                pil_in,
                degradation,
                seed=int(seed) + b,
                fog_texture_dir=_empty_to_none(fog_texture_dir),
                rain_texture_dir=_empty_to_none(rain_texture_dir),
                enable_density_averaging=not d_disable,
                enable_realesrgan_degradation=not r_disable,
                model_input_size=int(model_input_size),
                reflection_ckpt_path=_empty_to_none(reflection_ckpt_path),
                reflection_dir=_empty_to_none(reflection_dir),
                reflection_type=refl_t,
            )
            metas.append(result.metadata[0])
            out_frames.append(_pil_to_comfy_image(result.images[0]))

        out = torch.stack(out_frames, dim=0)
        meta_str = json.dumps(metas, ensure_ascii=False, indent=2)
        return (out, meta_str)


NODE_CLASS_MAPPINGS = {
    "RealRestorerLoadPipeline": RealRestorerLoadPipeline,
    "RealRestorerApplyDevices": RealRestorerApplyDevices,
    "RealRestorerRun": RealRestorerRun,
    "RealRestorerRestore": RealRestorerRestore,
    "RealRestorerDegrade": RealRestorerDegrade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealRestorerLoadPipeline": "RealRestorer Load Pipeline",
    "RealRestorerApplyDevices": "RealRestorer Apply Devices",
    "RealRestorerRun": "RealRestorer Run",
    "RealRestorerRestore": "RealRestorer Restore (all-in-one)",
    "RealRestorerDegrade": "RealRestorer Degrade (synthetic)",
}
