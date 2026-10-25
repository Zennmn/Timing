"""Pinned frazer/becruily BS-Roformer, direct mixture to lead vocal."""
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MODELS
from .audio import validate_stems
from .device import require_gpu, empty_cache

FOLDER = MODELS / "bs-karaoke"
CHECKPOINT = "bs_roformer_karaoke_frazer_becruily.ckpt"
CONFIG = "config_karaoke_frazer_becruily.yaml"
SHA256 = "eb90ee24c1154d83fbcfd27e96182f19e061557cc6e4746953125e08c29389f9"


def ready():
    return all((FOLDER / name).is_file() for name in (
        CHECKPOINT, CONFIG, "assets.json", "runtime/models/bs_roformer/bs_roformer.py"))


def separate(source: Path, output_dir: Path, log=print):
    import torch
    import yaml
    from types import SimpleNamespace
    from mel_band_roformer.utils import demix_track

    if not ready():
        raise RuntimeError("BS-Karaoke 模型未就绪，请运行 scripts/download_bs_karaoke.py。")
    device = require_gpu("主唱分离")
    manifest = json.loads((FOLDER / "assets.json").read_text(encoding="utf-8"))
    # Fail closed if the checkpoint or executable runtime was altered.
    for relative, asset in manifest["assets"].items():
        if hashlib.sha256((FOLDER / relative).read_bytes()).hexdigest() != asset["sha256"]:
            raise RuntimeError(f"BS-Karaoke 文件校验失败：{relative}")
    if manifest["assets"][CHECKPOINT]["sha256"] != SHA256:
        raise RuntimeError("BS-Karaoke 权重与固定版本不一致。")
    mix_path = output_dir / "bs_input.wav"
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                           "-i", str(source), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(mix_path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode:
        raise ValueError("音频准备失败：" + proc.stderr[-1000:])
    sys.path.insert(0, str(FOLDER / "runtime"))
    from models.bs_roformer.bs_roformer import BSRoformer
    cfg = yaml.load((FOLDER / CONFIG).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in cfg.items()})
    log("BS-Roformer Karaoke：从所选音轨提取主唱…")
    model = BSRoformer(**cfg["model"])
    try:
        model.load_state_dict(torch.load(FOLDER / CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        wave, rate = sf.read(mix_path, dtype="float32", always_2d=True)
        sources, _ = demix_track(config, model, torch.from_numpy(wave.T.copy()), device)
        lead = sources["Vocals"]
        if lead.ndim == 3:
            lead = lead[0]
        lead = lead.T
        if lead.shape != wave.shape or not np.isfinite(lead).all():
            raise RuntimeError("主唱模型输出不完整，已停止。")
        lead_path, residual_path = output_dir / "lead_vocals.wav", output_dir / "residual.wav"
        sf.write(lead_path, lead, rate, subtype="FLOAT")
        sf.write(residual_path, wave - lead, rate, subtype="FLOAT")
        info = validate_stems(mix_path, lead_path, residual_path)
        info.update(model="frazer/becruily BS-Roformer Karaoke", revision=manifest["revision"],
                    checkpoint_sha256=SHA256, input=str(source), route="bs_lead",
                    residual_includes_accompaniment=True,
                    lead=lead_path.name, residual=residual_path.name, quality="unreviewed",
                    note="残余轨包含伴奏与和声，不能当作独立背景人声。")
        (output_dir / "lead_separation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return lead_path, info
    finally:
        model.cpu()
        del model
        gc.collect()
        empty_cache()
