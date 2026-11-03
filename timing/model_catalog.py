"""Installation metadata only; importing this module never loads/downloads models."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    label: str
    folder: str
    purpose: str
    installer: str


# The four required model groups. Installation always provisions every group;
# there are no optional profiles and no experimental entries.
MODELS = {
    "ctc": ModelSpec("MMS-300M", "ctc", "Multilingual MMS alignment and TIFA sentence location", "download_ctc.py"),
    "tifa": ModelSpec("TIFA-1.0-ST", "tifa", "Mandarin/Japanese/English singing alignment", "download_tifa.py"),
    "bs_lead": ModelSpec("BS-Roformer Karaoke", "bs-karaoke", "Lead vocals from the original mix or an imported stem", "download_bs_karaoke.py"),
    "separator": ModelSpec("Kim MelBand vocals", "separator/melband-roformer-kim-vocals", "Accompaniment removal keeping harmonies; vocals mode and background lyrics", "download_separator.py"),
}


def folder_size(folder):
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) if folder.is_dir() else 0


def inventory(models_dir: Path):
    """Read-only folder sizes for the four required groups; never deletes."""
    rows = []
    for name, spec in MODELS.items():
        folder = models_dir / spec.folder
        rows.append({"id": name, "label": spec.label, "folder": spec.folder,
                     "present": folder.is_dir(), "bytes": folder_size(folder)})
    return rows
