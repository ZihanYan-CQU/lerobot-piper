"""Prepare a pinned, MIT-licensed Menagerie model without modifying source assets."""

import hashlib
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import default_model_path

REVISION = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
BASE_URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/agilex_piper"


def prepare_assets(destination: Path | None = None, source: Path | None = None) -> Path:
    """Copy an existing model, or download only files referenced by its MJCF.

    Downloads are fixed to a commit, never executed, and recorded with SHA-256 hashes.
    Existing destination files are never silently overwritten with different contents.
    """
    destination = (destination or default_model_path().parent).resolve()
    if source is not None:
        source = source.resolve()
        if source == destination:
            raise ValueError("Source and destination must be different")
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    def obtain(relative: str) -> bytes:
        if source is not None:
            return (source / relative).read_bytes()
        with urllib.request.urlopen(f"{BASE_URL}/{relative}", timeout=60) as response:
            return response.read()

    def save(relative: str, content: bytes):
        path = destination / relative
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Different asset already exists: {path}; use another destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})

    xml = obtain("piper.xml")
    root = ET.fromstring(xml)
    files = sorted({f"assets/{mesh.attrib['file']}" for mesh in root.findall("./asset/mesh")})
    save("piper.xml", xml)
    for name in ["LICENSE", "README.md", *files]:
        save(name, obtain(name))
    provenance = {
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "revision": REVISION if source is None else "local-copy; see original-source.json when present",
        "files": records,
    }
    if source is not None and (source / "SOURCE.json").is_file():
        shutil.copy2(source / "SOURCE.json", destination / "original-source.json")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination / "piper.xml"
