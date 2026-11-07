"""Tabletop scene built around an unmodified Menagerie robot description."""

import xml.etree.ElementTree as ET
from pathlib import Path


def scene_xml(model_path: Path, timestep: float, width: int, height: int) -> str:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Piper model not found: {model_path}. Run piper-sim prepare first, "
            "or provide model_path / PIPER_SIM_ASSETS."
        )
    model = ET.parse(model_path).getroot()
    model.set("model", "lerobot_piper_pick_place")
    model.find("compiler").set("meshdir", str(model_path.parent / "assets"))
    model.find("option").set("timestep", str(timestep))
    # In this pinned vendor revision the finger box colliders extend toward
    # the wrist, opposite to the visible finger meshes. Mirror their local
    # Y offsets in the constructed scene so contact occurs at the fingertips.
    # The cached/vendor XML and all visual meshes remain unmodified.
    for body_name in ("link7", "link8"):
        for geom in model.findall(f".//body[@name='{body_name}']/geom[@type='box']"):
            position = [float(value) for value in geom.attrib["pos"].split()]
            position[1] *= -1
            geom.set("pos", " ".join(map(str, position)))
    # Reset is owned by the simulation, not the vendor's eight-coordinate keyframe.
    keyframe = model.find("keyframe")
    if keyframe is not None:
        model.remove(keyframe)
    ET.SubElement(model, "statistic", center="0.1 0 0.95", extent="1.1")
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth=str(max(width, 640)), offheight=str(max(height, 480)))
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3")
    asset = model.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="floor_texture",
        type="2d",
        builtin="checker",
        rgb1="0.17 0.20 0.24",
        rgb2="0.22 0.25 0.29",
        width="256",
        height="256",
    )
    ET.SubElement(asset, "material", name="floor_material", texture="floor_texture", texrepeat="3 3")
    world = model.find("worldbody")
    world.find("body[@name='base_link']").set("pos", "-0.24 0 0.72")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", material="floor_material")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0.15 0 0.68",
        size="0.65 0.48 0.04",
        rgba="0.63 0.49 0.34 1",
        friction="1 0.005 0.0001",
    )
    ET.SubElement(world, "light", pos="0.5 -0.8 2", dir="-0.2 0.3 -1", diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "body", name="camera_target", pos="0.08 0 0.90")
    for name, pos in {
        "top": "0.85 -0.9 1.5",
        "agentview": "0.85 -0.9 1.5",
        "down": "0.15 0 1.65",
        "sideview": "0.15 1.25 1.3",
    }.items():
        ET.SubElement(
            world, "camera", name=name, pos=pos, mode="targetbody", target="camera_target", fovy="50"
        )
    wrist = model.find(".//body[@name='link6']")
    ET.SubElement(wrist, "camera", name="wrist", pos="-0.08 0 0.025", xyaxes="0 -1 0 -0.6 0 0.8", fovy="70")
    ET.SubElement(
        wrist, "site", name="piper_grasp_site", pos="0 0 0.18", size="0.003", rgba="0.1 0.8 0.3 0.6"
    )
    cube = ET.SubElement(world, "body", name="cube", pos="0.12 -0.06 0.746")
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(
        cube,
        "geom",
        name="cube_geom",
        type="box",
        size="0.02 0.02 0.02",
        mass="0.05",
        rgba="0.85 0.1 0.07 1",
        friction="1.2 0.01 0.001",
        condim="4",
    )
    ET.SubElement(
        world,
        "site",
        name="goal",
        type="box",
        pos="0.12 0.18 0.721",
        size="0.055 0.055 0.001",
        rgba="0.1 0.8 0.2 0.5",
    )
    return ET.tostring(model, encoding="unicode")
