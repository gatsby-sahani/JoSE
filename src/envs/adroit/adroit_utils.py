import copy
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from importlib_resources import files

_ADROIT_SRC = pathlib.Path(__file__).parent / "src"
_ASSETS_SRC = _ADROIT_SRC / "assets.xml"
_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_ASSETS_OUT = _ASSETS_DIR / "adroit_assets_include.xml"
_DICE_SRC = pathlib.Path(str(files("myosuite").joinpath("envs/myo/assets/hand/dice.png")))
_DICE_OUT = _ASSETS_DIR / "dice.png"

_CHAIN_SRC = _ADROIT_SRC / "chain.xml"
_TENDON_SRC = _ADROIT_SRC / "tendon_torque_actuation.xml"

_SKIP_TAGS = {"compiler", "size", "option", "visual"}

# Skin-toned material referenced by the pen task's default class.
# Not present in Adroit assets.xml; defined here so geoms that inherit it render.
_MATSKIN = ET.fromstring(
    '<material name="MatSkin" specular="0.3" shininess="0.3"'
    ' reflectance="0.0" rgba="0.7 0.6 0.5 1"/>'
)

# Hand-mount position is fixed across all Adroit task XMLs.
_MOUNT_POS = "0 -0.3 1.45"

# ---------------------------------------------------------------------------
# Coordinate-frame helpers for programmatic object placement
# ---------------------------------------------------------------------------

# Default R_x(-pi/2) mount rotation (used as the starting point for computing
# per-task mount corrections).
_R_MOUNT_DEFAULT = np.array([[1., 0., 0.],
                              [0., 0., 1.],
                              [0., -1., 0.]])
_P_MOUNT = np.array([0., -0.3, 1.45])

# Chain body offsets at zero joint angles (forearm + wrist + palm, from chain.xml).
_P_PALM_CHAIN = np.array([0., 0.01, 0.290])

# S_grasp site position in Adroit palm-body local frame (from chain.xml).
_SGRASP_POS_LOCAL = np.array([0.007, -0.04, 0.07])

# Adroit fingertip offsets from S_grasp in MOUNT-LOCAL frame (R_MOUNT_DEFAULT^T @
# world_offset). These are fixed by the hand anatomy and independent of mount euler.
# Measured from the Adroit gym env at the default mount euler.
_ADROIT_MF_LOCAL = np.array([ 0.004,  0.040, 0.125])   # S_mftip − S_grasp
_ADROIT_FF_LOCAL = np.array([ 0.026,  0.040, 0.121])   # S_fftip − S_grasp (index)
_ADROIT_LF_LOCAL = np.array([-0.041,  0.040, 0.114])   # S_lftip − S_grasp (little)


def _mat_to_euler_xyz(R):
    """3×3 rotation matrix → XYZ Euler angles (rad) for MuJoCo's euler= convention.

    MuJoCo interprets euler="rx ry rz" as R = R_x(rx) @ R_y(ry) @ R_z(rz).
    Extracts rx, ry, rz from that factorisation.
    """
    # R[0,2] = sin(ry)
    sy = np.clip(R[0, 2], -1.0, 1.0)
    cy = np.sqrt(max(1.0 - sy**2, 0.0))
    if cy > 1e-6:
        rx = np.arctan2(-R[1, 2], R[2, 2])
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock (ry = ±π/2)
        rx = np.arctan2(R[2, 1], R[1, 1])
        ry = np.arctan2(sy, cy)
        rz = 0.0
    return np.array([rx, ry, rz])


def _anatomical_frame(sg_pos, mf_tip, ff_tip, lf_tip):
    """Rotation matrix whose columns are consistent anatomical axes.

    col 0 (x): lateral — little-finger → index-finger direction
    col 1 (y): distal  — S_grasp → middle-fingertip direction
    col 2 (z): dorsal  — palm outward normal = cross(x, y)

    All inputs are world-frame positions (or any common frame).
    """
    y = mf_tip - sg_pos
    y /= np.linalg.norm(y)
    x_raw = ff_tip - lf_tip          # index − little: lateral
    z = np.cross(x_raw, y)
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, y, z])


# Adroit anatomical frame in mount-local coordinates (constant — depends only on
# the fixed hand anatomy, not the mount euler).
_R_ANAT_LOCAL = _anatomical_frame(
    np.zeros(3), _ADROIT_MF_LOCAL, _ADROIT_FF_LOCAL, _ADROIT_LF_LOCAL
)


def _fmt_vec(v):
    """Format a length-3 array as a space-separated string for XML attributes."""
    return ' '.join(f'{x:.4f}' for x in v)


def _transfer_myo_to_adroit(myo_xml_rel, pro_sup, body_names=(), site_names=(),
                             mount_correction=None):
    """Transfer MyoHand placements into Adroit world coords, with matching hand pose.

    Two-step process:
      1. Compute the Adroit mount euler via the RELATIVE rotation between the palm-up
         base pose (pro_sup = -1.57, matching Adroit's default mount) and the target
         pro_sup.  This preserves the Adroit hand's natural world orientation and only
         applies the same pronation/supination change that pro_sup applies in MyoHand.
      2. Transfer object positions and orientations via the shared anatomical frame
         (middle-finger direction, lateral axis, palm normal) so they end up in the
         same relationship to the Adroit palm as they have to the MyoHand palm.

    Args:
        myo_xml_rel : path relative to the myosuite package root
        pro_sup     : pro_sup joint angle (rad) at reset in the MyoHand gym env
        body_names  : body names whose pos/euler to transfer
        site_names  : site names whose pos to transfer

    Returns:
        dict with key 'mount_euler' (np.array(3)) plus one entry per requested
        body/site: {name: {'pos': np.array(3), 'euler': np.array(3)}}
    """
    import mujoco as _mujoco

    xml = str(files("myosuite").joinpath(myo_xml_rel))
    m = _mujoco.MjModel.from_xml_path(xml)

    def _run(angle):
        d = _mujoco.MjData(m)
        pid = _mujoco.mj_name2id(m, _mujoco.mjtObj.mjOBJ_JOINT, "pro_sup")
        if pid >= 0:
            d.qpos[m.jnt_qposadr[pid]] = angle
        _mujoco.mj_forward(m, d)
        return d

    def _site_pos(d, name):
        sid = _mujoco.mj_name2id(m, _mujoco.mjtObj.mjOBJ_SITE, name)
        return d.site_xpos[sid].copy()

    # Base pose: pro_sup = -1.57 (palm-up).  Adroit default mount (-1.5708 0 0)
    # corresponds to this posture, so R_delta = I at pro_sup = -1.57.
    d_base = _run(-1.57)
    p_sg_base = _site_pos(d_base, "S_grasp")
    R_anat_base = _anatomical_frame(
        p_sg_base,
        _site_pos(d_base, "MFtip"),
        _site_pos(d_base, "IFtip"),
        _site_pos(d_base, "LFtip"),
    )

    # Target pose at the task-specific pro_sup.
    d = _run(pro_sup)
    p_sg_myo = _site_pos(d, "S_grasp")
    R_anat_myo = _anatomical_frame(
        p_sg_myo,
        _site_pos(d, "MFtip"),
        _site_pos(d, "IFtip"),
        _site_pos(d, "LFtip"),
    )

    # Relative rotation: how much does the palm rotate from base to target?
    # Using world-frame anatomical frames cancels out the full-body posture.
    R_delta = R_anat_myo @ R_anat_base.T

    # Apply that relative rotation to the Adroit default mount.
    # At pro_sup = -1.57: R_delta = I → R_mount = R_MOUNT_DEFAULT (-1.5708 0 0).
    # mount_correction (optional) is applied in mount-local frame (on the right)
    # to flip thumb direction without changing the hand-object relationship.
    _mc = mount_correction if mount_correction is not None else np.eye(3)
    R_mount = R_delta @ _R_MOUNT_DEFAULT @ _mc
    mount_euler = _mat_to_euler_xyz(R_mount)

    # Adroit S_grasp world position with the new mount.
    p_sg_adroit = _P_MOUNT + R_mount @ (_P_PALM_CHAIN + _SGRASP_POS_LOCAL)

    # Adroit anatomical frame in world coords.
    R_anat_adroit = R_mount @ _R_ANAT_LOCAL

    result = {"mount_euler": mount_euler}

    for bname in body_names:
        bid = _mujoco.mj_name2id(m, _mujoco.mjtObj.mjOBJ_BODY, bname)
        p_myo = d.xpos[bid].copy()
        R_myo = d.xmat[bid].reshape(3, 3).copy()
        # Transfer via anatomical frame: same relative position/orientation to the palm.
        p_local = R_anat_myo.T @ (p_myo - p_sg_myo)
        R_local = R_anat_myo.T @ R_myo
        result[bname] = {
            "pos":   p_sg_adroit + R_anat_adroit @ p_local,
            "euler": _mat_to_euler_xyz(R_anat_adroit @ R_local),
        }

    for sname in site_names:
        sid2 = _mujoco.mj_name2id(m, _mujoco.mjtObj.mjOBJ_SITE, sname)
        p_myo = d.site_xpos[sid2].copy()
        p_local = R_anat_myo.T @ (p_myo - p_sg_myo)
        result[sname] = {
            "pos": p_sg_adroit + R_anat_adroit @ p_local,
        }

    return result


def _apply_psj_to_chain(root):
    """Insert PSJ hinge joint before WRJ1 in the wrist body (in-place)."""
    for body in root.iter("body"):
        if body.get("name") == "wrist":
            for i, child in enumerate(body):
                if child.tag == "joint" and child.get("name") == "WRJ1":
                    psj = ET.Element("joint", attrib={
                        "name": "PSJ", "type": "hinge",
                        "axis": "0 0 1", "range": "-1.5708 1.5708", "damping": "0.5",
                    })
                    body.insert(i, psj)
                    return


def _apply_psj_to_assets(root):
    """Insert T_PSJp/T_PSJs fixed tendons before T_WRJ1r in the tendon section (in-place)."""
    for tendon_sec in root:
        if tendon_sec.tag == "tendon":
            for i, child in enumerate(tendon_sec):
                if child.tag == "fixed" and child.get("name") == "T_WRJ1r":
                    t_psjp = ET.Element("fixed", attrib={"name": "T_PSJp", "range": "-.032 0.032"})
                    ET.SubElement(t_psjp, "joint", attrib={"joint": "PSJ", "coef": "0.018"})
                    t_psjs = ET.Element("fixed", attrib={"name": "T_PSJs", "range": "-.032 0.032"})
                    ET.SubElement(t_psjs, "joint", attrib={"joint": "PSJ", "coef": "-0.018"})
                    tendon_sec.insert(i, t_psjs)
                    tendon_sec.insert(i, t_psjp)
                    return


def _apply_psj_to_tendon(root):
    """Insert A_PSJp/A_PSJs general actuators before A_WRJ1r in the actuator section (in-place)."""
    for actuator_sec in root:
        if actuator_sec.tag == "actuator":
            for i, child in enumerate(actuator_sec):
                if child.tag == "general" and child.get("name") == "A_WRJ1r":
                    a_psjp = ET.Element("general", attrib={
                        "name": "A_PSJp", "class": "Adroit", "tendon": "T_PSJp",
                        "ctrlrange": "-1 0", "gainprm": "125", "gear": "1",
                    })
                    a_psjs = ET.Element("general", attrib={
                        "name": "A_PSJs", "class": "Adroit", "tendon": "T_PSJs",
                        "ctrlrange": "-1 0", "gainprm": "125", "gear": "1",
                    })
                    actuator_sec.insert(i, a_psjs)
                    actuator_sec.insert(i, a_psjp)
                    return


def _build_psj_xml_files():
    """Generate PSJ-patched XML helper files in _ASSETS_DIR.

    Writes adroit_chain_psj.xml, adroit_assets_include_psj.xml, and
    adroit_tendon_psj.xml — all in the tracked src/envs/assets/ directory.
    Must be called after adroit_assets_include.xml already exists.
    """
    chain_root = ET.parse(_CHAIN_SRC).getroot()
    _apply_psj_to_chain(chain_root)
    ET.indent(chain_root, space="\t")
    (_ASSETS_DIR / "adroit_chain_psj.xml").write_text(
        ET.tostring(chain_root, encoding="unicode")
    )

    assets_root = ET.parse(_ASSETS_OUT).getroot()
    _apply_psj_to_assets(assets_root)
    ET.indent(assets_root, space="    ")
    (_ASSETS_DIR / "adroit_assets_include_psj.xml").write_text(
        ET.tostring(assets_root, encoding="unicode")
    )

    tendon_root = ET.parse(_TENDON_SRC).getroot()
    _apply_psj_to_tendon(tendon_root)
    ET.indent(tendon_root, space="\t")
    (_ASSETS_DIR / "adroit_tendon_psj.xml").write_text(
        ET.tostring(tendon_root, encoding="unicode")
    )


def build_adroit_assets_include():
    """Generate adroit_assets_include.xml, dice.png, and all Adroit task XMLs.

    adroit_assets_include.xml is built from the Adroit repo's assets.xml:
    strips top-level elements that conflict with task-level XML settings
    (<compiler>, <size>, <option>, <visual>) and rewrites mesh file paths to be
    relative to the output file.

    Task XMLs are generated from the corresponding MyoSuite XMLs, replacing the
    MyoHand body include with the Adroit hand mount and overriding object
    positions/orientations to match the Adroit palm's world frame.

    Object placements are computed programmatically: the MyoHand model is loaded
    via the mujoco C-API at the standard palm-up pose (pro_sup = -1.57 rad), and
    S_grasp is used as the anatomical reference frame common to both hands.
    """
    if not _ASSETS_SRC.exists():
        raise FileNotFoundError(
            f"Adroit source files not found at {_ADROIT_SRC}."
        )

    _ASSETS_DIR.mkdir(exist_ok=True)
    src_dir = _ADROIT_SRC
    out_dir = _ASSETS_OUT.parent
    src_root = ET.parse(_ASSETS_SRC).getroot()

    out = ET.Element("mujocoinclude")
    for child in src_root:
        if child.tag not in _SKIP_TAGS:
            out.append(child)

    # Rewrite file paths in <mesh> and <texture> elements: assets.xml uses
    # paths relative to its own location; make them relative to _ASSETS_OUT.
    for elem in out.iter():
        if "file" in elem.attrib:
            abs_path = (src_dir / elem.attrib["file"]).resolve()
            elem.set("file", os.path.relpath(abs_path, out_dir))

    ET.indent(out, space="    ")
    _ASSETS_OUT.write_text(
        ET.tostring(out, encoding="unicode", xml_declaration=False)
    )

    shutil.copy2(_DICE_SRC, _DICE_OUT)

    # ------------------------------------------------------------------
    # Compute per-task mount euler and object placements.
    #
    # pro_sup values come from the MyoHand gym env at reset (measured):
    #   baoding / die: -1.57 rad (standard palm-up supination)
    #   pen:           -1.50 rad (slightly less supinated)
    #   keyturn:        0.0 rad  (neutral; env zeroes all joints)
    #
    # Step 1: _transfer_myo_to_adroit computes the Adroit mount euler that
    #   aligns the Adroit anatomical frame (middle finger direction, lateral,
    #   palm normal) with the MyoHand anatomical frame at the given pro_sup.
    # Step 2: object positions are transferred as world-frame offsets from
    #   S_grasp — valid once both frames are aligned.
    # ------------------------------------------------------------------
    _baoding = _transfer_myo_to_adroit(
        "envs/myo/assets/hand/myohand_baoding.xml",
        pro_sup=-1.57,
        body_names=["ball1", "ball2"],
        site_names=["target1_site", "target2_site"],
    )
    _die = _transfer_myo_to_adroit(
        "envs/myo/assets/hand/myohand_die.xml",
        pro_sup=-1.57,
        body_names=["Object"],
    )
    # Rz(-pi) flips thumb from bottom to top (thumb-up, matching MyoHand convention).
    _Rz_neg_pi = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    _keyturn = _transfer_myo_to_adroit(
        "envs/myo/assets/hand/myohand_keyturn.xml",
        pro_sup=0.0,
        body_names=["key"],
        mount_correction=_Rz_neg_pi,
    )
    _pen = _transfer_myo_to_adroit(
        "envs/myo/assets/hand/myohand_pen.xml",
        pro_sup=-1.50,
        body_names=["Object"],
    )

    _baoding_xml = pathlib.Path(str(files("myosuite").joinpath(
        "envs/myo/assets/hand/myohand_baoding.xml"
    )))
    _baoding_target_ball = ET.Element("site", attrib={
        "name": "target_ball", "type": "sphere", "size": "0.052",
        "rgba": "2 0 0.2 0.1", "group": "1", "pos": "0.16 0.04 1.52",
    })
    _baoding_kwargs = dict(
        task_name="baoding balls",
        myosuite_xml=_baoding_xml,
        mount_euler=_fmt_vec(_baoding["mount_euler"]),
        body_overrides={
            "ball1": {"pos": _fmt_vec(_baoding["ball1"]["pos"])},
            "ball2": {"pos": _fmt_vec(_baoding["ball2"]["pos"])},
        },
        site_overrides={},
        extra_worldbody=[
            _baoding_target_ball,
            ET.fromstring(
                f'<site name="target1_site" type="sphere" size="0.007"'
                f' rgba="1 .8 .31 1" group="3"'
                f' pos="{_fmt_vec(_baoding["target1_site"]["pos"])}"/>'
            ),
            ET.fromstring(
                f'<site name="target2_site" type="sphere" size="0.007"'
                f' rgba=".84 .59 .53 1" group="3"'
                f' pos="{_fmt_vec(_baoding["target2_site"]["pos"])}"/>'
            ),
        ],
    )
    _build_adroit_task_xml(**_baoding_kwargs, out_path=_ASSETS_DIR / "adroit_baoding.xml")
    _build_adroit_task_xml(**_baoding_kwargs, out_path=_ASSETS_DIR / "adroit_baoding_psj.xml", add_psj=True)

    _die_xml = pathlib.Path(str(files("myosuite").joinpath(
        "envs/myo/assets/hand/myohand_die.xml"
    )))
    _die_kwargs = dict(
        task_name="die reorient",
        myosuite_xml=_die_xml,
        mount_euler=_fmt_vec(_die["mount_euler"]),
        body_overrides={
            "Object": {
                "pos":   _fmt_vec(_die["Object"]["pos"]),
                "euler": _fmt_vec(_die["Object"]["euler"]),
            },
            "target": {"pos": "0.10 0.04 1.52"},
        },
        site_overrides={},
        body_site_overrides={"target": {"target_ball": {"size": "0.052"}}},
    )
    _build_adroit_task_xml(**_die_kwargs, out_path=_ASSETS_DIR / "adroit_die.xml")
    _build_adroit_task_xml(**_die_kwargs, out_path=_ASSETS_DIR / "adroit_die_psj.xml", add_psj=True)

    _keyturn_xml = pathlib.Path(str(files("myosuite").joinpath(
        "envs/myo/assets/hand/myohand_keyturn.xml"
    )))
    _keyturn_target_ball = ET.Element("site", attrib={
        "name": "target_ball", "type": "sphere", "size": "0.052",
        "rgba": "2 0 0.2 0.1", "group": "1", "pos": "0.16 0.04 1.52",
    })
    _keyturn_kwargs = dict(
        task_name="key turn",
        myosuite_xml=_keyturn_xml,
        mount_euler=_fmt_vec(_keyturn["mount_euler"]),
        body_overrides={
            "key": {
                "pos":   _fmt_vec(_keyturn["key"]["pos"]),
                "euler": _fmt_vec(_keyturn["key"]["euler"]),
            },
        },
        site_overrides={},
        body_site_overrides={"key": {"": {"rgba": "0 0 0 0"}}},
        extra_worldbody=[_keyturn_target_ball],
    )
    _build_adroit_task_xml(**_keyturn_kwargs, out_path=_ASSETS_DIR / "adroit_keyturn.xml")
    _build_adroit_task_xml(**_keyturn_kwargs, out_path=_ASSETS_DIR / "adroit_keyturn_psj.xml", add_psj=True)

    _pen_pos = _fmt_vec(_pen["Object"]["pos"])
    _pen_euler = _fmt_vec(_pen["Object"]["euler"])
    _pen_xml = pathlib.Path(str(files("myosuite").joinpath(
        "envs/myo/assets/hand/myohand_pen.xml"
    )))
    _pen_target_ball = ET.Element("site", attrib={
        "name": "target_ball", "type": "sphere", "size": "0.052",
        "rgba": "0.2 1.7 0.2 0.1", "group": "1",
    })
    _pen_kwargs = dict(
        task_name="pen twirl",
        myosuite_xml=_pen_xml,
        mount_euler=_fmt_vec(_pen["mount_euler"]),
        body_overrides={
            "Object": {"pos": _pen_pos, "euler": _pen_euler},
            "target": {"pos": "0.16 0.01 1.52"},
        },
        site_overrides={
            "eps_ball": {"pos": _pen_pos},
        },
        body_extra_children={
            "target": [_pen_target_ball],
        },
        extra_assets=[_MATSKIN],
    )
    _build_adroit_task_xml(**_pen_kwargs, out_path=_ASSETS_DIR / "adroit_pen.xml")
    _build_adroit_task_xml(**_pen_kwargs, out_path=_ASSETS_DIR / "adroit_pen_psj.xml", add_psj=True)

    # Build PSJ-patched XML helper files (must come after adroit_assets_include.xml exists).
    _build_psj_xml_files()


def _build_adroit_task_xml(
    task_name,
    myosuite_xml,
    mount_euler,
    body_overrides,
    site_overrides,
    out_path,
    extra_assets=None,
    extra_worldbody=None,
    body_extra_children=None,
    body_site_overrides=None,
    add_psj=False,
):
    """Generate an Adroit task XML from a MyoSuite task XML.

    mount_euler         : space-separated XYZ euler string for the hand-mount body,
                          computed to match the MyoHand palm orientation at reset.
    body_overrides      : {body_name: {xml_attr: value_string}}
    site_overrides      : {site_name: {xml_attr: value_string}}
    extra_assets        : ET.Element list appended to <asset> (e.g. MatSkin)
    extra_worldbody     : ET.Element list appended to <worldbody> after task objects
    body_extra_children : {body_name: [ET.Element, ...]} inserted at start of that body
    body_site_overrides : {body_name: {site_name: {attr: val}}} overrides attrs on sites inside a body
    add_psj          : if True, use PSJ-patched XML files from assets dir instead of
                       the gitignored Adroit repo files
    """
    src_root = ET.parse(myosuite_xml).getroot()

    if add_psj:
        assets_file = "adroit_assets_include_psj.xml"
        chain_rel = "adroit_chain_psj.xml"
        tendon_rel = "adroit_tendon_psj.xml"
    else:
        assets_file = "adroit_assets_include.xml"
        chain_rel = os.path.relpath(_CHAIN_SRC.resolve(), out_path.parent)
        tendon_rel = os.path.relpath(_TENDON_SRC.resolve(), out_path.parent)

    out = ET.Element("mujoco", attrib={"model": f"Adroit hand for {task_name}"})

    ET.SubElement(out, "compiler", attrib={"angle": "radian"})
    ET.SubElement(out, "option", attrib={"timestep": "0.002"})
    ET.SubElement(out, "include", attrib={"file": assets_file})
    ET.SubElement(out, "include", attrib={"file": tendon_rel})

    # Tags that we handle explicitly; everything else is passed through verbatim
    # (e.g. <tendon> for visual ball-to-target strings in baoding).
    _HANDLED = {"include", "compiler", "option", "default", "asset", "worldbody"}

    # Task-specific defaults (e.g. class="key", class="pen").
    for elem in src_root:
        if elem.tag == "default":
            out.append(copy.deepcopy(elem))

    # Assets: standard scene textures/materials + optional extras + task assets.
    asset_out = ET.SubElement(out, "asset")
    ET.SubElement(asset_out, "texture", attrib={
        "name": "texplane", "type": "2d", "builtin": "checker",
        "rgb1": ".2 .3 .4", "rgb2": ".1 0.15 0.2",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset_out, "material", attrib={
        "name": "MatGnd", "reflectance": "0.5", "texture": "texplane",
        "texrepeat": "2 2", "texuniform": "true",
    })
    for elem in extra_assets or []:
        asset_out.append(copy.deepcopy(elem))
    for elem in src_root:
        if elem.tag == "asset":
            for child in elem:
                child_copy = copy.deepcopy(child)
                # Fix the dice.png path: MyoSuite uses a deep relative path;
                # we auto-copy dice.png to _ASSETS_DIR so just use the filename.
                if child_copy.get("file", "").endswith("dice.png"):
                    child_copy.set("file", "dice.png")
                asset_out.append(child_copy)

    # Worldbody: standard scene + Adroit hand mount + task objects from MyoSuite.
    worldbody = ET.SubElement(out, "worldbody")
    ET.SubElement(worldbody, "light", attrib={
        "directional": "false", "diffuse": ".8 .8 .8", "specular": "0.3 0.3 0.3",
        "pos": "0 1.0 4.0", "dir": "0 -1.0 -4",
    })
    ET.SubElement(worldbody, "geom", attrib={
        "name": "ground", "pos": "0 0 0", "size": "1 1 5", "material": "MatGnd",
        "type": "plane", "contype": "1", "conaffinity": "1",
    })
    hand_mount = ET.SubElement(worldbody, "body", attrib={
        "name": "hand mount", "pos": _MOUNT_POS, "euler": mount_euler,
    })
    ET.SubElement(hand_mount, "inertial", attrib={
        "mass": "0.100", "pos": "0 0 0", "diaginertia": "0.001 0.001 0.001",
    })
    ET.SubElement(hand_mount, "include", attrib={"file": chain_rel})

    for elem in src_root:
        if elem.tag != "worldbody":
            continue
        for child in elem:
            if child.tag == "include":
                continue  # skip <include file=".../myohand_body.xml"/>
            child_copy = copy.deepcopy(child)
            name = child_copy.get("name", "")
            if child.tag == "body" and name in body_overrides:
                for attr, val in body_overrides[name].items():
                    child_copy.set(attr, val)
                if body_site_overrides and name in body_site_overrides:
                    for site_elem in child_copy.iter("site"):
                        sname = site_elem.get("name", "")
                        if sname in body_site_overrides[name]:
                            for attr, val in body_site_overrides[name][sname].items():
                                site_elem.set(attr, val)
                if body_extra_children and name in body_extra_children:
                    for i, extra in enumerate(body_extra_children[name]):
                        child_copy.insert(i, copy.deepcopy(extra))
            elif child.tag == "site" and name in site_overrides:
                for attr, val in site_overrides[name].items():
                    child_copy.set(attr, val)
            worldbody.append(child_copy)

    for elem in extra_worldbody or []:
        worldbody.append(copy.deepcopy(elem))

    # Pass through any other top-level elements verbatim (e.g. <tendon> for
    # the visual strings connecting baoding balls to their target sites).
    for elem in src_root:
        if elem.tag not in _HANDLED:
            out.append(copy.deepcopy(elem))

    ET.indent(out, space="    ")
    out_path.write_text(ET.tostring(out, encoding="unicode", xml_declaration=False))
