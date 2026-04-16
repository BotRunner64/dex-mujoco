# Assets and Models

## External Asset Model

Runtime assets are hosted externally and downloaded on demand — they are **not stored in Git**.

### Asset Groups

| Group | Remote entry | Local destination |
| --- | --- | --- |
| **`mjcf`** | `archives/mjcf_assets.tar.gz` | `assets/mjcf/` |
| **`mediapipe`** | `models/hand_landmarker.task` | `assets/models/hand_landmarker.task` |
| **`examples`** | `archives/reference_assets.tar.gz` | `assets/` |
| **`examples`** | `archives/sample_recordings.tar.gz` | `recordings/` |

### Download Commands

```bash
# Minimum required for running
python scripts/setup/download_assets.py --only mjcf mediapipe

# Sample recordings and reference assets
python scripts/setup/download_assets.py --only examples

# Everything
python scripts/setup/download_assets.py
```

### Asset Repositories

| Source | Repository |
| --- | --- |
| **ModelScope** (default) | `BingqianWu/somehand-assets` |
| **HuggingFace** | `12e21/somehand-assets` |

---

## Local Asset Paths

| Asset | Expected path |
| --- | --- |
| MJCF models | `assets/mjcf/<model_name>/model.xml` |
| MediaPipe landmarker | `assets/models/hand_landmarker.task` |
| Sample recordings | `recordings/` |

> If a required asset is missing, the runtime will point you to `python scripts/setup/download_assets.py`.

---

## Supported Hand Models (20+)

Current config stems in the repository:

| Category | Models |
| --- | --- |
| **LinkerHand** | `linkerhand_l6`, `linkerhand_l10`, `linkerhand_l20`, `linkerhand_l20pro`, `linkerhand_l21`, `linkerhand_l25`, `linkerhand_l30`, `linkerhand_lhg20`, `linkerhand_o6`, `linkerhand_o7`, `linkerhand_t12` |
| **Inspire** | `inspire_dfq`, `inspire_ftp` |
| **Others** | `dex5`, `dexhand021`, `omnihand`, `revo2`, `rohand`, `sharpa_wave`, `wujihand` |

**Coverage notes:**

- `left/` and `right/` directories are not perfectly symmetric
- `bihand/` contains only a subset of paired configs
- Real-backend support is narrower than config coverage

Use `configs/retargeting/{left,right,bihand}` as the source of truth for current availability.

---

## URDF to MJCF Conversion

For models starting as URDF:

```bash
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --urdf path/to/model.urdf --output assets/mjcf/my_hand
```

After generating a new MJCF asset:

1. Store the artifact in the external asset repository
2. Add or update the matching config under `configs/retargeting/`
3. Verify the config works before documenting it
