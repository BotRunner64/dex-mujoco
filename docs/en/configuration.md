# Configuration

## Directory Layout

```
configs/retargeting/
├── base/       ← reusable model-level constraint definitions
├── left/       ← left-hand runtime configs
├── right/      ← right-hand runtime configs
└── bihand/     ← bi-hand viewer/replay configs
```

**CLI defaults:**

| Mode | Default config |
| --- | --- |
| Single hand | `configs/retargeting/right/linkerhand_l20_right.yaml` |
| Bi-hand | `configs/retargeting/bihand/linkerhand_l20_bihand.yaml` |

---

## Single-Hand Config

Typical runtime configs are thin wrappers around a base config:

```yaml
extends: "../base/linkerhand_l20.yaml"

hand:
  name: "linkerhand_l20_right"
  side: "right"
  mjcf_path: "../../../assets/mjcf/linkerhand_l20_right/model.xml"
```

The loader resolves relative paths from the config file location and supports **chained `extends`**.

### Main Sections

#### `hand` — model identity

| Field | Description |
| --- | --- |
| `name` | Model identifier |
| `side` | `left` or `right` |
| `mjcf_path` | Path to the MJCF model file |
| `urdf_source` | (optional) Source URDF metadata |

#### `controller` — backend defaults

| Field | Description |
| --- | --- |
| `backend` | `viewer`, `sim`, or `real` |
| `model_family` | Hardware model family for real backend |
| `control_rate_hz` | Control loop frequency |
| `sim_rate_hz` | Simulation frequency |
| `transport` | Communication transport |
| `can_interface` | CAN bus interface |
| `modbus_port` | Modbus serial port |
| `sdk_root` | Path to LinkerHand SDK |
| `default_speed` / `default_torque` | Hardware defaults |

#### `retargeting` — retargeting behavior

| Pattern | When to use |
| --- | --- |
| `preset: universal` | Default universal constraint preset |
| Explicit `vector_constraints`, `frame_constraints`, `distance_constraints`, `angle_constraints` | Custom models |
| `vector_loss`, `preprocess`, `solver` | Tuning solver behavior |

---

## Bi-Hand Config

Bi-hand configs are separate and compose a left + right config pair:

```yaml
left:
  config: "../left/linkerhand_l20_left.yaml"

right:
  config: "../right/linkerhand_l20_right.yaml"
```

Also supports a `viewer` section for panel size, camera placement, and per-hand pose.

---

## Schema Notes

- `retargeting.preset` only accepts `universal` when set
- **Removed**: `human_vector_pairs`, `origin_link_names`, `task_link_names`, `vector_weights` (legacy vector schema)
- **Rejected**: `position_constraints`, `pinch` (removed sections)
- Runtime validation checks backend names, transport names, and positive control/sim rates

---

## Which File to Edit?

| Goal | Edit |
| --- | --- |
| Change reusable model-level constraints | `base/` |
| Bind a runtime side to a specific asset path | `left/` or `right/` |
| Change bi-hand viewer/replay behavior | `bihand/` |
