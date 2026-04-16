# Runtime Modes

## CLI Overview

| Command | Purpose | Input | Output |
| --- | --- | --- | --- |
| **`webcam`** | Live hand tracking from camera | Webcam device | viewer / sim / real |
| **`video`** | Offline tracking from video file | MP4 etc. | viewer / sim / real |
| **`replay`** | Replay a saved recording | `.pkl` file | viewer / sim / real |
| **`dump-video`** | Render a recording to MP4 | `.pkl` file | MP4 file |
| **`pico`** | Live tracking via XRoboToolkit | XRoboToolkit stream | viewer / sim / real |
| **`hc-mocap`** | Live tracking from hc_mocap UDP | UDP packets | viewer / sim / real |

---

## Common Options

Shared by most commands:

| Option | Description |
| --- | --- |
| `--config` | Retargeting config YAML path |
| `--hand {left,right,both}` | Hand selector |
| `--backend {viewer,sim,real}` | Output backend |
| `--record-output` | Save input as replayable `.pkl` recording |

Additional control options (live commands):

`--control-rate` · `--sim-rate` · `--transport` · `--can-interface` · `--modbus-port` · `--sdk-root` · `--model-family`

---

## Mode Details

### `webcam`

Live retargeting from a webcam.

```bash
somehand webcam --camera 0
```

| Option | Description |
| --- | --- |
| `--swap-hands` | Fix inverted left/right labels from MediaPipe |
| `--record-output <path>` | Save the session for later replay |

### `video`

Offline retargeting from a video file.

```bash
somehand video --video input.mp4
```

Use `--swap-hands` for mirrored or mislabeled footage.

### `replay`

Replay a saved recording in real time.

```bash
somehand replay --recording recordings/webcam_hand.pkl
```

Use `--loop` for continuous replay.

### `dump-video`

Render a recording to MP4 (as fast as possible, not real-time).

```bash
somehand dump-video \
    --recording recordings/webcam_hand.pkl \
    --output recordings/webcam_hand_replay.mp4
```

### `pico`

Live hand tracking via XRoboToolkit / PICO.

```bash
somehand pico --hand right
```

| Option | Description |
| --- | --- |
| `--signal-fps` | Resample the incoming stream |
| `--pico-timeout` | Startup waiting time |

> Requires XRoboToolkit installed and actively publishing hand-tracking data.

### `hc-mocap`

Live hand data from hc_mocap UDP.

```bash
somehand hc-mocap --hand right --udp-port 1118
```

| Option | Description |
| --- | --- |
| `--signal-fps` | Resample the live stream |
| `--reference-bvh` | Override the built-in joint ordering |
| `--udp-host` / `--udp-port` | Network settings |
| `--udp-timeout` / `--udp-stats-every` | Connection tuning |

---

## Important Limitations

- **Bi-hand (`--hand both`)** is only supported with `--backend viewer` for live and replay commands
- **`dump-video`** supports bi-hand rendering, but it is recording-based, not live
- **Real backend** is currently single-hand only
- **`pico`** depends on XRoboToolkit availability
- **LinkerHand real control** depends on LinkerHand SDK and correct `model_family` mapping
