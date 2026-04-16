# 资产与模型

## 外部资产模式

运行时资产托管在外部，按需下载 —— **不存放在 Git 仓库中**。

### 资产分组

| 分组 | 远端条目 | 本地目标路径 |
| --- | --- | --- |
| **`mjcf`** | `archives/mjcf_assets.tar.gz` | `assets/mjcf/` |
| **`mediapipe`** | `models/hand_landmarker.task` | `assets/models/hand_landmarker.task` |
| **`examples`** | `archives/reference_assets.tar.gz` | `assets/` |
| **`examples`** | `archives/sample_recordings.tar.gz` | `recordings/` |

### 下载命令

```bash
# 运行所需的最小集合
python scripts/setup/download_assets.py --only mjcf mediapipe

# 样例录制和参考资产
python scripts/setup/download_assets.py --only examples

# 全部下载
python scripts/setup/download_assets.py
```

### 资产仓库

| 来源 | 仓库地址 |
| --- | --- |
| **ModelScope**（默认） | `BingqianWu/somehand-assets` |
| **HuggingFace** | `12e21/somehand-assets` |

---

## 本地资产路径

| 资产 | 预期路径 |
| --- | --- |
| MJCF 模型 | `assets/mjcf/<model_name>/model.xml` |
| MediaPipe landmarker | `assets/models/hand_landmarker.task` |
| 样例录制 | `recordings/` |

> 如果缺少必要资产，运行时会提示你执行 `python scripts/setup/download_assets.py`。

---

## 支持的手模型（20+）

仓库当前可见的配置 stem：

| 类别 | 模型 |
| --- | --- |
| **LinkerHand** | `linkerhand_l6`、`linkerhand_l10`、`linkerhand_l20`、`linkerhand_l20pro`、`linkerhand_l21`、`linkerhand_l25`、`linkerhand_l30`、`linkerhand_lhg20`、`linkerhand_o6`、`linkerhand_o7`、`linkerhand_t12` |
| **Inspire** | `inspire_dfq`、`inspire_ftp` |
| **其他** | `dex5`、`dexhand021`、`omnihand`、`revo2`、`rohand`、`sharpa_wave`、`wujihand` |

**覆盖说明：**

- `left/` 与 `right/` 目录不完全对称
- `bihand/` 只包含部分成对配置
- 真机 backend 的实际支持范围比"有配置文件"更窄

当前可用性应以 `configs/retargeting/{left,right,bihand}` 为准。

---

## URDF 转 MJCF

如果模型起点是 URDF：

```bash
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --urdf path/to/model.urdf --output assets/mjcf/my_hand
```

生成新的 MJCF 资产后：

1. 把运行时资产放到外部资产仓
2. 在 `configs/retargeting/` 下新增或更新匹配配置
3. 验证配置确实可用后再写入文档
