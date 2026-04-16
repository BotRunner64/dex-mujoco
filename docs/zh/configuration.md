# 配置说明

## 目录结构

```
configs/retargeting/
├── base/       ← 可复用的模型级约束定义
├── left/       ← 左手运行配置
├── right/      ← 右手运行配置
└── bihand/     ← 双手 viewer/replay 配置
```

**CLI 默认路径：**

| 模式 | 默认配置 |
| --- | --- |
| 单手 | `configs/retargeting/right/linkerhand_l20_right.yaml` |
| 双手 | `configs/retargeting/bihand/linkerhand_l20_bihand.yaml` |

---

## 单手配置

典型运行配置是对基础配置的轻量包装：

```yaml
extends: "../base/linkerhand_l20.yaml"

hand:
  name: "linkerhand_l20_right"
  side: "right"
  mjcf_path: "../../../assets/mjcf/linkerhand_l20_right/model.xml"
```

加载器会从配置文件位置解析相对路径，并支持**链式 `extends`**。

### 主要字段

#### `hand` —— 模型身份

| 字段 | 说明 |
| --- | --- |
| `name` | 模型标识 |
| `side` | `left` 或 `right` |
| `mjcf_path` | MJCF 模型文件路径 |
| `urdf_source` | （可选）源 URDF 元数据 |

#### `controller` —— backend 默认值

| 字段 | 说明 |
| --- | --- |
| `backend` | `viewer`、`sim` 或 `real` |
| `model_family` | 真机 backend 的硬件型号族 |
| `control_rate_hz` | 控制频率 |
| `sim_rate_hz` | 仿真频率 |
| `transport` | 通信方式 |
| `can_interface` | CAN 总线接口 |
| `modbus_port` | Modbus 串口 |
| `sdk_root` | LinkerHand SDK 路径 |
| `default_speed` / `default_torque` | 硬件默认值 |

#### `retargeting` —— retargeting 行为

| 用法 | 适用场景 |
| --- | --- |
| `preset: universal` | 默认通用约束 preset |
| 显式 `vector_constraints`、`frame_constraints`、`distance_constraints`、`angle_constraints` | 自定义模型 |
| `vector_loss`、`preprocess`、`solver` | 调参 |

---

## 双手配置

双手配置独立于单手，通过组合左右手配置实现：

```yaml
left:
  config: "../left/linkerhand_l20_left.yaml"

right:
  config: "../right/linkerhand_l20_right.yaml"
```

还支持 `viewer` 段，用于定义面板大小、相机位置和左右手姿态。

---

## Schema 约束

- `retargeting.preset` 如果设置，只能是 `universal`
- **已废弃**：`human_vector_pairs`、`origin_link_names`、`task_link_names`、`vector_weights`（旧版向量 schema）
- **被拒绝**：`position_constraints`、`pinch`（已移除的字段）
- 运行时校验会检查 backend 名称、transport 名称，以及控制/仿真频率必须为正

---

## 该改哪个文件？

| 目标 | 修改位置 |
| --- | --- |
| 调整可复用的模型级约束 | `base/` |
| 绑定某一侧的运行配置与资产路径 | `left/` 或 `right/` |
| 调整双手 viewer/replay 行为 | `bihand/` |
