## 文档：为 `EasyHOI` 项目准备优化流程的输入数据

### 1. 优化流程概述

`EasyHOI` 的核心优化流程 (`optim_easyhoi.py`) 分为三个主要阶段，依次执行：

1. **第一阶段 (Camera Setup - `optim_obj_cam`)**:

   * **目标**: 固定手部姿态，优化一个虚拟相机来观察3D重建的物体，使其渲染出的2D剪影与图像中的物体分割掩码（mask）对齐。
   * **产出**: 获得物体在统一相机视角下的正确位姿，并计算出物体表面的潜在接触点（`obj_contact`）和多层深度图（`object_depth`）。
2. **第二阶段 (Contact Alignment - `run_handpose_global`)**:

   * **目标**: 固定物体位姿，主要优化**整只手**的**全局参数**（平移、旋转、缩放），使手部模型的2D剪影与图像对齐，并通过ICP算法使手-物三维接触点对齐。
   * **产出**: 得到手部模型在场景中一个粗略但物理上合理的全局位姿。
3. **第三阶段 (Hand Refine - `run_handpose_refine`)**:

   * **目标**: 固定物体的位姿和手的**全局**位姿，精细优化手的**局部**关节姿态（`fullpose`的手指部分）和形状参数（`betas`）。
   * **产出**: 得到最终的、手指姿态自然的、与物体紧密交互的精确手部姿态。

**结论先行**: 第三阶段的输入，是前两个阶段优化后的系统状态。因此，准备输入数据，就是要准备能让第一阶段顺利启动的**所有原始数据**。

### 2. 数据依赖关系

第三阶段 `run_handpose_refine` 的直接输入是 `HOI_Sync` 对象在经历了一、二阶段后的内部状态，关键数据包括：

* `self.data`: 从始至终携带所有原始输入数据的字典。
* `self.mano_params`: 经过第二阶段全局对齐后的手部参数。
* `self.global_params`: 经过第二阶段全局对齐后的手部变换参数。
* `self.obj_contact`: 在第一阶段计算出的、固定的物体接触点。
* `self.data['object_sdf']`: 描述物体形状的符号距离场。

### 3. 核心：如何准备初始输入数据

为了运行完整的优化流程，您需要按照特定的目录结构和文件格式准备以下数据。

#### 3.1. 推荐目录结构

建议您将所有相关数据按以下结构组织：

```
<your_base_data_dir>/
│
├── input_images/                 # 存放原始的、未经处理的输入图像
│   ├── example1.png
│   └── example2.jpg
│
├── hamer_output/                 # 存放初始手部姿态估计的结果 (如 HAMER)
│   ├── example1.pt
│   ├── example1_cam.json
│   ├── example2.pt
│   └── example2_cam.json
│
├── segmentation_masks/           # 存放所有2D分割掩码
│   │
│   ├── hand_mask/                # 手部掩码
│   │   ├── example1.png
│   │   └── example2.png
│   │
│   ├── obj_mask/                 # 物体掩码
│   │   ├── example1.png
│   │   └── example2.png
│   │
│   └── inpaint_mask/             # 手物交互区域的inpainting掩码
│       ├── example1.png
│       └── example2.png
│
└── object_recon/                 # 存放3D物体重建的结果
    │
    ├── inpaint/                  # (可选) inpainting模型的相关输出
    │   └── hoi_box/
    │       ├── example1.json
    │       └── example2.json
    │
    ├── example1/
    │   ├── fixed.obj             # 【必须】重建的3D物体模型
    │   └── sdf.npy               # (可选) 预计算的SDF体素
    │
    └── example2/
        ├── fixed.obj
        └── sdf.npy
```

#### 3.2. 文件格式详解

##### a. 原始数据 (`input_images/`)

* **格式**: 标准图像格式，如 `.png` 或 `.jpg`。
* **内容**: 包含手与物体交互场景的原始照片。

##### b. 分割掩码 (`segmentation_masks/`)

* **格式**: 8位灰度图 (`L` mode), `.png` 格式。
* **`hand_mask/`**: 手部区域的掩码。代码逻辑中，**手部区域像素值为0**，背景为255。
* **`obj_mask/`**: 物体区域的掩码。**物体区域像素值 > 0**，背景为0。
* **`inpaint_mask/`**: 移除了手的inpainting结果的有效区域掩码。**有效区域（即原图中的物体+手）像素值 > 0**，背景为0。

##### c. 初始手部姿态 (`hamer_output/`)

这是来自一个预训练模型（如[HAMER](https://github.com/yu-log/HAMER)）的输出，为优化提供了初始的手部姿态。

* **`<name>.pt`**:

  * **格式**: 使用 `torch.save()` 保存的PyTorch字典。
  * **必须包含的键 (Keys)**:
    * `mano_params`: 一个字典，包含MANO模型参数的PyTorch张量。
      * `global_orient`: `(N, 1, 3, 3)` 旋转矩阵。
      * `hand_pose`: `(N, 1, 15, 3, 3)` 旋转矩阵。
      * `betas`: `(N, 10)` 手部形状参数。
    * `is_right`: `(N)` 布尔值张量，`True`表示右手。
    * `cam_transl`: `(N, 3)` 手部在相机坐标系下的平移。
    * `boxes`: `(N, 4)` 检测到的手部边界框。
    * `pred_cam`: `(N, 3)` 预测的相机参数(s, tx, ty)。
* **`<name>_cam.json`**:

  * **格式**: JSON文件。
  * **内容**: 描述HAMER输出时所用的相机参数，例如：`{"fov": 60, "cam_t": [0.0, 0.4]}`。

##### d. 3D物体模型 (`object_recon/`)

这是来自一个3D重建模型（如[InstantMesh](https://github.com/TencentARC/InstantMesh)或TripoSR）的输出。

* **`fixed.obj`**:
  * **格式**: 标准的 `.obj` 三维模型文件。
  * **要求**:
    * 模型需要被放置在一个规范坐标系下（例如，Y轴朝上）。`optim_easyhoi.py` 会根据是否为 `is_tripo` 对其进行旋转以统一坐标系。
    * 强烈建议包含**顶点颜色**（vertex colors），这样最终的渲染和导出结果会更好看。
* **`sdf.npy`** (可选):
  * **格式**: 使用 `np.save()` 保存的NumPy文件。
  * **内容**: 一个 `(64, 64, 64)` 的三维数组，表示物体的符号距离场（Signed Distance Field）体素网格。
  * **说明**: 如果此文件不存在，脚本会使用 `mesh_to_sdf` 库在首次运行时自动计算并保存，但这会花费一些时间。预先计算可以加快后续运行速度。
* **`hoi_box/<name>.json`** (可选，非TripoSR时需要):
  * **格式**: JSON文件。
  * **内容**: 一个列表，表示用于重建的物体在原图中的边界框 `[x_min, y_min, x_max, y_max]`。

### 4. 配置优化脚本

最后，在您的配置文件（例如 `src/configs/optim_teaser.yaml`）中，需要将 `data` 部分的路径指向您准备好的数据目录：

```yaml
data:
  base_dir: /path/to/your_base_data_dir/
  input_dir: ${data.base_dir}/input_images
  hand_dir: ${data.base_dir}/hamer_output
  obj_dir: ${data.base_dir}/object_recon
  hand_mask_dir: ${data.base_dir}/segmentation_masks/hand_mask
  obj_mask_dir: ${data.base_dir}/segmentation_masks/obj_mask
  inpaint_dir: ${data.base_dir}/segmentation_masks/inpaint_mask
  split: train # Or any other split name
```

---

只要您按照上述说明准备好完整的输入数据，`optim_easyhoi.py` 脚本就能顺利地从第一阶段运行到第三阶段，其中间结果会自动在各个阶段之间传递，最终完成手-物姿态的精细化对齐。
