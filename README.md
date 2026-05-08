# Raw Alchemy (C++ Edition)

A high-performance C++ command-line tool for processing camera RAW files through cinematic log color pipelines with 3D LUT color grading.

> **本项目是 [Raw-Alchemy](https://github.com/shenmintao/Raw-Alchemy)（Python）项目的 C++ 重写版本。**
>
> This project is a C++ rewrite of the [Raw-Alchemy](https://github.com/shenmintao/Raw-Alchemy) Python project by [shenmintao](https://github.com/shenmintao).

---

## 致谢 / Acknowledgments

本项目的核心设计、处理管线、色彩科学算法以及测光策略均源自 [Raw-Alchemy](https://github.com/shenmintao/Raw-Alchemy) 项目。感谢原作者的设计与实现，使得本 C++ 移植版本得以完成。

The core design, processing pipeline, color science algorithms, and metering strategies are all derived from the [Raw-Alchemy](https://github.com/shenmintao/Raw-Alchemy) project. Credit goes to the original author for the architecture that made this C++ port possible.

---

## 与参考项目的对比 / Comparison with Reference Project

### 功能对比 / Feature Comparison

| 功能 / Feature | Raw-Alchemy (Python) | 本项目 / This Project (C++) |
|---|---|---|
| RAW 文件解码 / RAW decoding | rawpy (LibRaw bindings) | LibRaw (直接调用 C API) |
| 色域变换 / Gamut transform | colour-science 库 | 预计算 constexpr 3×3 矩阵 (colour-science 离线生成) |
| Log 曲线编码 / Log curve encoding | colour-science `cctf_encoding()` | 内联分段函数 (switch-case, 零开销) |
| 3D LUT 插值 / 3D LUT interpolation | Numba JIT 四面体插值 | C++ 四面体插值 + OpenMP 并行 |
| 镜头校正 / Lens correction | ctypes 调用 Lensfun 共享库 | 直接编译 Lensfun 源码 + GLib2 shim |
| 自动曝光测光 / Auto exposure metering | 5 种策略 (Protocol 模式) | 5 种策略 (函数重载 + 子采样) |
| 图像输出 / Image output | TIFF / HEIF / JPEG | TIFF (16-bit) / JPEG (8-bit) |
| CLI 界面 / CLI interface | Click 框架 | 原生命令行参数解析 |
| GUI 界面 / GUI | Tkinter GUI + matplotlib 预览 | 无 (仅 CLI) |
| 批处理 / Batch processing | ProcessPoolExecutor 并行 | 无 (单文件处理) |
| 实时预览 / Live preview | matplotlib 异步渲染 | 无 |
| HEIF 输出 / HEIF output | pillow-heif (10-bit) | 不支持 |
| 跨平台验证 / Cross-validation | — | 内置交叉验证脚本 (对比 Python 输出) |
| 分发 / Distribution | PyInstaller 打包 | CMake 静态链接构建 |
| 语言 / Language | Python ≥ 3.11 | C++17 |
| 并行加速 / Parallelism | Numba `@njit(parallel=True)` | OpenMP |
| 许可证 / License | GNU AGPL v3 | — |

### 设计与实现差异 / Design & Implementation Differences

#### 1. 色彩空间数据生成方式

| | Raw-Alchemy | C++ Edition |
|---|---|---|
| **方式** | 运行时调用 `colour-science` 库计算矩阵和曲线 | 离线用 `colour-science` 生成 `color_data.h`，编译时嵌入 |
| **优势** | 灵活，可运行时切换 | 零运行时依赖，constexpr 矩阵无计算开销 |
| **工具** | Python 运行时依赖 | `scripts/gen_color_data.py` 代码生成器 |

#### 2. 性能优化策略

| | Raw-Alchemy | C++ Edition |
|---|---|---|
| **像素级操作** | Numba JIT 编译 (`@njit(parallel=True, fastmath=True)`) | 手写 C++ 循环 + OpenMP `#pragma omp parallel for` |
| **矩阵变换** | NumPy 扁平化 + 展开乘法 | 逐像素内联 3×3 乘法 |
| **LUT 查找** | Numba 四面体插值 | 相同算法，C++ 原生实现 |
| **镜头校正重采样** | SciPy `map_coordinates` (双三次) | 手写 Catmull-Rom 双三次插值 (4×4 核) |
| **内存布局** | NumPy ndarray (H×W×3) | `std::vector<float>` 平铺存储 (W×H×3) |

#### 3. 镜头校正集成方式

| | Raw-Alchemy | C++ Edition |
|---|---|---|
| **Lensfun 加载** | 运行时 ctypes 动态加载 `.dll/.so/.dylib` | 直接编译 Lensfun 源码到可执行文件 |
| **GLib2 依赖** | 依赖系统安装的 GLib2 | **完全消除 GLib2 依赖** — 自写 GLib2 兼容 shim |
| **GLib2 shim** | 不适用 | 556 行 C++17 实现，将 GLib2 API 映射到 std::string / std::mutex / pugixml 等 |

#### 4. 架构差异

| | Raw-Alchemy | C++ Edition |
|---|---|---|
| **模块耦合** | Python 模块间函数调用 | 头文件 + 编译单元，静态库 `raw_alchemy_core` |
| **入口点** | CLI (`cli.py`) + GUI (`gui.py`) 双入口 | 仅 CLI (`main.cpp`) |
| **管线编排** | `core.py` 函数式编排 | `main.cpp` 内联管线 |
| **日志** | 自定义 Logger (print/Queue/callback) | stderr 直出 |
| **测光接口** | Protocol 策略模式 + 类继承 | 函数式，通过枚举 + switch 分发 |

---

## 处理管线 / Processing Pipeline

```
RAW 文件
  │
  ▼
[步骤 0] 元数据提取 ─── extractMetadata() via LibRaw
  │         → CameraMetadata (相机、镜头、焦距、光圈、ISO)
  ▼
[步骤 1] 标准化解码 ──── decodeRaw() via LibRaw
  │         RAW → ProPhoto RGB (线性), float32 [0.0, 1.0]
  ▼
[步骤 1.5] 镜头校正 (可选)  applyLensCorrection() via Lensfun
  │           暗角 → 畸变 + 色差 (Catmull-Rom 双三次重采样)
  ▼
[步骤 2] 曝光控制 ────── computeAutoGain() 或手动 EV
  │         5 种测光模式: 平均 / 中央重点 / 高光安全 / 混合 / 矩阵
  ▼
[步骤 3] 相机匹配增强 ── applySaturationContrast()
  │         饱和度 (×1.25) + 对比度 (×1.10)
  ▼
[步骤 4] Log 信号准备 ── applyLogTransform()
  │         4a: 色域变换 (ProPhoto → 目标色域, 3×3 矩阵)
  │         4b: Log 曲线编码 (线性 → Log OETF)
  ▼
[步骤 5] 3D LUT 应用 (可选)  loadCubeLUT() + applyLUT3D()
  │         四面体插值 (6-case, 每像素仅读 4 顶点)
  ▼
[输出] 保存 ────── 16-bit TIFF 或 8-bit JPEG
```

---

## 支持的 Log 空间 / Supported Log Spaces

本项目支持 12 种电影级 Log 色彩空间，每种空间配对对应的色域矩阵（ProPhoto RGB → 目标色域，使用 CAT02 色适应变换预计算）：

| Log 曲线 | 配对色域 |
|---|---|
| Fujifilm F-Log | F-Gamut |
| Fujifilm F-Log2 | F-Gamut |
| Fujifilm F-Log2C | F-Gamut C |
| Panasonic V-Log | V-Gamut |
| Nikon N-Log | N-Gamut |
| Leica L-Log | ITU-R BT.2020 |
| Canon Log 2 | Cinema Gamut |
| Canon Log 3 | Cinema Gamut |
| Sony S-Log3 | S-Gamut3 |
| Sony S-Log3.Cine | S-Gamut3.Cine |
| ARRI Arri LogC3 | Wide Gamut 3 |
| ARRI Arri LogC4 | Wide Gamut 4 |
| RED Log3G10 | REDWideGamutRGB |
| DJI D-Log | D-Gamut |

---

## 项目结构 / Project Structure

```
Raw_Log_Lut/
├── CMakeLists.txt                  # CMake 构建系统 (C++17, 静态链接)
├── .gitmodules                     # 6 个第三方库子模块
├── .gitignore
│
├── include/                        # 公共头文件
│   ├── common.h                    #   ImageBuffer 核心数据结构 (float32 RGB)
│   ├── color_data.h                #   色域矩阵 + Log 曲线 (自动生成)
│   ├── raw_decoder.h               #   RAW 解码接口 (DecodeParams, CameraMetadata)
│   ├── log_transform.h             #   色域变换 + Log 编码接口
│   ├── lut_applier.h               #   3D LUT 加载与应用接口
│   ├── metering.h                  #   自动曝光测光接口 (5 种模式)
│   ├── stylize.h                   #   饱和度/对比度增强接口
│   ├── lens_correction.h           #   镜头校正接口 (Lensfun)
│   ├── tiff_writer.h               #   16-bit TIFF 输出接口
│   └── jpeg_writer.h               #   8-bit JPEG 输出接口
│
├── src/                            # 源代码
│   ├── main.cpp                    #   CLI 入口 + 管线编排
│   ├── raw_decoder.cpp             #   LibRaw RAW 解码实现
│   ├── log_transform.cpp           #   色域矩阵乘法 + Log OETF 实现
│   ├── lut_applier.cpp             #   .cube 解析 + 四面体插值实现
│   ├── metering.cpp                #   5 种测光策略实现
│   ├── stylize.cpp                 #   饱和度/对比度增强实现
│   ├── lens_correction.cpp         #   Lensfun 镜头校正实现
│   ├── tiff_writer.cpp             #   16-bit TIFF 输出实现 (libtiff)
│   ├── jpeg_writer.cpp             #   8-bit JPEG 输出实现 (libjpeg-turbo)
│   └── verify.cpp                  #   独立 TIFF 验证工具
│
├── scripts/
│   └── gen_color_data.py           #   色域矩阵 + Log 曲线代码生成器
│
├── Test/                           # 交叉验证测试
│   ├── cross_validate.py           #   RAW 解码步骤对比验证
│   ├── cross_validate_log.py       #   Log 变换步骤对比验证
│   ├── cross_validate_lut.py       #   LUT 应用步骤对比验证
│   ├── cross_validate_jpeg.py      #   JPEG 全管线对比验证
│   ├── cross_validate_lens.py      #   镜头校正效果验证
│   ├── Sample.NEF                  #   测试 RAW 文件 (Nikon NEF)
│   └── FLog2C_to_CLASSIC-Neg._65grid_V.1.00.cube  #   测试用 3D LUT
│
└── third_party/                    # 第三方依赖 (git 子模块 + 自定义)
    ├── LibRaw/                     #   RAW 解码库 (子模块)
    ├── LibRaw-cmake/               #   LibRaw CMake 封装 (子模块)
    ├── libtiff/                    #   TIFF I/O 库 (子模块)
    ├── libjpeg-turbo/              #   JPEG 编码库 (子模块)
    ├── lensfun/                    #   镜头校正库 (子模块)
    ├── pugixml/                    #   XML 解析库 (子模块, 供 GLib2 shim 使用)
    ├── glib_shim/                  #   ★ 自定义 GLib2 兼容层
    │   ├── include/glib.h          #     GLib2 API 头文件 (237 行)
    │   ├── include/glib/gstdio.h   #     重定向到 glib.h
    │   └── src/glib_shim.cpp       #     C++17 实现 (556 行)
    └── lensfun_config/
        └── config.h.in             #   Lensfun config.h CMake 模板
```

---

## 模块对应关系 / Module Mapping

下表展示本项目各模块与参考项目 (Raw-Alchemy) Python 源文件的对应关系：

| 本项目 (C++) | 参考项目 (Python) | 功能说明 |
|---|---|---|
| `src/main.cpp` | `cli.py` + `orchestrator.py` + `core.py` | CLI 入口 + 管线编排 |
| `src/raw_decoder.cpp` + `include/raw_decoder.h` | `core.py` (Step 1: RAW decode) | RAW 文件解码至 ProPhoto 线性空间 |
| `src/lens_correction.cpp` + `include/lens_correction.h` | `lensfun_wrapper.py` + `utils.py` (lens correction) | 镜头畸变/色差/暗角校正 |
| `src/metering.cpp` + `include/metering.h` | `metering.py` | 5 种自动曝光测光策略 |
| `src/stylize.cpp` + `include/stylize.h` | `utils.py` (`apply_saturation_contrast_inplace`) | 饱和度 + 对比度增强 |
| `src/log_transform.cpp` + `include/log_transform.h` | `core.py` (Step 4: gamut + log) + `utils.py` (`apply_matrix_inplace`) | 色域变换 + Log 曲线编码 |
| `include/color_data.h` | `config.py` + `colour-science` 运行时计算 | 色域矩阵 + Log 曲线定义 |
| `src/lut_applier.cpp` + `include/lut_applier.h` | `utils.py` (`apply_lut_inplace`) + `colour.LUT` | 3D LUT 加载 + 四面体插值 |
| `src/tiff_writer.cpp` + `include/tiff_writer.h` | `file_io.py` (TIFF 分支) | 16-bit TIFF 输出 |
| `src/jpeg_writer.cpp` + `include/jpeg_writer.h` | `file_io.py` (JPEG 分支) | 8-bit JPEG 输出 |
| `src/verify.cpp` | (无对应) | 独立 TIFF 统计验证工具 (C++ 独有) |
| `scripts/gen_color_data.py` | `colour-science` 运行时依赖 | 色彩数据离线代码生成器 |
| `include/common.h` | NumPy ndarray (隐式) | ImageBuffer 数据结构定义 |
| `third_party/glib_shim/` | (无对应 — Python 不需要 GLib2) | GLib2 兼容层 (C++ 编译 Lensfun 所需) |
| `Test/cross_validate_*.py` | (无对应) | 交叉验证脚本 (C++ 独有) |

### 说明

- **`core.py` 被拆分**：Python 版的 `core.py` 是单一管线文件，包含全部 6 步处理流程。C++ 版将其拆分为独立模块 (`raw_decoder`, `log_transform`, `lut_applier`, `metering`, `stylize`, `lens_correction`)，管线编排由 `main.cpp` 负责。
- **`utils.py` 被拆分**：Python 版的 `utils.py` 包含所有 Numba 加速核函数（矩阵变换、LUT 插值、增益、饱和度/对比度等）。C++ 版将这些功能分别归入对应的源文件。
- **`lensfun_wrapper.py` → 直接编译**：Python 版通过 ctypes 动态加载 Lensfun 共享库；C++ 版直接将 Lensfun 源码编译进项目，并自写了 GLib2 兼容 shim 以消除外部 GLib2 依赖。
- **`config.py` → `color_data.h`**：Python 版在 `config.py` 中定义 log 空间映射，运行时调用 `colour-science` 计算矩阵和曲线。C++ 版通过 `scripts/gen_color_data.py` 离线生成 `color_data.h`，将所有矩阵和曲线嵌入编译产物。
- **无 GUI**：Python 版包含完整的 Tkinter GUI + matplotlib 实时预览。C++ 版仅提供 CLI。
- **无 HEIF 输出**：Python 版支持 10-bit HEIF 输出。C++ 版仅支持 TIFF 和 JPEG。
- **独有的交叉验证**：C++ 版包含一组 Python 交叉验证脚本，逐步对比 C++ 输出与 Python (rawpy + colour-science) 参考输出，确保移植精度。

---

## 依赖 / Dependencies

所有依赖均通过 git 子模块引入，无需系统安装：

| 库 | 用途 |
|---|---|
| [LibRaw](https://www.libraw.org/) | RAW 文件解码 (支持所有主流相机格式) |
| [libtiff](http://www.libtiff.org/) | 16-bit TIFF 输出 |
| [libjpeg-turbo](https://www.libjpeg-turbo.org/) | 8-bit JPEG 输出 (SIMD 加速) |
| [Lensfun](https://lensfun.github.io/) | 镜头校正 (畸变、色差、暗角) |
| [pugixml](https://pugixml.org/) | XML 解析 (供 GLib2 shim 使用) |

**可选**：OpenMP（自动检测，用于像素级操作的并行加速）。

**构建时**需要 Python 3 + `colour-science` 库（仅用于运行 `scripts/gen_color_data.py` 生成色彩数据头文件，非运行时依赖）。

---

## 构建 / Build

```bash
# 克隆（含子模块）
git clone --recursive <repo-url>
cd Raw_Log_Lut

# 配置
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build --config Release -j$(nproc)

# 产出
#   build/raw_alchemy_cli       — CLI 主程序
#   build/raw_alchemy_verify    — TIFF 验证工具
```

---

## 用法 / Usage

```
raw_alchemy_cli <input.raw> <output> --log-space <space> [options]
```

**输出格式**由文件扩展名自动判定：`.tif` / `.tiff` → 16-bit TIFF，`.jpg` / `.jpeg` → 8-bit JPEG。

**主要选项：**

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--log-space <space>` | (必需) | 目标 Log 空间 (见上方支持列表) |
| `--lut <path>` | 无 | .cube 3D LUT 文件路径 |
| `--exposure <ev>` | 自动 | 手动曝光值 (EV)，覆盖自动测光 |
| `--metering <mode>` | hybrid | 测光模式: average / center-weighted / highlight-safe / hybrid / matrix |
| `--lens-correct` / `--no-lens-correct` | 启用 | 镜头校正开关 |
| `--custom-lensfun-db <path>` | 系统默认 | 自定义 Lensfun XML 数据库路径 |
| `--half-size` | 关闭 | 半尺寸快速解码 (预览用) |
| `--format <fmt>` | 自动 | 强制输出格式: tif / jpg |
| `--jpeg-quality <n>` | 95 | JPEG 压缩质量 (1-100) |
| `--info` | 关闭 | 仅输出元数据，不处理 |
