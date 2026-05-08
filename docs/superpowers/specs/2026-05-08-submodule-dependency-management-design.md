# Design: Git Submodule 依赖管理 — LibRaw & libtiff

**日期**: 2026-05-08
**状态**: Approved

## 背景

当前项目通过 `pkg_check_modules` + 手动 `find_library`/`find_path` 查找 LibRaw 和 libtiff。用户必须预先在系统上安装这些库，跨平台构建（尤其是 Windows、Android、交叉编译）体验差。

## 目标

- clone + `git submodule update --init` + cmake + build 即可完成编译，无需手动配置外部依赖
- 支持尽可能多的平台：Linux、macOS、Windows、Android
- 保留回退机制：用户仍可选择使用系统安装的库

## 方案

### 依赖来源

| 依赖 | Git 仓库 | 分支/Tag | 说明 |
|------|----------|----------|------|
| LibRaw 源码 | `LibRaw/LibRaw` | `0.22.1` | 官方源码（无 CMake 支持） |
| LibRaw CMake 包装 | `LibRaw/LibRaw-cmake` | latest master | 社区维护，专为 add_subdirectory 设计 |
| libtiff | `libtiff/libtiff`（GitLab 镜像） | `v4.7.1` | 原生支持 add_subdirectory |

### 目录结构

```
third_party/
├── LibRaw/          (git submodule: LibRaw/LibRaw, tag 0.22.1)
├── LibRaw-cmake/    (git submodule: LibRaw/LibRaw-cmake)
└── libtiff/         (git submodule: libtiff/libtiff, tag v4.7.1)
```

### CMakeLists.txt 变更

#### 1. 新增 `USE_SYSTEM_LIBS` option

```cmake
option(USE_SYSTEM_LIBS "Use system-installed libraries instead of building from source" OFF)
```

- **OFF（默认）**: 使用 submodule 从源码构建
- **ON**: 回退到现有的 pkg-config/find_library 逻辑

#### 2. LibRaw 构建（USE_SYSTEM_LIBS=OFF 时）

```cmake
# 指向 LibRaw 源码路径
set(LIBRAW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/LibRaw" CACHE STRING "")

# 最小化构建选项
set(ENABLE_LCMS     OFF CACHE BOOL "" FORCE)
set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_JASPER   OFF CACHE BOOL "" FORCE)
set(ENABLE_RAWSPEED OFF CACHE BOOL "" FORCE)
set(LIBRAW_INSTALL  OFF CACHE BOOL "" FORCE)

# ENABLE_OPENMP: 不设置，使用默认值 ON，让 LibRaw-cmake 自行检测编译器支持

add_subdirectory(third_party/LibRaw-cmake)
# 链接目标: libraw::libraw_r (线程安全版本)
```

**关键选项说明**:
- `ENABLE_LCMS=OFF`: 项目有完整的手写色彩管线（gamut 矩阵 + log OETF + LUT），不依赖 LibRaw 的 LCMS 色彩管理
- `ENABLE_OPENMP`: 使用默认值 ON，LibRaw-cmake 会自动检测编译器是否支持 OpenMP，与项目自身的 `find_package(OpenMP)` 互不冲突
- 使用 `libraw::libraw_r`（线程安全版本）而非 `libraw::libraw`

#### 3. libtiff 构建（USE_SYSTEM_LIBS=OFF 时）

```cmake
# 最小化构建
set(tiff-tools      OFF CACHE BOOL "" FORCE)
set(tiff-tests      OFF CACHE BOOL "" FORCE)
set(tiff-contrib    OFF CACHE BOOL "" FORCE)
set(tiff-docs       OFF CACHE BOOL "" FORCE)
set(tiff-install    OFF CACHE BOOL "" FORCE)
set(tiff-cxx        OFF CACHE BOOL "" FORCE)

# 关闭不需要的编解码器
set(jpeg  OFF CACHE BOOL "" FORCE)
set(jbig  OFF CACHE BOOL "" FORCE)
set(lerc  OFF CACHE BOOL "" FORCE)
set(lzma  OFF CACHE BOOL "" FORCE)
set(zstd  OFF CACHE BOOL "" FORCE)
set(webp  OFF CACHE BOOL "" FORCE)

# 保留 zlib（TIFF deflate 压缩，DNG 文件需要）

add_subdirectory(third_party/libtiff)
# 链接目标: tiff (或别名 TIFF::tiff)
```

#### 4. 链接方式统一

将链接从变量方式改为 CMake target 方式：

```cmake
# 之前:
# target_include_directories(raw_alchemy_lib PUBLIC ${LIBRAW_INCLUDE_DIRS} ${LIBTIFF_INCLUDE_DIRS})
# target_link_libraries(raw_alchemy_lib PUBLIC ${LIBRAW_LIBRARIES} ${LIBTIFF_LIBRARIES})

# 之后:
target_link_libraries(raw_alchemy_lib PUBLIC libraw::libraw_r tiff)
```

CMake target 自动传递 include 目录和链接依赖，不再需要手动管理变量。

#### 5. 保留项目自身 OpenMP

```cmake
# 保持不变
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raw_alchemy_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(raw_alchemy_lib PUBLIC RA_USE_OPENMP)
endif()
```

这用于并行化 stylize.cpp、lut_applier.cpp、log_transform.cpp 中的逐像素处理循环。

### .gitignore 变更

不需要额外变更——submodule 本身是 git 追踪的引用，不会污染仓库。

### 构建流程

```bash
git clone <repo>
cd Raw_Log_Lut
git submodule update --init --recursive
cmake -B build
cmake --build build
```

如需使用系统库：
```bash
cmake -B build -DUSE_SYSTEM_LIBS=ON
```

## Trade-offs

| 方面 | 优势 | 代价 |
|------|------|------|
| 用户体验 | clone + submodule init + build，零手动配置 | submodule 操作多一步 |
| 可重现性 | 锁定特定 tag，所有平台构建相同版本源码 | clone 时间略增（submodule 按需下载） |
| 跨平台 | 不依赖系统包管理器，Android/交叉编译友好 | LibRaw-cmake 是社区维护，非官方 |
| 灵活性 | USE_SYSTEM_LIBS=ON 可随时回退 | 维护两套查找逻辑 |

## 已知风险

1. **LibRaw-cmake 社区维护**: 非官方维护，可能滞后于 LibRaw 主仓库更新。darktable 项目也采用相同方案，降低了风险。
2. **libtiff `enable_testing()` 污染**: libtiff 即使作为 subproject 也会调用 `enable_testing()`，可能干扰父项目的 CTest 配置。实际影响微小。
