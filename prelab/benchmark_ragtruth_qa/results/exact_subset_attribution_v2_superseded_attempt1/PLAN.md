# Exact subset attribution v2：Windows WDDM GPU 门禁修复

v2 只修 GPU 入场判定并写入新目录。它只读复用 v1 的无标签 prepared inputs，不改 8 子集、模型、特征或评测定义。

CPU 顺序：`initialize -> prepare -> check -> audit`。审核后才可单独运行 `gpu-smoke -> extract`。当前不得运行 GPU 命令。
