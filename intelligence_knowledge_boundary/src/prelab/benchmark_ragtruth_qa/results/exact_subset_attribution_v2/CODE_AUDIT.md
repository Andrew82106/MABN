# Exact subset attribution v2：WDDM 门禁审计

通过 CPU 审计。v2 只读绑定 v1 的 3,839 条无标签 prepared inputs；v1 现有文件哈希全部未变。

门禁只放行同时满足三项的外部条目：显存字段严格等于 `[N/A]`、NVIDIA 类型为 `G/C+G`、程序名在冻结 GUI 清单。数值显存、纯 `C`、未知类型/程序名、其他 `N/A` 写法均拦截。此外保留全局排他锁、CUDA 初始化前后两次扫描和 6 GiB 空闲显存门槛。

当前未运行 GPU，未读取标签或 test，未修改 baseline。审核后可依次执行 `gpu-smoke` 与 `extract`。
