# NCCL profiler 与流量聚合工具

把这个文件夹复制到训练机器即可使用。它只负责采集 NCCL 事件和离线聚合，
直接接入你已有的 `torchrun`、MPI 或其他训练启动方式，不依赖 control-plane、
Megatron、原实验 manifest、固定主机地址或两机八卡配置。
同一插件适用于 tied 和 untied 模型，不修改模型参数、网络配置或通信调度。

## 1. 包内文件与环境要求

| 文件 | 用途 |
| --- | --- |
| `plugin.cc`、`trace_writer.h` | 原始 NCCL profiler 插件与缓冲 JSONL 输出 |
| `nccl/` | 构建所需的接口头文件，保留上游版权声明 |
| `Makefile` | 构建共享库、运行 CPU 自检 |
| `aggregate.py`、`trace_reader.py`、`time_windows.py` | 原始 trace 校验、rank 流量矩阵和时间窗聚合 |
| `context.py` | 可选的 step / microbatch / phase 打标接口 |
| `test_plugin.cc`、`check_plugin.py`、`test_aggregate.py` | CPU 回归检查 |
| `PROVENANCE.json` | 源码来源、文件哈希及交付验证范围 |

采集运行环境：Linux、支持 profiler v5/v6 ABI 的 NCCL、原有的 GPU 训练环境。
原项目采集使用 NCCL **2.29.2**；本插件只导出 `ncclProfiler_v5` 和
`ncclProfiler_v6`。其他版本需要实际确认加载和事件输出，不能仅凭版本大小认定兼容。
PyTorch 可用 `python -c 'import torch; print(torch.cuda.nccl.version())'` 查看其 NCCL 版本。

构建只需 C++17 编译器、make，无需额外 CUDA SDK 头文件；Python 聚合只使用
Python **3.9+ 标准库**，不需要安装 numpy、pandas 或 Megatron。
一个进程对应一个 GPU / global rank；单进程控制多个 GPU 的情况不在本包支持范围内。

## 2. 构建

在实际训练容器或 Linux 环境中执行：

```bash
cd /absolute/path/nccl-profiler
make -j
make check
```

产物是 `build/libnccl-profiler-fine.so`。多机需分别构建，或分发同一目标平台上
构建的兼容共享库；Mac 上构建的文件不能复制到 Linux 使用。
`make check` 不使用 GPU，只验证回调生命周期和聚合逻辑，不能替代 NCCL 实际加载检查。

## 3. 必需环境变量

每个训练进程在**初始化 NCCL 之前**必须具备下面三个变量：

| 变量 | 设置方法 |
| --- | --- |
| `NCCL_PROFILER_PLUGIN` | 共享库的绝对路径，例如 `/opt/nccl-profiler/build/libnccl-profiler-fine.so` |
| `FINE_TRACE_DIR` | 本次运行的输出根目录，必须提前创建且可写 |
| `RANK` | 当前进程的 **global rank**，必须是 `0..WORLD_SIZE-1` 中的唯一编号 |

`torchrun` 会给子进程设置 `RANK`，不要在启动前把所有 worker 的 `RANK` 固定为同一个值。
MPI / Slurm 启动时，需要在每个 worker 内分别把 `OMPI_COMM_WORLD_RANK` /
`SLURM_PROCID` 转为 `RANK`；`LOCAL_RANK` 不能代替 global rank。
插件不读取 `WORLD_SIZE`；聚合时通过 `--world-size` 显式核验应有的 rank 数。

在每个节点的训练环境里准备：

```bash
export NCCL_PROFILER_PLUGIN=/absolute/path/nccl-profiler/build/libnccl-profiler-fine.so
# 所有节点属于同一次 job；替换为每次运行唯一的名称。
# 使用各主机的本地目录；也可放共享文件系统的不同主机子目录下。
export FINE_TRACE_DIR=/absolute/path/traces/job-001/host-a
mkdir -p "$(dirname "$FINE_TRACE_DIR")"
mkdir "$FINE_TRACE_DIR"

# 推荐用于确认插件加载；不是启用插件的必需变量。
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET
export NCCL_DEBUG_FILE="$FINE_TRACE_DIR/nccl.%h.%p.log"
```

另一个节点使用 `host-b` 等不同子目录。然后执行**原本的训练命令**。
例如单节点现有命令是 `torchrun --standalone --nproc-per-node=4 train.py ...`，
则在上面 export 后照常执行它；多节点保留原来的 rendezvous、节点数和 node rank 设置。
通过 Docker、SSH、MPI 启动时，要确保这些变量确实进入训练容器和每个 worker。

插件会写入：

```text
host-a/
├── nccl.<hostname>.<pid>.log
├── rank-0/nccl-events.jsonl
└── rank-1/nccl-events.jsonl
host-b/
├── rank-2/nccl-events.jsonl
└── rank-3/nccl-events.jsonl
```

**每个 job / 重启 attempt 使用新目录。** 插件以追加模式打开文件；复用目录会混入
旧 communicator 和重复 event ID。elastic 重启也要换目录。不要在同一个 rank 文件里
拼接多次运行。训练结束、所有通信完成并销毁 process group / communicator 后再回收数据。
自建子 process group 也应按框架要求正常销毁。

不需要 `LD_PRELOAD`，也不需要另设 profiler event mask；插件初始化时会请求
collective、P2P、proxy operation 和 proxy step 事件。

## 4. 网络相关变量如何设置

上述三个变量只负责启用采集。网络选择沿用你已经验证的训练配置：

| 变量 | 作用及何时设置 |
| --- | --- |
| `NCCL_NET=IB`、`NCCL_IB_DISABLE=0` | 使用 NCCL 内置 IB/RoCE 数据路径时的配置；是否适用取决于现有训练环境 |
| `NCCL_IB_HCA` | 选择已验收的 HCA/端口；按现场配置，不要照搬其他机器的 `mlx5_0` |
| `NCCL_IB_GID_INDEX` | RoCE GID 选择；若原环境需要固定，沿用已验收值 |
| `NCCL_SOCKET_IFNAME` | socket 建连/控制接口；按现有配置设置 |
| `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1` | 仅在实验明确要求关闭 GPU P2P 和共享内存路径时设置；会改变训练通信路径和性能 |

本插件统计有发送 proxy chunk 回调的通信，**不保证覆盖 NVLink/P2P、SHM、NVLS
等所有通信实现**。默认训练中机内通信可能没有这些事件，矩阵里的零不能直接理解为
“两个 GPU 之间没有通信”。如果研究问题要求所有 rank 间流量走网络，需要另行配置
并验证数据路径。插件不会帮你禁用 P2P/SHM，也不会改变 HCA 绑定。

## 5. 聚合整个运行的通信量（无需修改训练代码）

把**同一次完整 job 的所有主机数据**收齐，保留 host 和 `rank-N` 目录。
不用复制训练代码或原项目实验 manifest。下面以四个 global rank 为例：

```bash
python3 aggregate.py \
  --input host-a=/data/job-001/host-a \
  --input host-b=/data/job-001/host-b \
  --world-size 4 \
  --output results/job-001-total
```

`--input` 每个主机给一次；HOST 是你指定的唯一主机标签。一个目录只包含一个 job，
脚本递归读取其中的 `rank-N/nccl-events.jsonl`。单机只给一个 `--input`。
`--output` 必须是新目录，不覆盖已有结果。这里的“聚合”是**将通信事件累加成流量矩阵**，
不包含光电拓扑/Gurobi 求解器。

| 输出 | 含义 |
| --- | --- |
| `matrix_bytes.csv` | 行=发送 global rank，列=接收 global rank，值=选中事件的发送字节总量 |
| `matrix_by_step.csv` | `step,src_rank,dst_rank,bytes`；不打标时 step 为 `-1` |
| `matrix_by_operation.csv` | 按 communicator、collective/P2P、函数和 rank pair 分组 |
| `runtime.json` | communicator local rank → global rank 的观测映射，以及 rank → host 标签 |
| `validation.json` | 校验状态、输入及聚合源码 SHA256、选择条件、总字节和事件计数 |

只有 `validation.json` 的 `status=completed` 才表示通过本工具的完整性校验。
脚本检查所有 global rank、communicator 成员和 finalize 是否齐全、错误计数是否为零、
事件 ID 是否重复、API 父子归属，以及 proxy 与 chunk 字节守恒。
peer 通过实际 communicator membership 映射，不能直接把 `peer_local` 当 global rank。
无发送 chunk 或筛选后无字节会报错，不会输出一个貌似有效的全零矩阵。
这类校验不证明训练收敛、物理路径正确，或所有传输类型都被覆盖。

## 6. 可选：按 step 打标并排除 warmup

只配环境变量即可获得全程总量；若要区分训练 step，在自己的训练循环里加下面的接口。
将本包目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/absolute/path/nccl-profiler${PYTHONPATH:+:$PYTHONPATH}
```

```python
from context import ProfilerContext

profiler = ProfilerContext()  # 使用 NCCL_PROFILER_PLUGIN 指向的同一份共享库
for step in range(1, num_steps + 1):
    profiler.set(step)        # 放在本 step 发起 NCCL 操作之前
    train_one_step()          # 你自己的训练逻辑
profiler.set(-1)              # 后续 teardown 等不归入训练 step
profiler.flush()
# 按现有框架要求等待通信完成、销毁 process groups，然后正常退出。
```

可进一步调用 `profiler.set(step, microbatch=0, phase=1)`；phase 为
`0=未区分, 1=forward, 2=backward`。这些标签由用户赋予，插件不会自动推断。
标签在 NCCL enqueue 回调处采样并传给异步 proxy 子事件；接口是**进程全局状态**，
若有多个线程并发发起不同 step 的操作，需要应用自行协调，不能依赖这段简单循环精确归属。
`flush()` 仅刷缓冲，不等待 GPU，不替代 communicator finalize。

例如保留 step 6–25：

```bash
python3 aggregate.py \
  --input host-a=/data/job-001/host-a \
  --input host-b=/data/job-001/host-b \
  --world-size 4 --step-min 6 --step-max 25 \
  --output results/job-001-step6-25
```

上下界均包含。不打标时不要传 step 筛选，否则 `-1` 会被排除。
筛选范围不保证每一步都产生通信；实际观测标签在 `validation.json` 中。
总矩阵是选中范围的**总字节数**；确认采集了 20 个完整 step 后才能自行除以 20 求均值。
本包不自动把 communicator 分类为 TP/PP/DP；这需要你的训练框架提供进程组语义。

## 7. 可选：按 1 ms / 100 µs 时间窗聚合

```bash
python3 aggregate.py \
  --input host-a=/data/job-001/host-a \
  --input host-b=/data/job-001/host-b \
  --world-size 4 --window-us 1000 \
  --output results/job-001-1ms
```

`--window-us 100` 则为 100 µs，可以与 step 筛选组合。额外输出压缩 CSV
`bandwidth_windows.csv.gz`，仅保留非零窗口，不删除任何小流量。

- 窗口以各 host 的 `CLOCK_MONOTONIC` 零点对齐，边界为绝对单调时钟纳秒；不是 step 相对时间。
- `estimated_bytes`：把 chunk 字节均匀分摊到 `[submit_ns,end_ns)` 所覆盖的窗口。
- `estimated_gbps = estimated_bytes × 8 / window_ns`，是十进制 Gbit/s。
- `completion_bytes`：把全部字节记在 `end_ns-1` 所属窗口；它是完成事件直方图，可能有批量尖峰。
- 窗口同时保留 `clock_domain,step,src_rank,dst_rank`；两种口径分别守恒到选中总字节。

这是主机 proxy 提交至观测完成区间上的**带宽估计**，不是逐包线速、实测 RX 到达时序或 GPU kernel 时间。
**不同 host 的窗口不能直接拼成同一全局时间轴**，即使使用相同窗口宽度、step 编号，
或系统显示相同墙上时间也不成立。跨主机全局窗口需要另行测量单调时钟映射与误差；
本工具包没有带入原项目的时钟校准实验编排。
各主机的字节总量可以直接汇总，时间窗仅在同一 `clock_domain` 内比较。

## 8. 采集自检与常见问题

先用已有训练做一次短采集：查看 NCCL 日志确认成功加载 profiler，正常结束后核对
各 rank 的 JSONL 包含 `comm_member`、`api`、有网络发送时的 `chunk` / `proxy`，
以及 `comm_finalize`。然后运行聚合检查。该插件使用缓冲写入，运行中暂时看不到尾部事件不代表没有采集。

| 症状 | 检查项 |
| --- | --- |
| 没生成 JSONL | 三个必需变量是否进入每个 worker、目录存在且可写、共享库路径及 ABI 是否正确；缺 `RANK`/目录时插件可能静默不采集 |
| 数据都挤在 rank-0 | 是否误用 local rank，或给所有进程设置同一个 `RANK` |
| 有 API 无 chunk | 实际传输路径是否产生 proxy step、NCCL 是否兼容；不要据此宣称通信量为零 |
| 缺 finalize / orphan / 守恒失败 | 训练是否仍运行、被强杀、文件没收齐或混入不同 job；保留原始文件排查 |
| 重复 communicator / event ID | 输出目录是否复用，或把同一份 trace 输入两次 |
| 按 step 筛选无字节 | 是否接入 `context.py`，标签范围是否正确 |

总字节只加 **send chunk**，不重复叠加 proxy 汇总或接收端，不含 Ethernet/RDMA 包头、重传等物理开销。
AllReduce 的 API tensor 大小与算法实际运输字节是不同口径；本包聚合后者。
采集本身会有 CPU、内存和 I/O 开销，长作业事件量可能很大；要评价训练性能时应另测开启/关闭 profiler 的影响。
