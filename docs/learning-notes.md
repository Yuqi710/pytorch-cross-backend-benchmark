# Learning Notes

本文件记录项目实施过程中掌握的推理系统知识、工程边界和阶段性理解。
实际执行结果和实验数据记录在[experiment-log.md](experiment-log.md)。

## Day 1: PyTorch CPU 与 MPS 基线

### `model.eval()` 和 `torch.inference_mode()`
 - `model.eval()`控制Dropout、BatchNorm等模型层的推理行为
 - `torch.inference_mode()`关闭Autograd记录，减少推理时的状态维护和开销
 - 二者职责不同，不可相互替代
 - 一个标准推理入口通常需要模型处于`eval`模式，并在`inference_mode`上下文中执行

### MPS 异步执行
 - CPU向MPS提交计算任务后可能继续执行后续代码
 - 性能计时必须在正确的测量边界调用`torch.mps.synchronize()`
 - 如果不进行同步，计时结果可能主要反映任务提交时间，而不是设备完成计算的时间

### 后端生命周期
正式推理后端应逐步拆分以下阶段：
1. 模型加载或创建
2. 后端/session 初始化
3. 模型和权重传输
4. 输入准备
5. 预热
6. 正式执行
7. 异步设备同步
8. 输出传输
9. 正确性检查
10. 结果持久化

只有明确测量边界，延迟数据才有可解释性

### 正确性判断
 - 不同硬件后端的浮点结果不要求逐位相同
 - 同时检查Shape、NaN/Inf、绝对误差、相对误差与Top-1
 - 本次CPU/MPS最大绝对误差为`0.000007`，Top-1一致

### 测试边界
 - 单元测试不依赖预训练权重和网络下载
 - 冒烟测试使用真实权重验证完整链路
 - 当前结果不代表模型准确率或性能提升

### 当前限制
 - 仅测试一个模型和一个随机输入
 - 尚未进行预热、多轮采样和延迟统计
 - 尚未区分冷启动时间和稳定推理时间
 - 尚未加入ONNX Runtime

---

## Day 2: PyTorch 到 ONNX 的导出与图检查

### 处理链路
 - ONNX是一种模型计算图表示格式，它的主要作用是将模型的计算过程、参数、输入输出信息表示为
   标准化计算图，使模型能在不同框架、运行时和硬件后端之间传递。
 - ONNX不是硬件执行，也不是推理加速器；真正执行ONNX模型的组件是ONNX Runtime、TensorRT
   等运行时或编译器。因此，成功导出ONNX ≠ 成功执行ONNX，成功执行ONNX ≠ 推理速度更快。
 - 本项目的处理链路为：

```text
PyTorch nn.Module（CPU / eval）
    ↓ torch.onnx.export(dynamo=True)
ONNX ModelProto
    ↓ 序列化
.onnx 模型文件
    ↓ ONNX Checker
模型结构与算子规范校验
    ↓ Shape Inference
张量类型与形状补全
    ↓ Graph Inspector
输入输出、节点、Initializer 与算子统计
    ↓
JSON Graph Summary
```

 - 其中，`torch.onnx.export()`内部依次完成Dynamo图捕获、算子分解、ONNX算子翻译和导出图优化

### ONNX模型的基本结构

```text
ModelProto
├── 模型元数据
├── IR version
├── opset imports
├── producer 信息
└── GraphProto
    ├── inputs
    ├── outputs
    ├── initializers
    ├── nodes
    └── value_info
```

关键概念：

 - `ModelProto`：完整ONNX模型的顶层结构
 - `GraphProto`：计算图，包含节点、输入、输出和权重
 - `NodeProto`：一次ONNX算子调用，例如Conv或Gemm
 - `TensorProto`：权重、常量等张量数据
 - `ValueInfoProto`：张量名称、类型和shape信息
 - `initializer`：随模型保存的静态权重或常量

### IR version 与 opset version
 - IR version 描述ONNX文件、图、节点和张量等数据结构的版本
 - opset version 描述默认ONNX算子的定义和语义版本

### Initializer 与 运行时输入
Initializer通常包括：
 - 卷积权重
 - 卷积偏置
 - 全连接权重
 - 全连接偏置
 - 常量张量
 - 推理优化后生成或折叠的常量

项目中设置：

```python
 - include_initializers=True
 - keep_initializers_as_inputs=False
 ```

含义是：
 - 权重写入ONNX文件
 - 权重不要求调用者在每次推理时提供
 - 运行时只需要接收真正的业务输入input
 - 静态权重有利于常量折叠和后端图优化

### 固定输入形状
 - 减少初期实验变量
 - 便于不同后端使用完全相同的输入契约
 - 简化内存规划和输出验证
 - 避免把动态shape的重新编译、缓存或算子支持问题引入基础版本

固定形状本身不证明推理一定更快，只是为后续可重复比较建立受控条件；
动态batch和动态空间尺寸将在基础benchmark稳定后单独实验。

### Logits不是概率
当前输出是

```text
logits: [1, 1000]
```

Logits是模型对1000个类别产生的原始分数，不满足概率和为1的约束。
如果业务需要概率，应显示执行Softmax，并说明Softmax是否属于测量图的一部分。

### Checker、Shape Inference 与 运行时验证
 - Checker 检查模型结构、节点schema、属性和图连接是否合法；它不执行模型
 - Shape Inference 根据算子规则推导张量类型和形状，并将结果补充到value_info中
   它不保证所有中间shape都可以推导，也不能证明运行结果正确
 - 运行时数值验证 必须使用相同模型权重和输入张量，分别执行PyTorch与ONNX Runtime
   并检查输出shape、NaN/Inf、最大和平均误差、逐元素接近程度、Top-1或任务级输出

### 导出图观察
本次导出的MobileNetV3 Small包含122个节点：
| Operator | Count | 主要含义 |
|---|---:|---|
| Conv | 52 | 空间卷积、depthwise 和 pointwise 卷积 |
| HardSwish | 19 | MobileNetV3 激活函数 |
| Relu | 14 | 部分卷积块激活 |
| ReduceMean | 10 | 全局空间聚合、SE squeeze |
| HardSigmoid | 9 | 激活或 SE 门控 |
| Mul | 9 | 通道门控权重与特征相乘 |
| Add | 6 | 残差连接 |
| Gemm | 2 | 分类器全连接计算 |
| Reshape | 1 | 特征与分类头之间的形状转换 |

算子数量之和等于节点总数，说明统计逻辑覆盖了图中的全部节点。

 - ONNX节点数不等于PyTorch层数，因为module在导出过程中可能被分解、折叠或优化。
 - 图中没有独立的`BatchNormalization`节点，说明BatchNorm参数可能已经折叠进相邻
   卷积的权重和偏置；但仅凭算子统计不能完全证明具体融合过程，需要结合Netron、优化
   前后图和initializer进一步确认。
 - 当前`optimize=True`发生在PyTorch ONNX导出阶段，后续ONNX Runtime可能会执行
   自己的常量折叠、节点消除和算子融合。

### 当前验证范围
Day2 已验证：
 - Dynamo ONNX导出成功
 - 模型通过ONNX Checker
 - 输入、输出和固定shape正确
 - opset为18
 - 图节点和算子统计完整

当前尚未验证：
 - ONNX Runtime能否执行
 - PyTorch与ONNX Runtime是否数值一致
 - ONNX是否比PyTorch更快
 - 动态Shape、FP16或INT8
 - 冷启动、预热后延迟、吞吐量和尾延迟

---

## Day 3: ONNX Runtime CPU 执行与数值一致性

### Execution Provider
ONNX Runtime 通过 Execution Provider 将计算图节点映射到具体执行后端。
本机可用的Provider包括：

```text
CoreMLExecutionProvider
AzureExecutionProvider
CPUExecutionProvider
```

可用Provider不等于实际参与执行的Provider。本次显示指定：

```python
providers=["CPUExecutionProvider"]
```

因此只有 CPU EP 参与执行，避免实验环境变化后静默切换后端。

### InferenceSession 生命周期
创建`InferenceSession`可能包含模型解析、图优化、节点分配、Kernel初始化和
线程池准备。因此，正式benchmark应区分以下阶段：

```text
Session初始化
预热
稳定推理
端到端请求
```

Day3仅验证模型能否正确执行以及输出是否一致，不进行性能计时。

### 数据边界
当前数据链路是：

```text
PyTorch CPU Tensor
    ↓ detach / contiguous / NumPy
NumPy input
    ↓ session.run
ONNX Runtime CPU
    ↓ NumPy output
NumPy array
    ↓ copy / torch.from_numpy
PyTorch CPU Tensor
```

Tensor连续化、设备转换或显式`copy()`都可能产生数据复制。
后续性能测试必须分别测量：
- 仅包含`session.run()`的核心推理延迟
- 包含输入准备和输出转换的端到端延迟

### 数值一致性
PyTorch推理和ONNX导出复用同一个模型实例，并使用相同权重和输入，从而将输出差异
限定在算子实现、计算顺序和浮点舍入范围内。

逐元素比较采用：
```text
|actual - expected| <= atol + rtol × |expected|
```

正确性判断需要同时覆盖：
 - 输出Shape
 - 输出dtype
 - NaN/Inf
 - Top-1
 - 最大绝对误差
 - 平均绝对误差
 - 逐元素容差

不能仅依赖Top-1判断完整输出是否一致，因为非最大类别的logits仍可能存在明显误差。

### ORT 图优化
当前Session使用`ORT_ENABLE_ALL`。ORT可能在Session初始化期间执行：
 - 常量折叠
 - 冗余节点消除
 - Conv/Add等算子融合
 - CPU EP特定融合
 - layout优化

因此，Day 2 统计的序列化ONNX图不一定等于ONNX Runtime最终执行的内部图。
只要优化保持算子语义，输出应继续满足数值容差。
后续可以导出ORT优化模型或启用profiling，分析优化后的图结构和实际执行节点。

### 当前限制
 - 尚未分析每个节点的Provider分配
 - 尚未区分Session初始化、数据转换和核心推理耗时
 - 尚未研究线程数、执行模式和图优化等级

 ---

 ## Day 4: 统一后端抽象

 ### 统一接口与原生数据
 统一接口的目的是让上层流程以相同的方式调用不同后端。

后端内部适合保留运行时的数据格式：
 - PyTorch CPU: CPU Tensor
 - PyTorch MPS: MPS Tensor
 - ORT CPU: Numpy array

`run()`返回原生输出，`to_cpu()`再统一为CPU Tensor。
这样可以单独定义执行和输出转换的测量边界。

### Protocol 的作用与边界
`InferenceBackend`用Protocol描述方法及其类型关系。

具体类不必显式继承Protocol，只要满足相应结构，就可以被静态
类型检查器视为兼容实现。

类型注解不会自动检查运行时输入。dtype、snape、device等约束
仍需要显式验证，后端行为则需要测试覆盖。

当前测试通过不等同于已经通过mypy或pyright静态检查。

### 模型所有权
`nn.Module.to()`和`eval()`会修改模型状态。

如果CPU和MPS后端共享一个模型对象，设备迁移可能改变另一后端
持有的模型。因此每个PyTorch后端取得独立模型副本。

模型复制和设备放置属于初始化成本，应在稳定推理计时之外完成。

当前deepcopy方案适用于本项目的torchvision模型。后续接入
包含特殊资源或不可复制状态的模型时，需要重新设计模型构造方式。

### 输入所有权
CPU Tensor 转 Numpy 可能共享底层存储。
`detach()`也不会自动复制存储。

本项目在`prepare_input()`中取得独立快照，因此调用者随后
修改原始Tensor，不会改变已准备的输入。

这个选择引入了明确的准备成本，同时降低了别名和输入修改带来的
隐形错误。该成本应计入端到端指标，而不计入准备好输入后的执行指标。

### 两个同步点
run_once()调用顺序：

```text
prepare_input()
synchronize()
run()
synchronize()
to_cpu()
```

第一个同步点等待准备阶段完成；第二个同步点等待执行阶段完成。

MPS 使用显式设备同步。当前 PyTorch CPU 和 ORT CPU 同步调用
完成后已经可以使用输出，因此对应同步方法为空。

MPS 全局同步可能等待其他MPS工作，后续计时应避免同进程并发提交
无关任务。

### 正确性入口与计时入口
`run_once()`包含完整生命周期，适合正确性验证。

后续准备好输入后的同步延迟测量应覆盖：

```text
开始计时
run()
synchronize()
结束计时
```

输入准备、模型初始化和`to_cpu()`在这个指标之外。

该延迟仍包含Python调用、运行时调度和输出分配等成本，不能称为
纯 GPU Kernel 时间。预热、采样方式和统计方法需要另外实现。

### 本次结果的含义
与直接PyTorch CPU参考结果相比：
 - CPU 适配器最大绝对误差：0
 - ORT CPU 最大绝对误差：约1.19e-05
 - PyTorch MPS 最大绝对误差：约9.11e-06
 - 三个后端Top-1均为574，逐元素容差检查通过

这支持统一接口在本次输入上保留了数值行为。
误差更小不表示任务准确率越高，也不能推导性能优劣。

### 线程配置的解释
PyTorch记录intra-op=6、inter-op=18。
ORT记录intra-op=0、inter-op=0，表示采用运行时默认设置。

intra-op控制算子内部并行，inter-op涉及算子之间并行。
ORT_SEQUENTIAL不等于整个推理只使用一个CPU线程。

记录到的配置值不等于实际活跃线程数；不同运行时的线程数也不能
直接相加或当作硬件核心数。

后续需要区分默认配置基线与显式控制线程数的实验。

### 证据边界
 - 29项测试通过说明已覆盖场景通过，不证明任意模型都兼容。
 - 生命周期顺序测试不等于设备计时精度测试。
 - MPS Tensor设备检查不等于逐算子GPU执行证明。
 - 环境变量为null指标时读取时未设置。
 - 当前分类比较函数以 batch size 1 为前提。

---

## Day5: 规范 Benchmark

### 首先明确测量边界
 - 本阶段测量的是“输入已经转换为后端原生格式之后，一次同步推理调用的延迟”。
 - 计时区间包括`backend.run()`和完成边界上的`backend.synchronize()`。
 - 模型加载、ONNX导出、Session初始化、输入准备、输出回传、正确性检查和结果落盘均不计时。
 - 因而这个结果不能称为端到端延迟，也不能直接代表真实应用响应时间。

### 异步后端必须同步
 - MPS等异步后端可能只在调用时提交任务，CPU随后继续运行。
   如果结束计时前不执行同步，测到的主要是任务提交开销，而不是设备完成推理所需的时间。
 - 计时前同步用于清理前序任务；执行后同步用于建立明确的完成边界。

### 正确性门禁必须位于性能结论之前
 - 性能结论必须以正确性为前提。
 - 正式采样前后都要检查输出 Shape、NaN/Inf、Top-1、一致性容差和误差指标。

### 预热次数不等于稳态证明
 - Warm-up可以降低初始化、缓存填充、图优化、设备唤醒等一次性成本。
   但固定预热次数只是一种实验策略，不是稳态已经建立的证据。
 - 本次MPS即使将进程内warm-up从50次提高到1000次，两次独立进程的平均延时仍相差12.59%。
   这说明部分性能状态可能存在于Python进程之外，固定进程内预热无法完全控制它。
 - 更严格的后续方案可以依据滚动窗口的延迟变化判断延迟是否收敛，而不是只依赖固定次数。

### 延迟统计 P50 P90 P99
 - P50 描述典型请求，P90/P99 描述尾部延迟。
 - 300个样本下，P99主要由最高的少数几个样本决定，对后台调度和系统活动更敏感。
 - 本实验保留原始纳秒样本，使所有统计量都可以重新计算和审计。

### 派生串行速率不等于吞吐率
 - `1000 / mean_latency_ms` 表示batch size 1、请求串行执行时的理论服务速率。
 - 它没有施加并发负载，也没有测量排队、资源竞争和持续吞吐，因此应称为derived serial
   service rate，不能直接写成QPS或并发吞吐率。

### 多轮顺序与复现证据
 - 三轮测试采用不同后端执行顺序，降低固定顺序对缓存、温度和设备状态造成的系统性偏差。
 - 每轮记录系统、运行时、电源状态、线程配置、Git commit、源码哈希、模型权重哈希和输入哈希。
 - 哈希一致只能证明被测内容一致，不能证明硬件频率、缓存、运行时内部状态和后台负载完全相同。

### 默认线程配置与公平性边界
 - PyTorch CPU 使用 6 个计算线程和 8 个inter-op线程；
   ONNX Runtime 的 intra-op 和 inter-op 均为0，表示采用运行时默认策略。
 - 因此 ORT CPU 相对于 PyTorch CPU 约 46.68 倍延迟差异是当前运行时默认配置下的观察结果，
   不能解释为等线程条件下的纯算子性能差异。后续线程扫描应作为独立实验。

### MPS案例说明
 - MPS在受控AC条件下出现约 2.74-3.51 ms的跨进程均值区间。
 - 连续运行、充电状态、显式预激活、1000次长预热都分别被检查，但没有得到可重复的单值。
 - 在无法进一步定位 MPS/Metal 内部状态时，明确报告结果范围和实验限制。
 - Benchmark的目标是建立边界明确、正确性可验证、原始证据可审计、能够发现不稳定的测量系统。

### 后续实验方向
 - 使用滚动窗口变化判断延迟是否收敛，代替固定次数 warm-up
 - 使用 PyTorch Profiler、ONNX Runtime Profiling 等工具定位算子和运行时开销
 - 扫描不同 CPU 线程数，进行等线程比较
 - 增加不同计算结构的模型，例如 ResNet、ViT 或 Transformer
 - 测试不同 batch size、精度和输入尺寸
 - 增加并发请求与持续负载测试，测试真实吞吐量和排队延迟
 - 将解码、预处理、数据传输、推理和后处理纳入端到端延迟