# Learning Notes

本文件记录项目实施过程中掌握的推理系统知识、工程边界和阶段性理解。
实际执行结果和实验数据记录在[experiment-log.md](experiment-log.md)。

## Day 1: PyTorch CPU 与 MPS 基线

### model.eval() 和 torch.inference_mode()
 - eval()控制Dropout、BatchNorm等模型层的推理行为
 - inference_mode()关闭Autograd记录，减少推理时的状态维护和开销
 - 二者职责不同，不可相互替代
 - 一个标准推理入口通常需要模型处于eval模式，并在inference_mode上下文中执行

### MPS 异步执行
 - CPU向MPS提交计算任务后可能继续执行后续代码
 - 性能计时必须在正确的测量边界调用torch.mps.synchronize()
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
 - 本次CPU/MPS最大绝对误差为0.000007，Top-1一致

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

 - 其中，torch.onnx.export()内部依次完成Dynamo图捕获、算子分解、ONNX算子翻译和导出图优化

### ONNX模型的基本结构

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

关键概念：
 - ModelProto：完整ONNX模型的顶层结构
 - GraphProto：计算图，包含节点、输入、输出和权重
 - NodeProto：一次ONNX算子调用，例如Conv或Gemm
 - TensorProto：权重、常量等张量数据
 - ValueInfoProto：张量名称、类型和shape信息
 - initializer：随模型保存的静态权重或常量

### IR version 与 opset version
 - IR version描述ONNX文件、图、节点和张量等数据结构的版本
 - opset version描述默认ONNX算子的定义和语义版本

### Initializer 与 运行时输入
Initializer通常包括：
 - 卷积权重
 - 卷积偏置
 - 全连接权重
 - 全连接偏置
 - 常量张量
 - 推理优化后生成或折叠的常量
项目中设置：
 - include_initializers=True
 - keep_initializers_as_inputs=False
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
当前输出是`logits: [1, 1000]`
Logits是模型对1000个类别产生的原始分数，不满足概率和为1的约束。
如果业务需要概率，应显示执行Softmax，并说明Softmax是否属于测量图的一部分。

### Checker、Shape Inference 与 运行时验证
 - Checker检查模型结构、节点schema、属性和图连接是否合法；它不执行模型
 - Shape Inference根据算子规则推导张量类型和形状，并将结果补充到value_info中
   它不保证所有中间shape都可以推导，也不能证明运行结果正确
 - 运行时数值验证必须使用相同模型权重和输入张量，分别执行PyTorch与ONNX Runtime
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