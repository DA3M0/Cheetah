# xxx\_ops 通用算子目录 - 产品需求文档

## Overview

* **Summary**: 新增一个通用、可扩展的算子目录（`xxx_ops`），让任意模型架构都能通过组合已注册的基础算子完成前向计算；每个算子同时提供 native CPU 与 Vulkan 两种后端实现，保持两个后端能力对齐。单个模型注册文件还可携带自己的自定义 shader / 自定义算子。
* **Purpose**: 现有 `OpsToolbox`（`src/chat/native/ops.rs`）与 `shared_ops.comp` 的算子集合是固定的，新增一个算子就要改核心文件，无法支撑“多奇怪的架构都能组合算子”的目标。需要一个与 precision 注册表同构的、声明式、可自动发现的算子注册表，把“控制流（注册文件）”与“算子（目录）”彻底分离。
* **Target Users**: 在 cheetah 中接入新模型架构的开发者；需要在不改动核心分发代码的前提下组合/新增算子。

## Goals

* 建立 `src/chat/ops/` 算子注册表：每个算子文件声明 `OpRegistration`（算子 id + 后端算子表），`build.rs` 自动发现并生成 `REGISTRY`，运行时按 id 查找。
* 每个注册算子同时携带 **native 实现** 与 **vulkan 实现**（op code + 分发入口），两个后端功能对齐（“vulkan 跟进”）。
* 落地首批基础算子，覆盖：激活函数族、逐元素二元运算、归一化族、位置编码与其他（含两个有状态递归算子）。
* 支持单个模型注册文件自带自定义 `.comp` shader 与自定义 native 算子，并能在运行时被分发调用。
* 模型注册文件通过算子 id 查找并调用算子，实现任意架构的组合。
* 不破坏现有 7 个家族（llama/qwen2/qwen3/mixtral/mistral/gemma/phi）的 forward 与 Ant 契约；现有测试全绿。

## Non-Goals

* 本期**不重构**现有家族的 forward 去改用新目录（保持现有热路径稳定、性能不退化）；新目录以“加法”形式提供，并用测试 + 一处接入演示证明可用。
* 不替换精度注册表（`src/chat/precision`）的大矩阵乘法路径：线性投影 / lm\_head 仍走 precision registry 的存储精度融合核；新算子目录只承载“非大矩阵乘法”的逐元素/归一化/位置/有状态算子，统一在 f32 缓冲上运算（小参数如 norm 权重、bias 由调用方提前反量化到 f32，开销可忽略）。
* 不在本期新增模型家族；只提供算子基础设施与一批算子。
* 不涉及 CUDA/Metal（本仓库这两个后端走 Candle，不在范围）。

## Background & Context

* 现有两套注册表示例（均由 `build.rs` 自动发现目录 → 生成 `REGISTRY` → 运行时 lookup 派生）：
  * 模型注册表：`src/chat/registrations/*.rs` → `OUT_DIR/model_registry.rs`（`REGISTRY` / `ALL_ARCHITECTURES`），见 `src/chat/registry.rs` 与 `docs/REGISTRY.md`。
  * 精度注册表：`src/chat/precision/*.rs` → `OUT_DIR/precision_registry.rs`，每个格式文件同时声明 `NativeOps` 与 `VulkanOps` 两个 fn-pointer 表；测试断言每个格式两个后端都齐全（`every_format_is_registered_for_both_backends`）。
* native 侧算子现状：`src/chat/native/ops.rs` 的自由函数（`gemv`/`rmsnorm`/`silu_multiply`/`gelu`/`rope_heads`/`softmax`/`l2norm_heads`/`causal_conv1d_step`/`gated_delta_step` 等）+ `OpsToolbox` 方法；名称冻结，被 `native/kernels.rs` 再导出给 precision 注册表用。
* vulkan 侧算子现状：`shared_ops.comp` 用一个 push-constant `op`（0..9）分发到 `run_rms_norm`/`run_layer_norm`/`run_elementwise`(含子操作 1/2/3/4)/`run_rotary`/`run_attention`/`run_embedding_gather`/`run_bias_add`；C 侧 `cheetah_vulkan.c` 的 `vk_qwen_*` 系列函数负责绑定 buffer、写 push constant、派发。`build.rs` 已会自动发现 `src/chat/registrations/` 下的 `.comp` 文件（`discover_shaders`）。
* 强约定（见 `docs/CODEBASE.md` §6 不变量）：注册文件是唯一真相源；核心代码无家族分支；控制流与算子分离；`Disk` 层必须是真实释放页；位运算变更需要隔离测试。新设计必须遵守。

## Functional Requirements

* **FR-1（注册表骨架）**: 新增 `src/chat/ops/` 目录与 `src/chat/ops.rs` 模块；定义 `OpRegistration` / `NativeOp` / `VulkanOp` 共享类型；`build.rs` 新增第三个注册表规范（同构于 model/precision），扫描该目录生成 `OUT_DIR/ops_registry.rs` 的 `REGISTRY` 与 `ALL_OP_IDS`，并 `include!` 进 `ops.rs`。新增算子 = 加一个文件，不改核心分发。
* **FR-2（按 id 查找）**: 提供 `registration_for_op(id)` / `native_op_for(id)` / `vulkan_op_for(id)` 运行时 lookup，全部由生成的 `REGISTRY` 派生，不硬编码算子数据。
* **FR-3（算子签名族）**: 定义有限个算子签名族（unary 就地 / binary 输出到第三缓冲 / norm 带权重与 eps / softmax 按行 / 有状态步），`NativeOp` 用枚举携带与族匹配的具体 fn 指针；`VulkanOp` 携带 op code（u32）+ 族标签，供通用分发器绑定 buffer/参数。
* **FR-4（首批算子清单）**: 落地以下算子（每个都有 native + vulkan 实现与一致性测试）：
  * 激活（unary 就地）：`act.relu`、`act.leaky_relu`、`act.tanh`、`act.sigmoid`、`act.silu`、`act.gelu_tanh`、`act.gelu_erf`、`act.softplus`；门控乘（就地 gate ← act(gate)⊙up）：`act.swiglu`、`act.geglu`。
  * 逐元素二元（输出到 out，rhs 可广播到最后一维）：`ew.add`、`ew.sub`、`ew.mul`、`ew.div`、`ew.max`、`ew.min`。
  * 归一化：`norm.rmsnorm`、`norm.rmsnorm_offset`（gemma `(1+weight)`）、`norm.layernorm`、`norm.groupnorm`、`norm.l2norm`（按 head）、`norm.softmax`（按行）。
  * 位置/其他：`pos.rope`（参数化）、`embedding.gather`、`bias.add`（三者 vulkan 已存在于 shared\_ops，补注册与 native 对齐封装）。
  * 有状态递归：`conv.causal1d_step`、`attn.gated_delta_step`（native 已存在，补 vulkan 对等实现）。
* **FR-5（vulkan 分发）**: 扩展 `shared_ops.comp` 增加新 op code 分支（激活族、二元族、groupnorm、l2norm、softmax、rmsnorm\_offset、两个有状态算子）；`cheetah_vulkan.c` 增加通用分发 C 入口并在 `cheetah_vulkan.h` 声明；Rust 侧 `vulkan` 模块提供按 `VulkanOp` 调用的通用封装。op code 分配要稳定、可覆盖（沿用 `CHEETAH_QWEN_OPS_SPV` / `CHEETAH_*_OPS_SPV` 机制）。
* **FR-6（注册自带自定义 shader/算子）**: 复用 `build.rs` 已有的 `src/chat/registrations/*.comp` 发现能力；提供机制让注册文件声明自定义算子（native fn + 自定义 shader 的 op code 或独立 pipeline），并能在运行时按 id 分发。自定义 shader 编译失败时给出运行时可覆盖路径（与现有 shader 覆盖机制一致），不阻断构建。
* **FR-7（注册表不变量测试）**: 断言：所有 op id 全局唯一；每个注册算子都同时声明 native 与 vulkan 后端（对齐不变量，仿 `every_format_is_registered_for_both_backends`）；新增文件自动被发现（目录即契约）。
* **FR-8（native/vulkan 一致性测试）**: 对每个算子，native 结果与 vulkan（设备可用时）结果在容差内一致；无 Vulkan 设备时该测试自动跳过（不阻塞 CPU 环境）。
* **FR-9（不回归）**: `cargo test`（默认特性）全绿；`cargo test --features vulkan` 全绿（设备门控测试除外）；现有 7 个家族 forward 与 Ant 行为不变。

## Non-Functional Requirements

* **NFR-1**: 遵循仓库约定：注册文件是唯一真相源、核心代码无家族分支、控制流与算子分离、低阶数学保持静态分发自由函数且名称冻结。
* **NFR-2**: 新增算子 = 新增文件；不得在 `native/ops.rs` / `shared_ops.comp` / `cheetah_vulkan.c` 的现有分支里硬编码新家族。
* **NFR-3**: 现有 forward 热路径性能不退化（现有家族不改用新目录；新算子走一次表查找 + fn 指针调用，开销与 precision registry 同级）。
* **NFR-4**: 无 Vulkan 设备 / 未编译 vulkan 特性时，native 算子完整可用，vulkan 相关测试优雅跳过。
* **NFR-5**: 文档同步：更新 `docs/CODEBASE.md` 的模块索引与 `docs/REGISTRY.md` 的“如何新增”章节，说明新算子目录。

## Constraints

* **Technical**: shader 编译依赖 `glslc`（build.rs 已有容错：失败给 warning + 运行时覆盖路径，不 panic 硬编码 shader）；C FFI 边界在 `cheetah_vulkan.c/.h`；三个注册表（model/precision/ops）必须同构，`build.rs` 抽象可复用。
* **Dependencies**: rayon（native 并行）、serde\_json、half；vulkan 特性开启时链接 `vulkan`。
* **Business**: 不引入新的第三方依赖；不改变现有二进制对外行为。

## Assumptions

* 新算子目录统一在 **f32 缓冲** 上运算（小参数 norm 权重/bias 由调用方反量化到 f32）；大矩阵乘法仍归 precision 注册表。这与现有 native `silu_multiply`/`softmax`/`rope_heads` 的做法一致，也让 vulkan/native 对称。
* 现有 `shared_ops.comp` 的 buffer 绑定模型（5 个 storage buffer + 8 个 push-constant 参数）足以承载新增算子；有状态算子用单 workgroup 递归派发即可正确（每 token 一步）。
* `build.rs` 已发现 `src/chat/registrations/*.comp`，FR-6 的“自定义 shader”主要是把发现结果暴露给运行时分发，而非全新的构建机制。

## Open Questions

* [ ] 有状态算子（`conv.causal1d_step` / `attn.gated_delta_step`）的 vulkan 实现是否必须在本期交付，还是可先注册 native + 标注 vulkan 为后续？（用户已明确“vulkan 跟进”，默认本期交付对等实现；若实现风险过高则在任务里降为 stretch 并显式记录。）
* [ ] 自定义 shader 的运行时分发粒度：按 op code 复用 shared\_ops pipeline，还是允许注册独立 pipeline？（默认先复用 shared\_ops pipeline + op code，独立 pipeline 留作后续扩展，避免过度设计。）

## Acceptance Criteria

### AC-1: 算子注册表骨架与自动发现

* **Type**: `rule`
* **Given**: `src/chat/ops/` 目录存在且 `build.rs` 注册了该目录
* **When**: 运行 `cargo build`
* **Then**: 生成 `OUT_DIR/ops_registry.rs`，其中 `REGISTRY` 指向每个算子文件的 `REGISTRATION`，`ALL_OP_IDS` 收集所有 id；`src/chat/ops.rs` 通过 `include!` 引入
* **Pass Condition**: 构建成功；`ops::REGISTRY` 非空；在目录下新增一个最小算子文件后重新构建，`REGISTRY` 自动包含它（无需改其他文件）
* **Evidence**: `cargo build` 成功输出；新增文件发现测试通过

### AC-2: 按 id 查找 native/vulkan 算子

* **Type**: `rule`
* **Given**: 一个已注册算子 id（如 `act.relu`）
* **When**: 调用 `ops::native_op_for("act.relu")` 与 `ops::vulkan_op_for("act.relu")`
* **Then**: 两者都返回 `Some(...)`，分别携带正确签名族的 fn 指针 / op code
* **Pass Condition**: 查找命中且类型与算子声明的族一致；未知 id 返回 `None` 并列出已注册 id
* **Evidence**: lookup 单元测试通过

### AC-3: 每个算子双后端齐全且 id 唯一

* **Type**: `rule`
* **Given**: 生成的 `ops::REGISTRY`
* **When**: 运行注册表不变量测试
* **Then**: 所有 op id 全局唯一；每个注册条目同时声明 native 与 vulkan 后端
* **Pass Condition**: 测试断言通过；不存在只有单后端的算子
* **Evidence**: `cargo test ops_registry` 类测试通过

### AC-4: 首批算子全部可用（native）

* **Type**: `rule`
* **Given**: FR-4 列出的全部算子 id
* **When**: 对每个 id 执行一次 native 调用并与参考实现对比
* **Then**: 结果在数值容差内匹配参考（或与现有自由函数结果一致）
* **Pass Condition**: 每个算子的 native 单测通过
* **Evidence**: `cargo test`（默认特性）中算子测试全部通过

### AC-5: 首批算子全部可用（vulkan）且与 native 一致

* **Type**: `rule`
* **Given**: FR-4 列出的全部算子 id 且存在 Vulkan 设备
* **When**: 对每个 id 分别跑 native 与 vulkan 并对比
* **Then**: 两者结果在容差内一致；无设备时测试自动跳过
* **Pass Condition**: 有设备时每个算子一致性断言通过；无设备时不报错
* **Evidence**: `cargo test --features vulkan` 中 `ops_parity_*` 测试通过或自动跳过

### AC-6: 模型注册可自带自定义 shader/算子

* **Type**: `rule`
* **Given**: 一个注册文件附带自定义 `.comp` 与自定义 native fn，并声明一个自定义算子 id
* **When**: 构建并在运行时按该 id 分发
* **Then**: 自定义算子可被 lookup 命中并在两个后端执行；自定义 shader 编译失败时回退到运行时覆盖路径且不阻断构建
* **Pass Condition**: 存在至少一个端到端演示（注册自带自定义算子的最小 forward 或测试）证明分发可用
* **Evidence**: 自定义算子发现 + 分发测试通过

### AC-7: 不回归

* **Type**: `rule`
* **Given**: 本次改动
* **When**: 运行 `cargo test` 与 `cargo test --features vulkan`
* **Then**: 现有全部测试通过，现有 7 个家族 forward / Ant 行为不变
* **Pass Condition**: 两套测试全绿（设备门控测试除外）
* **Evidence**: 两套 `cargo test` 输出无失败

### AC-8: 设计与文档质量

* **Type**: `rubric`
* **Dimension**: 新算子目录与现有注册表的同构程度、可扩展性、文档完整度
* **Scale**: 1-5
* **Anchors**: 1 = 核心代码硬编码家族、加算子要改多处、无文档；3 = 基本同构但有特例分支、文档部分缺失；5 = 与 model/precision 注册表完全同构、加算子=加文件、核心零家族分支、文档同步更新
* **Pass Threshold**: >= 4
* **Evidence**: 代码结构审查 + `docs/CODEBASE.md`/`docs/REGISTRY.md` 更新核对
