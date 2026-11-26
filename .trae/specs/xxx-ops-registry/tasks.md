# xxx_ops 通用算子目录 - 实现计划

## 总览
实现顺序遵循“骨架 → native 算子 → vulkan shader → vulkan 分发 → 双后端注册 → 自定义 shader 机制 → 测试 → 回归/文档”。所有任务完成后进入独立 Review。

---

## Task 1: 算子注册表骨架（build.rs + 共享类型 + lookups）
- **Status**: `completed`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - `build.rs` 新增第三个 `RegistrySpec`（`src/chat/ops` → `OUT_DIR/ops_registry.rs`，`type_name = OpRegistration`，`module_suffix = "_op"` 或空，`index = ("ALL_OP_IDS", "OP_IDS")`），与 model/precision 两个注册表同构；`println!("cargo:rerun-if-changed=src/chat/ops")`。
  - 新增 `src/chat/ops.rs`：定义 `OpRegistration`、`OpBackend`（`Native(NativeOp)` / `Vulkan(VulkanOp)`）、算子签名族枚举 `NativeOp`（`Unary(fn(&mut [f32]))`、`BinaryInPlace(fn(&mut [f32],&[f32]))`、`Binary(fn(&[f32],&[f32],&mut [f32], usize /*broadcast period or 0*/))`、`Norm(fn(&mut [f32],&[f32],f32))`、`Softmax(fn(&mut [f32],usize))`、`Stateful(fn(...) -> ...)`）与 `VulkanOp { op_code: u32, kind: OpKind, shader_env: Option<&'static str> }`；定义稳定的 op code 常量（保留 0..9 给现有 shared_ops，从 10 起分配）。
  - `include!(concat!(env!("OUT_DIR"), "/ops_registry.rs"))`；提供 `registration_for_op` / `native_op_for` / `vulkan_op_for` / `registered_op_ids` lookups，全部派生自 `REGISTRY`。
  - 加一个最小占位算子文件（如 `src/chat/ops/_bootstrap.rs` 注册 `act.relu` 的 native 桩），仅用于让骨架可编译、可被发现；后续 Task 2 替换为真实实现。
  - 在 `src/chat/mod.rs` 声明 `pub(crate) mod ops;`。
- **Acceptance Criteria Addressed**: AC-1, AC-2
- **Test Requirements**:
  - `rule` TR-1.1: `cargo build` 成功；`OUT_DIR/ops_registry.rs` 生成且含 `REGISTRY`/`ALL_OP_IDS`。证据：构建输出 + `find target -name ops_registry.rs`。
  - `rule` TR-1.2: 在 `src/chat/ops/` 新增一个临时最小算子文件后重新构建，`registered_op_ids()` 包含其 id，删除后消失——证明目录即契约。证据：单元/集成测试。
  - `rule` TR-1.3: `native_op_for("unknown")` 返回 `None` 且错误信息列出已注册 id。证据：lookup 测试。
- **Notes**: op code 常量集中定义，避免 shader/C/Rust 三处漂移。
- **Completion Evidence**:
  - TR-1.1: `cargo build` 成功；生成 `target/debug/build/cheetah/<hash>/out/ops_registry.rs` 含 `pub(crate) const REGISTRY: &[&OpRegistration] = &[ &_bootstrap::REGISTRATION ];`。
  - TR-1.2: 临时新增 `src/chat/ops/_probe.rs` 后重建，生成文件自动加入 `mod _probe;` 与 `&_probe::REGISTRATION`；删除后消失——目录即契约定律验证通过。
  - TR-1.3: `cargo test --bin cheetah ops::` 通过（`registry_is_non_empty`、`unknown_op_lookup_is_none_and_lists_registered`、`known_bootstrap_op_resolves_native`）。
  - 附加：`cargo build --features vulkan` 同样成功，证明 `#[cfg(feature = "vulkan")]` 门控正确。

## Task 2: native 侧首批算子实现与注册
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 在 `src/chat/ops/` 下按族分文件注册算子：`activations.rs`、`elementwise.rs`、`norms.rs`、`positional.rs`、`stateful.rs`。每个文件声明一个或多个 `OpRegistration`。
  - 复用 `src/chat/native/ops.rs` 已有的自由函数（`silu_multiply`→`act.swiglu`、`gelu_multiply`→`act.geglu`、`rmsnorm`/`rmsnorm_offset`/`layernorm`/`softmax`/`l2norm_heads`/`rope_heads`/`causal_conv1d_step`/`gated_delta_step`）；缺失的补实现：`relu`/`leaky_relu`/`tanh`/`sigmoid`/`silu`/`gelu_tanh`/`gelu_erf`/`softplus`（unary）、`ew.{add,sub,mul,div,max,min}`（broadcast）、`groupnorm`。
  - 统一封装成 `NativeOp` 族匹配的 fn 指针；norm 类算子的 weight/bias 由调用方以 f32 缓冲传入（小参数反量化开销可忽略）。
  - 移除 Task 1 的占位桩，替换为真实注册。
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `rule` TR-2.1: 每个 FR-4 算子 id 都有 native 单测，结果与独立参考实现（或已有自由函数）在容差内一致。证据：`cargo test ops_` 全过。
  - `rule` TR-2.2: `ew.*` 的 broadcast 行为测试（rhs 长度 = 1 与 = 最后一维两种）。证据：单测。
- **Notes**: 现有 `native/ops.rs` 自由函数保持不动（名称冻结）；新文件通过 `use crate::chat::native::ops as native_ops;` 复用或新增薄封装。
- **Completion Evidence**:
  - TR-2.1: 首批 27 个算子全部注册并有 native 单测（activations 10 / elementwise 6 / norms 6 / positional 3 / stateful 2）。`cargo test --bin cheetah ops::` → `38 passed; 0 failed`。
  - TR-2.2: `broadcast_scalar`（rhs 长度 1）与 `broadcast_last_dim_over_rows`（rhs 长度 = 最后一维）两个 broadcast 用例通过。
  - 附加：`cargo build` 与 `cargo build --features vulkan` 均成功；删除占位 `_bootstrap.rs` 后注册表改由 5 个真实文件组装。

## Task 3: 扩展 shared_ops.comp 的 GLSL 算子
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1（op code 常量定稿）
- **Description**:
  - 在 `shared_ops.comp` 增加新 op code 分支与 `run_*` 函数：unary 激活族（一个 `run_activation(index, sub_op)`，sub_op 区分 relu/tanh/sigmoid/silu/gelu_tanh/gelu_erf/softplus/leaky_relu）、二元族（`run_binary(index, sub_op)` 含 broadcast）、`run_rms_norm_offset`、`run_group_norm`、`run_l2_norm`、`run_softmax`、有状态的 `run_conv1d_step` / `run_gated_delta_step`。
  - 复用现有 5 buffer + 8 push-constant 模型；参数不够时扩展 push-constant 到 16 个 u32（同步更新 C 侧 pipeline layout）。
  - op code 与 Task 1 常量一致；保持 `CHEETAH_QWEN_OPS_SPV` 可覆盖。
- **Acceptance Criteria Addressed**: AC-5（shader 层）
- **Test Requirements**:
  - `rule` TR-3.1: `glslc -fshader-stage=compute shared_ops.comp -o /tmp/sp.spv` 编译无错误（CI/本地）。证据：构建日志无 shader warning 失败。
  - `rule` TR-3.2: 每个新 `run_*` 的 GLSL 数学与 native 参考在数值上一致（由 Task 7 端到端验证；此处保证 shader 可编译且分支可达）。证据：编译通过 + 代码走查。
- **Notes**: 有状态算子用单 workgroup（local_size 足够覆盖通道数）递归更新共享状态，正确性优先于性能。
- **Completion Evidence**:
  - TR-3.1: `glslc -fshader-stage=compute shared_ops.comp -o /tmp/opencode/shared_ops.spv` 成功；`cargo build --features vulkan` 无 shader 编译错误。
  - TR-3.2: `cargo test --features vulkan ops::` 中 `parity_unary_activations` / `parity_binary_elementwise` / `parity_norms` / `parity_l2_softmax_rope_embedding` / `parity_conv1d_step_multi_token` / `parity_gated_delta_step_multi_token` 全部通过（NVIDIA GT 730M）。

## Task 4: Vulkan C 分发入口 + Rust 通用封装
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - `cheetah_vulkan.c` 增加通用分发函数（如 `vk_run_op(uint32_t op, const float *b0, const float *b1, float *b2, float *b3, const float *b4, uint32_t count, uint32_t p1..p15, uint32_t local_size)`），绑定 5 个 storage buffer + 写 push-constant + 派发；现有 `vk_qwen_*` 保持不动（向后兼容）。
  - `cheetah_vulkan.h` 声明新入口。
  - Rust `src/chat/vulkan/` 增加 `ops` 子模块（或扩展现有），提供 `run_vulkan_op(vk_op: &VulkanOp, buffers...)` 封装，把 `VulkanOp.op_code` 与族翻译成 C 调用参数。
- **Acceptance Criteria Addressed**: AC-2（vulkan lookup 可用）, AC-5（分发层）
- **Test Requirements**:
  - `rule` TR-4.1: `cargo build --features vulkan` 成功，C 编译无 warning。证据：构建输出。
  - `rule` TR-4.2: 通用入口能正确派发一个已存在 op（如 rmsnorm op 0）并与旧 `vk_qwen_rms_norm` 结果一致，证明分发无回归。证据：vulkan 特性测试。
- **Notes**: push-constant 布局变更要同步 shader 与 C，避免错位。
- **Completion Evidence**:
  - TR-4.1: `cargo build --features vulkan` 成功；新增 `vk_run_op` / `vk_run_custom_op`（cheetah_vulkan.c/.h）与 Rust `run_op_raw` / `run_custom_op_raw`（src/native.rs）。
  - TR-4.2: `src/chat/vulkan/ops.rs` 的 `run_vulkan_op` 把 28 个已注册算子翻译到 5-buffer/8-push-constant 布局；vulkan 特性下全部 native↔vulkan parity 测试通过，包含复用旧 op code 的 `norm.rmsnorm`、`pos.rope`、`embedding.gather`、`bias.add`。

## Task 5: 首批算子 vulkan 注册 + 双后端齐全 + 有状态算子对等
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 2, Task 3, Task 4
- **Description**:
  - 为 Task 2 每个算子在其注册文件补全 `VulkanOp`（op code + kind + shader_env），使每个 `OpRegistration.backends` 同时含 native 与 vulkan。
  - 有状态算子（`conv.causal1d_step` / `attn.gated_delta_step`）的 vulkan 对等实现校验：与 native 在小尺寸输入上一致；状态跨 token 递推正确。
  - `pos.rope` / `embedding.gather` / `bias.add` 复用现有 shared_ops op code（3/8/9）注册，native 封装与 vulkan 对齐。
- **Acceptance Criteria Addressed**: AC-3, AC-5
- **Test Requirements**:
  - `rule` TR-5.1: 注册表不变量测试断言每个 op 双后端齐全、id 唯一。证据：`cargo test --features vulkan` 中 `every_op_has_both_backends` 类测试通过。
  - `rule` TR-5.2: 有状态算子 native↔vulkan 跨 3 个 token 递推一致。证据：一致性测试。
- **Notes**: 若有状态 vulkan 实现风险过高，记录为 stretch 并在 spec 开放问题里显式标注，但默认本期交付。
- **Completion Evidence**:
  - TR-5.1: `registered_op_ids_are_unique`、`every_registered_op_has_native_backend`、`every_registered_op_has_vulkan_backend`、`vulkan_kind_matches_native_shape` 全部通过；首批 27 个算子 + 自定义 `act.mish` 均为双后端。
  - TR-5.2: `parity_conv1d_step_multi_token` / `parity_gated_delta_step_multi_token` 跨 3 个 token 的状态与输出均与 native 一致。

## Task 6: 模型注册自带自定义 shader / 自定义算子机制 + 演示
- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 1, Task 4
- **Description**:
  - 复用 `build.rs` 已发现的 `src/chat/registrations/*.comp`：把发现到的自定义 shader 暴露给运行时（通过生成的常量或注册表字段），让注册文件能声明一个自定义算子 id → (native fn, 自定义 shader 的 op code 或 pipeline)。
  - 提供最小端到端演示：在 `src/chat/registrations/` 或 `src/chat/ops/` 下加一个自定义算子示例（自定义 unary 激活），带自定义 `.comp`（或复用 shared_ops 新 op code）+ native fn，证明可被 `vulkan_op_for` 命中并分发。
  - 自定义 shader 编译失败时走运行时覆盖路径，不阻断构建（沿用现有容错）。
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `rule` TR-6.1: 自定义算子 id 可被 `native_op_for`/`vulkan_op_for` 命中；端到端演示测试跑通（vulkan 有设备时）。证据：测试通过。
  - `rule` TR-6.2: 故意损坏自定义 `.comp` 语法时，`cargo build --features vulkan` 不失败（给 warning），运行时可用 `CHEETAH_*_OPS_SPV` 覆盖。证据：构建容错验证。
- **Notes**: 优先复用 shared_ops pipeline + 新 op code，避免引入独立 pipeline 的过度设计。
- **Completion Evidence**:
  - TR-6.1: 新增 `src/chat/ops/custom.rs`（`act.mish`，op code `CUSTOM_OP_BASE`）+ `src/chat/ops/shaders/mish.comp`；`VulkanOp::custom_shader: Option<CustomShader>` 携带 shader stem 与运行时覆盖变量；`run_vulkan_op` 对 op code ≥ 100 调用 `vk_run_custom_op`。`parity_custom_shader_op` 在有设备时通过，`mish_declares_its_custom_shader` 验证 lookup 元数据。
  - TR-6.2: 临时写入非法 `mish.comp` 后 `cargo build --features vulkan` 仅输出 `custom op shader ... failed to compile` warning 并成功结束；恢复 shader 后重新编译正常，运行时可通过 `CHEETAH_MISH_OPS_SPV` 覆盖。

## Task 7: 注册表不变量测试 + native/vulkan 一致性测试
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 5, Task 6
- **Description**:
  - 在 `src/chat/ops.rs`（或 `ops/mod` 测试模块）加注册表测试：id 唯一性、每算子双后端齐全、目录发现、未知 id 错误信息。
  - 加一致性测试模块（`#[cfg(all(test, feature = "vulkan"))]`）：对每个注册算子跑 native 与 vulkan，对比容差；用 `init_vulkan_device` 返回值判断设备是否可用，不可用时 `eprintln` 跳过（不 panic）。
- **Acceptance Criteria Addressed**: AC-3, AC-5
- **Test Requirements**:
  - `rule` TR-7.1: 默认特性 `cargo test` 下注册表测试全过。证据：测试输出。
  - `rule` TR-7.2: `cargo test --features vulkan` 下一致性测试：有设备则断言通过，无设备则优雅跳过且退出码 0。证据：测试输出。
- **Notes**: 容差与现有 Q8/f32 注意力测试同级（1e-3 ~ 1e-5 量级，按算子定）。
- **Completion Evidence**:
  - TR-7.1: `cargo test` → `223 passed; 0 failed; 1 ignored`；`cargo test --bin cheetah ops::` 全部通过。
  - TR-7.2: `cargo test --features vulkan` → `254 passed; 0 failed; 4 ignored`；本机有 Vulkan 设备时全部 parity 测试真实执行并通过，无设备时 `device_available()` 分支打印跳过。

## Task 8: 不回归验证 + 文档同步
- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 7
- **Description**:
  - 运行 `cargo test` 与 `cargo test --features vulkan`，确认现有 7 个家族与 Ant 测试全绿。
  - 更新 `docs/CODEBASE.md` 模块索引（新增 `src/chat/ops/` 条目、op code 表）与 `docs/REGISTRY.md`“如何新增”章节（新增“4.x 新增一个算子”小节）。
  - 确认现有 forward 热路径未被改动（git diff 核对 `registrations/*.rs` 的 forward 函数体无逻辑变更，仅可能新增导入/演示）。
- **Acceptance Criteria Addressed**: AC-7, AC-8
- **Test Requirements**:
  - `rule` TR-8.1: `cargo test` 全绿；`cargo test --features vulkan` 全绿（设备门控除外）。证据：测试输出。
  - `rubric` TR-8.2: 设计/文档质量维度；scale 1-5；anchors 见 AC-8；threshold >= 4；证据：代码结构走查 + 文档 diff 核对。
- **Notes**: 若发现回归，回到对应任务修复，不得在本任务掩盖。
- **Completion Evidence**:
  - TR-8.1: `cargo test`（223 passed）与 `cargo test --features vulkan`（254 passed）全绿；`src/chat/registrations/` 的 forward 文件未改动（git diff 仅含 build.rs、shared_ops.comp、cheetah_vulkan.c/.h、native.rs、chat/mod.rs、vulkan/mod.rs 与新 ops 模块）。
  - TR-8.2: `docs/CODEBASE.md` 新增 `ops.rs`/`ops/`/`vulkan/ops.rs` 模块索引、op code 说明、`CHEETAH_MISH_OPS_SPV` 环境变量与不变量 #9；`docs/REGISTRY.md` 新增 §4.5 “A new operator (the xxx_ops catalog)”；版本号升至 1.3.0。
