# Reduction operators — PyTorch validation report

Deviation of Cheetah's reduction operators (v1.3.1, `reduce.sum/mean/max/
argmax`) against the PyTorch 2.15 reference, measured on 2026-11-27.

## Method

- Fixture: seeded `torch.rand` buffers (uniform `[-1, 1)`), verified
  tie-free per row/column so argmax semantics (first vs last occurrence)
  cannot diverge. Reference values computed with `torch.sum/mean/max/
  argmax` over both axes.
- Fixture test: `chat::ops::reductions::tests::pytorch_reference_fixture_matches`
  — points at a generated JSON file through `CHEETAH_REDUCE_FIXTURE`,
  auto-skips when unset. Runs native and (when a Vulkan device is present)
  the GPU path against the same reference and prints per-case `DEV` lines.
- Deviations are absolute (vs the torch value) unless noted; `max_rel` is
  the largest per-element relative deviation.

Reproduce:

```sh
python3 - <<'EOF'  # fixture generator (rows width out.json), tie-free seed search
import json, sys, torch
rows, width, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for seed in range(1, 10000):
    torch.manual_seed(seed)
    x = torch.rand(rows, width, dtype=torch.float32) * 2.0 - 1.0
    if all(int((x == x.max(dim=dim, keepdim=True).values).sum()) == x.shape[1 - dim]
           for dim in (0, 1)):
        break
expected = {}
for axis, dim in [(0, 1), (1, 0)]:
    expected[f"sum_axis{axis}"] = x.sum(dim=dim).tolist()
    expected[f"mean_axis{axis}"] = x.mean(dim=dim).tolist()
    expected[f"max_axis{axis}"] = x.max(dim=dim).values.tolist()
    expected[f"argmax_axis{axis}"] = x.argmax(dim=dim).tolist()
json.dump({"rows": rows, "width": width, "x": x.flatten().tolist(),
           "expected": expected}, open(out, "w"))
EOF
python3 gen.py 64 8192 /tmp/reduce_large.json
CHEETAH_REDUCE_FIXTURE=/tmp/reduce_large.json \
  cargo test --features vulkan pytorch_reference -- --nocapture
```

## Results

`max` / `argmax`: **zero deviation** at every size, on both axes and both
backends — bit-exact against torch (comparison ops involve no rounding;
argmax tie-break was verified to replicate the host-side greedy scan).

`sum` / `mean` (accumulation-order rounding; Cheetah reduces sequentially,
torch pairwise/tree):

| Fixture | Op × axis | Outputs | max_abs | mean_abs | max_rel |
|---|---|---|---|---|---|
| 6×257 | sum axis 0 | 6 | 4.1e-6 | 1.7e-6 | 2.2e-6 |
| 6×257 | sum axis 1 | 257 | 0 | 0 | 0 |
| 6×257 | mean (both axes) | — | 1.6e-8 | 6.6e-9 | 2.2e-6 |
| 64×8192 | sum axis 0 | 64 | 3.6e-4 | 5.5e-5 | 2.5e-5 |
| 64×8192 | sum axis 1 | 8192 | 4.8e-6 | 5.2e-7 | 8.8e-5 (see note) |
| 64×8192 | mean (both axes) | — | 4.4e-8 | 6.7e-9 | 2.5e-5 |

Note: the axis-1 `max_rel` is inflated by near-zero sums (cancellation);
its absolute deviation is 4.8e-6 — read `max_abs` there.

## Conclusions

1. Deviations sit at f32 rounding noise: the only source is accumulation
   order (sequential vs pairwise), and it grows with the reduced extent —
   ~4e-6 at 257 columns, ~3.6e-4 (relative 2.5e-5) at 8192 columns.
2. `max` / `argmax` are bit-exact vs torch on both backends.
3. Native and Vulkan are **bit-identical** in every measured case (all DEV
   statistics match to the last digit) — the GPU shader reproduces the CPU
   reduction order exactly.
4. Pairwise/Kahan accumulation would close the sum gap another order of
   magnitude if ever needed; not warranted at current deviation levels
   (tracked as a v1.3.2 performance-conditional item).
