# SNAKEBYTE — Claude Instructions

## Project
CodinGame Winter 2026 — SNAKEBYTE bot in Rust.
Competition URL: https://www.codingame.com/ide/challenge/winter-challenge-2026-exotec

## My responsibilities
- Improve the bot's AI (heuristics, search depth, pruning)
- Fix simulation bugs when discovered
- Run benchmarks before and after changes to confirm improvement
- Update `memory/lessons.md` at the end of every session with what worked / what didn't

---

## Critical rules

### Running binaries
The project is on an NTFS-mounted partition (`/media/mjovani/Data/`).
Binaries built there **cannot be executed directly** (noexec mount).
**Always copy to `/tmp` first:**
```bash
install -m 755 target/release/simulate /tmp/sim && /tmp/sim [args]
```

### CG submission is a single file
`src/main.rs` uses `use snakebyte::*;` — it is NOT standalone.
To produce the submittable file run:
```bash
python scripts/bundle.py --out submission.rs
rustc --edition 2021 submission.rs -o /tmp/sub_test   # verify it compiles
```
Always verify the bundle compiles before calling it done.

### Build / test workflow
```bash
cargo build --release
install -m 755 target/release/simulate /tmp/sim
/tmp/sim -- --bench 20 --quiet          # 20-game benchmark, ~70s
/tmp/sim                                # 1 game with ASCII visualization
```

### Never skip benchmarks
Any change to `lib.rs` that affects `step()`, `heuristic()`, or a bot
must be validated with at least `--bench 10` before and after.

---

## Architecture

### Files
| File | Role |
|------|------|
| `src/lib.rs` | **Everything**: types, simulation, BFS, all bots |
| `src/main.rs` | CG IO only — reads stdin, runs bot, prints actions |
| `src/bin/simulate.rs` | Local harness — bots vs bots, ASCII vis, benchmarks |
| `scripts/bundle.py` | Concatenates lib + main into single `submission.rs` |

### Key types
```
Pos { x, y: i32 }
Dir { Up, Down, Left, Right }  →  delta(), opposite(), from/to_str
Snake { id: u8, body: VecDeque<Pos>, dir: Dir, player: u8 }
GameState { width, height, grid: Vec<bool>, power: HashSet<Pos>,
            snakes: Vec<Snake>, turn: u32 }
```

### Bots
| Name | Location | Notes |
|------|----------|-------|
| `WaitBot` | `src/lib.rs` | Baseline / sanity |
| `GreedyBot` | `src/lib.rs` | BFS to nearest food; used as opponent sim inside beam search |
| `BeamSearchBot` | `src/lib.rs` | Current main bot: width=120, horizon=8, 40ms limit |

### `GameState::step()` — 11 phases
1. Apply direction overrides from actions map
2. Compute proposed new head positions
3. Identify eaters (head lands on power source)
4. Build body obstacle set — tail excluded for non-eaters (it vacates), included for eaters (it stays)
5. Detect destroyed heads: OOB / platform / body collision / head-on-head (non-food)
6. Apply movement: push_front new head; pop_back tail unless eating
7. Remove consumed power sources
8. Drop fully-destroyed snakes via `retain`
9. `apply_gravity()` — simultaneous-fall fixpoint loop
10. Border removal — `retain` snakes with all parts in bounds
11. Increment `self.turn`

### Gravity detail
A snake can fall if **none** of its parts has support below it.
Support = bottom edge, platform, power source, or another snake's body part (not own body).
Runs as a loop: rebuild snapshot each iteration, fall all unsupported snakes, repeat until stable.

---

## Known borrow checker gotcha
Closures inside `retain` that reference `self.field` will fail to compile
in standalone `rustc` (no Cargo.toml edition) because the closure captures
`self` while `retain` holds `&mut self.vec`.

**Fix:** extract the fields before the closure:
```rust
let (w, h) = (self.width, self.height);
self.snakes.retain(|s| s.body.iter().all(|&p| p.in_bounds(w, h)));
```

---

## Rust performance guidelines

### Allocations (biggest bottleneck in hot loops)
- Minimize clones inside beam search — the inner loop clones `GameState` thousands of times per turn
- Prefer `Vec::retain` + reuse over allocate-new-vec patterns
- `smallvec` / `arrayvec` for collections that are almost always small (e.g. snake list ≤8 items)
- `sort_unstable()` over `sort()` — stable sort is rarely needed here

### Style: prefer iterators over for loops
Prefer iterator chains (`.iter()`, `.map()`, `.filter()`, `.for_each()`, `.sum()`, etc.)
over indexed `for` loops whenever the logic is equivalent. Reasons:
- Eliminates bounds checks that indexed loops can retain
- Easier for LLVM to auto-vectorise
- More idiomatic, composable Rust
Only fall back to an indexed `for` when you genuinely need the index for
non-iterator logic (e.g. `O(n²)` pairwise loops, or in-place mutation that
requires two indices simultaneously).

### General
- `#[inline]` on hot path methods (`Pos::in_bounds`, `Dir::delta`, `GameState::is_platform`)
- Iterative over recursive — no risk of stack overflow, easier to bound
- Complexity awareness: label O(?) on any non-trivial loop before writing it

### Heuristics priority (prefer earlier items first)
1. Greedy (currently: BFS to nearest food)
2. Beam search (current main bot)
3. Simulated annealing
4. Genetic / evolutionary

### I/O
Use `stdin().lock()` + `.lines()` — already in `src/main.rs`. Do **not** switch to BufReader unless profiling shows I/O is a bottleneck (it won't be — the game sends <50 lines per turn).

---

## Testing protocol

### Unit tests in `src/lib.rs`
```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_eating_grows_snake() { /* ... */ }
}
```
Run with `cargo test`.

### Correctness verification strategy
1. Build a minimal `GameState` by hand for the mechanic being tested
2. Call `state.step(&actions)` once
3. Assert exact snake positions, lengths, power source count
4. Cover: eating, head destruction (len<3 / len>=3), head-on-head, gravity cascade, border removal

### Stress testing
- Use `--bench N` in `simulate` to catch regressions over many games
- Before any PR/submission: run `--bench 20 --quiet` and confirm win rate doesn't drop

### Profiling
```bash
cargo flamegraph --bin simulate -- --bench 5 --quiet
# flamegraph.svg shows where time is spent (likely GameState::clone or apply_gravity)
```

---

## Lessons log
See `memory/lessons.md` for a running record of what worked and what didn't.
**Update this file at the end of every session.**

## Project architecture doc
See `/home/mjovani/.claude/projects/-media-mjovani-Data-Documents-coding-game/memory/project_snakebyte.md`
for the full memory-system entry.
