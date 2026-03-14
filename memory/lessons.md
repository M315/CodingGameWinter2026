# Lessons log

Each session should append an entry here.
Format: `## YYYY-MM-DD — <topic>`

---

## 2026-03-13 — Initial implementation

### What was built
- Full game engine in `src/lib.rs`: `GameState::step()` with all mechanics
  (movement, collision, eating, gravity, border removal)
- Three bots: WaitBot, GreedyBot, BeamSearchBot
- Local simulation harness (`src/bin/simulate.rs`) with ASCII vis + benchmark mode
- CG submission bundler (`scripts/bundle.py`)

### What worked
- **Beam search (width=120, horizon=8, 40ms limit)** wins 100% vs GreedyBot
  in 20-game benchmark on the default test map
- **Release mode timing**: 10–22ms per turn — comfortably under 50ms CG limit
- **`VecDeque<Pos>` for snake body**: O(1) push_front / pop_back is clean and fast
- **Standalone greedy_actions() function** inside beam search avoids borrow-checker
  issues with `Box<dyn Bot>` fields requiring `&mut self`
- **Simultaneous gravity** (snapshot → fall all unsupported → repeat) correctly
  handles cascading falls and stacking

### What didn't work / gotchas
- **noexec mount**: binaries in `/media/mjovani/Data/` can't be run directly.
  Must `install -m 755 target/release/binary /tmp/name && /tmp/name`.
  Discovered on first run attempt.
- **Borrow checker in bundled single file**: `self.snakes.retain(|s| { ... self.width ... })`
  compiles as a lib crate (Cargo manages edition + NLL) but fails with standalone `rustc`.
  Fix: extract field copies before the closure.
  Affected line: `step()` Phase 10 border removal.
- **Default test map**: games always run 200 turns (food never exhausted).
  Snakes end at 6 vs 3 parts. Some food items appear inaccessible — worth
  improving the test map or adding more snakes.

### Open questions / next steps
- [ ] Improve heuristic: currently penalises distance to nearest food but doesn't
      account for opponent blocking or food competition
- [ ] Minimax/adversarial search: beam search assumes greedy opponent which may
      be exploitable at higher league tiers
- [x] Test maps with multiple snakes per player → done via exotec_arena map
- [ ] Verify `step()` correctness against the official game replays once we have
      CG submission results
- [ ] Consider Alpha-Beta or MCTS for deeper search without exponential blowup

---

## 2026-03-14 — Gravity-aware heuristic + Exotec Arena map

### What was built
- **`maps/exotec_arena.txt`**: realistic 21×11 map reconstructed from gameplay
  screenshots. Left/right pillars (x=1-2 and x=17-18, y=4-9), two center shelves
  (x=5-7 and x=13-15, y=8-9). 3 snakes per player, 10 food items spread at
  multiple heights. Validated against known open coordinates from screenshot tooltips:
  (3,7), (4,8), (9,5), (16,9), (0,10), (2,10), (3,10).
- **Gravity-aware BFS** (`game.rs`): `is_grounded_cell()`, `bfs_dist_grounded()`,
  `bfs_first_step_grounded()` — intermediate cells must be supported from below
  (platform, power source, or bottom edge). Targets (food) are always reachable.
- **Updated heuristic** (`bots/mod.rs`):
  - `food_bonus` now uses `bfs_dist_grounded` — stops treating mid-air shortcuts
    as valid routes. Unreachable penalty raised from -30 → -50.
  - Added `stability` term: -120 per player snake that is currently unsupported
    (same logic as `apply_gravity`). Rewards grounded positions, punishes floating.
- **Updated `greedy_actions`**: switched to `bfs_first_step_grounded` so the
  opponent model also avoids planning through open air.

### What worked
- **Beam vs Greedy** (exotec arena, 10 games): 100% win rate ✓
- **Beam vs Beam** (exotec arena, 10 games): 50/50 — correct on symmetric map ✓
- **Beam vs Greedy** (default map, 20 games): 100% win rate maintained ✓
- **Bundle compiles**: `python scripts/bundle.py | rustc --edition 2021` → OK ✓
- **Greedy vs Greedy** (both maps): 100% draws — expected, symmetric+deterministic

### What didn't work / gotchas
- **Timing on 3-snake maps**: local simulation hits ~80-120ms/turn (3 snakes per
  player → 27 combos vs 3 with 1 snake). CG server runs 3-5× faster — screenshots
  confirm 8-28ms on CG for the same map. Not a real issue for submission.
- **Gravity-aware BFS is conservative**: it can't model the snake's OWN body
  providing support while the head extends into the air. A long snake can legally
  be in positions the BFS considers unreachable. Over-penalises some valid food
  paths. Acceptable approximation for now — beam search corrects via simulation.

### Open questions / next steps
- [ ] Beam width scaling for multi-snake games: 3 snakes → 27 combos, 4 snakes → 81.
      Consider capping combos per snake or reducing beam width dynamically.
- [ ] Snake-body-aware gravity BFS: let a snake's own body count as support for
      higher cells when planning (improves accuracy of food distance estimate).
- [ ] Food competition heuristic: penalise food that the opponent can reach faster.
- [ ] Opponent blocking: reward moves that cut off opponent paths to food.
- [ ] Gravity heuristic hurts on flat maps (see 2026-03-14 session note below).

---

## 2026-03-14 — Performance optimisation (flat arrays, eliminate hashing)

### What was profiled
- Callgrind (valgrind) on 1 game, beam vs greedy.
- 3.7 billion instructions total.

### Top bottlenecks found
1. **SipHash on `Pos` keys — 37% of all instructions**: every `HashSet<Pos>` /
   `HashMap<Pos, *>` operation in BFS, `apply_gravity`, and `heuristic` was
   hashing a 32-bit pair using a cryptographic hash. Completely unnecessary.
2. **`apply_gravity` — 8%**: rebuilt two `HashSet<Pos>` (occupied + own) inside
   the fixpoint loop on every call. Called after every `step()` in beam search
   (~2880 times/turn at width=120, horizon=8).
3. **Heap allocations — 9%**: direct consequence of #1+#2. malloc/free dominated
   by HashSet/HashMap construction in hot paths.
4. **`build_obstacles` called twice**: once inside `heuristic()` AND once inside
   `greedy_actions()` for the same state — now eliminated by passing precomputed
   grids.

### What was changed (`game.rs`)
- **All 4 BFS functions**: signatures changed from `&HashSet<Pos>` to `&[bool]`
  (flat grid). Internals: `HashMap<Pos, Dir/i32>` → `Vec<u8>/Vec<i32>` visited
  array; `VecDeque<Pos>` → `VecDeque<usize>` (cell index queue). Zero hashing.
- **`build_obstacles()`**: returns `Vec<bool>` instead of `HashSet<Pos>`.
- **Added `power_grid()`**: flat `Vec<bool>` for power source positions.
- **Added `snake_grid()`**: flat `Vec<u8>` mapping cell → snake index.
- **`apply_gravity()`**: pre-allocates `snake_at: Vec<u8>` outside the fixpoint
  loop; `.fill(u8::MAX)` clears it each iteration. Zero HashSet allocations.
- **`step()`**: `eaters: HashSet<usize>` → `[bool; 32]`; `body_cells: HashSet<Pos>`
  → `Vec<bool>` flat grid; `head_map: HashMap<Pos, Vec<usize>>` → O(n²) loop
  (n ≤ 8, so O(64) max — cheaper than any HashMap).
- **`is_grounded_cell_ci()`**: new private helper using flat index arithmetic,
  used by grounded BFS variants.

### What was changed (`bots/`)
- `greedy_actions()`: calls `power_grid()` once, passes `&pow` to BFS.
- `heuristic()`: calls `build_obstacles()` + `power_grid()` + `snake_grid()`
  once each. Stability check uses flat `sng` grid — no `HashSet` for `occupied`
  or `own`.
- `BeamSearchBot` / `OldBeamSearchBot`: turn 0 now uses 950 ms budget (full CG
  initialisation window) instead of 40 ms.

### Benchmark results (default map, beam vs greedy, 20 games)
| Version | Wall time | Win rate |
|---|---|---|
| Before optimisation | 142 600 ms | 100% |
| After optimisation  |  10 500 ms | 100% |
| **Speedup** | **13.5×** | — |

### New vs old beam (heuristic comparison)
| Map | beam P0 vs old_beam P1 (20g) | beam P0 vs old_beam P1 (exotec, 10g) |
|---|---|---|
| Default (flat) | old_beam 100% | — |
| Exotec Arena   | — | beam 60%, old_beam 20%, draws 20% |

Gravity heuristic helps on the platform map but hurts on the flat map because
`bfs_dist_grounded` marks elevated food as unreachable when the only path goes
through air — but the snake body itself provides support (known limitation).
Next fix: replace grounded BFS with a body-aware version, or fall back to plain
BFS when all food is near the floor.

### Remaining allocation hotspots (not yet fixed)
- BFS visited arrays (`Vec<u8>`, `Vec<i32>`) still allocated per BFS call.
  Could be eliminated with thread-local scratch buffers.
- `HashMap<u8, Dir>` for action combos in beam search (minor — not in top-10).
- `GameState::clone()` inside beam search — the dominant remaining cost.
