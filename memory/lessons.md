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

---

## 2026-03-14 — OOB panic fix + style preference

### Bug fixed
- **`apply_gravity` OOB panic** (seen in battle.log: `len=264, index=266`):
  In the grounding check, `below_y >= self.height` guarded the y-axis but
  `p.x` was never checked against `[0, self.width)`. Added early return `false`
  when `p.x < 0 || p.x >= self.width` before computing `below_ci`.

### Style preference recorded
- User prefers **iterator chains** over `for` loops for equivalent logic.
  Reasons: bounds-check elimination, better LLVM vectorisation hints, more
  idiomatic Rust. Exception: indexed `for` is fine when the index is genuinely
  needed (e.g., O(n²) pairwise loops). Documented in `CLAUDE.md`.

### Next optimization investigation
Three directions identified for further speedup:
1. **Minimize `GameState::clone()`** — dominant remaining cost; see open questions.
2. **Turn-0 precomputation + subtree reuse** — carry beam tree across turns.
3. **Transposition table** — cache heuristic scores for repeated board states.

### Open questions / next steps
- [ ] `Arc<grid>` — platforms never change; sharing the grid allocation across
      beam clones would cut clone size proportionally.
- [ ] Reversible `step()` / undo-redo — eliminates clones in inner loop entirely
      (most impactful but complex).
- [ ] Subtree reuse: store beam tree between turns, on turn N+1 find the branch
      that matches actual game outcome and resume from it.
- [ ] Transposition table: Zobrist hash of `GameState` → cached (depth, score).

---

## 2026-03-15 — Clone cost reduction + beam correctness fixes

### What was built
- **`Arc<Vec<bool>>` for grid** — platforms never change after construction;
  all beam-search clones share the allocation via refcount bump. Zero memcpy,
  zero alloc for the grid on every clone.
- **`food: Vec<bool>` + `food_count: u32`** — replaced `HashSet<Pos>` power set.
  Clone is memcpy(~264B) instead of rehash+alloc. Direct index lookup throughout.
  `add_food()` / `clear_food()` replace `insert` / `clear`. `food_count` keeps
  `is_over()` O(1).
- **`&state.food` in hot paths** — eliminated `power_grid()` allocation in
  `heuristic()`, `old_heuristic()`, `greedy_actions()`, and `apply_gravity()`.
  ~1000+ Vec allocations per turn gone from beam inner loop.
- **Inner-loop time check** — was only checked at horizon boundaries; one depth
  expansion (50 states × 27 combos = 1350 ops) could blow 10ms budget entirely.
  Added `if t0.elapsed() >= limit { break; }` inside the per-state loop. Safe
  because beam is sorted best-first — breaking discards only the lowest-scoring tail.

### Benchmark results
| Version | Wall time (20g, beam vs greedy, --time-limit 10) |
|---|---|
| Before clone refactor | 18 380ms |
| After Arc+flat food   |  8 308ms (2.2× speedup) |

### Bugs fixed
1. **`gen_action_combos` U-turn filter** — was filtering `s.dir.opposite()` but
   `s.dir` goes stale after a head-destruction+pop. A snake like `[(4,5),(4,6)]`
   with `s.dir=Right` (from before the pop) would allow `Down` → `(4,6)` = neck.
   Fix: check proposed cell against `body[1]` (the actual neck) instead.

2. **`mem::take` empty-action bug** — when the inner time check fired before
   expanding any state, `next` was empty → `beam = []` → `unwrap_or_default()`
   returned `{}` → `"WAIT"` → snake continued into walls / past food.
   Fix: initialise `result` from depth-0 best action; update each completed depth;
   `if next.is_empty() { break; }` preserves the previous depth's result.

3. **`cap` limit on combos per state removed** — was capping combos to
   `beam_width` per parent, which could silently drop food-eating combos when
   combo count > beam_width. Removed: beam truncation after scoring is the
   correct place to enforce width.

### Lessons
- **`mem::take` in loop bodies is dangerous** — if you take a vec and the loop
  breaks early before producing output, you've destroyed your previous state with
  nothing to show for it. Always save a fallback before taking, or guard with
  `if next.is_empty() { break; }`.
- **`s.dir` is not reliable for U-turn detection** after head-destruction — always
  use `body[1]` (neck position) as the source of truth.
- **Time budget must be enforced inside inner loops**, not just at outer depth
  boundaries, when each individual expansion can exceed the total budget.
- **`power_grid()`** was an allocation hotspot inside the beam inner loop — now
  that `food` IS the flat grid, callers pass `&state.food` directly.

### Open questions / next steps
- [ ] Reversible `step()` / undo-redo — eliminates clones in inner loop; hardest
      remaining optimization, highest ceiling.
- [ ] Subtree reuse: carry beam tree between turns, resume from matching branch.
- [ ] Transposition table: Zobrist hash → cached (depth, score).
- [ ] Improve heuristic for flat maps: gravity-aware BFS is too conservative when
      snake body itself provides support (known limitation from 2026-03-14).

---

## 2026-03-15 — MCTS exploration + strategy decision

### What was investigated
- Explored MCTS as an alternative to beam search.
- Reference repo (tczajka/wazir-drop) turned out to be alpha-beta + NNUE (not MCTS),
  but provided useful time-management patterns (hard/soft deadlines, iterative deepening).
- Implemented a full MCTS bot (`src/bots/mcts.rs`) with:
  - Arena-based flat Vec<MctsNode> tree (no per-node heap allocation)
  - DUCT (Decoupled UCT) for simultaneous multi-player moves
  - Greedy rollout × depth-6 + heuristic leaf evaluation (sigmoid-normalized)
  - Turn-0 extended budget (950ms), 50k node cap, time-check every 128 iters

### Benchmark results
| Matchup | Result |
|---|---|
| MCTS vs Greedy (default map, 10g) | MCTS 100% |
| MCTS vs Beam (default map, 10g) | 100% draws (both optimal on symmetric map) |
| MCTS vs Beam (exotec arena, 10g) | Beam 100% |

### Why MCTS underperforms on multi-snake maps
- With 3 snakes per player: 27 my_combos × 27 op_combos = 729 joint actions/turn.
- With ~4000 nodes/turn budget, tree barely reaches depth 1 (729 pairs at depth 1).
- `nodes == iters` always: every iteration creates 1 new node, UCB never revisits.
- Beam search reaches depth 8 with width 120 — far more signal per time budget.

### Why MCTS+NN is not worth pursuing here
- Good hand-crafted heuristic already exists → MCTS rollout substitution not needed.
- Game is deterministic → no averaging benefit over beam.
- NN needs self-play training infra + weeks of iteration — overkill for contest.
- Modifications needed (greedy opponent, progressive widening, heuristic leaf, subtree
  reuse) converge toward reinventing beam search with UCB selection.

### Decision
Shelve MCTS. Focus on beam search with better heuristics and performance.
See `memory/plan.md` for the prioritized improvement roadmap.

---

## 2026-03-15 — heuristic_v2: food competition + friendly assignment

### What was built
- `bfs_dist_map_grounded()` in `game.rs`: full-grid BFS returning `Vec<i32>`;
  one call per snake covers all food distances, no repeated per-food BFS calls.
- `heuristic_v2()` in `bots/beam.rs`:
  - Greedy food assignment (sort triples by dist, claim greedily):
    each food goes to the closest friendly snake; a snake with a nearer
    alternative elsewhere yields the contested item automatically.
  - Competitive scoring per food: +20 uncontested, +10 winning race,
    -5 tied, -15 losing, -50 no food reachable.
  - `stability_score()` extracted as a shared helper.
- Heuristic versioning protocol in place: `beam` = latest, `beam_vN` = pinned alias.
  Git tags `heuristic-v1` and `heuristic-v2` mark each version.

### Benchmark results
| Map | v2 vs v1 (20g, time-limit 40) |
|---|---|
| Default (flat, 1 snake/player) | 20 draws — no regression |
| Exotec arena (3 snakes/player) | **v2 60% / v1 30% / draws 10%** |

### Lessons
- **Food competition is the biggest blind spot in v1**: the bot was chasing food the
  opponent would reach first, wasting turns. Competitive scoring fixed this.
- **Greedy assignment handles friendly coordination as a free by-product**: sorting
  (snake, food, dist) triples means a snake with a closer alternative naturally
  yields contested food — no special tie-break logic required.
- **`flat_map` with nested closures fights the borrow checker**: `move` in the inner
  closure tries to move `dist_my` out of the outer closure. Fix: plain `for` loops.
- **No regression on flat maps**: v2 draws v1 on the symmetric map — the new terms
  don't hurt when there's no multi-snake competition.

### Next steps (from plan.md)
- [ ] T2-A: thread-local BFS scratch buffers (eliminate Vec alloc per BFS call)
- [ ] T2-C: combo pruning (expand top-K combos per node, not all 27)
- [ ] T2-B: territory / flood-fill liberty count

---

## 2026-03-15 — heuristic_v2/v3 food competition — null result, reverted to v1

### What was tried
- **v2**: competitive food scoring + greedy friendly assignment using per-snake
  `bfs_dist_map_grounded` calls (N_my + N_op full BFS maps per evaluation).
- **v3**: same logic but replaced per-snake BFS with 2 `bfs_multisource_dist_map`
  calls — one per team. Intended to be faster than even v1.

### Benchmark results (exotec arena, 50 games, --time-limit 40)
| Matchup | P0 wins | P1 wins | Draws |
|---|---|---|---|
| v2 (P0) vs v1 (P1) | 36% | 54% | 10% |
| v3 (P0) vs v1 (P1) | 46% | 46% |  8% |
| v1 (P0) vs v1 (P1) | 45% | 35% | 20% (baseline) |

v2 was a clear regression (6 slow BFS calls per evaluation → fewer beam iterations).
v3 recovered performance but still lost: P0 advantage is ~10pp so 46% as P0 = ~9pp below v1.
At 200ms (generous budget): v3 loses 0/20, v1 wins 20/20 — confirms v3 slightly worse.

### Root cause (key lesson)
**Beam search already captures food competition implicitly.** The greedy-opponent
simulation inside beam search steps the opponent toward food each turn; the
`score_delta` at the leaf already reflects who ends up with more food. Adding
an explicit "who wins the race" term at the leaf **double-counts** information
the search already has. At shallow depth (5ms) it helps marginally; at beam's
normal depth (8 steps / 40ms) it adds noise.

### When competition signal WOULD help
Only at depths beyond beam's horizon (>8 turns ahead) where the leaf can't yet
"see" who wins a specific food race. That requires either deeper search or a
learned value function trained on outcomes.

### What was learned about the benchmarking process
- **20 games is not enough** on the exotec arena: v2 showed 60/30 at 20g but
  reversed to 36/54 at 50g. Always run ≥50 games before drawing conclusions.
- **Test at multiple time limits**: the 5ms / 40ms / 200ms trifecta diagnoses
  whether a regression is a signal problem or a performance problem cleanly.
  - 5ms equal + 200ms regression → performance is the bottleneck
  - 5ms equal + 40ms equal + 200ms regression → signal misleads at depth
- **Account for P0/P1 map bias** when interpreting results: this map has ~10pp
  P0 advantage even with identical bots. Always run a v1 vs v1 baseline.
- **Update lessons BEFORE committing**, not after.

### Plan update
T1-A food competition approach shelved. Moving to:
- T2-A: BFS scratch buffers (perf, safe win regardless of signal quality).
- Shelve further leaf-heuristic experiments until perf baseline is improved.

### Lessons
- **MCTS suits domains where eval function is hard**: Go, Hex, stochastic games.
  When a good heuristic exists + game is deterministic, beam/AB dominates.
- **Branching factor kills MCTS**: 27^3 = 19k nodes at depth 3. With 4k budget
  you never get past depth 1–2 on multi-snake maps.
- **DUCT is theoretically correct for simultaneous games** but practically unusable
  here due to (my_combos × op_combos) explosion. Greedy opponent assumption
  (same as beam) is the right practical choice.
- **Arena-based MCTS implementation is clean in Rust**: flat Vec<Node> with usize
  indices avoids all borrow-checker issues with tree traversal.

---

## 2026-03-15 — heuristic_v4 territory/Voronoi — null result; simulation improvements

### heuristic_v4 (territory Voronoi) — failed
- 2 multi-source BFS calls per evaluation (same overhead as v3), territory = Voronoi
  cell count (mine − opponent) × 3, added on top of v1 food proximity + stability.
- 5ms: v4 40% vs v1 36% (mildly better at shallow depth)
- 40ms: v4 26% vs v1 48% (performance overhead causes regression)
- 200ms: v4 26% vs v1 50% (even at generous budget, territory adds noise)
- Root cause: double failure — (1) performance overhead from extra BFS, (2) territory
  at depth-8 leaf doesn't correlate with outcome (opponent adapts, score_delta already
  captures food race implicitly). This is worse than v2/v3 which only failed on perf.

### Simulation improvements shipped
1. **External map files**: Maps moved to `maps/*.txt` (sorted by name, rotated in bench).
   Add/edit maps without touching Rust. Format: same `#/.`, `P x y`, `S player id coords`.
2. **Map pool**: 01_default, 02_small_multi, 03_wide, 04_tall_multi, 05_exotec_arena.
3. **Timing root cause**: exotec_arena has 3 snakes/player → 3³=27 combos → 9× beam
   cost vs default (3 combos). ~10s/game vs ~1s/game. Maps 02/04 (2/player, 9 combos)
   take ~2-4s/game.
4. **Bounded vs open maps**: Maps 01-04 have full border walls (snakes can't exit).
   05_exotec_arena has NO border walls (open top/sides — snakes can exit by moving OOB).
   Gravity never causes map exit (bottom edge = support even without wall).

### Benchmark workflow
- Quick iteration (maps 01-04): use `--map maps/01_default.txt` or add a `maps/quick/` subdir
- Competition-accurate (all 5 maps): default `--bench 50` takes ~100s (10 exotec games)
- Use `--map maps/05_exotec_arena.txt` for single-map exotec testing

### Next priority
T2-A: BFS scratch buffers — eliminate Vec<i32>/Vec<bool> alloc per BFS call.
Each heuristic_v1 call does N_my early-exit BFS, each allocating W×H Vec.
Thread-local reusable buffers + generation counter would eliminate these allocs
entirely. Expected 20-30% speed gain → more beam iterations → better decisions.

---

## 2026-03-15 — Open-border map redesign

### What was done
Rewrote maps 01–04 to match real competition map design (open borders, void gaps).
Previously all four maps had full border walls — snakes couldn't fall off the edge.

### New design rules (applied to all maps)
- **No lateral/top walls**: edges are open `'.'`; snake heads that cross the boundary
  are destroyed by border removal in `step()`.
- **Ground platform with edge gaps**: ground spans `x=left_gap..width-right_gap`
  (typically 2-cell gap left + right). Snakes near the edge fall to the bottom row
  and can then walk off-map in one move.
- **Interior stepping-stone platforms**: small `###` clusters at varying heights for
  multi-level routing and gravity traps.
- **Snake starting columns verified**: none of the start x positions coincide with
  any interior platform cell (body would start inside `'#'`).

### Map summary
| Map | Size | Snakes/player | Ground y | Void gap x | Snakes land |
|-----|------|--------------|----------|------------|-------------|
| 01_default     | 20×15 | 1 | 11-12 | 0,1 and 18,19 | y=8,9,10 |
| 02_small_multi | 16×12 | 2 |  9-10 | 0,1 and 14,15 | y=6,7,8  |
| 03_wide        | 30×15 | 1 | 11-12 | 0-2 and 27-29 | y=8,9,10 |
| 04_tall_multi  | 22×18 | 2 | 15-16 | 0,1 and 20,21 | y=12,13,14 |

### Lessons
- **Gravity grounding is "bottom edge OR platform below"**: snakes never
  gravity-fall off the map — `below_y >= height` is always "grounded". Exit
  requires an explicit horizontal move into an OOB cell.
- **Platform overlap check**: for a snake body starting at y=1,2,3, check every
  cell `(x, y)` in the initial body against the grid. Also trace the full fall path:
  falling body cells pass through platform ROWS but must not land in platform CELLS.
- **Parallel bench confirmed**: `std::thread::scope` + `&[String]` (Copy rebind)
  gives ~3× wall-time speedup on 20-game bench with 4 threads.

---

## 2026-03-15 — T2-A/B/C: BFS scratch buffers + DirArr combo optimization

### T2-A: Thread-local BFS scratch buffers — shipped

**Problem**: every BFS call allocates 2 Vecs (dist: Vec<i32>, visited: Vec<u8>) + a
VecDeque queue — ~3 allocs per call, ~3000 BFS calls/turn at beam width=120.

**Solution**: `thread_local! { static BFS_SCRATCH: RefCell<BfsScratch> }` where
`BfsScratch` holds reusable `i32_buf`, `u8_buf`, and `queue`. After `borrow_mut()`,
field-borrow splitting works: `let dist = &mut sc.i32_buf[..size]; let queue = &mut sc.queue;`
Both fields are accessible simultaneously through the single `&mut BfsScratch` reference.

**Separate RefCells for separate statics**: `OBS_SCRATCH`, `SNG_SCRATCH`, `BFS_SCRATCH`
are three independent `RefCell` statics, so nested borrows from different contexts
never conflict.

**Benchmark**: all-maps bench, 20 games, --time-limit 10, 4 threads
| Before T2-A | After T2-A |
|-------------|------------|
| 5193ms      | 3327ms (1.56× speedup) |

### T2-B: Closure-based build_obstacles / snake_grid — regression, reverted

**Problem**: wanted to expose `with_obstacles(|obs| …)` closure API so callers could
borrow TLS scratch without returning a Vec.

**Result**: regression (3327ms → 4256ms). Root causes:
1. Closure + `borrow_mut()` overhead outweighs malloc savings for 231-300 byte buffers.
2. Making `build_obstacles()` delegate to `with_obstacles(|obs| obs.to_vec())` is WORSE
   than direct calloc: TLS borrow + fill + malloc + memcpy > calloc + fill.

**Lesson**: TLS scratch helps only when the caller owns the buffer across multiple
operations (like BFS which uses it for the entire BFS traversal). A thin wrapper that
immediately converts back to Vec has negative ROI.

### T2-C: DirArr combo generation — shipped

**Problem**: `gen_action_combos` returned `Vec<HashMap<u8,Dir>>`, allocating a new
HashMap per combo (~27K allocs/turn on 3-snake maps).

**Solution**:
- `type DirArr = [Option<Dir>; 8]` — 8 bytes, `Copy`, zero heap.
- `Option<Dir>` = 1 byte via niche (Dir has 4 variants, 252 spare patterns for None).
- `gen_combos()` returns `Vec<DirArr>` — `let mut c = existing;` (Copy, no clone).
- `greedy_dirmap()` returns `DirArr` instead of `HashMap<u8,Dir>`.
- `dirmap_to_hashmap()` converts at output time only (once per turn).

**Double iteration bug** (initial version): called `apply_dirs()` then `ns.step(&empty)`.
`apply_dirs` set snake dirs; `step()` Phase 1 also iterated all snakes with `HashMap::get`
(no-op with empty map but still one full snake scan). Result: 4238ms vs 3327ms T2-A.

**Fix**: added `step_arr(&DirArr)` to `GameState` that handles Phase 1 inline with a
direct array index, then calls private `step_phases_2_to_11()` shared with `step()`.
`step()` calls `step_phases_2_to_11()` after its HashMap Phase 1. In `beam.rs`, two
DirArrs (mine + opponent) are merged via:
```rust
fn merge_dirs(a: &DirArr, b: &DirArr) -> DirArr {
    let mut out = *a;
    for i in 0..8 { if out[i].is_none() { out[i] = b[i]; } }
    out
}
```
then `ns.step_arr(&merged)` — single pass, no HashMap, no double iteration.

**Benchmark** (20 games, all maps, --time-limit 10, 4 threads):
| T2-A baseline | T2-C (broken, double iter) | T2-C (step_arr fixed) |
|---------------|---------------------------|----------------------|
| 3327ms        | 4238ms (regression)       | 3595ms (~≈ T2-A) |

Net result: T2-C's main benefit was eliminating HashMap allocs in combo generation;
the step_arr fix recovers the double-iteration overhead but doesn't provide additional
speedup over T2-A because the per-call cost of one extra O(n) snake scan is negligible
vs the actual step body. The DirArr path is cleaner and avoids heap allocs — correct
to keep.

### Maps/quick directory created
`maps/quick/` contains symlinks to 01–04 only. Use `--maps-dir maps/quick/` for fast
iteration without the ~10s/game exotec map.

### Next priority: see microbench session below.

---

## 2026-03-15 — Microbench: precise per-component cost breakdown

### What was built
`src/bin/microbench.rs` — single-threaded, fixed-state, fixed-depth benchmark.
Measures each component in isolation on a real map position.
Avoids game outcome noise, positional bias, and thread contention.
Usage: `/tmp/mb maps/05_exotec_arena.txt`

### Key design choices
- `std::hint::black_box` prevents dead-code elimination
- Warmup of 10% before timing
- Fixed-depth beam (no time limit): measures work done, not budget allowed
- Clone isolated separately to split "clone cost" from "step logic cost"
- `BeamHashMapBot` added as benchmark control (same heuristic, old HashMap path)

### Confounding lesson: old_beam vs beam comparison was wrong
The initial "beam vs old_beam" game benchmark was confounded: beam uses `heuristic_v1`
(gravity-aware BFS) while old_beam uses `old_heuristic` (plain BFS). old_heuristic wins
on these maps because gravity-aware BFS is too conservative (marks food as unreachable
when snake's own body would provide support). The structural difference (DirArr vs HashMap)
was drowned by the heuristic quality difference. Always isolate one variable at a time.

### Per-call measurements (exotec arena, 3 snakes/player, 27 combos)

| Component | Old (HashMap) | New (DirArr) | Speedup |
|---|---|---|---|
| `gen_combos` / `gen_action_combos` | 2.56 µs | 0.40 µs | **6.4×** |
| `greedy_dirmap` / `greedy_actions` | 0.34 µs | 0.28 µs | 1.2× |
| `state.clone()` (isolated) | — | 0.20 µs | baseline |
| `step` (includes clone) | 1.36 µs | 1.27 µs | 1.07× |
| `heuristic_v1` | — | 0.34 µs | baseline |

### Per-node cost breakdown (beam inner loop, DirArr path)
One "node" = clone + step_arr + heuristic_v1 + amortised combo/greedy overhead.

| Sub-cost | µs | % of node |
|---|---|---|
| step phases 2-11 (pure logic) | 1.07 | 67% |
| heuristic_v1 | 0.34 | 21% |
| state.clone() | 0.20 | 12% |
| gen_combos (÷ 27 combos) | ~0.015 | <1% |
| greedy_dirmap (÷ 27 combos) | ~0.010 | <1% |

**Correction to prior estimate**: clone was previously estimated as the "dominant remaining
cost". Microbench shows it's only 12% of per-node time. The real dominant cost is
step phases 2-11 (67%), primarily `apply_gravity` + `body_cells` allocation inside step.

### Fixed-depth beam speedup (DirArr vs HashMap, exotec)
| Depth | DirArr (ms) | HashMap (ms) | Speedup |
|---|---|---|---|
| 1 | <1 | <1 | — |
| 2 | 1 | 2 | 2× |
| 3 | 9 | 9 | 1.0× |
| 4 | 19 | 20 | 1.05× |
| 5 | 31 | 28 | ~1× |

DirArr's advantage concentrates in the first 1-2 depths (combo gen cost dominates there).
By depth 3+ the beam is truncated to 120 nodes and the per-node clone+step+heuristic
cost dominates — where DirArr provides only 1.07×.

### Why the game benchmark showed 70% win rate at 5ms
At 5ms budget, depth 1-2 is where most of the DirArr advantage lives. A bot that can
consistently reach depth 2 reliably (DirArr) beats one that is often stuck at depth 1
(HashMap). At 40ms both bots reach full depth 5-8, equalising.

### Next targets (priority order)

**T3-A — Stack arrays in step_phases_2_to_11** (quick win, low risk)
`proposed: Vec<Pos>` → `[Pos; 8]`, `head_destroyed: vec![false; n]` → `[bool; 8]`,
`should_remove: vec![false; n]` → `[bool; 8]`. Eliminates 3 small heap allocs per step.
Expected saving: ~5-10% of step cost.

**T3-B — TLS scratch for body_cells in step** (medium effort)
`body_cells: vec![false; grid_size]` inside step_phases_2_to_11 allocates W×H = 231 bytes
every step call. Make it TLS with a separate `BODY_SCRATCH` RefCell. Since step doesn't
call BFS, there's no nesting conflict. Expected saving: ~10-15% of step cost.

**T3-C — TLS scratch for snake_at in apply_gravity** (same effort as T3-B)
`snake_at: Vec<u8>` in apply_gravity is allocated fresh per call. If gravity runs 2+ fixpoint
iterations it's refilled (`.fill(u8::MAX)`) but the initial alloc is still wasted. TLS
eliminates the alloc; `.fill` keeps the reset. Expected saving: ~5% of step cost.

**Q1 — Body-aware BFS in heuristic_v1** (quality improvement)
Current `bfs_dist_grounded` marks cells as unreachable if the path passes through air, even
when the snake's own body segments provide support. This makes elevated food appear
unreachable. Fix: include the snake's own body as a support source in the grounding
check. Risk: changes heuristic behaviour — benchmark before/after.

**T4 — Reversible step / undo-redo** (complex, 12% gain)
Now that microbench shows clone is only 12% of node cost, the ROI is lower than
originally estimated. Reversible step would save 0.20 µs per node = ~14% speedup.
Complexity is high (undo stack for food changes, snake body deques, destroyed snakes).
Deprioritised relative to T3-A/B/C which target the 67% step-logic cost more directly.

---

## 2026-03-15 — T3 optimizations + heuristic rollback + old_heuristic TLS

### T3-A: Stack arrays in step_phases_2_to_11
Replaced `proposed: Vec<Pos>`, `head_destroyed: vec![false; n]`, `should_remove: vec![false; n]`
with fixed-size stack arrays `[Pos; 8]`, `[bool; 8]`, `[bool; 8]`. Eliminates 3 small heap
allocs per step call. Quick win, minimal risk.

### T3-B: TLS scratch for body_cells in step
Added `STEP_SCRATCH: RefCell<Vec<bool>>` — reuses W×H obstacle grid across step calls.
Avoids allocating 231 bytes every call. No nesting conflict: step doesn't call BFS.

### T3-C: TLS scratch for snake_at in apply_gravity
Added `GRAV_SCRATCH: RefCell<Vec<u8>>` — reuses W×H snake-index grid across gravity fixpoint
iterations. Stack `to_fall: [usize; 8]` + explicit `to_fall_len` counter replaces the
per-iteration `Vec<usize>` alloc.

### Heuristic rollback: old_heuristic as main
`heuristic_v1` uses `bfs_dist_grounded` which is too conservative — marks food as unreachable
when path traverses air cells, even when the snake's own body provides support. On all test
maps beam with `heuristic_v1` lost 97% of games to old_beam. Rolled `beam` alias back to
`old_heuristic` (plain BFS, -30 unreachable penalty). old_beam and beam now use the same
heuristic; old_beam is the HashMap-structure control.

### Position bias lesson
On exotec map, P1 wins 67–77% regardless of bot quality. Always run both P0→P1 AND P1→P0
and average when comparing bots; or use microbench to avoid game-level confounders entirely.

### old_heuristic: build_obstacles → with_obstacles (T3-D)
`old_heuristic` called `build_obstacles()` which allocates a new `Vec<bool>` per call.
Replaced with `with_obstacles(|obs| ...)` (TLS reuse, same `OBS_SCRATCH` already used by
`bfs_first_step_grounded`). Result: 0.65 µs → 0.50 µs (23% faster). `bfs_dist` takes
`&[bool]` so the signature is compatible with zero change to the BFS logic. No nesting
conflict since `bfs_dist` reads obs but doesn't call `with_obstacles` again.

### Updated microbench numbers (exotec, after all T3 changes)

| Component | µs/call | Notes |
|---|---|---|
| `old_heuristic` | 0.50 | current main; was 0.65 before T3-D |
| `heuristic_v1` | 0.36 | gravity-aware BFS; broken on these maps |
| `gen_combos` (DirArr) | 0.41 | was 2.56 µs (HashMap) — 6.1× gain |
| `greedy_dirmap` (DirArr) | 0.29 | vs 0.37 µs (HashMap) |
| `state.clone()` | 0.24 | |
| `step_arr` (DirArr+clone) | 1.07 | ~0.83 µs pure step logic |
| `step` (HashMap+clone) | 1.06 | near parity after T3 alloc removal |

### Per-node cost (DirArr beam, after T3)

| Sub-cost | µs | % |
|---|---|---|
| step phases 2-11 (pure logic) | ~0.83 | 53% |
| old_heuristic | 0.50 | 32% |
| state.clone() | 0.24 | 15% |
| combo/greedy overhead | <0.02 | <1% |

**old_heuristic is now the dominant per-node cost (32%), overtaking step (53% combined).**
Optimizing the heuristic further would have more impact than further step micro-tuning.

### greedy_dirmap revert
Changed `greedy_dirmap` to use plain `bfs_first_step` (not grounded) to match old greedy
behaviour — this made it 0.54 µs (worse!). Reverted to `bfs_first_step_grounded`: 0.29 µs.
Lesson: grounded BFS is faster because it prunes more cells, not because it's simpler.

### Next priorities (updated)
1. **Improve old_heuristic quality**: still uses plain BFS which ignores gravity constraints.
   A gravity-aware variant that correctly uses the snake's own body as support would be more
   accurate without the `bfs_dist_grounded` over-pruning bug. Biggest quality lever.
2. **Heuristic: multi-food BFS**: compute distance to ALL food in a single BFS pass per snake
   (BFS from head, collect all food hits) instead of BFS-to-nearest. Better for multi-food maps.
3. **Beam width/horizon tuning**: step is now 27% faster than before T2/T3 changes. The 40ms
   budget can now explore wider or deeper. Retest optimal width=120, horizon=8 settings.
4. **T4 reversible step**: low ROI given 15% clone cost — defer unless other gains plateau.

---

## 2026-03-16 — Multi-food BFS (reverse BFS) — null result / regression

### What was tried
Added `with_food_dist_map(obs, |food_dist| …)` to `GameState` in `game.rs`:
a single reverse multi-source BFS seeded from all food cells simultaneously.
The goal: replace N_my per-snake forward early-exit BFS calls in `old_heuristic`
with one reverse BFS + O(1) head lookups, saving N_my-1 BFS initialisations.

`with_food_dist_map` is implemented using BFS_SCRATCH (TLS, zero-alloc), seeded at
every food cell (dist=0), expanding outward. Snake-head cells are in obs so
`food_dist[head_ci] == -1`; heuristic recovers via `min(food_dist[neighbor]+1)`.

### Benchmark result
| Component | Before | After |
|---|---|---|
| `old_heuristic` | 0.50 µs | 7.33 µs (14× slower) |

Reverted `old_heuristic` immediately. `with_food_dist_map` kept in `game.rs`
(may be useful for future experiments).

### Root cause — why the reverse BFS is slower
Early-exit forward BFS terminates when it hits the first food cell. On the exotec
map with 10 food cells on a 231-cell grid, the average nearest-food distance is
3–8 cells, so each forward BFS visits only ~10–50 cells before stopping.
Three per-snake calls = ~30–150 cell visits total.

The reverse BFS ALWAYS visits the entire reachable grid (~200 cells) regardless
of food proximity. It only beats forward BFS when:
  1. N_my is very large (saves N_my-1 fill+init overheads), AND
  2. Food is far from all snakes (forward BFS would also visit many cells)
Neither condition holds during normal play where snakes are actively hunting food.

### Lesson
**Reverse BFS is NOT a drop-in replacement for per-snake early-exit BFS.**
On small maps with food close to snakes, early-exit dominates by a large margin.
The break-even is roughly N_my ≥ 8 AND avg food distance ≥ W×H / N_food.
For this game (N_my ≤ 3, food dense), never use reverse BFS in the main heuristic.

`with_food_dist_map` may still be useful for:
- A post-search territory or Voronoi term that needs a full distance map
- One-off diagnostics (where call count is low)
Never call it inside the beam inner loop.

### Next step candidates (unchanged)
- Body-aware gravity BFS — lets us restore gravity-aware heuristic on platform maps
- Beam width/horizon retune — step is 27% faster than T2/T3 baseline
- Combo pruning (top-K actions) — effective branching factor reduction
- Territory/flood-fill liberty count — detect trapped-snake scenarios

---

## 2026-03-16 — Combo pruning (COMBO_CAP=9) — clear improvement

### What was built
Added `rank_and_prune_combos` in `beam.rs`, called in the inner loop (depth ≥ 1)
after `gen_combos` whenever `my_combos.len() > COMBO_CAP`.

**Scoring per combo:**
- +100 per snake whose proposed head lands on a food cell (always retained)
- +2 per snake whose direction matches `old_greedy_dirmap` preference
Sorted descending, truncated to `COMBO_CAP = 9`.

**When pruning fires:** only for 3+ snakes per player (3^3=27 > 9). For 1-2 snake
players the combos are already ≤ 9 so the guard short-circuits with no overhead.

**Overhead when pruning fires:** one extra `old_greedy_dirmap` call per node
(~0.33 µs, same cost as the opponent call we already make). The scoring and
sort are O(27 × 8) operations — negligible.

**Search depth gained (3 snakes, 40ms budget):**
- Before: 120 nodes × 27 combos × ~1.5 µs ≈ 4.9ms/depth → ~8 depths
- After:  120 nodes × 9 combos × ~1.5 µs ≈ 1.6ms/depth → ~24 depths

### Benchmark results (exotec arena, 50 games per orientation)
| Match-up | P0 wins | P1 wins | Draws |
|---|---|---|---|
| beam vs beam (baseline, both pruned) | 30% | 60% | 10% |
| beam (P0) vs old_beam (P1) | **72%** | 22% | 6% |
| old_beam (P0) vs beam (P1) | 20% | **60%** | 20% |

Map has a P1 positional advantage (~60% for P1 regardless of bot quality).
Beam as P0 overcomes the disadvantage with +42pp vs baseline (30% → 72%).
Beam as P1 holds the position's natural rate (60%), old_beam couldn't break it.

### Lessons
- **Combo pruning is the biggest quality lever found so far**: 3× more search depth
  for free, no signal loss because food-eating combos are always preserved.
- **Greedy-alignment ranking is surprisingly effective**: the +100 food bonus keeps
  critical combos; the +2 greedy alignment sorts the rest. Only 27 combos and a
  tiny sort — negligible cost.
- **Pruning ONLY fires at depth ≥ 1**: depth-0 still evaluates all 27 combos for
  the initial beam population. This preserves first-move diversity while maximising
  depth at all subsequent steps.
- **P1 map bias can mask quality improvements**: the exotec arena gives P1 ~60% win
  rate for equal bots, so beam as P1 shows "0pp improvement" even though it's
  clearly better (the P0 test reveals the truth). Always run both orientations.

### Next priorities
- Beam width/horizon retune: with 3× more depth, optimal (width, horizon) may shift
- Body-aware gravity BFS: restores gravity signal in the heuristic
- Territory/flood-fill: detect trapped-snake scenarios

