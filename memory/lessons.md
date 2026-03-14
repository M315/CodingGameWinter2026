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
