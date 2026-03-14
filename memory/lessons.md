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
- [ ] Test maps with multiple snakes per player (the game allows 1–4 per player)
- [ ] Verify `step()` correctness against the official game replays once we have
      CG submission results
- [ ] Consider Alpha-Beta or MCTS for deeper search without exponential blowup
