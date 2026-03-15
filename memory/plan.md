# SNAKEBYTE — Improvement Plan

**Direction**: beam search + heuristic refinement. MCTS shelved (see lessons.md 2026-03-15).

Each item has a benchmark gate: run `--bench 10` before + after and confirm win rate
doesn't regress. Exotec arena is the primary test map (3 snakes/player, platforms).

---

## Tier 1 — Highest ROI

### ✅ T1-A: Food competition heuristic — DONE (heuristic-v2, 2026-03-15)
**What**: penalise food items that the opponent reaches faster than us.
For each food cell: compute `my_dist` and `opp_dist` via BFS.
If `opp_dist < my_dist`: apply a penalty scaled by the advantage.
If `my_dist < opp_dist`: small bonus (we "own" that food).
**Why**: current heuristic completely ignores food competition. On contested maps
the bot chases food it will never reach while ceding food it could win.
**Effort**: low — pure heuristic change in `bots/beam.rs::heuristic()`.
**Benchmark gate**: beam+food_competition vs beam (exotec, 20g) — expect improvement.

### T1-B: Reversible step() / undo-redo
**What**: replace `state.clone()` in the beam inner loop with an undo stack.
Each `step()` records a delta (moved heads, eaten food, fallen snakes) that can be
reversed with `unstep()`. The beam search applies step+evaluate+unstep instead of
clone+step+evaluate+drop.
**Why**: `GameState::clone()` is still the dominant cost in the beam inner loop
(~2.2× speedup already seen from Arc+flat-food in a previous session).
Eliminating it entirely would unlock 3–5× more beam iterations per turn.
**Effort**: high — step() has 11 phases, several are hard to reverse (gravity cascade,
border removal). Consider incremental approach: undo for simple cases, clone only
when gravity fires.
**Benchmark gate**: beam_undo vs beam (default + exotec, 20g each) — same win rate,
faster wall time (measure ms/game).

---

## Tier 2 — Good gains, moderate effort

### T2-A: Thread-local BFS scratch buffers
**What**: `bfs_dist_grounded` and `bfs_first_step_grounded` each allocate a
`Vec<u8>` or `Vec<i32>` of size `width*height` on every call. Move these to
`thread_local!` scratch buffers (reset with `.fill()` instead of re-allocating).
**Why**: heuristic calls BFS 2× per snake per evaluation; beam inner loop evaluates
thousands of states per turn. Allocation dominates in the BFS hot path.
**Effort**: medium — requires refactoring BFS signatures to accept scratch ref.
**Benchmark gate**: measure ms/game improvement, win rate unchanged.

### T2-B: Territory / flood-fill liberty count
**What**: BFS flood-fill from each snake's head (treating opponent snakes as walls).
Count reachable cells. Reward more open territory, penalise getting cornered.
A snake with 3 reachable cells is in serious danger; one with 50 is safe.
**Why**: current heuristic has no notion of "trapped" or "cornered". Beam search
sometimes chooses food-optimal paths that lead into dead ends.
**Effort**: medium — new BFS variant; tune weight against existing terms.
**Benchmark gate**: beam+territory vs beam (exotec, 20g).

### T2-C: Smarter combo pruning (action policy)
**What**: before expanding all 27 combos in the beam inner loop, rank them by a
cheap pre-heuristic (e.g. greedy BFS direction) and only expand the top K (e.g. K=9).
This halves the branching factor, effectively doubling beam depth.
**Why**: with 3 snakes, 27 combos per node means beam width=120 is spread thin.
Cutting to 9 keeps the best combos and allows 3× more depth in the same budget.
**Effort**: low-medium — add a pre-scoring step before combo generation.
**Benchmark gate**: beam_pruned(K=9) vs beam (exotec, 20g).

---

## Tier 3 — Worth doing for higher leagues

### T3-A: Better opponent model in beam
**What**: instead of greedy opponent, simulate the opponent with a shallow beam
(width=10, depth=2) to anticipate non-greedy responses.
**Why**: at higher CG leagues, opponents will not play greedily. The current model
can be exploited by a smarter adversary.
**Effort**: high — nested search, time budget management gets complex.

### T3-B: Subtree reuse between turns
**What**: after committing action A on turn N, find the beam branch that
matches the actual game outcome on turn N+1 and resume from it instead of
starting fresh. Turn 0's 950ms budget compounds over 200 turns.
**Effort**: medium-high — need to match state identity (snake positions + food).

### T3-C: Gravity heuristic body-awareness
**What**: the current `bfs_dist_grounded` treats mid-air cells as unreachable,
but a snake's own body provides support. Fix: allow cells above the snake's
current body segments as passable in grounded BFS.
**Why**: known limitation from 2026-03-14. Over-penalises some valid food paths
on platform maps (seen in flat-map regression where old_beam beat new beam).

---

## Execution order

1. T1-A (food competition) — pure heuristic, fast to validate
2. T2-A (BFS scratch buffers) — low risk perf win, unlocks more beam iterations
3. T2-C (combo pruning) — deeper search, high impact on multi-snake maps
4. T1-B (reversible step) — hardest, save for when easier wins are exhausted
5. T2-B (territory) — tune alongside T1-A improvements
6. T3-* — league-dependent, revisit after seeing CG results

---

## Benchmark baseline (2026-03-15, beam vs greedy, --time-limit 40)

| Map | Result | Wall time (20g) |
|---|---|---|
| Default (flat) | Beam 100% | ~8 300ms |
| Exotec arena | Beam 100% vs greedy; 50/50 vs beam | ~? |

Run `--bench 20 --quiet` to refresh before starting each item.
