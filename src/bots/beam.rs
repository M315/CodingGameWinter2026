use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::{Bot, GameState, Dir, Pos, DirArr, old_greedy_dirmap, gen_combos, dirmap_to_hashmap,
            greedy_actions, gen_action_combos};

pub struct BeamSearchBot {
    pub beam_width:   usize,
    pub horizon:      usize,
    pub time_limit:   Duration,
    pub heuristic_fn: fn(&GameState, u8) -> i32,
}

impl BeamSearchBot {
    pub fn new(
        beam_width: usize,
        horizon: usize,
        time_limit_ms: u64,
        heuristic_fn: fn(&GameState, u8) -> i32,
    ) -> Self {
        BeamSearchBot {
            beam_width,
            horizon,
            time_limit: Duration::from_millis(time_limit_ms),
            heuristic_fn,
        }
    }
}

/// V1 heuristic: score delta + gravity-aware food distance + stability penalty.
pub fn heuristic_v1(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;
    let w   = state.width as usize;

    // with_obstacles / with_snake_grid borrow separate TLS statics (OBS_SCRATCH,
    // SNG_SCRATCH); BFS inside borrows BFS_SCRATCH — three independent RefCells,
    // so nested borrows are safe.
    let obs = state.build_obstacles();
    let sng = state.snake_grid();

    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist_grounded(s.head(), &state.food, &obs);
            if d == i32::MAX { -50 } else { 20 - d.min(20) }
        })
        .sum();

    let stability = stability_score(state, player, &sng, w);
    my * 100 - opp * 80 + food_bonus + stability
}

// ── Shared stability helper ───────────────────────────────────────────────────

fn stability_score(state: &GameState, player: u8, sng: &[u8], w: usize) -> i32 {
    state.snakes.iter().enumerate()
        .filter(|(_, s)| s.player == player)
        .map(|(snake_idx, s)| {
            let grounded = s.body.iter().any(|&p| {
                let below_y = p.y + 1;
                if below_y >= state.height { return true; }
                if p.x < 0 || p.x >= state.width { return false; }
                let below_ci = below_y as usize * w + p.x as usize;
                if state.grid[below_ci] { return true; }
                if state.food[below_ci] { return true; }
                let sat = sng[below_ci];
                sat != u8::MAX && sat as usize != snake_idx
            });
            if grounded { 0 } else { -120 }
        })
        .sum()
}

/// V2 heuristic: score delta + competitive food assignment + stability.
///
/// Key improvements over v1:
///   1. Food competition — each food item is scored by whether we or the
///      opponent wins the race to it (my_dist vs opp_dist).
///   2. Friendly coordination — greedy assignment ensures no two of our
///      snakes "double-count" the same food.  The snake with a better
///      alternative elsewhere naturally yields the contested item because
///      its closer food is assigned first in the sorted pass.
pub fn heuristic_v2(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my_score  = state.score(player) as i32;
    let opp_score = state.score(1 - player) as i32;
    let score_delta = my_score * 100 - opp_score * 80;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();
    let w   = state.width as usize;

    let stability = stability_score(state, player, &sng, w);

    // Collect food cell indices once.
    let food_cis: Vec<usize> = state.food.iter().enumerate()
        .filter(|(_, &b)| b)
        .map(|(ci, _)| ci)
        .collect();

    if food_cis.is_empty() {
        return score_delta + stability;
    }

    let my_snakes: Vec<_> = state.snakes.iter().filter(|s| s.player == player).collect();
    let op_snakes: Vec<_> = state.snakes.iter().filter(|s| s.player != player).collect();
    let n_my   = my_snakes.len();
    let n_food = food_cis.len();

    // One BFS distance-map per snake (gravity-aware).  O((n_my+n_op) * w*h).
    let dist_my: Vec<Vec<i32>> = my_snakes.iter()
        .map(|s| state.bfs_dist_map_grounded(s.head(), &state.food, &obs))
        .collect();
    let dist_op: Vec<Vec<i32>> = op_snakes.iter()
        .map(|s| state.bfs_dist_map_grounded(s.head(), &state.food, &obs))
        .collect();

    // For each food: best (minimum) distance from any opponent snake.
    let best_op_dist: Vec<i32> = food_cis.iter().map(|&ci| {
        dist_op.iter()
            .map(|d| d[ci])
            .filter(|&d| d >= 0)
            .min()
            .unwrap_or(i32::MAX)
    }).collect();

    // Greedy food assignment — prevents two friendly snakes competing for the
    // same item.  Sort all (my_snake, food, dist) triples by distance so that
    // the closest reachable pair is assigned first.  A snake that has a closer
    // food elsewhere is assigned that one first and naturally yields the farther
    // contested item to its teammate.
    // Build (food_idx, snake_idx, dist) triples — plain loop avoids
    // nested-closure borrow issues with dist_my.
    let mut triples: Vec<(usize, usize, i32)> =
        Vec::with_capacity(n_food * n_my);
    for fi in 0..n_food {
        let ci = food_cis[fi];
        for si in 0..n_my {
            let d = dist_my[si][ci];
            if d >= 0 { triples.push((fi, si, d)); }
        }
    }
    triples.sort_unstable_by_key(|&(_, _, d)| d);

    let mut food_claimed  = vec![false; n_food];
    let mut snake_food: Vec<Option<(usize, i32)>> = vec![None; n_my]; // (food_idx, dist)
    for (fi, si, d) in triples {
        if food_claimed[fi] || snake_food[si].is_some() { continue; }
        food_claimed[fi] = true;
        snake_food[si]   = Some((fi, d));
    }

    // Score each of my snakes based on its assigned food and the competition.
    let food_score: i32 = (0..n_my).map(|si| {
        match snake_food[si] {
            // No reachable food for this snake — heavy penalty.
            None => -50,

            Some((fi, my_d)) => {
                let op_d      = best_op_dist[fi];
                let proximity = 20 - my_d.min(20); // 20 at dist=0, 0 at dist≥20

                if op_d == i32::MAX {
                    // Only we can reach this food — full proximity bonus.
                    proximity + 20
                } else if my_d < op_d {
                    // We win the race.
                    proximity + 10
                } else if my_d == op_d {
                    // Contested — proximity still useful but small risk penalty.
                    proximity - 5
                } else {
                    // Opponent wins — we'll arrive too late, wasted effort.
                    -15
                }
            }
        }
    }).sum();

    score_delta + food_score + stability
}

/// V3 heuristic: same logic as v2 but using multi-source BFS.
///
/// v2 ran N_my + N_op per-snake BFS calls (up to 6 on exotec).
/// v3 replaces them with exactly 2 multi-source calls — one per team.
/// This is faster than even v1 (which ran N_my early-exit calls),
/// freeing beam iterations to search deeper for the same 40ms budget.
///
/// Assignment approximation: the multi-source BFS records which of my
/// snakes is CLOSEST to each cell.  Greedy sort by distance then assigns
/// each food to that snake, skipping conflicts.  Ties in the BFS
/// initialisation order break in favour of the snake listed first, which
/// is an acceptable approximation — exact assignment needs per-snake BFS
/// (v2) but is too expensive at beam scale.
pub fn heuristic_v3(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let score_delta = state.score(player) as i32 * 100
                    - state.score(1 - player) as i32 * 80;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();
    let w   = state.width as usize;

    let stability = stability_score(state, player, &sng, w);

    let food_cis: Vec<usize> = state.food.iter().enumerate()
        .filter(|(_, &b)| b)
        .map(|(ci, _)| ci)
        .collect();

    if food_cis.is_empty() {
        return score_delta + stability;
    }

    // Collect head positions for each team.
    let my_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| s.head())
        .collect();
    let op_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player != player)
        .map(|s| s.head())
        .collect();

    let n_my = my_heads.len();

    // Two BFS calls total (vs N_my + N_op in v2, N_my in v1).
    let (my_dist, my_src) = state.bfs_multisource_dist_map(&my_heads, &state.food, &obs);
    let (op_dist, _)      = state.bfs_multisource_dist_map(&op_heads, &state.food, &obs);

    // Greedy assignment: sort foods by my team's distance, claim each food
    // for the snake that reaches it first (my_src).  A snake already claimed
    // for a closer food is skipped — it naturally yields this one.
    let mut snake_food: Vec<Option<(usize, i32)>> = vec![None; n_my];
    let mut food_assigned = vec![false; food_cis.len()];

    let mut sorted: Vec<(usize, usize, i32)> = // (food_idx, snake_idx, dist)
        food_cis.iter().enumerate()
            .filter_map(|(fi, &ci)| {
                let d = my_dist[ci];
                let s = my_src[ci];
                if d >= 0 && s != u8::MAX { Some((fi, s as usize, d)) } else { None }
            })
            .collect();
    sorted.sort_unstable_by_key(|&(_, _, d)| d);

    for (fi, si, d) in sorted {
        if food_assigned[fi] || snake_food[si].is_some() { continue; }
        food_assigned[fi]  = true;
        snake_food[si]     = Some((fi, d));
    }

    // Score each of my snakes on its assigned food + competitive standing.
    let food_score: i32 = (0..n_my).map(|si| {
        match snake_food[si] {
            None => -50,
            Some((fi, my_d)) => {
                let op_d = {
                    let d = op_dist[food_cis[fi]];
                    if d < 0 { i32::MAX } else { d }
                };
                let proximity = 20 - my_d.min(20);
                if      op_d == i32::MAX { proximity + 20 }   // uncontested
                else if my_d < op_d      { proximity + 10 }   // winning
                else if my_d == op_d     { proximity -  5 }   // tied
                else                     { -15             }   // losing
            }
        }
    }).sum();

    score_delta + food_score + stability
}

/// V4 heuristic: score delta + food proximity (v1) + Voronoi territory.
///
/// Territory is the key signal v1–v3 all lack: how much of the board can
/// each team reach?  Two multi-source BFS calls (one per team, same cost as
/// v3) partition every reachable cell by which team arrives first.  The
/// signed territory count (mine − opponent) is weighted modestly so that a
/// score lead still dominates, but positional squeeze or space dominance
/// shifts the beam toward safer lines.
///
/// Cost: 2 multi-source BFS + N_my early-exit BFS (food proximity).
///       ≈ same wall-clock as v3; territory replaces the (failed) competition
///       signal with a genuinely novel one.
pub fn heuristic_v4(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let score_delta = state.score(player) as i32 * 100
                    - state.score(1 - player) as i32 * 80;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();
    let w   = state.width as usize;

    let stability = stability_score(state, player, &sng, w);

    // Food proximity — identical to v1 (early-exit per snake, O(W*H) each).
    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist_grounded(s.head(), &state.food, &obs);
            if d == i32::MAX { -50 } else { 20 - d.min(20) }
        })
        .sum();

    // Territory — Voronoi partition via two multi-source BFS calls.
    // Each cell goes to whichever team arrives first; ties are neutral.
    let my_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| s.head())
        .collect();
    let op_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player != player)
        .map(|s| s.head())
        .collect();

    let (my_dist, _) = state.bfs_multisource_dist_map(&my_heads, &state.food, &obs);
    let (op_dist, _) = state.bfs_multisource_dist_map(&op_heads, &state.food, &obs);

    // Count cells: +1 for mine, -1 for opponent, 0 for tied/unreachable.
    let territory: i32 = my_dist.iter().zip(op_dist.iter())
        .map(|(&md, &od)| match (md, od) {
            (m, o) if m < 0 && o < 0 => 0,
            (m, _) if m < 0          => -1, // only opponent reaches it
            (_, o) if o < 0          =>  1, // only I reach it
            (m, o) if m < o          =>  1, // I arrive first
            (m, o) if o < m          => -1, // opponent arrives first
            _                        =>  0, // tied
        })
        .sum();

    score_delta + food_bonus + territory * 3 + stability
}

impl Bot for BeamSearchBot {
    fn name(&self) -> &str { "BeamSearchBot" }

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {
        // Turn 0 gets the full 1s initialisation budget; subsequent turns use time_limit.
        let limit = if state.turn == 0 {
            Duration::from_millis(950)
        } else {
            self.time_limit
        };
        let t0 = Instant::now();
        // DirArr = [Option<Dir>; 8]: Copy, 8 bytes, zero heap alloc.
        // Avoids ~27K HashMap allocs/turn on 3-snake maps.
        type BeamItem = (DirArr, GameState, i32);

        /// Merge two DirArrs (one per player) into a single array.
        /// Each snake ID is owned by exactly one player, so slots never conflict.
        #[inline]
        fn merge_dirs(a: &DirArr, b: &DirArr) -> DirArr {
            let mut out = *a;
            for i in 0..8 { if out[i].is_none() { out[i] = b[i]; } }
            out
        }

        let first_combos = gen_combos(state, player);
        if first_combos.is_empty() { return HashMap::new(); }

        let opp = old_greedy_dirmap(state, 1 - player);
        let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
            let mut ns = state.clone();
            ns.step_arr(&merge_dirs(&first, &opp));
            let score = (self.heuristic_fn)(&ns, player);
            (first, ns, score)
        }).collect();
        beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        beam.truncate(self.beam_width);

        // Always keep the depth-0 best action as fallback. If the inner time check fires
        // before expanding any state at a deeper level, we return this rather than an
        // empty HashMap (which would silently produce "WAIT" and ignore food/walls).
        let mut result: HashMap<u8, Dir> = dirmap_to_hashmap(player, &beam[0].0);

        for _depth in 1..self.horizon {
            if t0.elapsed() >= limit { break; }

            // mem::take so we own the vec and can break early without drain issues.
            // beam is already sorted best-first, so breaking early expands the most
            // promising states and discards only the lower-scoring tail.
            let cur_beam = std::mem::take(&mut beam);
            let mut next: Vec<BeamItem> = Vec::with_capacity(cur_beam.len() * 9);
            for (first_acts, cur, _) in cur_beam {
                if t0.elapsed() >= limit { break; }
                if cur.is_over() {
                    let score = (self.heuristic_fn)(&cur, player);
                    next.push((first_acts, cur, score));
                    continue;
                }
                let my_combos = gen_combos(&cur, player);
                let opp_acts  = old_greedy_dirmap(&cur, 1 - player);
                for combo in my_combos {
                    let mut ns = cur.clone();
                    ns.step_arr(&merge_dirs(&combo, &opp_acts));
                    let score = (self.heuristic_fn)(&ns, player);
                    next.push((first_acts, ns, score)); // first_acts is Copy
                }
            }
            // If nothing was expanded (timed out on very first state), keep result from
            // the previous depth and stop — beam is already empty from mem::take.
            if next.is_empty() { break; }
            next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            next.truncate(self.beam_width);
            result = dirmap_to_hashmap(player, &next[0].0);
            beam = next;
        }

        result
    }
}

// ── Benchmark control: same heuristic, old HashMap dispatch ──────────────────
//
// `BeamHashMapBot` is identical to `BeamSearchBot` except it uses the old
// `gen_action_combos` + `greedy_actions` + `step(&HashMap)` code path.
// Exists only for isolated benchmarking of DirArr vs HashMap overhead.
// NOT submitted to CG — benchmark comparison only.

pub struct BeamHashMapBot {
    pub beam_width:   usize,
    pub horizon:      usize,
    pub time_limit:   Duration,
    pub heuristic_fn: fn(&GameState, u8) -> i32,
}

impl BeamHashMapBot {
    pub fn new(
        beam_width: usize,
        horizon: usize,
        time_limit_ms: u64,
        heuristic_fn: fn(&GameState, u8) -> i32,
    ) -> Self {
        BeamHashMapBot { beam_width, horizon, time_limit: Duration::from_millis(time_limit_ms), heuristic_fn }
    }
}

impl Bot for BeamHashMapBot {
    fn name(&self) -> &str { "BeamHashMapBot" }

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {
        let limit = if state.turn == 0 {
            Duration::from_millis(950)
        } else {
            self.time_limit
        };
        let t0 = Instant::now();
        type BeamItem = (HashMap<u8, Dir>, GameState, i32);

        let first_combos = gen_action_combos(state, player);
        if first_combos.is_empty() { return HashMap::new(); }

        let opp = greedy_actions(state, 1 - player);
        let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
            let mut combined = first.clone();
            for (&k, &v) in &opp { combined.entry(k).or_insert(v); }
            let mut ns = state.clone();
            ns.step(&combined);
            let score = (self.heuristic_fn)(&ns, player);
            (first, ns, score)
        }).collect();
        beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        beam.truncate(self.beam_width);

        let mut result: HashMap<u8, Dir> = beam.first()
            .map(|(a, _, _)| a.clone())
            .unwrap_or_default();

        for _depth in 1..self.horizon {
            if t0.elapsed() >= limit { break; }
            let cur_beam = std::mem::take(&mut beam);
            let mut next: Vec<BeamItem> = Vec::with_capacity(cur_beam.len() * 9);
            for (first_acts, cur, _) in cur_beam {
                if t0.elapsed() >= limit { break; }
                if cur.is_over() {
                    let score = (self.heuristic_fn)(&cur, player);
                    next.push((first_acts, cur, score));
                    continue;
                }
                let my_combos = gen_action_combos(&cur, player);
                let opp_acts  = greedy_actions(&cur, 1 - player);
                for combo in my_combos {
                    let mut combined = combo;
                    for (&k, &v) in &opp_acts { combined.entry(k).or_insert(v); }
                    let mut ns = cur.clone();
                    ns.step(&combined);
                    let score = (self.heuristic_fn)(&ns, player);
                    next.push((first_acts.clone(), ns, score));
                }
            }
            if next.is_empty() { break; }
            next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            next.truncate(self.beam_width);
            result = next[0].0.clone();
            beam = next;
        }

        result
    }
}
