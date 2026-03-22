use std::collections::HashMap;
// Re-export game types so sub-modules can use `use super::*;`
pub use crate::game::*;

pub mod wait;
pub mod greedy;
pub mod beam;
pub mod old_beam;
pub mod mcts;

// ============================================================
// Bot trait
// ============================================================

pub trait Bot {
    fn name(&self) -> &str;
    /// Return a map of snake_id → Dir for snakes to redirect this turn.
    /// Snakes absent from the map keep their current direction.
    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir>;
}

// ============================================================
// Shared utilities (used by multiple bots)
// ============================================================

/// Greedy action selection: BFS each snake toward the nearest power source.
/// Uses gravity-aware BFS so paths through open air are not considered.
/// Standalone function so BeamSearchBot can call it without trait overhead.
pub fn greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {
    // with_obstacles (OBS_SCRATCH) and bfs_first_step_grounded (BFS_SCRATCH)
    // borrow independent TLS RefCells — no nesting conflict.
    let obs = state.build_obstacles();
    // state.food IS the power grid — pass directly, no allocation
    state.snakes.iter()
        .filter(|s| s.player == player)
        .filter_map(|s| {
            state.bfs_first_step_grounded(s.head(), &state.food, &obs)
                .map(|d| (s.id, d))
        })
        .collect()
}

/// Convert a `DirArr` back to `HashMap<u8, Dir>` for the output of `choose_actions`.
/// Only called once per turn (not in the hot loop).
pub fn dirmap_to_hashmap(player: u8, arr: &DirArr) -> HashMap<u8, Dir> {
    // We only need the moves for *our* snakes — the caller doesn't apply opponent dirs.
    // But since we stored the full first combo in BeamItem, just emit everything set.
    let _ = player; // reserved for future filtering
    arr.iter().enumerate()
        .filter_map(|(id, &opt)| opt.map(|d| (id as u8, d)))
        .collect()
}

/// Zero-alloc action generators for the beam search hot path.
///
/// `DirArr` is `Copy` (8 bytes, stack-only) so combos need no `.clone()`.
/// `gen_combos` / `greedy_dirmap` replace `gen_action_combos` / `greedy_actions`
/// inside the beam inner loop to eliminate ~27K HashMap allocs/turn on 3-snake maps.

/// Fast variant of `greedy_actions` returning a stack-allocated `DirArr`.
/// Uses gravity-aware BFS — good for our own snakes (we plan grounded moves).
pub fn greedy_dirmap(state: &GameState, player: u8) -> DirArr {
    let obs = state.build_obstacles();
    let mut result: DirArr = [None; 8];
    state.snakes.iter()
        .filter(|s| s.player == player)
        .for_each(|s| {
            if let Some(d) = state.bfs_first_step_grounded(s.head(), &state.food, &obs) {
                result[s.id as usize] = Some(d);
            }
        });
    result
}

/// Plain-BFS variant of `greedy_dirmap` — for opponent modelling.
/// Matches `old_greedy_actions`: uses `bfs_first_step` (air-aware) + TLS obs buffer.
/// Opponents CAN move through air and fall, so plain BFS is a more accurate model.
pub fn old_greedy_dirmap(state: &GameState, player: u8) -> DirArr {
    state.with_obstacles(|obs| {
        let mut result: DirArr = [None; 8];
        state.snakes.iter()
            .filter(|s| s.player == player)
            .for_each(|s| {
                if let Some(d) = state.bfs_first_step(s.head(), &state.food, obs) {
                    result[s.id as usize] = Some(d);
                }
            });
        result
    })
}

/// Fast direction map using precomputed food distance cache.
///
/// O(4 × N_snakes) per call vs O(grid_size × N_snakes) for `old_greedy_dirmap`.
/// For each snake, picks the accessible non-body neighbour with the minimum
/// `cached_food_dist` value.  Requires `state.cache_food_dist()` to have been
/// called this turn (done at top of `BeamSearchBot::choose_actions`).
pub fn greedy_dirmap_fast(state: &GameState, player: u8) -> DirArr {
    let w = state.width as usize;
    state.with_obstacles(|obs| {
        let mut result: DirArr = [None; 8];
        state.snakes.iter()
            .filter(|s| s.player == player)
            .for_each(|s| {
                let head = s.head();
                let best = Dir::all().iter()
                    .filter_map(|&dir| {
                        let (dx, dy) = dir.delta();
                        let (nx, ny) = (head.x + dx, head.y + dy);
                        if nx < 0 || ny < 0 || nx >= state.width || ny >= state.height {
                            return None;
                        }
                        let ni = ny as usize * w + nx as usize;
                        if state.grid[ni] || obs[ni] { return None; }
                        let d = state.cached_food_dist(Pos::new(nx as i32, ny as i32));
                        // d == -1 means food enclosed by walls on all sides — treat as far
                        Some((dir, if d == -1 { i32::MAX } else { d }))
                    })
                    .min_by_key(|&(_, d)| d)
                    .map(|(dir, _)| dir);
                result[s.id as usize] = best;
            });
        result
    })
}

/// Fast variant of `gen_action_combos` returning `Vec<DirArr>` (no HashMap allocs).
pub fn gen_combos(state: &GameState, player: u8) -> Vec<DirArr> {
    let ids: Vec<u8> = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| s.id)
        .collect();

    if ids.is_empty() { return vec![[None; 8]]; }

    let mut combos: Vec<DirArr> = vec![[None; 8]];
    for id in ids {
        let s = state.snakes.iter().find(|s| s.id == id).unwrap();
        let neck = (s.len() >= 2).then(|| s.body.get(1).unwrap());
        let dirs: Vec<Dir> = Dir::all().iter()
            .filter(|&&d| {
                let (dx, dy) = d.delta();
                Some(s.head().translate(dx, dy)) != neck
            })
            .copied()
            .collect();
        let mut next = Vec::with_capacity(combos.len() * dirs.len());
        for &existing in &combos {
            for &d in &dirs {
                let mut c = existing; // Copy — no clone
                c[id as usize] = Some(d);
                next.push(c);
            }
        }
        combos = next;
    }
    combos
}

/// All valid direction combos for `player`'s snakes (U-turns pruned).
///
/// U-turn check uses the neck position (body[1]) rather than s.dir.opposite()
/// — s.dir can become stale after a head-destruction+pop in a previous step,
/// causing the wrong direction to be filtered.
pub fn gen_action_combos(state: &GameState, player: u8) -> Vec<HashMap<u8, Dir>> {
    let ids: Vec<u8> = state.snakes.iter()
        .filter(|s| s.player == player).map(|s| s.id).collect();

    if ids.is_empty() { return vec![HashMap::new()]; }

    let mut combos: Vec<HashMap<u8, Dir>> = vec![HashMap::new()];
    for id in ids {
        let s = state.snakes.iter().find(|s| s.id == id).unwrap();
        // Neck position: the cell the snake cannot re-enter this turn.
        // Only defined for snakes with 2+ segments; 1-segment snakes can go anywhere.
        let neck = (s.len() >= 2).then(|| s.body.get(1).unwrap());
        let dirs: Vec<Dir> = Dir::all().iter()
            .filter(|&&d| {
                let (dx, dy) = d.delta();
                Some(s.head().translate(dx, dy)) != neck
            })
            .copied()
            .collect();
        let mut next = Vec::with_capacity(combos.len() * dirs.len());
        for existing in &combos {
            for &d in &dirs {
                let mut c = existing.clone();
                c.insert(id, d);
                next.push(c);
            }
        }
        combos = next;
    }
    combos
}
