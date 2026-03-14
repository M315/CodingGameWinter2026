use std::collections::HashMap;
// Re-export game types so sub-modules can use `use super::*;`
pub use crate::game::*;

pub mod wait;
pub mod greedy;
pub mod beam;
pub mod old_beam;

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
        let neck = (s.len() >= 2).then(|| s.body[1]);
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
