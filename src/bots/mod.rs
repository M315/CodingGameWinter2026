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
    let obs = state.build_obstacles(); // Vec<bool>
    let pow = state.power_grid();      // Vec<bool>
    let mut actions = HashMap::new();
    for s in state.snakes.iter().filter(|s| s.player == player) {
        if let Some(d) = state.bfs_first_step_grounded(s.head(), &pow, &obs) {
            actions.insert(s.id, d);
        }
    }
    actions
}

/// All valid direction combos for `player`'s snakes (U-turns pruned).
pub fn gen_action_combos(state: &GameState, player: u8) -> Vec<HashMap<u8, Dir>> {
    let ids: Vec<u8> = state.snakes.iter()
        .filter(|s| s.player == player).map(|s| s.id).collect();

    if ids.is_empty() { return vec![HashMap::new()]; }

    let mut combos: Vec<HashMap<u8, Dir>> = vec![HashMap::new()];
    for id in ids {
        let s = state.snakes.iter().find(|s| s.id == id).unwrap();
        let dirs: Vec<Dir> = Dir::all().iter()
            .filter(|&&d| d != s.dir.opposite()).copied().collect();
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
