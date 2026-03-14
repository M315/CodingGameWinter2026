use std::collections::HashMap;
// Re-export game types so sub-modules can use `use super::*;`
pub use crate::game::*;

pub mod wait;
pub mod greedy;
pub mod beam;

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
/// Standalone function so BeamSearchBot can call it without trait overhead.
pub fn greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {
    let obs = state.build_obstacles();
    let mut actions = HashMap::new();
    for s in state.snakes.iter().filter(|s| s.player == player) {
        if let Some(d) = state.bfs_first_step(s.head(), &state.power, &obs) {
            actions.insert(s.id, d);
        }
    }
    actions
}

/// Heuristic score for `player` in `state`. Higher = better for player.
pub fn heuristic(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    let obs = state.build_obstacles();
    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist(s.head(), &state.power, &obs);
            if d == i32::MAX { -30 } else { 20 - d.min(20) }
        })
        .sum();

    my * 100 - opp * 80 + food_bonus
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
