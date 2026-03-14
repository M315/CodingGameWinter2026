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
/// Uses gravity-aware BFS so paths through open air are not considered.
/// Standalone function so BeamSearchBot can call it without trait overhead.
pub fn greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {
    let obs = state.build_obstacles();
    let mut actions = HashMap::new();
    for s in state.snakes.iter().filter(|s| s.player == player) {
        if let Some(d) = state.bfs_first_step_grounded(s.head(), &state.power, &obs) {
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

    // Gravity-aware food distance: only count paths along grounded cells.
    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist_grounded(s.head(), &state.power, &obs);
            if d == i32::MAX { -50 } else { 20 - d.min(20) }
        })
        .sum();

    // Stability: penalise snakes that are currently unsupported (will fall next turn).
    // Uses the same grounding logic as apply_gravity().
    let occupied: std::collections::HashSet<_> = state.snakes.iter()
        .flat_map(|s| s.body.iter().copied()).collect();
    let stability: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let own: std::collections::HashSet<_> = s.body.iter().copied().collect();
            let grounded = s.body.iter().any(|&p| {
                let below = p.translate(0, 1);
                p.y + 1 >= state.height
                    || state.is_platform(below)
                    || state.power.contains(&below)
                    || (occupied.contains(&below) && !own.contains(&below))
            });
            if grounded { 0 } else { -120 }
        })
        .sum();

    my * 100 - opp * 80 + food_bonus + stability
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
