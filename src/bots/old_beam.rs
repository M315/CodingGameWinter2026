use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::{Bot, GameState, Dir, gen_action_combos};

/// Beam search bot using the old (pre-gravity-aware) heuristic and greedy opponent model.
/// Used only for benchmarking against the current BeamSearchBot.
pub struct OldBeamSearchBot {
    pub beam_width: usize,
    pub horizon:    usize,
    pub time_limit: Duration,
}

impl OldBeamSearchBot {
    pub fn new(beam_width: usize, horizon: usize, time_limit_ms: u64) -> Self {
        OldBeamSearchBot { beam_width, horizon, time_limit: Duration::from_millis(time_limit_ms) }
    }
}

/// Old greedy: plain BFS (no grounding restriction).
pub fn old_greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {
    let obs = state.build_obstacles();
    let pow = state.power_grid();
    let mut actions = HashMap::new();
    for s in state.snakes.iter().filter(|s| s.player == player) {
        if let Some(d) = state.bfs_first_step(s.head(), &pow, &obs) {
            actions.insert(s.id, d);
        }
    }
    actions
}

/// Old heuristic: plain BFS food distance, no stability term, -30 unreachable penalty.
pub fn old_heuristic(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    let obs = state.build_obstacles();
    let pow = state.power_grid();

    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist(s.head(), &pow, &obs);
            if d == i32::MAX { -30 } else { 20 - d.min(20) }
        })
        .sum();

    my * 100 - opp * 80 + food_bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GameState, Pos, Snake, Dir};
    use std::time::Instant;

    fn flat_state() -> GameState {
        // 20×15 with solid floor at y=14 — snakes near floor stay grounded
        let w = 20i32;
        let h = 15i32;
        let mut grid = vec![false; (w * h) as usize];
        for x in 0..w as usize {
            grid[14 * w as usize + x] = true; // floor
        }
        let mut s = GameState::new(w, h, grid);
        // Two snakes on the floor row (y=13, grounded because y=14 is platform below)
        s.snakes.push(Snake::new(0, vec![Pos::new(2, 13), Pos::new(3, 13), Pos::new(4, 13)], 0));
        s.snakes.push(Snake::new(1, vec![Pos::new(17, 13), Pos::new(16, 13), Pos::new(15, 13)], 1));
        // A handful of food items
        for &(x, y) in &[(5, 13), (10, 13), (14, 13), (7, 13), (12, 13)] {
            s.power.insert(Pos::new(x, y));
        }
        s
    }

    #[test]
    fn test_old_heuristic_favors_closer_food() {
        let mut s = flat_state();
        // Move snake 0 closer to food at (5,13): head is at (2,13), food at (5,13) = 3 away
        let score_far = old_heuristic(&s, 0);
        // Move head to (4,13) — 1 step from food
        s.snakes[0].body[0] = Pos::new(4, 13);
        let score_near = old_heuristic(&s, 0);
        assert!(score_near > score_far, "closer food should score higher: {} vs {}", score_near, score_far);
    }

    #[test]
    fn test_old_heuristic_dead_player_returns_min() {
        let mut s = flat_state();
        s.snakes.retain(|sn| sn.player != 0); // kill player 0
        assert_eq!(old_heuristic(&s, 0), i32::MIN / 2);
    }

    #[test]
    fn test_beam_returns_action() {
        let s = flat_state();
        let mut bot = OldBeamSearchBot::new(120, 8, 40);
        let acts = bot.choose_actions(&s, 0);
        assert!(!acts.is_empty(), "beam search must return at least one action");
        // The action must be for one of player 0's snakes
        for (&id, _) in &acts {
            let snake_ids: Vec<u8> = s.snakes.iter()
                .filter(|sn| sn.player == 0)
                .map(|sn| sn.id)
                .collect();
            assert!(snake_ids.contains(&id));
        }
    }

    #[test]
    fn test_beam_turn1_uses_40ms_not_950ms() {
        // state.turn = 1 → must use self.time_limit (40ms), not the turn-0 950ms budget
        let mut s = flat_state();
        s.turn = 1;
        let mut bot = OldBeamSearchBot::new(120, 8, 40);
        let t0 = Instant::now();
        let _ = bot.choose_actions(&s, 0);
        let ms = t0.elapsed().as_millis();
        assert!(ms < 200, "non-first turn used {}ms — expected ≤ ~40ms + overhead", ms);
    }

    #[test]
    fn test_beam_turn0_uses_extended_budget() {
        // state.turn = 0 → should use 950ms budget and search deeper
        // We just check it doesn't crash and returns an action; timing is environment-dependent
        let s = flat_state(); // turn = 0
        let mut bot = OldBeamSearchBot::new(10, 20, 40); // narrow beam, deep horizon
        let _ = bot.choose_actions(&s, 0); // must not panic
    }

    #[test]
    fn test_step_throughput() {
        // 10 000 step() calls must complete in < 2 s on any reasonable machine.
        // This is a regression guard: if step() gets 10× slower, this fails.
        let base = flat_state();
        let acts: std::collections::HashMap<u8, Dir> = [(0, Dir::Right), (1, Dir::Left)]
            .iter().cloned().collect();
        let t0 = Instant::now();
        for _ in 0..10_000 {
            let mut s = base.clone();
            s.step(&acts);
        }
        let ms = t0.elapsed().as_millis();
        assert!(ms < 2000, "10 000 step() calls took {}ms — expected < 2000ms", ms);
    }
}

impl Bot for OldBeamSearchBot {
    fn name(&self) -> &str { "OldBeamSearchBot" }

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

        let opp = old_greedy_actions(state, 1 - player);
        let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
            let mut combined = first.clone();
            for (&k, &v) in &opp { combined.entry(k).or_insert(v); }
            let mut ns = state.clone();
            ns.step(&combined);
            let score = old_heuristic(&ns, player);
            (first, ns, score)
        }).collect();
        beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        beam.truncate(self.beam_width);

        for _depth in 1..self.horizon {
            if t0.elapsed() >= limit { break; }

            let mut next: Vec<BeamItem> = Vec::with_capacity(beam.len() * 9);
            for (first_acts, cur, _) in beam.drain(..) {
                if cur.is_over() {
                    let score = old_heuristic(&cur, player);
                    next.push((first_acts, cur, score));
                    continue;
                }
                let my_combos = gen_action_combos(&cur, player);
                let opp_acts  = old_greedy_actions(&cur, 1 - player);
                let cap = self.beam_width.min(my_combos.len());
                for combo in my_combos.into_iter().take(cap) {
                    let mut combined = combo;
                    for (&k, &v) in &opp_acts { combined.entry(k).or_insert(v); }
                    let mut ns = cur.clone();
                    ns.step(&combined);
                    let score = old_heuristic(&ns, player);
                    next.push((first_acts.clone(), ns, score));
                }
            }
            next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            next.truncate(self.beam_width);
            beam = next;
        }

        beam.into_iter().next().map(|(a, _, _)| a).unwrap_or_default()
    }
}
