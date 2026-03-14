use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::{Bot, GameState, Dir, gen_action_combos};

/// Beam search bot using the old (pre-gravity-aware) heuristic and greedy opponent model.
/// Use this to compare against the current BeamSearchBot.
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

/// Old (pre-gravity) greedy action selection: plain BFS, no grounding restriction.
pub fn old_greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {
    let obs = state.build_obstacles();
    let mut actions = HashMap::new();
    for s in state.snakes.iter().filter(|s| s.player == player) {
        if let Some(d) = state.bfs_first_step(s.head(), &state.power, &obs) {
            actions.insert(s.id, d);
        }
    }
    actions
}

/// Old (pre-gravity) heuristic: plain BFS food distance, no stability term.
pub fn old_heuristic(state: &GameState, player: u8) -> i32 {
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

impl Bot for OldBeamSearchBot {
    fn name(&self) -> &str { "OldBeamSearchBot" }

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {
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
            if t0.elapsed() >= self.time_limit { break; }

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