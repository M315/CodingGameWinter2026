use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::{Bot, GameState, Dir, greedy_actions, heuristic, gen_action_combos};

pub struct BeamSearchBot {
    pub beam_width: usize,
    pub horizon:    usize,
    pub time_limit: Duration,
}

impl BeamSearchBot {
    pub fn new(beam_width: usize, horizon: usize, time_limit_ms: u64) -> Self {
        BeamSearchBot {
            beam_width,
            horizon,
            time_limit: Duration::from_millis(time_limit_ms),
        }
    }
}

impl Bot for BeamSearchBot {
    fn name(&self) -> &str { "BeamSearchBot" }

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {
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
            let score = heuristic(&ns, player);
            (first, ns, score)
        }).collect();
        beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        beam.truncate(self.beam_width);

        for _depth in 1..self.horizon {
            if t0.elapsed() >= self.time_limit { break; }

            let mut next: Vec<BeamItem> = Vec::with_capacity(beam.len() * 9);
            for (first_acts, cur, _) in beam.drain(..) {
                if cur.is_over() {
                    let score = heuristic(&cur, player);
                    next.push((first_acts, cur, score));
                    continue;
                }
                let my_combos = gen_action_combos(&cur, player);
                let opp_acts  = greedy_actions(&cur, 1 - player);
                let cap = self.beam_width.min(my_combos.len());
                for combo in my_combos.into_iter().take(cap) {
                    let mut combined = combo;
                    for (&k, &v) in &opp_acts { combined.entry(k).or_insert(v); }
                    let mut ns = cur.clone();
                    ns.step(&combined);
                    let score = heuristic(&ns, player);
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
