use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::{Bot, GameState, Dir, greedy_actions, gen_action_combos};

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

    // state.food IS the power grid — no allocation needed for pow
    let obs = state.build_obstacles(); // Vec<bool>: body obstacles
    let sng = state.snake_grid();      // Vec<u8>:   snake index per cell
    let w = state.width as usize;

    // Gravity-aware food distance
    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist_grounded(s.head(), &state.food, &obs);
            if d == i32::MAX { -50 } else { 20 - d.min(20) }
        })
        .sum();

    // Stability: penalise snakes not grounded
    let stability: i32 = state.snakes.iter().enumerate()
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
        .sum();

    my * 100 - opp * 80 + food_bonus + stability
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

        // Always keep the depth-0 best action as fallback. If the inner time check fires
        // before expanding any state at a deeper level, we return this rather than an
        // empty HashMap (which would silently produce "WAIT" and ignore food/walls).
        let mut result: HashMap<u8, Dir> = beam.first()
            .map(|(a, _, _)| a.clone())
            .unwrap_or_default();

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
            // If nothing was expanded (timed out on very first state), keep result from
            // the previous depth and stop — beam is already empty from mem::take.
            if next.is_empty() { break; }
            next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            next.truncate(self.beam_width);
            result = next[0].0.clone();
            beam = next;
        }

        result
    }
}
