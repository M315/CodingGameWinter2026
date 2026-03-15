use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::{Bot, GameState, Dir, gen_action_combos, greedy_actions};
use super::beam::heuristic_v1 as heuristic;

// ── Arena node ───────────────────────────────────────────────────
//
// The MCTS tree lives in a flat Vec<MctsNode> (arena).  All cross-references
// are plain `usize` indices — no heap-pointer chasing, no borrow-checker fights.
//
// Action combos are cached per node (computed once when the node is created).
// "my" / "op" always refer to the player passed to `choose_actions`; player
// identity is fixed for the whole MCTS run.
//
// DUCT (Decoupled UCT) is used for simultaneous moves:
//   • each player independently selects their combo via UCB1
//   • the joint (my_idx, op_idx) pair determines which child to visit/create
//   • backprop updates per-player stats separately
struct MctsNode {
    state: GameState,

    visits: u32,
    score:  f32,   // cumulative [0,1] from "my player" perspective

    // Cached action combos (generated once, reused for all children)
    my_combos: Vec<HashMap<u8, Dir>>,
    op_combos: Vec<HashMap<u8, Dir>>,

    // DUCT UCB stats — one entry per combo per player
    my_visits: Vec<u32>,
    my_score:  Vec<f32>,
    op_visits: Vec<u32>,
    op_score:  Vec<f32>,  // from opponent's perspective (1 - my_score)

    // Sparse child list: (my_combo_idx, op_combo_idx, arena_index)
    children: Vec<(u8, u8, u32)>,

    game_over: bool,
}

impl MctsNode {
    fn new(state: GameState, player: u8) -> Self {
        let game_over = state.is_over();
        // Game-over nodes get a single dummy combo so the stat vecs are non-empty.
        let my_combos = if game_over { vec![HashMap::new()] }
                        else         { gen_action_combos(&state, player) };
        let op_combos = if game_over { vec![HashMap::new()] }
                        else         { gen_action_combos(&state, 1 - player) };
        let nm = my_combos.len();
        let no = op_combos.len();
        MctsNode {
            state,
            visits: 0,
            score:  0.0,
            my_combos,
            op_combos,
            my_visits: vec![0u32;   nm],
            my_score:  vec![0.0f32; nm],
            op_visits: vec![0u32;   no],
            op_score:  vec![0.0f32; no],
            children: Vec::new(),
            game_over,
        }
    }
}

// ── UCB1 ─────────────────────────────────────────────────────────

/// Return the combo index with the highest UCB1 value.
/// Unvisited combos (visits == 0) always win with +∞.
#[inline]
fn ucb_select(visits: &[u32], scores: &[f32], parent_visits: u32, c: f32) -> usize {
    let ln_p = (parent_visits as f32).ln();
    visits.iter().zip(scores).enumerate()
        .map(|(i, (&v, &s))| {
            let val = if v == 0 {
                f32::INFINITY
            } else {
                s / v as f32 + c * (ln_p / v as f32).sqrt()
            };
            (i, val)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Rollout / leaf evaluation ─────────────────────────────────────

/// Evaluate `state` by simulating `depth` greedy steps then calling
/// `heuristic()`.  Returns a score in [0, 1] for `player`.
fn rollout(state: &GameState, player: u8, depth: usize) -> f32 {
    // Short-circuit: terminal states have a known result.
    if state.is_over() {
        let mine = state.score(player) as i32;
        let opp  = state.score(1 - player) as i32;
        return match mine.cmp(&opp) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less    => 0.0,
            std::cmp::Ordering::Equal   => 0.5,
        };
    }

    let mut s = state.clone();
    for _ in 0..depth {
        if s.is_over() { break; }
        let mut acts = greedy_actions(&s, 0);
        for (k, v) in greedy_actions(&s, 1) { acts.entry(k).or_insert(v); }
        s.step(&acts);
    }

    // Sigmoid with temperature 300 maps the heuristic's typical range to (0, 1).
    // At h = ±600 (decisive lead) the sigmoid saturates at ≈ 0.88 / 0.12.
    1.0 / (1.0 + (-(heuristic(&s, player) as f32) / 300.0).exp())
}

// ── MctsBot ──────────────────────────────────────────────────────

pub struct MctsBot {
    pub time_limit:    Duration,
    pub rollout_depth: usize,
    pub exploration:   f32,    // C in UCB1 (√2 ≈ 1.41 is the standard starting point)
    pub max_nodes:     usize,  // arena capacity cap — prevents OOM on long turn-0 budget
}

impl MctsBot {
    pub fn new(time_limit_ms: u64, rollout_depth: usize, exploration: f32) -> Self {
        MctsBot {
            time_limit: Duration::from_millis(time_limit_ms),
            rollout_depth,
            exploration,
            max_nodes: 50_000,
        }
    }

    /// One full MCTS iteration: selection → expansion → rollout → backprop.
    fn run_iteration(&self, arena: &mut Vec<MctsNode>, player: u8) {
        // ── 1. Selection ──────────────────────────────────────────────────
        // Walk down the tree via UCB1 until we reach an unvisited node (leaf)
        // or a terminal, or until we expand a new child.
        //
        // `path` records (node_idx, my_combo_idx, op_combo_idx) for backprop.
        // It does NOT include the final `cur` node — that gets updated separately.
        let mut path: Vec<(usize, usize, usize)> = Vec::new();
        let mut cur = 0usize;

        loop {
            let (game_over, visits) = {
                let n = &arena[cur];
                (n.game_over, n.visits)
            };

            // Leaf (never visited) or terminal: stop here and roll out.
            if game_over || visits == 0 { break; }

            // DUCT: each player independently picks their best combo via UCB1.
            let (mi, oi) = {
                let n = &arena[cur];
                let mi = ucb_select(&n.my_visits, &n.my_score, visits, self.exploration);
                let oi = ucb_select(&n.op_visits, &n.op_score, visits, self.exploration);
                (mi, oi)
            };

            // Look up the child for this (my_combo, op_combo) pair.
            let child_opt = arena[cur].children.iter()
                .find(|&&(m, o, _)| m == mi as u8 && o == oi as u8)
                .map(|&(_, _, ci)| ci as usize);

            path.push((cur, mi, oi));

            match child_opt {
                // Already expanded: descend.
                Some(ci) => { cur = ci; }

                // ── 2. Expansion ─────────────────────────────────────────────
                // Build the joint action map, step the state, create the child node.
                None => {
                    let combined = {
                        let n = &arena[cur];
                        let mut m = n.my_combos[mi].clone();
                        for (&k, &v) in &n.op_combos[oi] { m.entry(k).or_insert(v); }
                        m
                    };
                    let mut child_state = arena[cur].state.clone();
                    child_state.step(&combined);
                    let child_idx = arena.len();
                    arena.push(MctsNode::new(child_state, player));
                    arena[cur].children.push((mi as u8, oi as u8, child_idx as u32));
                    cur = child_idx;
                    break;
                }
            }
        }

        // ── 3. Rollout ────────────────────────────────────────────────────
        let score = rollout(&arena[cur].state, player, self.rollout_depth);

        // ── 4. Backpropagation ────────────────────────────────────────────
        // Update the leaf/new-child node first.
        {
            let n = &mut arena[cur];
            n.visits += 1;
            n.score  += score;
        }

        // Then walk the path back to the root, updating each node's aggregate
        // stats AND the per-combo (DUCT) stats for the choice made at that node.
        for &(ni, mi, oi) in path.iter().rev() {
            let n = &mut arena[ni];
            n.visits          += 1;
            n.score           += score;
            n.my_visits[mi]   += 1;
            n.my_score[mi]    += score;
            n.op_visits[oi]   += 1;
            n.op_score[oi]    += 1.0 - score; // opponent maximises (1 - my_score)
        }
    }
}

impl Bot for MctsBot {
    fn name(&self) -> &str { "MctsBot" }

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {
        // Turn 0 gets the full 1 s CG initialisation budget.
        let limit = if state.turn == 0 {
            Duration::from_millis(950)
        } else {
            self.time_limit
        };
        let t0 = Instant::now();

        let root = MctsNode::new(state.clone(), player);
        if root.game_over { return HashMap::new(); }

        let mut arena: Vec<MctsNode> = vec![root];
        let mut iters = 0u32;

        loop {
            // Check time every 128 iterations to amortise the syscall cost.
            if iters % 128 == 0 && t0.elapsed() >= limit { break; }
            if arena.len() >= self.max_nodes { break; }
            self.run_iteration(&mut arena, player);
            iters += 1;
        }

        // Robust-child criterion: return the my_combo with the most visits.
        // More stable than max-score when sample counts are uneven.
        let root = &arena[0];
        let best = root.my_visits.iter().enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0);

        eprintln!(
            "[MCTS] turn {} | iters {} | nodes {} | elapsed {}ms",
            state.turn, iters, arena.len(), t0.elapsed().as_millis()
        );

        root.my_combos[best].clone()
    }
}
