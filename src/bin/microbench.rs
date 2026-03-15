/// Micro-benchmark: measures raw per-call throughput of individual components.
///
/// Unlike the game benchmark, this:
///   - Uses a single fixed GameState (no game variance)
///   - Runs single-threaded (no resource contention between bots)
///   - Times exact wall-clock per operation via N repeated calls
///   - Uses std::hint::black_box to prevent dead-code elimination
///   - For beam: runs to a FIXED DEPTH (no time limit) so we measure work, not budget
///
/// Usage (from project root):
///   cargo build --release
///   install -m 755 target/release/microbench /tmp/mb && /tmp/mb [map_path]
///   /tmp/mb maps/05_exotec_arena.txt   # default
///   /tmp/mb maps/01_default.txt        # simpler (fewer combos)

use snakebyte::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

// ── Map loader (same format as simulate.rs) ──────────────────────────────────

fn load_map(path: &str) -> GameState {
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Cannot read map: {}", path));

    let mut lines: Vec<&str> = src.lines()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with("//") && !l.is_empty())
        .collect();

    // Grid rows: lines that contain only '.', '#', and spaces
    let grid_lines: Vec<&str> = lines.iter()
        .take_while(|l| l.chars().all(|c| c == '.' || c == '#'))
        .copied()
        .collect();
    let height = grid_lines.len() as i32;
    let width  = grid_lines.iter().map(|l| l.len()).max().unwrap_or(0) as i32;
    let mut grid = vec![false; (width * height) as usize];
    for (y, row) in grid_lines.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            if ch == '#' { grid[y * width as usize + x] = true; }
        }
    }
    lines.drain(..grid_lines.len());

    let mut state = GameState::new(width, height, grid);

    for line in &lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["P", x, y] => {
                state.add_food(Pos::new(x.parse().unwrap(), y.parse().unwrap()));
            }
            ["S", player, id, coords @ ..] => {
                let player: u8 = player.parse().unwrap();
                let id: u8     = id.parse().unwrap();
                let body: Vec<Pos> = coords.iter().flat_map(|s| {
                    s.split(',').collect::<Vec<_>>().chunks(2)
                        .map(|c| Pos::new(c[0].parse().unwrap(), c[1].parse().unwrap()))
                        .collect::<Vec<_>>()
                }).collect();
                // Infer initial direction from first two body segments
                let dir = if body.len() >= 2 {
                    let (h, n) = (body[0], body[1]);
                    if      h.x > n.x { Dir::Right }
                    else if h.x < n.x { Dir::Left  }
                    else if h.y > n.y { Dir::Down  }
                    else               { Dir::Up    }
                } else { Dir::Right };
                state.snakes.push(Snake::new(id, body, player));
                if let Some(s) = state.snakes.last_mut() { s.dir = dir; }
            }
            _ => {}
        }
    }

    // Apply gravity so snakes land on their starting platform
    state.apply_gravity();
    state.turn = 1; // turn 0 triggers 950ms budget; use turn 1 for fair timing
    state
}

// ── Benchmark helpers ─────────────────────────────────────────────────────────

/// Run `f` N times, print mean µs/call and total ms.
fn bench_n<F, T>(label: &str, n: usize, mut f: F)
where F: FnMut() -> T
{
    // Warmup: 10% of N, minimum 10
    let warmup = (n / 10).max(10);
    for _ in 0..warmup { black_box(f()); }

    let t0 = Instant::now();
    for _ in 0..n { black_box(f()); }
    let micros = t0.elapsed().as_micros() as f64;
    println!("  {:<40}  {:>8.2} µs/call  ({} calls, {:.1} ms total)",
        label, micros / n as f64, n, micros / 1000.0);
}

// ── Fixed-depth beam expansion (returns nodes expanded) ──────────────────────
//
// Unlike the timed BeamSearchBot, these expand exactly `depth` levels with no
// time check.  Both DirArr and HashMap variants use the same beam width so the
// comparison is purely structural overhead.

const BEAM_WIDTH: usize = 120;

fn beam_dirarr_depth(state: &GameState, player: u8, depth: usize) -> usize {
    type BeamItem = (DirArr, GameState, i32);

    #[inline]
    fn merge(a: &DirArr, b: &DirArr) -> DirArr {
        let mut out = *a;
        for i in 0..8 { if out[i].is_none() { out[i] = b[i]; } }
        out
    }

    let first_combos = gen_combos(state, player);
    if first_combos.is_empty() { return 0; }

    let opp = greedy_dirmap(state, 1 - player);
    let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
        let mut ns = state.clone();
        ns.step_arr(&merge(&first, &opp));
        let score = heuristic_v1(&ns, player);
        (first, ns, score)
    }).collect();
    beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    beam.truncate(BEAM_WIDTH);
    let mut nodes = beam.len();

    for _ in 1..depth {
        let cur_beam = std::mem::take(&mut beam);
        let mut next: Vec<BeamItem> = Vec::with_capacity(cur_beam.len() * 9);
        for (first_acts, cur, _) in cur_beam {
            if cur.is_over() { continue; }
            let my_combos = gen_combos(&cur, player);
            let opp_acts  = greedy_dirmap(&cur, 1 - player);
            for combo in my_combos {
                let mut ns = cur.clone();
                ns.step_arr(&merge(&combo, &opp_acts));
                let score = heuristic_v1(&ns, player);
                next.push((first_acts, ns, score));
            }
        }
        nodes += next.len();
        next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        next.truncate(BEAM_WIDTH);
        beam = next;
        if beam.is_empty() { break; }
    }
    nodes
}

fn beam_hashmap_depth(state: &GameState, player: u8, depth: usize) -> usize {
    type BeamItem = (HashMap<u8, Dir>, GameState, i32);

    let first_combos = gen_action_combos(state, player);
    if first_combos.is_empty() { return 0; }

    let opp = greedy_actions(state, 1 - player);
    let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
        let mut combined = first.clone();
        for (&k, &v) in &opp { combined.entry(k).or_insert(v); }
        let mut ns = state.clone();
        ns.step(&combined);
        let score = heuristic_v1(&ns, player);
        (first, ns, score)
    }).collect();
    beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    beam.truncate(BEAM_WIDTH);
    let mut nodes = beam.len();

    for _ in 1..depth {
        let cur_beam = std::mem::take(&mut beam);
        let mut next: Vec<BeamItem> = Vec::with_capacity(cur_beam.len() * 9);
        for (first_acts, cur, _) in cur_beam {
            if cur.is_over() { continue; }
            let my_combos = gen_action_combos(&cur, player);
            let opp_acts  = greedy_actions(&cur, 1 - player);
            for combo in my_combos {
                let mut combined = combo;
                for (&k, &v) in &opp_acts { combined.entry(k).or_insert(v); }
                let mut ns = cur.clone();
                ns.step(&combined);
                let score = heuristic_v1(&ns, player);
                next.push((first_acts.clone(), ns, score));
            }
        }
        nodes += next.len();
        next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        next.truncate(BEAM_WIDTH);
        beam = next;
        if beam.is_empty() { break; }
    }
    nodes
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let map_path = std::env::args().nth(1)
        .unwrap_or_else(|| "maps/05_exotec_arena.txt".to_string());

    let state = load_map(&map_path);
    let player = 0u8;

    let n_snakes_mine = state.snakes.iter().filter(|s| s.player == player).count();
    let n_food        = state.food.iter().filter(|&&b| b).count();
    let combos_mine   = (3usize).pow(n_snakes_mine as u32); // approx (3 dirs each)

    println!();
    println!("=== Micro-benchmark: {} ===", map_path);
    println!("  Grid: {}×{}  |  My snakes: {}  |  Food: {}  |  ~{} combos/player",
        state.width, state.height, n_snakes_mine, n_food, combos_mine);
    println!("  Single-threaded, no time limit.");
    println!();

    // ── 1. Heuristic ──────────────────────────────────────────────────────────
    println!("[ Heuristic evaluation ]");
    bench_n("old_heuristic (current main)", 5_000, || old_heuristic(black_box(&state), player));
    bench_n("heuristic_v1             ", 5_000, || heuristic_v1(black_box(&state), player));
    println!();

    // ── 2. Combo generation ───────────────────────────────────────────────────
    println!("[ Action combo generation (player {}) ]", player);
    bench_n("gen_combos   (DirArr, new)", 10_000, || gen_combos(black_box(&state), player));
    bench_n("gen_action_combos (HashMap, old)", 10_000, || gen_action_combos(black_box(&state), player));
    println!();

    // ── 3. Greedy opponent directions ─────────────────────────────────────────
    println!("[ Greedy opponent direction (player {}) ]", 1 - player);
    bench_n("greedy_dirmap (DirArr, new)", 10_000, || greedy_dirmap(black_box(&state), 1 - player));
    bench_n("greedy_actions (HashMap, old)", 10_000, || greedy_actions(black_box(&state), 1 - player));
    println!();

    // ── 4. Clone cost in isolation ────────────────────────────────────────────
    println!("[ GameState::clone() in isolation ]");
    bench_n("state.clone()", 10_000, || black_box(state.clone()));
    println!();

    // ── 5. Step: applying one action set ─────────────────────────────────────
    println!("[ step() — advancing state by 1 turn (includes clone) ]");
    // Use the first combo from each generator as a representative action
    let combo_arr: DirArr = gen_combos(&state, player)[0];
    let opp_arr:   DirArr = greedy_dirmap(&state, 1 - player);
    let mut merged_arr = combo_arr;
    for i in 0..8 { if merged_arr[i].is_none() { merged_arr[i] = opp_arr[i]; } }

    let combo_map: HashMap<u8, Dir> = gen_action_combos(&state, player)[0].clone();
    let opp_map:   HashMap<u8, Dir> = greedy_actions(&state, 1 - player);
    let mut merged_map = combo_map.clone();
    for (&k, &v) in &opp_map { merged_map.entry(k).or_insert(v); }

    bench_n("step_arr (DirArr, new)", 10_000, || {
        let mut s = state.clone();
        s.step_arr(black_box(&merged_arr));
        s
    });
    bench_n("step (HashMap, old)", 10_000, || {
        let mut s = state.clone();
        s.step(black_box(&merged_map));
        s
    });
    println!();

    // ── 5. Fixed-depth beam expansion ────────────────────────────────────────
    println!("[ Fixed-depth beam (width={}, same heuristic) ]", BEAM_WIDTH);
    println!("  Nodes column = total beam items scored across all depths.");
    println!();
    println!("  {:>5}  {:>12}  {:>12}  {:>8}  {:>8}  {:>8}",
        "depth", "DirArr (ms)", "HashMap (ms)", "speedup", "DirArr N", "HashMap N");
    println!("  {}", "-".repeat(65));

    for depth in 1..=5 {
        // DirArr run
        let t0 = Instant::now();
        let nodes_arr = black_box(beam_dirarr_depth(&state, player, depth));
        let ms_arr = t0.elapsed().as_millis();

        // HashMap run
        let t1 = Instant::now();
        let nodes_map = black_box(beam_hashmap_depth(&state, player, depth));
        let ms_map = t1.elapsed().as_millis();

        let speedup = if ms_arr > 0 { ms_map as f64 / ms_arr as f64 } else { 0.0 };
        println!("  {:>5}  {:>12}  {:>12}  {:>7.2}×  {:>8}  {:>8}",
            depth, ms_arr, ms_map, speedup, nodes_arr, nodes_map);
    }

    println!();
}
