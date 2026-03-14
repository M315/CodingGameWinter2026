/// Local simulation harness – pit any two bots against each other.
///
/// Usage:
///   cargo run --bin simulate
///   cargo run --bin simulate -- --p0 beam --p1 greedy
///   cargo run --bin simulate -- --p0 greedy --p1 greedy --bench 20 --quiet
///   cargo run --bin simulate -- --p0 beam --p1 beam --bench 5
///   cargo run --bin simulate -- --map path/to/map.txt
///
/// Available bots: wait | greedy | beam

use snakebyte::*;
use std::collections::HashMap;
use std::env;
use std::time::Instant;

// ============================================================
// Bot factory
// ============================================================

fn make_bot(name: &str) -> Box<dyn Bot> {
    match name {
        "wait"   => Box::new(WaitBot),
        "greedy" => Box::new(GreedyBot),
        "beam"   => Box::new(BeamSearchBot::new(120, 8, 40)),
        _ => {
            eprintln!("Unknown bot '{}'. Available: wait | greedy | beam", name);
            std::process::exit(1);
        }
    }
}

// ============================================================
// Entry point
// ============================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    let arg = |flag: &str| args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str());

    let p0_name = arg("--p0").unwrap_or("beam");
    let p1_name = arg("--p1").unwrap_or("greedy");
    let bench   = arg("--bench").and_then(|s| s.parse().ok()).unwrap_or(1usize);
    let quiet   = args.contains(&"--quiet".to_string()) || bench > 1;
    let map_str = match arg("--map") {
        Some(path) => std::fs::read_to_string(path).expect("Cannot read map file"),
        None       => default_map().to_string(),
    };

    let mut p0_wins = 0usize;
    let mut p1_wins = 0usize;
    let mut draws   = 0usize;
    let total_t = Instant::now();

    for game_n in 0..bench {
        let mut state = build_state_from_map(&map_str);
        let mut bots: [Box<dyn Bot>; 2] = [make_bot(p0_name), make_bot(p1_name)];

        if !quiet {
            eprintln!("=== Game {} | P0={} vs P1={} ===", game_n + 1, p0_name, p1_name);
            eprintln!("{}", visualize(&state));
        }

        while !state.is_over() && state.snakes_alive(0) && state.snakes_alive(1) {
            let t0 = Instant::now();
            let acts0 = bots[0].choose_actions(&state, 0);
            let p0_ms = t0.elapsed().as_millis();
            let acts1 = bots[1].choose_actions(&state, 1);

            let mut combined: HashMap<u8, Dir> = acts0;
            for (k, v) in acts1 { combined.entry(k).or_insert(v); }
            state.step(&combined);

            if !quiet {
                eprintln!("{}", visualize(&state));
                eprintln!("  P0 ({}) decision: {}ms", p0_name, p0_ms);
            }
        }

        let s0 = state.score(0);
        let s1 = state.score(1);
        if s0 > s1 { p0_wins += 1; } else if s1 > s0 { p1_wins += 1; } else { draws += 1; }

        if !quiet || game_n == bench - 1 {
            eprintln!(
                "Game {:3} | turn {:3} | P0 {} ({}) | P1 {} ({})",
                game_n + 1, state.turn, s0, p0_name, s1, p1_name
            );
        }
    }

    println!("\n=== {} games in {}ms ===", bench, total_t.elapsed().as_millis());
    println!("  P0 {:>10} : {:3} wins ({:.1}%)", p0_name, p0_wins, 100.0 * p0_wins as f64 / bench as f64);
    println!("  P1 {:>10} : {:3} wins ({:.1}%)", p1_name, p1_wins, 100.0 * p1_wins as f64 / bench as f64);
    println!("  draws      : {:3}      ({:.1}%)", draws, 100.0 * draws as f64 / bench as f64);
}

// ============================================================
// Map loader
// ============================================================

/// Build a GameState from a plain-text map string.
///
/// Format:
///   Lines starting with '#' or '.'  → grid rows (top to bottom)
///   P x y                           → power source at (x, y)
///   S <player> <id>  x,y x,y ...   → snake, head first
///   Lines starting with '//'        → comments
fn build_state_from_map(map: &str) -> GameState {
    let mut grid_rows: Vec<&str> = Vec::new();
    let mut power_lines: Vec<&str> = Vec::new();
    let mut snake_lines: Vec<&str> = Vec::new();

    for line in map.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") { continue; }
        match t.chars().next() {
            Some('#') | Some('.') => grid_rows.push(t),
            Some('P') => power_lines.push(t),
            Some('S') => snake_lines.push(t),
            _ => {}
        }
    }

    let height = grid_rows.len() as i32;
    let width  = grid_rows.iter().map(|r| r.len()).max().unwrap_or(0) as i32;
    let mut grid = vec![false; (width * height) as usize];
    for (y, row) in grid_rows.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            if ch == '#' { grid[y * width as usize + x] = true; }
        }
    }

    let mut state = GameState::new(width, height, grid);

    for line in &power_lines {
        let mut p = line[1..].trim().split_whitespace();
        let x: i32 = p.next().unwrap().parse().unwrap();
        let y: i32 = p.next().unwrap().parse().unwrap();
        state.power.insert(Pos::new(x, y));
    }
    for line in &snake_lines {
        let mut p = line[1..].trim().split_whitespace();
        let player: u8 = p.next().unwrap().parse().unwrap();
        let id: u8     = p.next().unwrap().parse().unwrap();
        let body: Vec<Pos> = p.map(|seg| {
            let mut c = seg.split(',');
            Pos::new(c.next().unwrap().parse().unwrap(), c.next().unwrap().parse().unwrap())
        }).collect();
        state.snakes.push(Snake::new(id, body, player));
    }

    state.apply_gravity();
    state
}

// ============================================================
// Default test map (20×15)
// ============================================================

fn default_map() -> &'static str {
    r"
####################
#..................#
#..................#
#....###...........#
#..................#
#.........###......#
#..................#
#...###............#
#..................#
#...........###....#
#..................#
#..................#
#....###...........#
#..................#
####################

P 3 1
P 7 2
P 14 1
P 2 5
P 17 5
P 5 8
P 13 8
P 3 11
P 16 11
P 9 6

S 0 0  2,1 2,2 2,3
S 1 1  17,1 17,2 17,3
"
}
