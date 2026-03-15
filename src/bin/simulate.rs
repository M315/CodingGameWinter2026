/// Local simulation harness – pit any two bots against each other.
///
/// Usage:
///   cargo run --bin simulate
///   cargo run --bin simulate -- --p0 beam --p1 greedy
///   cargo run --bin simulate -- --p0 greedy --p1 greedy --bench 20 --quiet
///   cargo run --bin simulate -- --p0 beam --p1 beam --bench 5
///   cargo run --bin simulate -- --map path/to/map.txt
///   cargo run --bin simulate -- --maps-dir path/to/maps/
///   cargo run --bin simulate -- --p0 beam --p1 old_beam --bench 50 --time-limit 10
///
/// Available bots: wait | greedy | beam | beam_v1 | beam_v2 | beam_v3 | beam_v4 | old_beam | mcts
///
/// Heuristic versioning protocol:
///   • `beam`      always points to the LATEST heuristic
///   • `beam_vN`   is a permanent alias for heuristic version N
///   • To compare versions: --p0 beam --p1 beam_v1
///   • Before adding a new heuristic: git tag heuristic-vN
///
/// --time-limit <ms>  Per-turn budget for all beam bots (default: 40).
///                    Lower values make benchmarks faster while preserving
///                    relative win rates. Disables the 950ms first-turn bonus.
///
/// Maps directory:
///   Default: ./maps/ (relative to cwd — run from project root).
///   Override with --maps-dir <path>.
///   In bench mode all *.txt files in the directory are loaded (sorted by name)
///   and rotated through so results aren't the same deterministic game repeated N times.
///
/// Timing guide (beam vs beam, 40ms budget):
///   01/03  (1 snake/player,  3 combos): ~1s/game
///   02/04  (2 snakes/player, 9 combos): ~2-4s/game
///   05     (3 snakes/player,27 combos): ~10s/game  ← real contest map
///
///   Quick 20-game bench (maps 01-04 only):
///     /tmp/sim -- --bench 20 --maps-dir maps/quick/
///   Full competition-accurate bench (all 5 maps):
///     /tmp/sim -- --bench 50
///   Single real-map test:
///     /tmp/sim -- --map maps/05_exotec_arena.txt

use snakebyte::*;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::thread;
use std::time::Instant;

// ============================================================
// Bot factory
// ============================================================

fn make_bot(name: &str, time_limit_ms: u64) -> Box<dyn Bot> {
    match name {
        "wait"     => Box::new(WaitBot),
        "greedy"   => Box::new(GreedyBot),
        // `beam` = fast structure (DirArr + TLS scratch) with old_heuristic (plain BFS).
        // old_heuristic is proven better than heuristic_v1 on open-border maps.
        // `beam_vN` = permanent versioned aliases.
        "beam"          => Box::new(BeamSearchBot::new(120, 8, time_limit_ms, old_heuristic)),
        "beam_v1"       => Box::new(BeamSearchBot::new(120, 8, time_limit_ms, heuristic_v1)),
        "beam_v2"       => Box::new(BeamSearchBot::new(120, 8, time_limit_ms, heuristic_v2)),
        "beam_v3"       => Box::new(BeamSearchBot::new(120, 8, time_limit_ms, heuristic_v3)),
        "beam_v4"       => Box::new(BeamSearchBot::new(120, 8, time_limit_ms, heuristic_v4)),
        "old_beam"      => Box::new(OldBeamSearchBot::new(120, 8, time_limit_ms)),
        // Same heuristic as old beam_v1 but uses gen_action_combos + step(&HashMap) — benchmark only.
        "beam_hashmap"  => Box::new(BeamHashMapBot::new(120, 8, time_limit_ms, heuristic_v1)),
        "mcts"          => Box::new(MctsBot::new(time_limit_ms, 6, 1.41)),
        _ => {
            eprintln!("Unknown bot '{}'. Available: wait | greedy | beam | beam_v1 | beam_hashmap | old_beam | mcts", name);
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

    let p0_name    = arg("--p0").unwrap_or("beam");
    let p1_name    = arg("--p1").unwrap_or("greedy");
    let bench      = arg("--bench").and_then(|s| s.parse().ok()).unwrap_or(1usize);
    let quiet      = args.contains(&"--quiet".to_string()) || bench > 1;
    let time_limit = arg("--time-limit").and_then(|s| s.parse().ok()).unwrap_or(40u64);
    // Parallel worker count for bench runs. Default: physical cores, capped at 4.
    let jobs = arg("--jobs")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4)
        });

    // Build the map pool.
    // --map <file>  → single fixed map for all games (useful for focused testing)
    // --maps-dir <dir> → load all *.txt files from that dir (overrides default)
    // default       → load from ./maps/ directory
    let pool: Vec<String> = if let Some(path) = arg("--map") {
        vec![std::fs::read_to_string(path).expect("Cannot read map file")]
    } else {
        let dir = arg("--maps-dir").unwrap_or("maps");
        load_maps_dir(dir)
    };

    if pool.is_empty() {
        eprintln!("No maps found. Place *.txt map files in ./maps/ or use --map <file>.");
        std::process::exit(1);
    }

    let total_t = Instant::now();

    // Single game: sequential with full visualization.
    // Multiple games: parallel across `jobs` threads — each game is independent.
    if bench == 1 {
        let map_str = &pool[0];
        let mut state = build_state_from_map(map_str);
        let mut bots: [Box<dyn Bot>; 2] = [make_bot(p0_name, time_limit), make_bot(p1_name, time_limit)];

        if !quiet {
            eprintln!("=== Game 1 | P0={} vs P1={} ===", p0_name, p1_name);
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
        let outcome = if s0 > s1 { "P0 wins" } else if s1 > s0 { "P1 wins" } else { "draw" };
        eprintln!("Game   1 | turn {:3} | P0 {} ({}) | P1 {} ({}) | {}", state.turn, s0, p0_name, s1, p1_name, outcome);

        println!("\n=== 1 game in {}ms ===", total_t.elapsed().as_millis());
        let (p0w, p1w, dw) = if s0 > s1 { (1,0,0) } else if s1 > s0 { (0,1,0) } else { (0,0,1) };
        println!("  P0 {:>10} : {:3} wins ({:.1}%)", p0_name, p0w, 100.0 * p0w as f64);
        println!("  P1 {:>10} : {:3} wins ({:.1}%)", p1_name, p1w, 100.0 * p1w as f64);
        println!("  draws      : {:3}      ({:.1}%)", dw, 100.0 * dw as f64);
        return;
    }

    // Multi-game parallel bench.
    // Divide game indices into `jobs` chunks; each chunk runs on its own thread.
    // std::thread::scope keeps borrows of pool/p0_name/p1_name valid without cloning.
    eprintln!("Running {} games across {} threads...", bench, jobs);

    struct GameResult { game_n: usize, s0: usize, s1: usize, turn: u32 }

    // Rebind as slice references so the `move` closures below copy the pointer,
    // not the owned Vec/String — thread::scope guarantees these outlive the threads.
    let pool: &[String] = &pool;
    let mut results: Vec<GameResult> = thread::scope(|s| {
        let handles: Vec<_> = (0..jobs)
            .map(|worker| {
                // Interleave game indices across workers (worker gets 0,J,2J,… or 1,J+1,2J+1,…)
                // so expensive maps (e.g. exotec every 5th game) are spread evenly rather than
                // clustering in one thread's contiguous chunk.
                s.spawn(move || {
                    (worker..bench).step_by(jobs).map(|game_n| {
                        let map_str = &pool[game_n % pool.len()];
                        let mut state = build_state_from_map(map_str);
                        // Skip 950ms first-turn bonus — distorts timings without affecting win rates.
                        state.turn = 1;
                        let mut bots: [Box<dyn Bot>; 2] = [
                            make_bot(p0_name, time_limit),
                            make_bot(p1_name, time_limit),
                        ];
                        while !state.is_over() && state.snakes_alive(0) && state.snakes_alive(1) {
                            let acts0 = bots[0].choose_actions(&state, 0);
                            let acts1 = bots[1].choose_actions(&state, 1);
                            let mut combined: HashMap<u8, Dir> = acts0;
                            for (k, v) in acts1 { combined.entry(k).or_insert(v); }
                            state.step(&combined);
                        }
                        GameResult { game_n, s0: state.score(0), s1: state.score(1), turn: state.turn }
                    }).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker thread panicked")).collect()
    });

    results.sort_unstable_by_key(|r| r.game_n);

    let mut p0_wins = 0usize;
    let mut p1_wins = 0usize;
    let mut draws   = 0usize;
    for r in &results {
        if r.s0 > r.s1 { p0_wins += 1; } else if r.s1 > r.s0 { p1_wins += 1; } else { draws += 1; }
    }

    // Print last game summary (mirrors pre-parallel behaviour in quiet mode).
    if let Some(last) = results.last() {
        eprintln!(
            "Game {:3} | turn {:3} | P0 {} ({}) | P1 {} ({})",
            last.game_n + 1, last.turn, last.s0, p0_name, last.s1, p1_name
        );
    }

    println!("\n=== {} games in {}ms ({} threads) ===", bench, total_t.elapsed().as_millis(), jobs);
    println!("  P0 {:>10} : {:3} wins ({:.1}%)", p0_name, p0_wins, 100.0 * p0_wins as f64 / bench as f64);
    println!("  P1 {:>10} : {:3} wins ({:.1}%)", p1_name, p1_wins, 100.0 * p1_wins as f64 / bench as f64);
    println!("  draws      : {:3}      ({:.1}%)", draws, 100.0 * draws as f64 / bench as f64);
}

// ============================================================
// Map directory loader
// ============================================================

/// Load all *.txt files from `dir`, sorted by filename.
/// Exits with an error message if the directory can't be read.
fn load_maps_dir(dir: &str) -> Vec<String> {
    let path = Path::new(dir);
    if !path.is_dir() {
        eprintln!(
            "Maps directory '{}' not found. Run from the project root, or use --maps-dir <path>.",
            dir
        );
        std::process::exit(1);
    }
    let mut entries: Vec<_> = std::fs::read_dir(path)
        .expect("Cannot read maps directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("txt"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    if entries.is_empty() {
        eprintln!("No *.txt files found in '{}'.", dir);
        std::process::exit(1);
    }
    entries.iter().map(|e| {
        std::fs::read_to_string(e.path())
            .unwrap_or_else(|_| panic!("Cannot read map file: {:?}", e.path()))
    }).collect()
}

// ============================================================
// Map loader
// ============================================================

/// Build a GameState from a plain-text map string.
///
/// Format (see maps/*.txt for examples):
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
        state.add_food(Pos::new(x, y));
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
