use snakebyte::*;
use std::collections::HashMap;
use std::io::{self, BufRead};

macro_rules! read_line {
    ($lines:expr) => {{
        $lines.next().unwrap().unwrap()
    }};
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // ── Initialization ───────────────────────────────────────────────
    let my_id: u8 = read_line!(lines).trim().parse().unwrap();
    let width: i32 = read_line!(lines).trim().parse().unwrap();
    let height: i32 = read_line!(lines).trim().parse().unwrap();

    let mut grid = vec![false; (width * height) as usize];
    for y in 0..height {
        let row = read_line!(lines);
        for (x, ch) in row.trim().chars().enumerate() {
            if ch == '#' {
                grid[(y * width + x as i32) as usize] = true;
            }
        }
    }

    let snakebots_per_player: usize = read_line!(lines).trim().parse().unwrap();
    let mut my_ids: Vec<u8> = Vec::new();
    for _ in 0..snakebots_per_player {
        let id: u8 = read_line!(lines).trim().parse().unwrap();
        my_ids.push(id);
    }
    let mut opp_ids: Vec<u8> = Vec::new();
    for _ in 0..snakebots_per_player {
        let id: u8 = read_line!(lines).trim().parse().unwrap();
        opp_ids.push(id);
    }

    let my_id_set: std::collections::HashSet<u8> = my_ids.iter().cloned().collect();

    let mut state = GameState::new(width, height, grid);
    let mut bot: Box<dyn Bot> = Box::new(OldBeamSearchBot::new(120, 8, 40));

    // ── Game loop ────────────────────────────────────────────────────
    loop {
        let power_count: usize = read_line!(lines).trim().parse().unwrap();
        state.clear_food();
        for _ in 0..power_count {
            let line = read_line!(lines);
            let mut parts = line.trim().split_whitespace();
            let x: i32 = parts.next().unwrap().parse().unwrap();
            let y: i32 = parts.next().unwrap().parse().unwrap();
            state.add_food(Pos::new(x, y));
        }

        let snake_count: usize = read_line!(lines).trim().parse().unwrap();
        state.snakes.clear();
        for _ in 0..snake_count {
            let line = read_line!(lines);
            let mut parts = line.trim().splitn(2, ' ');
            let id: u8 = parts.next().unwrap().parse().unwrap();
            let body_str = parts.next().unwrap();

            let body: Vec<Pos> = body_str.trim().split(':').map(|seg| {
                let mut coords = seg.split(',');
                let x: i32 = coords.next().unwrap().parse().unwrap();
                let y: i32 = coords.next().unwrap().parse().unwrap();
                Pos::new(x, y)
            }).collect();

            let player: u8 = if my_id_set.contains(&id) { my_id } else { 1 - my_id };
            state.snakes.push(Snake::new(id, body, player));
        }

        let actions = bot.choose_actions(&state, my_id);
        state.turn += 1; // track turn so the bot knows turn-0 vs subsequent
        println!("{}", format_actions(&actions));
    }
}

fn format_actions(actions: &HashMap<u8, Dir>) -> String {
    if actions.is_empty() {
        return "WAIT".to_string();
    }
    let mut parts: Vec<String> = actions.iter()
        .map(|(&id, &dir)| format!("{} {}", id, dir.to_str()))
        .collect();
    parts.sort(); // deterministic output
    parts.join("; ")
}
