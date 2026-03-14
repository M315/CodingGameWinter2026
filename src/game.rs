use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================
// Primitive types
// ============================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

impl Pos {
    #[inline] pub fn new(x: i32, y: i32) -> Self { Pos { x, y } }
    #[inline] pub fn translate(self, dx: i32, dy: i32) -> Self { Pos { x: self.x + dx, y: self.y + dy } }
    #[inline] pub fn in_bounds(self, w: i32, h: i32) -> bool {
        self.x >= 0 && self.y >= 0 && self.x < w && self.y < h
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dir { Up, Down, Left, Right }

impl Dir {
    #[inline]
    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up    => (0, -1),
            Dir::Down  => (0,  1),
            Dir::Left  => (-1, 0),
            Dir::Right => (1,  0),
        }
    }
    pub fn from_str(s: &str) -> Option<Dir> {
        match s { "UP" => Some(Dir::Up), "DOWN" => Some(Dir::Down),
                  "LEFT" => Some(Dir::Left), "RIGHT" => Some(Dir::Right), _ => None }
    }
    pub fn to_str(self) -> &'static str {
        match self { Dir::Up => "UP", Dir::Down => "DOWN",
                     Dir::Left => "LEFT", Dir::Right => "RIGHT" }
    }
    pub fn all() -> [Dir; 4] { [Dir::Up, Dir::Down, Dir::Left, Dir::Right] }
    pub fn opposite(self) -> Dir {
        match self { Dir::Up => Dir::Down, Dir::Down => Dir::Up,
                     Dir::Left => Dir::Right, Dir::Right => Dir::Left }
    }
}

// ============================================================
// Snake
// ============================================================

#[derive(Clone, Debug)]
pub struct Snake {
    pub id: u8,
    /// body[0] = head, body[last] = tail
    pub body: VecDeque<Pos>,
    pub dir: Dir,
    pub player: u8,
}

impl Snake {
    pub fn new(id: u8, parts: Vec<Pos>, player: u8) -> Self {
        Snake { id, body: VecDeque::from(parts.clone()),
                dir: infer_dir(&parts), player }
    }
    #[inline] pub fn head(&self) -> Pos { self.body[0] }
    #[inline] pub fn len(&self) -> usize { self.body.len() }
}

/// Infer movement direction from head − neck positions.
pub fn infer_dir(parts: &[Pos]) -> Dir {
    if parts.len() < 2 { return Dir::Up; }
    match (parts[0].x - parts[1].x, parts[0].y - parts[1].y) {
        (0, -1) => Dir::Up,   (0, 1) => Dir::Down,
        (-1, 0) => Dir::Left, (1, 0) => Dir::Right,
        _ => Dir::Up,
    }
}

// ============================================================
// GameState
// ============================================================

#[derive(Clone, Debug)]
pub struct GameState {
    pub width:  i32,
    pub height: i32,
    /// Flat row-major: grid[y * width + x] = true if platform
    pub grid:   Vec<bool>,
    /// Live power sources
    pub power:  HashSet<Pos>,
    /// All living snakes (both players)
    pub snakes: Vec<Snake>,
    pub turn:   u32,
}

impl GameState {
    pub fn new(width: i32, height: i32, grid: Vec<bool>) -> Self {
        GameState { width, height, grid, power: HashSet::new(), snakes: Vec::new(), turn: 0 }
    }

    #[inline]
    pub fn is_platform(&self, p: Pos) -> bool {
        p.in_bounds(self.width, self.height)
            && self.grid[(p.y * self.width + p.x) as usize]
    }

    pub fn score(&self, player: u8) -> usize {
        self.snakes.iter().filter(|s| s.player == player).map(|s| s.len()).sum()
    }

    pub fn snakes_alive(&self, player: u8) -> bool {
        self.snakes.iter().any(|s| s.player == player)
    }

    pub fn is_over(&self) -> bool {
        self.turn >= 200 || self.power.is_empty() || self.snakes.is_empty()
    }

    // --------------------------------------------------------
    // Step: advance the game state by one turn in-place
    // --------------------------------------------------------

    pub fn step(&mut self, actions: &HashMap<u8, Dir>) {
        let n = self.snakes.len();
        if n == 0 { self.turn += 1; return; }

        // Phase 1 – apply direction overrides
        for s in &mut self.snakes {
            if let Some(&d) = actions.get(&s.id) { s.dir = d; }
        }

        // Phase 2 – proposed new head positions
        let proposed: Vec<Pos> = self.snakes.iter().map(|s| {
            let (dx, dy) = s.dir.delta();
            s.head().translate(dx, dy)
        }).collect();

        // Phase 3 – eaters: head lands on a power source
        let eaters: HashSet<usize> = proposed.iter().enumerate()
            .filter(|(_, &h)| self.power.contains(&h))
            .map(|(i, _)| i)
            .collect();

        // Phase 4 – body obstacle set
        // non-eaters: exclude tail (it vacates); eaters: include tail (it stays)
        let mut body_cells: HashSet<Pos> = HashSet::new();
        for (idx, s) in self.snakes.iter().enumerate() {
            let end = if eaters.contains(&idx) { s.len() } else { s.len().saturating_sub(1) };
            for i in 0..end { body_cells.insert(s.body[i]); }
        }

        // Phase 5 – detect destroyed heads
        let mut head_destroyed: Vec<bool> = vec![false; n];
        for (idx, &h) in proposed.iter().enumerate() {
            if eaters.contains(&idx) { continue; }
            if !h.in_bounds(self.width, self.height) || self.is_platform(h) || body_cells.contains(&h) {
                head_destroyed[idx] = true;
            }
        }
        // head-to-head: two+ non-eating heads at same cell → both destroyed
        let mut head_map: HashMap<Pos, Vec<usize>> = HashMap::new();
        for (idx, &h) in proposed.iter().enumerate() {
            if !head_destroyed[idx] && !eaters.contains(&idx) {
                head_map.entry(h).or_default().push(idx);
            }
        }
        for (_, idxs) in &head_map {
            if idxs.len() > 1 { for &i in idxs { head_destroyed[i] = true; } }
        }

        // Phase 6 – apply movement
        let mut should_remove: Vec<bool> = vec![false; n];
        for (idx, s) in self.snakes.iter_mut().enumerate() {
            if head_destroyed[idx] {
                if s.len() >= 3 { s.body.pop_front(); }
                else             { should_remove[idx] = true; }
            } else {
                s.body.push_front(proposed[idx]);
                if !eaters.contains(&idx) { s.body.pop_back(); }
            }
        }

        // Phase 7 – remove consumed power sources
        for &idx in &eaters {
            if !head_destroyed[idx] { self.power.remove(&proposed[idx]); }
        }

        // Phase 8 – remove fully destroyed snakes
        let mut ri = 0usize;
        self.snakes.retain(|_| { let keep = !should_remove[ri]; ri += 1; keep });

        // Phase 9 – gravity
        self.apply_gravity();

        // Phase 10 – border removal
        let (w, h) = (self.width, self.height);
        self.snakes.retain(|s| s.body.iter().all(|&p| p.in_bounds(w, h)));

        self.turn += 1;
    }

    /// Drop all unsupported snakes simultaneously until the state is stable.
    pub fn apply_gravity(&mut self) {
        loop {
            let occupied: HashSet<Pos> = self.snakes.iter()
                .flat_map(|s| s.body.iter().copied()).collect();

            let mut to_fall: Vec<usize> = Vec::new();
            for (idx, s) in self.snakes.iter().enumerate() {
                let own: HashSet<Pos> = s.body.iter().copied().collect();
                let grounded = s.body.iter().any(|&p| {
                    let below = p.translate(0, 1);
                    p.y + 1 >= self.height
                        || self.is_platform(below)
                        || self.power.contains(&below)
                        || (occupied.contains(&below) && !own.contains(&below))
                });
                if !grounded { to_fall.push(idx); }
            }

            if to_fall.is_empty() { break; }
            for &idx in &to_fall {
                for p in &mut self.snakes[idx].body { p.y += 1; }
            }
        }
    }

    // --------------------------------------------------------
    // BFS utilities
    // --------------------------------------------------------

    /// Obstacle set for pathfinding: all body parts except tails (which vacate).
    pub fn build_obstacles(&self) -> HashSet<Pos> {
        let mut obs = HashSet::new();
        for s in &self.snakes {
            for i in 0..s.len().saturating_sub(1) { obs.insert(s.body[i]); }
        }
        obs
    }

    /// First direction from `start` toward any position in `targets`, or None.
    pub fn bfs_first_step(&self, start: Pos, targets: &HashSet<Pos>, obs: &HashSet<Pos>) -> Option<Dir> {
        if targets.is_empty() { return None; }
        let mut first: HashMap<Pos, Dir> = HashMap::new();
        let mut q: VecDeque<Pos> = VecDeque::new();
        for d in Dir::all() {
            let (dx, dy) = d.delta();
            let n = start.translate(dx, dy);
            if !n.in_bounds(self.width, self.height) || self.is_platform(n) || obs.contains(&n) || first.contains_key(&n) { continue; }
            first.insert(n, d); q.push_back(n);
        }
        while let Some(pos) = q.pop_front() {
            if targets.contains(&pos) { return Some(first[&pos]); }
            let fd = first[&pos];
            for d in Dir::all() {
                let (dx, dy) = d.delta();
                let n = pos.translate(dx, dy);
                if !n.in_bounds(self.width, self.height) || self.is_platform(n) || obs.contains(&n) || first.contains_key(&n) { continue; }
                first.insert(n, fd); q.push_back(n);
            }
        }
        None
    }

    /// BFS distance from `start` to the nearest target, or i32::MAX if unreachable.
    pub fn bfs_dist(&self, start: Pos, targets: &HashSet<Pos>, obs: &HashSet<Pos>) -> i32 {
        if targets.is_empty() { return i32::MAX; }
        if targets.contains(&start) { return 0; }
        let mut dist: HashMap<Pos, i32> = HashMap::new();
        let mut q: VecDeque<Pos> = VecDeque::new();
        dist.insert(start, 0); q.push_back(start);
        while let Some(pos) = q.pop_front() {
            let d = dist[&pos];
            for dir in Dir::all() {
                let (dx, dy) = dir.delta();
                let n = pos.translate(dx, dy);
                if !n.in_bounds(self.width, self.height) || self.is_platform(n) || obs.contains(&n) || dist.contains_key(&n) { continue; }
                if targets.contains(&n) { return d + 1; }
                dist.insert(n, d + 1); q.push_back(n);
            }
        }
        i32::MAX
    }

    /// Returns true if position `p` is supported from below (platform, power source,
    /// or bottom edge). Does not check snake bodies — use for static support checks.
    #[inline]
    pub fn is_grounded_cell(&self, p: Pos) -> bool {
        let below = p.translate(0, 1);
        p.y + 1 >= self.height
            || self.is_platform(below)
            || self.power.contains(&below)
    }

    /// Gravity-aware BFS distance: intermediate cells must be grounded (supported
    /// from below by platform/power/bottom-edge). Targets (food) are always reachable
    /// regardless of support — the snake body provides support en-route in practice,
    /// but this approximation correctly penalises paths through open air.
    pub fn bfs_dist_grounded(&self, start: Pos, targets: &HashSet<Pos>, obs: &HashSet<Pos>) -> i32 {
        if targets.is_empty() { return i32::MAX; }
        if targets.contains(&start) { return 0; }
        let mut dist: HashMap<Pos, i32> = HashMap::new();
        let mut q: VecDeque<Pos> = VecDeque::new();
        dist.insert(start, 0); q.push_back(start);
        while let Some(pos) = q.pop_front() {
            let d = dist[&pos];
            for dir in Dir::all() {
                let (dx, dy) = dir.delta();
                let n = pos.translate(dx, dy);
                if !n.in_bounds(self.width, self.height) || self.is_platform(n) || obs.contains(&n) || dist.contains_key(&n) { continue; }
                // Allow targets (food) through; require intermediate cells to be grounded.
                if !targets.contains(&n) && !self.is_grounded_cell(n) { continue; }
                if targets.contains(&n) { return d + 1; }
                dist.insert(n, d + 1); q.push_back(n);
            }
        }
        i32::MAX
    }

    /// Gravity-aware first-step BFS: returns direction toward nearest reachable
    /// grounded target, or None. Same grounding restriction as bfs_dist_grounded.
    pub fn bfs_first_step_grounded(&self, start: Pos, targets: &HashSet<Pos>, obs: &HashSet<Pos>) -> Option<Dir> {
        if targets.is_empty() { return None; }
        let mut first: HashMap<Pos, Dir> = HashMap::new();
        let mut q: VecDeque<Pos> = VecDeque::new();
        // First step: allow any reachable neighbour (snake body still provides support here)
        for d in Dir::all() {
            let (dx, dy) = d.delta();
            let n = start.translate(dx, dy);
            if !n.in_bounds(self.width, self.height) || self.is_platform(n) || obs.contains(&n) || first.contains_key(&n) { continue; }
            first.insert(n, d); q.push_back(n);
        }
        while let Some(pos) = q.pop_front() {
            if targets.contains(&pos) { return Some(first[&pos]); }
            let fd = first[&pos];
            for d in Dir::all() {
                let (dx, dy) = d.delta();
                let n = pos.translate(dx, dy);
                if !n.in_bounds(self.width, self.height) || self.is_platform(n) || obs.contains(&n) || first.contains_key(&n) { continue; }
                if !targets.contains(&n) && !self.is_grounded_cell(n) { continue; }
                first.insert(n, fd); q.push_back(n);
            }
        }
        None
    }
}

// ============================================================
// Visualization (ASCII render for debugging / simulate binary)
// ============================================================

pub fn visualize(state: &GameState) -> String {
    let w = state.width as usize;
    let h = state.height as usize;
    let mut display: Vec<Vec<char>> = vec![vec!['.'; w]; h];

    for y in 0..h { for x in 0..w {
        if state.grid[y * w + x] { display[y][x] = '#'; }
    }}
    for &p in &state.power {
        if p.in_bounds(state.width, state.height) {
            display[p.y as usize][p.x as usize] = '*';
        }
    }
    for s in &state.snakes {
        for (i, &p) in s.body.iter().enumerate() {
            if p.in_bounds(state.width, state.height) {
                display[p.y as usize][p.x as usize] =
                    if i == 0 { (b'A' + s.id) as char } else { (b'a' + s.id) as char };
            }
        }
    }

    format!(
        "Turn {:3} | P0: {:3} parts | P1: {:3} parts | food: {}\n{}",
        state.turn, state.score(0), state.score(1), state.power.len(),
        display.iter().map(|r| r.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
    )
}
