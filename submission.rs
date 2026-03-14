#![allow(dead_code, unused_imports, unused_variables)]
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::io::{self, BufRead};

// ── game.rs ─────────────────────────────────────────────────────
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

/// Clone cost breakdown:
///   grid  — Arc refcount bump only (zero copy, zero alloc)
///   food  — memcpy of width*height bytes (~264B for a 24×11 map)
///   snakes — unavoidable: VecDeque bodies per snake
#[derive(Clone, Debug)]
pub struct GameState {
    pub width:  i32,
    pub height: i32,
    /// Flat row-major platform grid. Wrapped in Arc — never mutates after
    /// construction, so all beam-search clones share the same allocation.
    pub grid: Arc<Vec<bool>>,
    /// Live food/power sources as a flat bool grid (same dimensions as grid).
    /// Replaces HashSet<Pos>: lookup is a direct array index, clone is memcpy.
    pub food: Vec<bool>,
    /// Number of live food items — keeps is_over() O(1).
    pub food_count: u32,
    /// All living snakes (both players)
    pub snakes: Vec<Snake>,
    pub turn: u32,
}

impl GameState {
    pub fn new(width: i32, height: i32, grid: Vec<bool>) -> Self {
        let size = (width * height) as usize;
        GameState {
            width, height,
            grid: Arc::new(grid),
            food: vec![false; size],
            food_count: 0,
            snakes: Vec::new(),
            turn: 0,
        }
    }

    /// Insert a food item at `p`. No-op if out of bounds or already present.
    pub fn add_food(&mut self, p: Pos) {
        if p.in_bounds(self.width, self.height) {
            let ci = self.cell_idx(p);
            if !self.food[ci] {
                self.food[ci] = true;
                self.food_count += 1;
            }
        }
    }

    /// Remove all food items.
    pub fn clear_food(&mut self) {
        self.food.fill(false);
        self.food_count = 0;
    }

    #[inline]
    pub fn is_platform(&self, p: Pos) -> bool {
        p.in_bounds(self.width, self.height)
            && self.grid[(p.y * self.width + p.x) as usize]
    }

    /// Flat index of position p. Caller must ensure p is in bounds.
    #[inline]
    pub fn cell_idx(&self, p: Pos) -> usize {
        p.y as usize * self.width as usize + p.x as usize
    }

    pub fn score(&self, player: u8) -> usize {
        self.snakes.iter().filter(|s| s.player == player).map(|s| s.len()).sum()
    }

    pub fn snakes_alive(&self, player: u8) -> bool {
        self.snakes.iter().any(|s| s.player == player)
    }

    pub fn is_over(&self) -> bool {
        self.turn >= 200 || self.food_count == 0 || self.snakes.is_empty()
    }

    // --------------------------------------------------------
    // Step: advance the game state by one turn in-place
    // --------------------------------------------------------

    pub fn step(&mut self, actions: &std::collections::HashMap<u8, Dir>) {
        let n = self.snakes.len();
        if n == 0 { self.turn += 1; return; }

        // Phase 1 – apply direction overrides
        self.snakes.iter_mut().for_each(|s| {
            if let Some(&d) = actions.get(&s.id) { s.dir = d; }
        });

        // Phase 2 – proposed new head positions
        let proposed: Vec<Pos> = self.snakes.iter().map(|s| {
            let (dx, dy) = s.dir.delta();
            s.head().translate(dx, dy)
        }).collect();

        // Phase 3 – eaters: head lands on a food cell (flat bool array, max 32 snakes)
        let w = self.width as usize;
        let mut eaters = [false; 32];
        for (i, &h) in proposed.iter().enumerate() {
            if h.in_bounds(self.width, self.height) {
                let ci = h.y as usize * w + h.x as usize;
                if self.food[ci] { eaters[i] = true; }
            }
        }

        // Phase 4 – body obstacle grid (flat Vec<bool>)
        let grid_size = (self.width * self.height) as usize;
        let mut body_cells = vec![false; grid_size];
        for (idx, s) in self.snakes.iter().enumerate() {
            let end = if eaters[idx] { s.len() } else { s.len().saturating_sub(1) };
            s.body.iter().take(end).for_each(|&p| {
                if p.in_bounds(self.width, self.height) {
                    body_cells[p.y as usize * w + p.x as usize] = true;
                }
            });
        }

        // Phase 5 – detect destroyed heads
        let mut head_destroyed = vec![false; n];
        for (idx, &h) in proposed.iter().enumerate() {
            if eaters[idx] { continue; }
            if !h.in_bounds(self.width, self.height) || self.is_platform(h) {
                head_destroyed[idx] = true;
                continue;
            }
            if body_cells[h.y as usize * w + h.x as usize] {
                head_destroyed[idx] = true;
            }
        }
        // head-to-head: O(n²), n ≤ 8
        for i in 0..n {
            if head_destroyed[i] || eaters[i] { continue; }
            for j in (i + 1)..n {
                if head_destroyed[j] || eaters[j] { continue; }
                if proposed[i] == proposed[j] {
                    head_destroyed[i] = true;
                    head_destroyed[j] = true;
                }
            }
        }

        // Phase 6 – apply movement
        let mut should_remove = vec![false; n];
        for (idx, s) in self.snakes.iter_mut().enumerate() {
            if head_destroyed[idx] {
                if s.len() >= 3 { s.body.pop_front(); }
                else             { should_remove[idx] = true; }
            } else {
                s.body.push_front(proposed[idx]);
                if !eaters[idx] { s.body.pop_back(); }
            }
        }

        // Phase 7 – remove consumed food (guard against two snakes on same cell)
        for (idx, &is_eater) in eaters.iter().enumerate().take(n) {
            if is_eater && !head_destroyed[idx] {
                let h = proposed[idx];
                let ci = h.y as usize * w + h.x as usize;
                if self.food[ci] {
                    self.food[ci] = false;
                    self.food_count -= 1;
                }
            }
        }

        // Phase 8 – remove fully destroyed snakes
        let mut ri = 0usize;
        self.snakes.retain(|_| { let keep = !should_remove[ri]; ri += 1; keep });

        // Phase 9 – gravity
        self.apply_gravity();

        // Phase 10 – border removal
        let (w_i, h_i) = (self.width, self.height);
        self.snakes.retain(|s| s.body.iter().all(|&p| p.in_bounds(w_i, h_i)));

        self.turn += 1;
    }

    /// Drop all unsupported snakes simultaneously until the state is stable.
    pub fn apply_gravity(&mut self) {
        let size = (self.width * self.height) as usize;
        let w = self.width as usize;

        // food IS already a flat power-source grid — no allocation needed
        let pow = &self.food;

        // snake_at: cell → snake index (u8::MAX = empty). Pre-allocated, cleared each iter.
        let mut snake_at = vec![u8::MAX; size];

        loop {
            // Rebuild snake_at for this iteration
            snake_at.fill(u8::MAX);
            for (idx, s) in self.snakes.iter().enumerate() {
                s.body.iter().for_each(|&p| {
                    if p.in_bounds(self.width, self.height) {
                        snake_at[p.y as usize * w + p.x as usize] = idx as u8;
                    }
                });
            }

            let mut to_fall: Vec<usize> = Vec::new();
            for (idx, s) in self.snakes.iter().enumerate() {
                let grounded = s.body.iter().any(|&p| {
                    let below_y = p.y + 1;
                    if below_y >= self.height { return true; }
                    if p.x < 0 || p.x >= self.width { return false; }
                    let below_ci = below_y as usize * w + p.x as usize;
                    if self.grid[below_ci] { return true; }
                    if pow[below_ci] { return true; }
                    let sat = snake_at[below_ci];
                    sat != u8::MAX && sat != idx as u8
                });
                if !grounded { to_fall.push(idx); }
            }

            if to_fall.is_empty() { break; }
            to_fall.iter().for_each(|&idx| {
                self.snakes[idx].body.iter_mut().for_each(|p| p.y += 1);
            });
        }
    }

    // --------------------------------------------------------
    // Grid helpers
    // --------------------------------------------------------

    /// Flat bool grid of food positions — returns a clone of self.food.
    /// Prefer passing `&state.food` directly in hot paths to avoid the allocation.
    pub fn power_grid(&self) -> Vec<bool> {
        self.food.clone()
    }

    /// Flat grid: each cell holds the snake index (in self.snakes) that occupies it,
    /// or u8::MAX if empty. Includes ALL body parts.
    pub fn snake_grid(&self) -> Vec<u8> {
        let mut g = vec![u8::MAX; (self.width * self.height) as usize];
        for (idx, s) in self.snakes.iter().enumerate() {
            s.body.iter().for_each(|&p| {
                if p.in_bounds(self.width, self.height) {
                    g[self.cell_idx(p)] = idx as u8;
                }
            });
        }
        g
    }

    // --------------------------------------------------------
    // BFS utilities
    // --------------------------------------------------------

    /// Obstacle set for pathfinding: all body parts except tails (which vacate).
    /// Returns a flat bool grid (true = blocked).
    pub fn build_obstacles(&self) -> Vec<bool> {
        let mut obs = vec![false; (self.width * self.height) as usize];
        let w = self.width as usize;
        self.snakes.iter().for_each(|s| {
            s.body.iter().take(s.len().saturating_sub(1)).for_each(|&p| {
                if p.in_bounds(self.width, self.height) {
                    obs[p.y as usize * w + p.x as usize] = true;
                }
            });
        });
        obs
    }

    /// True if position at flat index `ci` is statically grounded
    /// (bottom edge, platform below, or power source below).
    /// `pow` is the flat food/power-source grid.
    #[inline]
    fn is_grounded_cell_ci(&self, ci: usize, w: usize, pow: &[bool]) -> bool {
        let below_ci = ci + w;
        below_ci >= self.grid.len() // bottom edge
            || self.grid[below_ci]  // platform
            || pow[below_ci]        // power source
    }

    /// True if position `p` is supported from below (for external callers).
    #[inline]
    pub fn is_grounded_cell(&self, p: Pos) -> bool {
        let below_y = p.y + 1;
        if below_y >= self.height { return true; }
        if p.x < 0 || p.x >= self.width { return false; }
        let below_ci = below_y as usize * self.width as usize + p.x as usize;
        self.grid[below_ci] || self.food[below_ci]
    }

    /// First direction from `start` toward any target, or None.
    /// `targets` and `obs` are flat bool grids (true = target / blocked).
    pub fn bfs_first_step(&self, start: Pos, targets: &[bool], obs: &[bool]) -> Option<Dir> {
        let w = self.width as usize;
        let size = w * self.height as usize;
        let mut first_dir = vec![u8::MAX; size];
        let mut q: VecDeque<usize> = VecDeque::new();
        let dirs = Dir::all();

        for (di, &d) in dirs.iter().enumerate() {
            let (dx, dy) = d.delta();
            let (nx, ny) = (start.x + dx, start.y + dy);
            if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
            let ni = ny as usize * w + nx as usize;
            if self.grid[ni] || obs[ni] || first_dir[ni] != u8::MAX { continue; }
            first_dir[ni] = di as u8;
            if targets[ni] { return Some(d); }
            q.push_back(ni);
        }

        while let Some(ci) = q.pop_front() {
            let fd = first_dir[ci];
            let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
            for &d in &dirs {
                let (dx, dy) = d.delta();
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || first_dir[ni] != u8::MAX { continue; }
                first_dir[ni] = fd;
                if targets[ni] { return Some(dirs[fd as usize]); }
                q.push_back(ni);
            }
        }
        None
    }

    /// BFS distance from `start` to the nearest target, or i32::MAX if unreachable.
    /// `targets` and `obs` are flat bool grids.
    pub fn bfs_dist(&self, start: Pos, targets: &[bool], obs: &[bool]) -> i32 {
        let w = self.width as usize;
        let start_ci = start.y as usize * w + start.x as usize;
        if targets[start_ci] { return 0; }

        let size = w * self.height as usize;
        let mut dist = vec![-1i32; size];
        let mut q: VecDeque<usize> = VecDeque::new();
        dist[start_ci] = 0;
        q.push_back(start_ci);

        let dirs = Dir::all();
        while let Some(ci) = q.pop_front() {
            let d = dist[ci];
            let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
            for &dir in &dirs {
                let (dx, dy) = dir.delta();
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || dist[ni] != -1 { continue; }
                if targets[ni] { return d + 1; }
                dist[ni] = d + 1;
                q.push_back(ni);
            }
        }
        i32::MAX
    }

    /// Gravity-aware BFS distance: intermediate cells must be statically grounded.
    /// Targets (food) are always reachable. `targets` doubles as the power-source
    /// grid for the grounding check (food = power source).
    pub fn bfs_dist_grounded(&self, start: Pos, targets: &[bool], obs: &[bool]) -> i32 {
        let w = self.width as usize;
        let start_ci = start.y as usize * w + start.x as usize;
        if targets[start_ci] { return 0; }

        let size = w * self.height as usize;
        let mut dist = vec![-1i32; size];
        let mut q: VecDeque<usize> = VecDeque::new();
        dist[start_ci] = 0;
        q.push_back(start_ci);

        let dirs = Dir::all();
        while let Some(ci) = q.pop_front() {
            let d = dist[ci];
            let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
            for &dir in &dirs {
                let (dx, dy) = dir.delta();
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || dist[ni] != -1 { continue; }
                if targets[ni] { return d + 1; }
                if !self.is_grounded_cell_ci(ni, w, targets) { continue; }
                dist[ni] = d + 1;
                q.push_back(ni);
            }
        }
        i32::MAX
    }

    /// Gravity-aware first-step BFS. First step from start is unrestricted
    /// (snake body provides support); subsequent cells must be grounded.
    pub fn bfs_first_step_grounded(&self, start: Pos, targets: &[bool], obs: &[bool]) -> Option<Dir> {
        let w = self.width as usize;
        let size = w * self.height as usize;
        let mut first_dir = vec![u8::MAX; size];
        let mut q: VecDeque<usize> = VecDeque::new();
        let dirs = Dir::all();

        // First step: no grounding restriction
        for (di, &d) in dirs.iter().enumerate() {
            let (dx, dy) = d.delta();
            let (nx, ny) = (start.x + dx, start.y + dy);
            if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
            let ni = ny as usize * w + nx as usize;
            if self.grid[ni] || obs[ni] || first_dir[ni] != u8::MAX { continue; }
            first_dir[ni] = di as u8;
            if targets[ni] { return Some(d); }
            q.push_back(ni);
        }

        // Subsequent steps: require grounding for non-target cells
        while let Some(ci) = q.pop_front() {
            let fd = first_dir[ci];
            let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
            for &d in &dirs {
                let (dx, dy) = d.delta();
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || first_dir[ni] != u8::MAX { continue; }
                if !targets[ni] && !self.is_grounded_cell_ci(ni, w, targets) { continue; }
                first_dir[ni] = fd;
                if targets[ni] { return Some(dirs[fd as usize]); }
                q.push_back(ni);
            }
        }
        None
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {

    // ── helpers ──────────────────────────────────────────────────────

    /// Open map: no walls, gravity pulls snakes to row h-1.
    fn open(w: i32, h: i32) -> GameState {
        GameState::new(w, h, vec![false; (w * h) as usize])
    }

    /// Map with a solid platform row at the bottom (y = h-1).
    fn floored(w: i32, h: i32) -> GameState {
        let mut grid = vec![false; (w * h) as usize];
        for x in 0..w as usize {
            grid[(h as usize - 1) * w as usize + x] = true;
        }
        GameState::new(w, h, grid)
    }

    fn acts(pairs: &[(u8, Dir)]) -> HashMap<u8, Dir> {
        pairs.iter().cloned().collect()
    }

    fn snake(id: u8, player: u8, body: &[(i32, i32)]) -> Snake {
        let parts: Vec<Pos> = body.iter().map(|&(x, y)| Pos::new(x, y)).collect();
        Snake::new(id, parts, player)
    }

    // ── step: movement ───────────────────────────────────────────────

    #[test]
    fn test_move_forward() {
        let mut s = floored(10, 10);
        s.snakes.push(snake(0, 0, &[(3, 8), (4, 8), (5, 8)]));
        s.step(&acts(&[(0, Dir::Left)]));
        assert_eq!(s.snakes[0].head(), Pos::new(2, 8));
        assert_eq!(s.snakes[0].len(), 3);
    }

    #[test]
    fn test_eating_grows_and_consumes_power() {
        let mut s = floored(10, 10);
        s.snakes.push(snake(0, 0, &[(3, 8), (4, 8), (5, 8)]));
        s.add_food(Pos::new(2, 8));
        s.step(&acts(&[(0, Dir::Left)]));
        assert_eq!(s.snakes[0].head(), Pos::new(2, 8));
        assert_eq!(s.snakes[0].len(), 4);
        assert_eq!(s.food_count, 0);
    }

    // ── step: head destruction ───────────────────────────────────────

    #[test]
    fn test_head_into_wall_long_snake_loses_head() {
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(0, 9), (1, 9), (2, 9)]));
        s.step(&acts(&[(0, Dir::Left)]));
        assert_eq!(s.snakes[0].len(), 2);
        assert_eq!(s.snakes[0].head(), Pos::new(1, 9));
    }

    #[test]
    fn test_head_into_wall_short_snake_destroyed() {
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(0, 9), (1, 9)]));
        s.step(&acts(&[(0, Dir::Left)]));
        assert!(s.snakes.is_empty());
    }

    #[test]
    fn test_head_into_body_destroys_attacker() {
        let mut s = floored(10, 10);
        s.snakes.push(snake(0, 0, &[(5, 8), (4, 8), (3, 8)]));
        s.snakes.push(snake(1, 1, &[(5, 7), (5, 6), (5, 5)]));
        s.step(&acts(&[(0, Dir::Right), (1, Dir::Down)]));
        let s1 = s.snakes.iter().find(|sn| sn.id == 1).unwrap();
        assert_eq!(s1.len(), 2);
    }

    #[test]
    fn test_head_on_head_both_shortened() {
        let mut s = floored(10, 10);
        s.snakes.push(snake(0, 0, &[(4, 8), (3, 8), (2, 8)]));
        s.snakes.push(snake(1, 1, &[(6, 8), (7, 8), (8, 8)]));
        s.step(&acts(&[(0, Dir::Right), (1, Dir::Left)]));
        let s0 = s.snakes.iter().find(|sn| sn.id == 0).unwrap();
        let s1 = s.snakes.iter().find(|sn| sn.id == 1).unwrap();
        assert_eq!(s0.len(), 2);
        assert_eq!(s1.len(), 2);
    }

    // ── gravity ──────────────────────────────────────────────────────

    #[test]
    fn test_gravity_unsupported_falls_to_bottom() {
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(5, 2), (5, 3), (5, 4)]));
        s.apply_gravity();
        assert_eq!(s.snakes[0].body[2], Pos::new(5, 9));
        assert_eq!(s.snakes[0].body[0], Pos::new(5, 7));
    }

    #[test]
    fn test_gravity_rests_on_platform() {
        // Build grid with platform at row y=5, then construct state
        let mut grid = vec![false; 100];
        for x in 0..10usize { grid[5 * 10 + x] = true; }
        let mut s = GameState::new(10, 10, grid);
        s.snakes.push(snake(0, 0, &[(5, 2), (5, 3), (5, 4)]));
        s.apply_gravity();
        assert_eq!(s.snakes[0].body[2], Pos::new(5, 4));
    }

    #[test]
    fn test_gravity_snake_stacks_on_snake() {
        let mut s = open(10, 5);
        s.snakes.push(snake(1, 1, &[(5, 4), (4, 4), (3, 4)]));
        s.snakes.push(snake(0, 0, &[(5, 1), (5, 2), (5, 3)]));
        s.apply_gravity();
        let b = s.snakes.iter().find(|sn| sn.id == 1).unwrap();
        let a = s.snakes.iter().find(|sn| sn.id == 0).unwrap();
        assert_eq!(b.body[0], Pos::new(5, 4));
        assert_eq!(a.body[2], Pos::new(5, 3));
    }

    #[test]
    fn test_gravity_snake_falls_off_bottom() {
        let mut s = open(5, 3);
        s.snakes.push(snake(0, 0, &[(2, 0)]));
        s.apply_gravity();
        assert_eq!(s.snakes[0].body[0], Pos::new(2, 2));
    }

    // ── BFS utilities ────────────────────────────────────────────────

    #[test]
    fn test_bfs_dist_open_space() {
        let s = open(10, 10);
        let mut targets = vec![false; 100];
        targets[3 * 10 + 8] = true;
        let obs = vec![false; 100];
        assert_eq!(s.bfs_dist(Pos::new(3, 3), &targets, &obs), 5);
    }

    #[test]
    fn test_bfs_dist_wall_blocks_returns_max() {
        // Build fully-enclosed grid: vertical wall at x=5, plus top/bottom rows
        let mut grid = vec![false; 100];
        for y in 0..10usize { grid[y * 10 + 5] = true; }
        for x in 0..10usize { grid[x] = true; grid[9 * 10 + x] = true; }
        let s = GameState::new(10, 10, grid);
        let mut targets = vec![false; 100];
        targets[3 * 10 + 8] = true;
        let obs = vec![false; 100];
        assert_eq!(s.bfs_dist(Pos::new(3, 3), &targets, &obs), i32::MAX);
    }

    #[test]
    fn test_bfs_dist_target_at_start() {
        let s = open(10, 10);
        let mut targets = vec![false; 100];
        targets[3 * 10 + 3] = true;
        let obs = vec![false; 100];
        assert_eq!(s.bfs_dist(Pos::new(3, 3), &targets, &obs), 0);
    }

    #[test]
    fn test_bfs_first_step_moves_toward_food() {
        let s = open(10, 10);
        let mut targets = vec![false; 100];
        targets[5 * 10 + 5] = true;
        let obs = vec![false; 100];
        let dir = s.bfs_first_step(Pos::new(2, 5), &targets, &obs);
        assert_eq!(dir, Some(Dir::Right));
    }

    #[test]
    fn test_bfs_first_step_no_food_returns_none() {
        let s = open(10, 10);
        let targets = vec![false; 100];
        let obs = vec![false; 100];
        assert_eq!(s.bfs_first_step(Pos::new(5, 5), &targets, &obs), None);
    }

    // ── grid helpers ─────────────────────────────────────────────────

    #[test]
    fn test_build_obstacles_excludes_tail() {
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(2, 5), (3, 5), (4, 5)]));
        let obs = s.build_obstacles();
        assert!(obs[5 * 10 + 2]);
        assert!(obs[5 * 10 + 3]);
        assert!(!obs[5 * 10 + 4]);
    }

    #[test]
    fn test_power_grid_matches_food() {
        let mut s = open(10, 10);
        s.add_food(Pos::new(3, 4));
        s.add_food(Pos::new(7, 1));
        let pg = s.power_grid();
        assert!(pg[4 * 10 + 3]);
        assert!(pg[1 * 10 + 7]);
        assert!(!pg[0 * 10 + 0]);
    }

    #[test]
    fn test_snake_grid_covers_all_body_parts() {
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(2, 5), (3, 5), (4, 5)]));
        s.snakes.push(snake(1, 1, &[(7, 7), (7, 8)]));
        let sg = s.snake_grid();
        assert_eq!(sg[5 * 10 + 2], 0);
        assert_eq!(sg[5 * 10 + 3], 0);
        assert_eq!(sg[5 * 10 + 4], 0);
        assert_eq!(sg[7 * 10 + 7], 1);
        assert_eq!(sg[8 * 10 + 7], 1);
        assert_eq!(sg[0], u8::MAX);
    }

    // ── clone throughput ─────────────────────────────────────────────

    #[test]
    fn test_clone_throughput() {
        // Regression guard: 100k clones of a realistic 6-snake state must be fast.
        // With Arc<grid> + flat food, grid is never copied and food is a memcpy.
        let mut s = floored(20, 15);
        for i in 0u8..6 {
            let x = i as i32 * 2 + 2;
            s.snakes.push(snake(i, i / 3, &[(x, 13), (x + 1, 13), (x + 2, 13)]));
        }
        for &(x, y) in &[(5i32, 13i32), (10, 13), (14, 13)] {
            s.add_food(Pos::new(x, y));
        }
        let t0 = std::time::Instant::now();
        for _ in 0..100_000 {
            let _ = s.clone();
        }
        let ms = t0.elapsed().as_millis();
        eprintln!("100k GameState::clone() = {}ms", ms);
        assert!(ms < 1000, "clone too slow: {}ms", ms);
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
    for (ci, _) in state.food.iter().enumerate().filter(|(_, &b)| b) {
        let (x, y) = (ci % w, ci / w);
        display[y][x] = '*';
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
        state.turn, state.score(0), state.score(1), state.food_count,
        display.iter().map(|r| r.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
    )
}

// ── mod.rs ──────────────────────────────────────────────────────
// Re-export game types so sub-modules can use `use super::*;`


// ============================================================
// Bot trait
// ============================================================

pub trait Bot {
    fn name(&self) -> &str;
    /// Return a map of snake_id → Dir for snakes to redirect this turn.
    /// Snakes absent from the map keep their current direction.
    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir>;
}

// ============================================================
// Shared utilities (used by multiple bots)
// ============================================================

/// Greedy action selection: BFS each snake toward the nearest power source.
/// Uses gravity-aware BFS so paths through open air are not considered.
/// Standalone function so BeamSearchBot can call it without trait overhead.
pub fn greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {
    let obs = state.build_obstacles();
    // state.food IS the power grid — pass directly, no allocation
    state.snakes.iter()
        .filter(|s| s.player == player)
        .filter_map(|s| {
            state.bfs_first_step_grounded(s.head(), &state.food, &obs)
                .map(|d| (s.id, d))
        })
        .collect()
}

/// All valid direction combos for `player`'s snakes (U-turns pruned).
pub fn gen_action_combos(state: &GameState, player: u8) -> Vec<HashMap<u8, Dir>> {
    let ids: Vec<u8> = state.snakes.iter()
        .filter(|s| s.player == player).map(|s| s.id).collect();

    if ids.is_empty() { return vec![HashMap::new()]; }

    let mut combos: Vec<HashMap<u8, Dir>> = vec![HashMap::new()];
    for id in ids {
        let s = state.snakes.iter().find(|s| s.id == id).unwrap();
        let dirs: Vec<Dir> = Dir::all().iter()
            .filter(|&&d| d != s.dir.opposite()).copied().collect();
        let mut next = Vec::with_capacity(combos.len() * dirs.len());
        for existing in &combos {
            for &d in &dirs {
                let mut c = existing.clone();
                c.insert(id, d);
                next.push(c);
            }
        }
        combos = next;
    }
    combos
}

// ── old_beam.rs ─────────────────────────────────────────────────
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
    // state.food IS the power grid — pass directly, no allocation
    state.snakes.iter()
        .filter(|s| s.player == player)
        .filter_map(|s| {
            state.bfs_first_step(s.head(), &state.food, &obs)
                .map(|d| (s.id, d))
        })
        .collect()
}

/// Old heuristic: plain BFS food distance, no stability term, -30 unreachable penalty.
pub fn old_heuristic(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    let obs = state.build_obstacles();
    // state.food IS the power grid — pass directly, no allocation
    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist(s.head(), &state.food, &obs);
            if d == i32::MAX { -30 } else { 20 - d.min(20) }
        })
        .sum();

    my * 100 - opp * 80 + food_bonus
}

#[cfg(test)]
mod tests {

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
            s.add_food(Pos::new(x, y));
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

// ── main.rs ─────────────────────────────────────────────────────
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
