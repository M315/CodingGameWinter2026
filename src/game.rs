use std::collections::{HashSet, VecDeque};

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
        self.turn >= 200 || self.power.is_empty() || self.snakes.is_empty()
    }

    // --------------------------------------------------------
    // Step: advance the game state by one turn in-place
    // --------------------------------------------------------

    pub fn step(&mut self, actions: &std::collections::HashMap<u8, Dir>) {
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

        // Phase 3 – eaters: head lands on a power source (flat bool array, max 32 snakes)
        let mut eaters = [false; 32];
        for (i, &h) in proposed.iter().enumerate() {
            if self.power.contains(&h) { eaters[i] = true; }
        }

        // Phase 4 – body obstacle grid (flat Vec<bool> instead of HashSet)
        let grid_size = (self.width * self.height) as usize;
        let w = self.width as usize;
        let mut body_cells = vec![false; grid_size];
        for (idx, s) in self.snakes.iter().enumerate() {
            let end = if eaters[idx] { s.len() } else { s.len().saturating_sub(1) };
            for i in 0..end {
                let p = s.body[i];
                if p.in_bounds(self.width, self.height) {
                    body_cells[p.y as usize * w + p.x as usize] = true;
                }
            }
        }

        // Phase 5 – detect destroyed heads
        let mut head_destroyed: Vec<bool> = vec![false; n];
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
        // head-to-head: O(n²), n ≤ 8 — cheaper than HashMap allocation
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
        let mut should_remove: Vec<bool> = vec![false; n];
        for (idx, s) in self.snakes.iter_mut().enumerate() {
            if head_destroyed[idx] {
                if s.len() >= 3 { s.body.pop_front(); }
                else             { should_remove[idx] = true; }
            } else {
                s.body.push_front(proposed[idx]);
                if !eaters[idx] { s.body.pop_back(); }
            }
        }

        // Phase 7 – remove consumed power sources
        for (idx, &is_eater) in eaters.iter().enumerate().take(n) {
            if is_eater && !head_destroyed[idx] {
                self.power.remove(&proposed[idx]);
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

        // Build power grid once (power set is small, ≤~20 items)
        let mut pow = vec![false; size];
        for &p in &self.power {
            if p.in_bounds(self.width, self.height) {
                pow[p.y as usize * w + p.x as usize] = true;
            }
        }

        // snake_at: cell → snake index (u8::MAX = empty). Pre-allocated, cleared each iter.
        let mut snake_at = vec![u8::MAX; size];

        loop {
            // Rebuild snake_at for this iteration
            snake_at.fill(u8::MAX);
            for (idx, s) in self.snakes.iter().enumerate() {
                for &p in &s.body {
                    if p.in_bounds(self.width, self.height) {
                        snake_at[p.y as usize * w + p.x as usize] = idx as u8;
                    }
                }
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
            for &idx in &to_fall {
                for p in &mut self.snakes[idx].body { p.y += 1; }
            }
        }
    }

    // --------------------------------------------------------
    // Grid helpers
    // --------------------------------------------------------

    /// Flat bool grid: true where a power source lives.
    pub fn power_grid(&self) -> Vec<bool> {
        let mut g = vec![false; (self.width * self.height) as usize];
        for &p in &self.power {
            if p.in_bounds(self.width, self.height) {
                g[self.cell_idx(p)] = true;
            }
        }
        g
    }

    /// Flat grid: each cell holds the snake index (in self.snakes) that occupies it,
    /// or u8::MAX if empty. Includes ALL body parts.
    pub fn snake_grid(&self) -> Vec<u8> {
        let mut g = vec![u8::MAX; (self.width * self.height) as usize];
        for (idx, s) in self.snakes.iter().enumerate() {
            for &p in &s.body {
                if p.in_bounds(self.width, self.height) {
                    g[self.cell_idx(p)] = idx as u8;
                }
            }
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
        for s in &self.snakes {
            for i in 0..s.len().saturating_sub(1) {
                let p = s.body[i];
                if p.in_bounds(self.width, self.height) {
                    obs[p.y as usize * w + p.x as usize] = true;
                }
            }
        }
        obs
    }

    /// True if position at flat index `ci` is statically grounded
    /// (bottom edge, platform below, or power source below).
    /// `pow` is the flat power-source grid.
    #[inline]
    fn is_grounded_cell_ci(&self, ci: usize, w: usize, pow: &[bool]) -> bool {
        let below_ci = ci + w;
        below_ci >= self.grid.len() // bottom edge
            || self.grid[below_ci] // platform
            || pow[below_ci]       // power source
    }

    /// True if position `p` is supported from below (for external callers).
    #[inline]
    pub fn is_grounded_cell(&self, p: Pos) -> bool {
        let below = p.translate(0, 1);
        p.y + 1 >= self.height
            || self.is_platform(below)
            || self.power.contains(&below)
    }

    /// First direction from `start` toward any target, or None.
    /// `targets` and `obs` are flat bool grids (true = target / blocked).
    pub fn bfs_first_step(&self, start: Pos, targets: &[bool], obs: &[bool]) -> Option<Dir> {
        let w = self.width as usize;
        let size = w * self.height as usize;
        // first_dir[ci] = direction index (0-3) to reach ci from start; u8::MAX = unvisited
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
        // dist[ci] = -1 means unvisited
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
    use super::*;
    use std::collections::HashMap;

    // ── helpers ──────────────────────────────────────────────────────

    /// Open map: no walls, gravity pulls snakes to row h-1.
    fn open(w: i32, h: i32) -> GameState {
        GameState::new(w, h, vec![false; (w * h) as usize])
    }

    /// Map with a solid platform row at the bottom (y = h-1).
    /// Cells at y = h-2 are always grounded — good for movement tests.
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
        // floored map so gravity doesn't interfere
        let mut s = floored(10, 10);
        // head at (3,8), body going right along floor row y=8
        s.snakes.push(snake(0, 0, &[(3, 8), (4, 8), (5, 8)]));
        s.step(&acts(&[(0, Dir::Left)]));
        assert_eq!(s.snakes[0].head(), Pos::new(2, 8));
        assert_eq!(s.snakes[0].len(), 3);
    }

    #[test]
    fn test_eating_grows_and_consumes_power() {
        let mut s = floored(10, 10);
        s.snakes.push(snake(0, 0, &[(3, 8), (4, 8), (5, 8)]));
        s.power.insert(Pos::new(2, 8)); // one step to the left
        s.step(&acts(&[(0, Dir::Left)]));
        assert_eq!(s.snakes[0].head(), Pos::new(2, 8));
        assert_eq!(s.snakes[0].len(), 4); // grew
        assert!(s.power.is_empty());      // food consumed
    }

    // ── step: head destruction ───────────────────────────────────────

    #[test]
    fn test_head_into_wall_long_snake_loses_head() {
        // len >= 3 hits OOB → loses front, survives with len-1
        let mut s = open(10, 10);
        // Snake on bottom row, head at x=0 moving left → OOB
        s.snakes.push(snake(0, 0, &[(0, 9), (1, 9), (2, 9)]));
        s.step(&acts(&[(0, Dir::Left)]));
        assert_eq!(s.snakes[0].len(), 2);
        assert_eq!(s.snakes[0].head(), Pos::new(1, 9));
    }

    #[test]
    fn test_head_into_wall_short_snake_destroyed() {
        // len < 3 hits OOB → removed entirely
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(0, 9), (1, 9)]));
        s.step(&acts(&[(0, Dir::Left)]));
        assert!(s.snakes.is_empty());
    }

    #[test]
    fn test_head_into_body_destroys_attacker() {
        let mut s = floored(10, 10);
        // Snake 0 horizontal, Snake 1 drives its head into snake 0's body
        s.snakes.push(snake(0, 0, &[(5, 8), (4, 8), (3, 8)]));
        s.snakes.push(snake(1, 1, &[(5, 7), (5, 6), (5, 5)])); // moving Down into (5,8)
        s.step(&acts(&[(0, Dir::Right), (1, Dir::Down)]));
        // Snake 1 head destroyed; len was 3 → len becomes 2
        let s1 = s.snakes.iter().find(|sn| sn.id == 1).unwrap();
        assert_eq!(s1.len(), 2);
    }

    #[test]
    fn test_head_on_head_both_shortened() {
        let mut s = floored(10, 10);
        // Snake 0 moving Right, snake 1 moving Left — meet at (5, 8)
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
        // Open 10×10 map, no platforms. Snake floats at y=2-4; should fall to y=7-9.
        let mut s = open(10, 10);
        s.snakes.push(snake(0, 0, &[(5, 2), (5, 3), (5, 4)]));
        s.apply_gravity();
        // bottom part must reach y=9 (bottom edge, 9+1 >= 10)
        assert_eq!(s.snakes[0].body[2], Pos::new(5, 9));
        assert_eq!(s.snakes[0].body[0], Pos::new(5, 7));
    }

    #[test]
    fn test_gravity_rests_on_platform() {
        // Snake sits directly on a platform — should not move.
        let mut s = open(10, 10);
        // Platform at row y=5
        for x in 0..10usize { s.grid[5 * 10 + x] = true; }
        // Snake at y=2-4; tail at y=4, below is y=5 (platform) → grounded
        s.snakes.push(snake(0, 0, &[(5, 2), (5, 3), (5, 4)]));
        s.apply_gravity();
        assert_eq!(s.snakes[0].body[2], Pos::new(5, 4)); // didn't move
    }

    #[test]
    fn test_gravity_snake_stacks_on_snake() {
        // Snake B is on bottom row (grounded). Snake A is floating above B.
        // Snake A should fall and come to rest on top of snake B.
        let mut s = open(10, 5);
        // B: grounded at bottom row (y=4, 4+1=5 >= 5 → bottom edge)
        s.snakes.push(snake(1, 1, &[(5, 4), (4, 4), (3, 4)]));
        // A: floating at y=1, above B's cell (5,4) — would need to land at y=3
        s.snakes.push(snake(0, 0, &[(5, 1), (5, 2), (5, 3)]));
        // After gravity: A's tail at (5,3) has below (5,4) occupied by B → grounded
        // A should NOT fall (already grounded via B's body)
        s.apply_gravity();
        let b = s.snakes.iter().find(|sn| sn.id == 1).unwrap();
        let a = s.snakes.iter().find(|sn| sn.id == 0).unwrap();
        assert_eq!(b.body[0], Pos::new(5, 4)); // B unchanged
        assert_eq!(a.body[2], Pos::new(5, 3)); // A's tail rests on B's head
    }

    #[test]
    fn test_gravity_snake_falls_off_bottom() {
        // Snake falls out of the 10-row map (bottom = y=9, falls past it)
        // Actually fall is capped at bottom edge. Let me test border removal instead:
        // A 3-tall map, snake at y=0 falls to y=2 (the only valid row).
        let mut s = open(5, 3);
        s.snakes.push(snake(0, 0, &[(2, 0)])); // single-cell snake floating
        // Single cell at y=0 — below y=1 is not bottom edge (h=3 so bottom edge is y=2).
        // Will fall to y=2 where y+1=3 >= 3 → grounded.
        s.apply_gravity();
        assert_eq!(s.snakes[0].body[0], Pos::new(2, 2));
    }

    // ── BFS utilities ────────────────────────────────────────────────

    #[test]
    fn test_bfs_dist_open_space() {
        let s = open(10, 10);
        let mut targets = vec![false; 100];
        targets[3 * 10 + 8] = true; // (8, 3)
        let obs = vec![false; 100];
        // Manhattan distance from (3,3) to (8,3) = 5
        assert_eq!(s.bfs_dist(Pos::new(3, 3), &targets, &obs), 5);
    }

    #[test]
    fn test_bfs_dist_wall_blocks_returns_max() {
        let mut s = open(10, 10);
        // Vertical wall at x=5, all rows
        for y in 0..10usize { s.grid[y * 10 + 5] = true; }
        let mut targets = vec![false; 100];
        targets[3 * 10 + 8] = true; // (8, 3) — behind the wall
        let obs = vec![false; 100];
        // No top/bottom gap so completely blocked
        // Actually BFS can go around y=0 or y=9 rows...
        // Let's also wall the top and bottom to guarantee no path
        for x in 0..10usize {
            s.grid[x] = true;       // top row
            s.grid[9 * 10 + x] = true; // bottom row
        }
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
        targets[5 * 10 + 5] = true; // food at (5, 5)
        let obs = vec![false; 100];
        // From (2, 5): food is to the Right
        let dir = s.bfs_first_step(Pos::new(2, 5), &targets, &obs);
        assert_eq!(dir, Some(Dir::Right));
    }

    #[test]
    fn test_bfs_first_step_no_food_returns_none() {
        let s = open(10, 10);
        let targets = vec![false; 100]; // no food
        let obs = vec![false; 100];
        assert_eq!(s.bfs_first_step(Pos::new(5, 5), &targets, &obs), None);
    }

    // ── grid helpers ─────────────────────────────────────────────────

    #[test]
    fn test_build_obstacles_excludes_tail() {
        let mut s = open(10, 10);
        // head=(2,5), body=(3,5), tail=(4,5)
        s.snakes.push(snake(0, 0, &[(2, 5), (3, 5), (4, 5)]));
        let obs = s.build_obstacles();
        assert!(obs[5 * 10 + 2]);  // head blocked
        assert!(obs[5 * 10 + 3]);  // body blocked
        assert!(!obs[5 * 10 + 4]); // tail NOT blocked (it vacates)
    }

    #[test]
    fn test_power_grid_matches_power_set() {
        let mut s = open(10, 10);
        s.power.insert(Pos::new(3, 4));
        s.power.insert(Pos::new(7, 1));
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
        assert_eq!(sg[5 * 10 + 2], 0); // snake index 0
        assert_eq!(sg[5 * 10 + 3], 0);
        assert_eq!(sg[5 * 10 + 4], 0);
        assert_eq!(sg[7 * 10 + 7], 1); // snake index 1
        assert_eq!(sg[8 * 10 + 7], 1);
        assert_eq!(sg[0], u8::MAX);    // empty cell
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
