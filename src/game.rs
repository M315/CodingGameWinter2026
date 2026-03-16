use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

/// Compact action map: `arr[snake_id]` = direction for that snake this turn.
/// Indexed by `snake.id as usize`; `None` means "keep current direction".
/// 8 bytes on the stack (niche-optimized `Option<Dir>` = 1 byte × 8), `Copy` — no heap use.
pub type DirArr = [Option<Dir>; 8];

// ============================================================
// Thread-local BFS scratch buffers
// ============================================================
//
// Eliminate the Vec<i32>/Vec<u8>/VecDeque allocs that previously happened on
// every BFS call inside the beam inner loop (~5 000+/turn on 3-snake maps).
// Each worker thread owns one BfsScratch via thread_local!; calls borrow it
// for the duration of a single BFS, reset (fill) the needed slice, and return
// a scalar/Option — no heap interaction after the first call on that thread.

struct BfsScratch {
    i32_buf: Vec<i32>,   // used as dist[] (sentinel -1)
    u8_buf:  Vec<u8>,    // used as first_dir[] (sentinel u8::MAX)
    queue:   VecDeque<usize>,
}

impl BfsScratch {
    #[inline]
    fn ensure_i32(&mut self, size: usize) {
        if self.i32_buf.len() < size { self.i32_buf.resize(size, -1); }
    }
    #[inline]
    fn ensure_u8(&mut self, size: usize) {
        if self.u8_buf.len() < size { self.u8_buf.resize(size, u8::MAX); }
    }
}

thread_local! {
    // Reusable body-obstacle grid for step_phases_2_to_11 phase 4.
    // Separate from OBS_SCRATCH (which is used by build_obstacles in heuristic/BFS callers).
    static STEP_SCRATCH: RefCell<Vec<bool>> = RefCell::new(Vec::new());
    // Reusable snake-index grid for apply_gravity.
    // Kept separate from SNG_SCRATCH (which is used by snake_grid in heuristic callers).
    static GRAV_SCRATCH: RefCell<Vec<u8>>   = RefCell::new(Vec::new());

    static BFS_SCRATCH: RefCell<BfsScratch> = RefCell::new(BfsScratch {
        i32_buf: Vec::new(),
        u8_buf:  Vec::new(),
        queue:   VecDeque::new(),
    });
    // Separate RefCells so with_obstacles / with_snake_grid can be borrowed
    // simultaneously with BFS_SCRATCH (all three are accessed together in heuristic_v1).
    static OBS_SCRATCH: RefCell<Vec<bool>> = RefCell::new(Vec::new());
    static SNG_SCRATCH: RefCell<Vec<u8>>   = RefCell::new(Vec::new());
}

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

    /// Hot-path variant of `step()` using a stack-allocated `DirArr`.
    /// Avoids the HashMap lookup overhead in Phase 1; zero heap use for action dispatch.
    #[inline]
    pub fn step_arr(&mut self, actions: &DirArr) {
        let n = self.snakes.len();
        if n == 0 { self.turn += 1; return; }
        self.snakes.iter_mut().for_each(|s| {
            if let Some(d) = actions[s.id as usize] { s.dir = d; }
        });
        self.step_phases_2_to_11();
    }

    pub fn step(&mut self, actions: &std::collections::HashMap<u8, Dir>) {
        let n = self.snakes.len();
        if n == 0 { self.turn += 1; return; }

        // Phase 1 – apply direction overrides
        self.snakes.iter_mut().for_each(|s| {
            if let Some(&d) = actions.get(&s.id) { s.dir = d; }
        });

        self.step_phases_2_to_11();
    }

    /// Phases 2–11 of `step()`, shared by both `step()` and `step_arr()`.
    /// Called after Phase 1 (direction overrides) has already been applied.
    fn step_phases_2_to_11(&mut self) {
        let n = self.snakes.len();

        // Phase 2 – proposed new head positions (stack array, n ≤ 8)
        let mut proposed = [Pos::new(0, 0); 8];
        self.snakes.iter().enumerate().for_each(|(i, s)| {
            let (dx, dy) = s.dir.delta();
            proposed[i] = s.head().translate(dx, dy);
        });

        // Phase 3 – eaters: head lands on a food cell (flat bool array, max 32 snakes)
        let w = self.width as usize;
        let mut eaters = [false; 32];
        for (i, &h) in proposed.iter().enumerate() {
            if h.in_bounds(self.width, self.height) {
                let ci = h.y as usize * w + h.x as usize;
                if self.food[ci] { eaters[i] = true; }
            }
        }

        // Phase 4 + 5 – body obstacle grid (TLS) + destroyed head detection
        let grid_size = (self.width * self.height) as usize;
        let mut head_destroyed = [false; 8];
        STEP_SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() < grid_size { g.resize(grid_size, false); }
            let body_cells = &mut g[..grid_size];
            body_cells.fill(false);
            // Phase 4 – mark body cells (TLS, no heap alloc after first call)
            for (idx, s) in self.snakes.iter().enumerate() {
                let end = if eaters[idx] { s.len() } else { s.len().saturating_sub(1) };
                s.body.iter().take(end).for_each(|&p| {
                    if p.in_bounds(self.width, self.height) {
                        body_cells[p.y as usize * w + p.x as usize] = true;
                    }
                });
            }
            // Phase 5 – OOB / platform / body collision
            for (idx, &h) in proposed.iter().enumerate().take(n) {
                if eaters[idx] { continue; }
                if !h.in_bounds(self.width, self.height) || self.is_platform(h) {
                    head_destroyed[idx] = true;
                    continue;
                }
                if body_cells[h.y as usize * w + h.x as usize] {
                    head_destroyed[idx] = true;
                }
            }
        });
        // head-to-head: O(n²), n ≤ 8 — outside TLS closure (no grid needed)
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

        // Phase 6 – apply movement (stack array, n ≤ 8)
        let mut should_remove = [false; 8];
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
        let w    = self.width as usize;
        let pow  = &self.food;

        // to_fall: stack array (n ≤ 8 snakes), with explicit count
        let mut to_fall     = [0usize; 8];
        let mut to_fall_len = 0usize;

        GRAV_SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();
            // snake_at: cell → snake index (u8::MAX = empty). TLS — no heap alloc after first call.
            if g.len() < size { g.resize(size, u8::MAX); }
            let snake_at = &mut g[..size];

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

                to_fall_len = 0;
                for (idx, s) in self.snakes.iter().enumerate() {
                    let grounded = s.body.iter().any(|&p| {
                        let below_y = p.y + 1;
                        if below_y >= self.height { return true; }
                        if p.x < 0 || p.x >= self.width { return false; }
                        let below_ci = below_y as usize * w + p.x as usize;
                        if self.grid[below_ci] { return true; }
                        if pow[below_ci]       { return true; }
                        let sat = snake_at[below_ci];
                        sat != u8::MAX && sat != idx as u8
                    });
                    if !grounded {
                        to_fall[to_fall_len] = idx;
                        to_fall_len += 1;
                    }
                }

                if to_fall_len == 0 { break; }
                for &idx in &to_fall[..to_fall_len] {
                    self.snakes[idx].body.iter_mut().for_each(|p| p.y += 1);
                }
            }
        });
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
    /// Snake-index grid — allocates. Prefer `with_snake_grid` in hot paths.
    pub fn snake_grid(&self) -> Vec<u8> {
        let mut g = vec![u8::MAX; (self.width * self.height) as usize];
        let w = self.width as usize;
        self.snakes.iter().enumerate().for_each(|(idx, s)| {
            s.body.iter().for_each(|&p| {
                if p.in_bounds(self.width, self.height) {
                    g[p.y as usize * w + p.x as usize] = idx as u8;
                }
            });
        });
        g
    }

    // --------------------------------------------------------
    // BFS utilities
    // --------------------------------------------------------

    /// Zero-alloc variant: reuses OBS_SCRATCH thread-local buffer.
    /// Passes a `&[bool]` obstacle grid to `f` without heap allocation.
    /// Use this in hot paths; `build_obstacles()` for callers needing ownership.
    pub fn with_obstacles<R>(&self, f: impl FnOnce(&[bool]) -> R) -> R {
        let size = (self.width * self.height) as usize;
        let w    = self.width as usize;
        OBS_SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() < size { g.resize(size, false); }
            g[..size].fill(false);
            self.snakes.iter().for_each(|s| {
                s.body.iter().take(s.len().saturating_sub(1)).for_each(|&p| {
                    if p.in_bounds(self.width, self.height) {
                        g[p.y as usize * w + p.x as usize] = true;
                    }
                });
            });
            f(&g[..size])
        })
    }

    /// Reverse multi-source BFS from all food/power-source cells.
    /// Passes a flat `&[i32]` distance map to `f`:
    ///   dist[ci] = min BFS distance from cell ci to the nearest food  (-1 = unreachable)
    ///
    /// Snake-head cells are in `obs` and therefore have dist == -1 even when adjacent to
    /// food.  Callers that need a head's distance should take `min(food_dist[neighbor]+1)`
    /// over the head's four accessible neighbours (see `old_heuristic`).
    ///
    /// Uses BFS_SCRATCH (TLS) — zero alloc after the first call on each thread.
    /// Must NOT be called from inside another BFS_SCRATCH closure.
    pub fn with_food_dist_map<R>(&self, obs: &[bool], f: impl FnOnce(&[i32]) -> R) -> R {
        let w    = self.width as usize;
        let size = w * self.height as usize;
        let dirs = Dir::all();
        BFS_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            sc.ensure_i32(size);
            let dist  = &mut sc.i32_buf[..size];
            let queue = &mut sc.queue;
            dist.fill(-1);
            queue.clear();
            // Seed every food cell at distance 0.
            for (ci, &b) in self.food.iter().enumerate() {
                if b {
                    dist[ci] = 0;
                    queue.push_back(ci);
                }
            }
            // BFS outward — respects platform walls and body obstacles.
            while let Some(ci) = queue.pop_front() {
                let d   = dist[ci];
                let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
                for &dir in &dirs {
                    let (dx, dy) = dir.delta();
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                    let ni = ny as usize * w + nx as usize;
                    if self.grid[ni] || obs[ni] || dist[ni] != -1 { continue; }
                    dist[ni] = d + 1;
                    queue.push_back(ni);
                }
            }
            f(&sc.i32_buf[..size])
        })
    }

    /// Zero-alloc variant: reuses SNG_SCRATCH thread-local buffer.
    /// Passes a `&[u8]` snake-index grid to `f` (u8::MAX = no snake).
    pub fn with_snake_grid<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        let size = (self.width * self.height) as usize;
        SNG_SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() < size { g.resize(size, u8::MAX); }
            g[..size].fill(u8::MAX);
            let w = self.width as usize;
            self.snakes.iter().enumerate().for_each(|(idx, s)| {
                s.body.iter().for_each(|&p| {
                    if p.in_bounds(self.width, self.height) {
                        g[p.y as usize * w + p.x as usize] = idx as u8;
                    }
                });
            });
            f(&g[..size])
        })
    }

    /// Obstacle set for pathfinding: all body parts except tails (which vacate).
    /// Returns an owned flat bool grid (true = blocked).
    /// Allocates — prefer `with_obstacles` in hot paths.
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
        let w    = self.width as usize;
        let size = w * self.height as usize;
        let dirs = Dir::all();
        BFS_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            sc.ensure_u8(size);
            let first_dir = &mut sc.u8_buf[..size];
            let queue     = &mut sc.queue;
            first_dir.fill(u8::MAX);
            queue.clear();

            for (di, &d) in dirs.iter().enumerate() {
                let (dx, dy) = d.delta();
                let (nx, ny) = (start.x + dx, start.y + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || first_dir[ni] != u8::MAX { continue; }
                first_dir[ni] = di as u8;
                if targets[ni] { return Some(d); }
                queue.push_back(ni);
            }

            while let Some(ci) = queue.pop_front() {
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
                    queue.push_back(ni);
                }
            }
            None
        })
    }

    /// BFS distance from `start` to the nearest target, or i32::MAX if unreachable.
    /// `targets` and `obs` are flat bool grids.
    pub fn bfs_dist(&self, start: Pos, targets: &[bool], obs: &[bool]) -> i32 {
        let w        = self.width as usize;
        let start_ci = start.y as usize * w + start.x as usize;
        if targets[start_ci] { return 0; }
        let size = w * self.height as usize;
        let dirs = Dir::all();
        BFS_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            sc.ensure_i32(size);
            let dist  = &mut sc.i32_buf[..size];
            let queue = &mut sc.queue;
            dist.fill(-1);
            dist[start_ci] = 0;
            queue.clear();
            queue.push_back(start_ci);

            while let Some(ci) = queue.pop_front() {
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
                    queue.push_back(ni);
                }
            }
            i32::MAX
        })
    }

    /// Gravity-aware BFS distance: intermediate cells must be statically grounded.
    /// Targets (food) are always reachable. `targets` doubles as the power-source
    /// grid for the grounding check (food = power source).
    pub fn bfs_dist_grounded(&self, start: Pos, targets: &[bool], obs: &[bool]) -> i32 {
        let w        = self.width as usize;
        let start_ci = start.y as usize * w + start.x as usize;
        if targets[start_ci] { return 0; }
        let size = w * self.height as usize;
        let dirs = Dir::all();
        BFS_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            sc.ensure_i32(size);
            let dist  = &mut sc.i32_buf[..size];
            let queue = &mut sc.queue;
            dist.fill(-1);
            dist[start_ci] = 0;
            queue.clear();
            queue.push_back(start_ci);

            while let Some(ci) = queue.pop_front() {
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
                    queue.push_back(ni);
                }
            }
            i32::MAX
        })
    }

    /// Multi-source gravity-aware BFS from a set of starting positions (snake heads).
    /// Returns `(dist, source)` flat grids:
    ///   dist[ci]   = minimum distance from any start to cell ci  (-1 = unreachable)
    ///   source[ci] = index into `starts` of the closest start    (u8::MAX = unreachable)
    ///
    /// Initialising all starts simultaneously means a single pass gives us both
    /// "which team snake is closest" and "what distance" for every food cell —
    /// replacing N separate per-snake BFS calls with one.
    pub fn bfs_multisource_dist_map(
        &self,
        starts:  &[Pos],
        targets: &[bool],
        obs:     &[bool],
    ) -> (Vec<i32>, Vec<u8>) {
        let w    = self.width as usize;
        let size = w * self.height as usize;

        let mut dist   = vec![-1i32;    size];
        let mut source = vec![u8::MAX;  size];
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        for (si, &start) in starts.iter().enumerate() {
            if !start.in_bounds(self.width, self.height) { continue; }
            let ci = start.y as usize * w + start.x as usize;
            if dist[ci] != -1 { continue; } // two heads on same cell
            dist[ci]   = 0;
            source[ci] = si as u8;
            q.push_back(ci);
        }

        let dirs = Dir::all();
        while let Some(ci) = q.pop_front() {
            let d   = dist[ci];
            let src = source[ci];
            let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
            for &dir in &dirs {
                let (dx, dy) = dir.delta();
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || dist[ni] != -1 { continue; }
                if !targets[ni] && !self.is_grounded_cell_ci(ni, w, targets) { continue; }
                dist[ni]   = d + 1;
                source[ni] = src;
                q.push_back(ni);
            }
        }
        (dist, source)
    }

    /// Full gravity-aware distance map from `start`.
    /// Returns a flat Vec<i32> where dist[ci] is the BFS distance to cell ci,
    /// or -1 if unreachable under the gravity-aware rules.
    /// Unlike `bfs_dist_grounded` (which returns the distance to the nearest
    /// target and exits early), this visits the whole reachable grid so the
    /// caller can look up distances to every food cell in one pass.
    pub fn bfs_dist_map_grounded(&self, start: Pos, targets: &[bool], obs: &[bool]) -> Vec<i32> {
        let w = self.width as usize;
        let size = w * self.height as usize;
        let start_ci = start.y as usize * w + start.x as usize;

        let mut dist = vec![-1i32; size];
        dist[start_ci] = 0;
        let mut q: VecDeque<usize> = VecDeque::new();
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
                // Food cells are always passable (they provide support and are targets).
                // Non-food cells must be statically grounded.
                if !targets[ni] && !self.is_grounded_cell_ci(ni, w, targets) { continue; }
                dist[ni] = d + 1;
                q.push_back(ni);
            }
        }
        dist
    }

    /// Gravity-aware first-step BFS. First step from start is unrestricted
    /// (snake body provides support); subsequent cells must be grounded.
    pub fn bfs_first_step_grounded(&self, start: Pos, targets: &[bool], obs: &[bool]) -> Option<Dir> {
        let w    = self.width as usize;
        let size = w * self.height as usize;
        let dirs = Dir::all();
        BFS_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            sc.ensure_u8(size);
            let first_dir = &mut sc.u8_buf[..size];
            let queue     = &mut sc.queue;
            first_dir.fill(u8::MAX);
            queue.clear();

            // First step: no grounding restriction
            for (di, &d) in dirs.iter().enumerate() {
                let (dx, dy) = d.delta();
                let (nx, ny) = (start.x + dx, start.y + dy);
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                let ni = ny as usize * w + nx as usize;
                if self.grid[ni] || obs[ni] || first_dir[ni] != u8::MAX { continue; }
                first_dir[ni] = di as u8;
                if targets[ni] { return Some(d); }
                queue.push_back(ni);
            }

            // Subsequent steps: require grounding for non-target cells
            while let Some(ci) = queue.pop_front() {
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
                    queue.push_back(ni);
                }
            }
            None
        })
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
