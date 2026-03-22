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
    // Separate RefCell for liberty_count flood-fill (bool visited grid + queue).
    // Safe to borrow simultaneously with BFS_SCRATCH, OBS_SCRATCH, SNG_SCRATCH.
    static LIB_SCRATCH: RefCell<(Vec<bool>, VecDeque<usize>)> =
        RefCell::new((Vec::new(), VecDeque::new()));
    // Cached food distance map (no snake-body obstacles, only platform walls).
    // Populated once per real turn by cache_food_dist_no_obs(); read O(1) by
    // cached_food_dist().  Separate from BFS_SCRATCH so it persists across BFS calls.
    static FOOD_DIST_CACHE: RefCell<Vec<i32>> = RefCell::new(Vec::new());
}

// ============================================================
// Bitboard helpers for bfs_dist_bits
// ============================================================

/// OR-accumulate a left-shift-by-k of `src[0..n]` into `dst[0..n]`.  k must be in 1..64.
#[inline]
fn bb_shl_or(src: &[u64; 32], k: usize, dst: &mut [u64; 32], n: usize) {
    let rk = 64 - k;
    dst[0] |= src[0] << k;
    for i in 1..n { dst[i] |= (src[i] << k) | (src[i-1] >> rk); }
}

/// OR-accumulate a right-shift-by-k of `src[0..n]` into `dst[0..n]`.  k must be in 1..64.
#[inline]
fn bb_shr_or(src: &[u64; 32], k: usize, dst: &mut [u64; 32], n: usize) {
    let lk = 64 - k;
    dst[n-1] |= src[n-1] >> k;
    for i in (0..n-1).rev() { dst[i] |= (src[i] >> k) | (src[i+1] << lk); }
}

/// Precomputed bitboard BFS setup: constant across all snake heads in one heuristic call.
/// Build with `GameState::prepare_bfs_bits`; query with `bfs_dist_bits_with`.
#[derive(Clone)]
pub struct BfsBitsSetup {
    pub blocked: [u64; 32],  // platform | obs | OOB bits
    pub tbits:   [u64; 32],  // target (food) bits
    pub rcol:    [u64; 32],  // right-column anti-wrap mask
    pub lcol:    [u64; 32],  // left-column anti-wrap mask
    pub n:       usize,      // number of active 64-bit words = ceil(size/64)
}

// ============================================================
// Primitive types
// ============================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Dir { #[default] Up, Down, Left, Right }

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
    #[inline] pub fn to_u8(self) -> u8 {
        match self { Dir::Up => 0, Dir::Down => 1, Dir::Left => 2, Dir::Right => 3 }
    }
    #[inline] pub fn from_u8(v: u8) -> Dir {
        match v { 0 => Dir::Up, 1 => Dir::Down, 2 => Dir::Left, _ => Dir::Right }
    }
}

// ============================================================
// SnakeBody — compact direction-sequence body storage
// ============================================================
//
// Encodes a snake body as (head, dirs[]) without heap allocation.
// Every beam-search clone is a plain memcpy — no malloc per snake.
// Max length 256: at 200 turns with start length 3 the body tops out at 203.
//
// Ring-buffer layout:
//   dirs[start], dirs[start+1], ..., dirs[start+len-2]  (all mod 256)
//   dirs[start+i] = direction FROM body[i] TO body[i+1]  (toward the tail)
//   head = cached body[0];  tail = cached body[len-1]

pub const MAX_SNAKE_LEN: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct SnakeBody {
    pub head:  Pos,
    pub tail:  Pos,
    pub len:   u16,
    start: u8,   // ring-buffer start index; arithmetic wraps at 256 naturally via u8
    dirs:  [u8; MAX_SNAKE_LEN],
}

impl SnakeBody {
    /// Build from an ordered slice of adjacent positions (head first).
    pub fn new(parts: &[Pos]) -> Self {
        assert!(!parts.is_empty() && parts.len() <= MAX_SNAKE_LEN,
                "snake body length {} out of range", parts.len());
        let mut b = SnakeBody {
            head:  parts[0],
            tail:  *parts.last().unwrap(),
            len:   parts.len() as u16,
            start: 0,
            dirs:  [0u8; MAX_SNAKE_LEN],
        };
        for i in 0..parts.len().saturating_sub(1) {
            let dx = parts[i + 1].x - parts[i].x;
            let dy = parts[i + 1].y - parts[i].y;
            b.dirs[i] = match (dx, dy) {
                (0, -1) => Dir::Up,
                (0,  1) => Dir::Down,
                (-1, 0) => Dir::Left,
                _       => Dir::Right,
            }.to_u8();
        }
        b
    }

    #[inline] pub fn head(&self) -> Pos { self.head }
    #[inline] pub fn tail(&self) -> Pos { self.tail }
    #[inline] pub fn len(&self) -> usize { self.len as usize }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Teleport the head to `p` (test-only; does not fix direction encoding).
    #[inline] pub fn set_head(&mut self, p: Pos) { self.head = p; }

    /// Walk body from head to tail without heap allocation.
    /// Returns `Pos` by value (unlike VecDeque::iter which yields &Pos).
    pub fn iter(&self) -> impl Iterator<Item = Pos> + '_ {
        let mut pos = self.head;
        let mut i   = 0u16;
        let dirs: &[u8; MAX_SNAKE_LEN] = &self.dirs;
        let start = self.start;
        let len   = self.len;
        std::iter::from_fn(move || {
            if i >= len { return None; }
            let cur = pos;
            if i + 1 < len {
                let idx = start.wrapping_add(i as u8) as usize;
                let (dx, dy) = Dir::from_u8(dirs[idx]).delta();
                pos = Pos { x: pos.x + dx, y: pos.y + dy };
            }
            i += 1;
            Some(cur)
        })
    }

    /// O(i) random access (needed for neck lookup at index 1).
    pub fn get(&self, i: usize) -> Option<Pos> {
        if i >= self.len as usize { return None; }
        let mut pos = self.head;
        for j in 0..i {
            let idx = self.start.wrapping_add(j as u8) as usize;
            let (dx, dy) = Dir::from_u8(self.dirs[idx]).delta();
            pos = Pos { x: pos.x + dx, y: pos.y + dy };
        }
        Some(pos)
    }

    /// Prepend `new_head`; old head becomes body[1].
    /// Direction stored: from `new_head` toward old head.
    #[inline]
    pub fn push_front(&mut self, new_head: Pos) {
        let dx = self.head.x - new_head.x;
        let dy = self.head.y - new_head.y;
        let d = match (dx, dy) {
            (0, -1) => Dir::Up,
            (0,  1) => Dir::Down,
            (-1, 0) => Dir::Left,
            _       => Dir::Right,
        };
        self.start = self.start.wrapping_sub(1);
        self.dirs[self.start as usize] = d.to_u8();
        self.head = new_head;
        self.len += 1;
    }

    /// Remove the tail (non-eating move: body shrinks from the back).
    #[inline]
    pub fn pop_back(&mut self) {
        if self.len <= 1 { self.len = 0; return; }
        // dirs[(start + len - 2) % 256] points from body[len-2] to body[len-1]=tail.
        // New tail = current tail - that delta.
        let last = self.start.wrapping_add((self.len - 2) as u8) as usize;
        let (dx, dy) = Dir::from_u8(self.dirs[last]).delta();
        self.tail = Pos { x: self.tail.x - dx, y: self.tail.y - dy };
        self.len -= 1;
    }

    /// Remove the head (head-destroyed shrink, only when len >= 3).
    #[inline]
    pub fn pop_front(&mut self) {
        if self.len <= 1 { self.len = 0; return; }
        let (dx, dy) = Dir::from_u8(self.dirs[self.start as usize]).delta();
        self.head = Pos { x: self.head.x + dx, y: self.head.y + dy };
        self.start = self.start.wrapping_add(1);
        self.len -= 1;
        if self.len == 1 { self.tail = self.head; }
    }

    /// Shift all body parts by `dy` rows — O(1) (only head and tail move).
    /// Correct because body positions are encoded as relative direction steps.
    #[inline]
    pub fn apply_dy(&mut self, dy: i32) {
        self.head.y += dy;
        self.tail.y += dy;
    }
}

// [u8; 256] doesn't implement Default via derive (stable only covers up to [T;32]).
impl Default for SnakeBody {
    fn default() -> Self {
        SnakeBody { head: Pos::default(), tail: Pos::default(),
                    len: 0, start: 0, dirs: [0u8; MAX_SNAKE_LEN] }
    }
}

// ============================================================
// Snake
// ============================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct Snake {
    pub id: u8,
    /// body[0] = head, body[last] = tail — compact direction-sequence, no heap alloc
    pub body: SnakeBody,
    pub dir: Dir,
    pub player: u8,
}

impl Snake {
    pub fn new(id: u8, parts: Vec<Pos>, player: u8) -> Self {
        Snake { id, body: SnakeBody::new(&parts),
                dir: infer_dir(&parts), player }
    }
    #[inline] pub fn head(&self) -> Pos { self.body.head() }
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
// SnakeVec — fixed-size inline snake storage (no heap alloc on clone)
// ============================================================
//
// Replaces Vec<Snake> so GameState::clone() needs only one remaining malloc
// (food: Vec<bool>). With Snake: Copy, the whole array copies as a memcpy.
// Max 8 snakes matches the DirArr capacity already used throughout the code.

pub const MAX_SNAKES: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct SnakeVec {
    data: [Snake; MAX_SNAKES],
    len:  u8,
}

impl SnakeVec {
    #[inline] pub fn new() -> Self {
        SnakeVec { data: [Snake::default(); MAX_SNAKES], len: 0 }
    }
    #[inline] pub fn push(&mut self, s: Snake) {
        self.data[self.len as usize] = s;
        self.len += 1;
    }
    #[inline] pub fn clear(&mut self) { self.len = 0; }
    /// In-place filter — same signature as Vec::retain.
    #[inline] pub fn retain(&mut self, mut f: impl FnMut(&Snake) -> bool) {
        let mut w = 0usize;
        for r in 0..self.len as usize {
            if f(&self.data[r]) { self.data[w] = self.data[r]; w += 1; }
        }
        self.len = w as u8;
    }
}

impl Default for SnakeVec {
    fn default() -> Self { SnakeVec::new() }
}

/// Deref to `[Snake]` gives iter/iter_mut/len/is_empty/find/any/Index for free.
impl std::ops::Deref for SnakeVec {
    type Target = [Snake];
    #[inline] fn deref(&self) -> &[Snake] { &self.data[..self.len as usize] }
}
impl std::ops::DerefMut for SnakeVec {
    #[inline] fn deref_mut(&mut self) -> &mut [Snake] { &mut self.data[..self.len as usize] }
}

// `for s in &snakevec` and `for s in &mut snakevec` — Deref alone isn't enough
// for IntoIterator lookup in for-loops, so we implement it explicitly.
impl<'a> IntoIterator for &'a SnakeVec {
    type Item = &'a Snake;
    type IntoIter = std::slice::Iter<'a, Snake>;
    #[inline] fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<'a> IntoIterator for &'a mut SnakeVec {
    type Item = &'a mut Snake;
    type IntoIter = std::slice::IterMut<'a, Snake>;
    #[inline] fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

// ============================================================
// GameState
// ============================================================

/// Clone cost breakdown:
///   grid  — Arc refcount bump only (zero copy, zero alloc)
///   food  — memcpy of width*height bytes (~264B for a 24×11 map)
///   snakes — memcpy of SnakeVec ([Snake;8] inline); zero heap alloc
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
    /// All living snakes (both players) — inline, no heap alloc on clone
    pub snakes: SnakeVec,
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
            snakes: SnakeVec::new(),
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
                s.body.iter().take(end).for_each(|p| {
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
        // eaters are NOT exempt: two snakes landing on the same food cell still collide.
        for i in 0..n {
            if head_destroyed[i] { continue; }
            for j in (i + 1)..n {
                if head_destroyed[j] { continue; }
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
        self.snakes.retain(|s| s.body.iter().all(|p| p.in_bounds(w_i, h_i)));

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
                    s.body.iter().for_each(|p| {
                        if p.in_bounds(self.width, self.height) {
                            snake_at[p.y as usize * w + p.x as usize] = idx as u8;
                        }
                    });
                }

                to_fall_len = 0;
                for (idx, s) in self.snakes.iter().enumerate() {
                    let grounded = s.body.iter().any(|p| {
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
                    self.snakes[idx].body.apply_dy(1);
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
            s.body.iter().for_each(|p| {
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
                s.body.iter().take(s.len().saturating_sub(1)).for_each(|p| {
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

    /// Precompute food distance map ignoring snake bodies (platform walls only).
    ///
    /// Runs a single reverse multi-source BFS from all food cells with no body
    /// obstacles — only fixed platform walls (`self.grid`) block the BFS.
    /// Result stored in `FOOD_DIST_CACHE` for O(1) lookup per beam node.
    ///
    /// The distances are a lower bound on the true obstacle-aware distance;
    /// the approximation improves naturally at deeper beam depths where bodies
    /// have vacated.  One full-grid BFS per real game turn (~50µs).
    pub fn cache_food_dist(&self) {
        let w    = self.width as usize;
        let size = w * self.height as usize;
        let dirs = Dir::all();
        BFS_SCRATCH.with(|bfs_cell| {
            let mut guard = bfs_cell.borrow_mut();
            let sc = &mut *guard;
            sc.ensure_i32(size);
            let dist  = &mut sc.i32_buf[..size];
            let queue = &mut sc.queue;
            dist.fill(-1);
            queue.clear();
            for (ci, &b) in self.food.iter().enumerate() {
                if b { dist[ci] = 0; queue.push_back(ci); }
            }
            while let Some(ci) = queue.pop_front() {
                let d = dist[ci];
                let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
                for &dir in &dirs {
                    let (dx, dy) = dir.delta();
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                    let ni = ny as usize * w + nx as usize;
                    if self.grid[ni] || dist[ni] != -1 { continue; }
                    dist[ni] = d + 1;
                    queue.push_back(ni);
                }
            }
            FOOD_DIST_CACHE.with(|cache_cell| {
                let mut cache = cache_cell.borrow_mut();
                if cache.len() < size { cache.resize(size, -1); }
                cache[..size].copy_from_slice(&sc.i32_buf[..size]);
            });
        });
    }

    /// O(1) food distance lookup using the map precomputed by `cache_food_dist`.
    ///
    /// Returns BFS distance from `pos` to nearest food (root-turn obstacles,
    /// head cells traversable).  Returns -1 if the cell is unreachable.
    #[inline]
    pub fn cached_food_dist(&self, pos: Pos) -> i32 {
        if !pos.in_bounds(self.width, self.height) { return -1; }
        let ci = pos.y as usize * self.width as usize + pos.x as usize;
        FOOD_DIST_CACHE.with(|cell| cell.borrow()[ci])
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
                s.body.iter().for_each(|p| {
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
            s.body.iter().take(s.len().saturating_sub(1)).for_each(|p| {
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

    /// Like `is_grounded_cell_ci` but also counts any snake body part below
    /// as support (matches actual gravity: any snake provides support to snakes above it).
    #[inline]
    fn is_grounded_cell_ci_sng(&self, ci: usize, w: usize, pow: &[bool], sng: &[u8]) -> bool {
        let below_ci = ci + w;
        below_ci >= self.grid.len()   // bottom edge
            || self.grid[below_ci]    // platform
            || pow[below_ci]          // power source
            || sng[below_ci] != u8::MAX // any snake body
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

    /// Precompute bitboard BFS setup that is constant across all snakes' BFS calls
    /// within one heuristic evaluation (same targets, same obs, same grid geometry).
    ///
    /// Call once per heuristic then pass to `bfs_dist_bits_with` for each snake.
    pub fn prepare_bfs_bits(&self, targets: &[bool], obs: &[bool]) -> BfsBitsSetup {
        let w    = self.width as usize;
        let size = w * self.height as usize;
        let n    = (size + 63) >> 6;
        assert!(n <= 32, "map too large for bitboard BFS: {} cells (max 2048)", size);

        let mut blocked = [!0u64; 32];
        let mut tbits   = [0u64;  32];
        for i in 0..size {
            if !self.grid[i] && !obs[i] { blocked[i >> 6] &= !(1u64 << (i & 63)); }
            if targets[i]               { tbits  [i >> 6] |=   1u64 << (i & 63);  }
        }

        let mut rcol = [0u64; 32];
        let mut lcol = [0u64; 32];
        for y in 0..self.height as usize {
            let ri = y * w + (w - 1);
            let li = y * w;
            rcol[ri >> 6] |= 1u64 << (ri & 63);
            lcol[li >> 6] |= 1u64 << (li & 63);
        }

        BfsBitsSetup { blocked, tbits, rcol, lcol, n }
    }

    /// Bitboard BFS distance from `start` using a pre-built setup.
    ///
    /// Call `prepare_bfs_bits` once per heuristic evaluation, then call this once
    /// per snake — avoids rebuilding blocked/tbits/rcol/lcol for each snake.
    pub fn bfs_dist_bits_with(&self, start: Pos, s: &BfsBitsSetup) -> i32 {
        let w  = self.width as usize;
        let n  = s.n;
        let start_ci = start.y as usize * w + start.x as usize;
        if s.tbits[start_ci >> 6] & (1u64 << (start_ci & 63)) != 0 { return 0; }

        let size = w * self.height as usize;
        let mut front = [0u64; 32];
        front[start_ci >> 6] |= 1u64 << (start_ci & 63);
        let mut visited = front;

        for dist in 1..=size as i32 {
            let mut exp = [0u64; 32];

            {
                let w0 = front[0] & !s.rcol[0];
                exp[0] = w0 << 1;
                for i in 1..n {
                    let wi = front[i] & !s.rcol[i];
                    exp[i] = (wi << 1) | ((front[i-1] & !s.rcol[i-1]) >> 63);
                }
            }
            {
                let wn = front[n-1] & !s.lcol[n-1];
                exp[n-1] |= wn >> 1;
                for i in (0..n-1).rev() {
                    let wi = front[i] & !s.lcol[i];
                    exp[i] |= (wi >> 1) | ((front[i+1] & !s.lcol[i+1]) << 63);
                }
            }
            bb_shl_or(&front, w, &mut exp, n);
            bb_shr_or(&front, w, &mut exp, n);

            let mut any_new = false;
            let mut new_front = [0u64; 32];
            for i in 0..n {
                let cell = exp[i] & !s.blocked[i] & !visited[i];
                new_front[i] = cell;
                if cell != 0 { any_new = true; }
            }

            if !any_new { return i32::MAX; }

            for i in 0..n {
                if new_front[i] & s.tbits[i] != 0 { return dist; }
            }

            for i in 0..n { visited[i] |= new_front[i]; }
            front = new_front;
        }
        i32::MAX
    }

    /// Convenience wrapper: build setup + run single BFS.  Use `bfs_dist_bits_with`
    /// directly when calling for multiple snakes in the same heuristic.
    pub fn bfs_dist_bits(&self, start: Pos, targets: &[bool], obs: &[bool]) -> i32 {
        let setup = self.prepare_bfs_bits(targets, obs);
        self.bfs_dist_bits_with(start, &setup)
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

    /// Like `bfs_dist_grounded` but also treats any snake body part below a cell
    /// as support.  Pass `sng = state.snake_grid()` (u8::MAX = empty).
    /// This fixes the over-pruning bug in heuristic_v1 where the snake's own body
    /// was not recognised as ground during BFS path evaluation.
    pub fn bfs_dist_grounded_sng(&self, start: Pos, targets: &[bool], obs: &[bool], sng: &[u8]) -> i32 {
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
                    if !self.is_grounded_cell_ci_sng(ni, w, targets, sng) { continue; }
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

    /// Non-gravity-aware flood-fill liberty count from `head`.
    ///
    /// Counts cells reachable from `head` treating only static platform walls
    /// (`self.grid`) and body obstacles (`obs`) as impassable.  Gravity is
    /// intentionally ignored: a snake can reach a non-grounded cell so long as
    /// some part of its body remains grounded — checking exact body support
    /// would require a per-step simulation, so plain connectivity is the right
    /// proxy for "how much space can this snake manoeuvre in".
    ///
    /// Returns `min(reachable_cells, cap)`.  Use a small cap (e.g. 30) to
    /// bound cost — if the snake has cap+ open cells it is "free" and the
    /// exact count does not matter for penalty purposes.
    ///
    /// Uses LIB_SCRATCH (TLS, zero alloc after first call).  Safe to call
    /// inside `with_obstacles` + alongside `bfs_dist` (all different RefCells).
    pub fn liberty_count(&self, head: Pos, obs: &[bool], cap: usize) -> usize {
        if !head.in_bounds(self.width, self.height) { return 0; }
        let w    = self.width as usize;
        let size = w * self.height as usize;
        LIB_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let (visited, queue) = &mut *guard;
            if visited.len() < size { visited.resize(size, false); }
            let vis = &mut visited[..size];
            vis.fill(false);
            queue.clear();

            let start = head.y as usize * w + head.x as usize;
            vis[start] = true;
            queue.push_back(start);
            let mut count = 0usize;

            while let Some(ci) = queue.pop_front() {
                count += 1;
                if count >= cap { return cap; }
                let (cx, cy) = ((ci % w) as i32, (ci / w) as i32);
                for &dir in &Dir::all() {
                    let (dx, dy) = dir.delta();
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height { continue; }
                    let ni = ny as usize * w + nx as usize;
                    if vis[ni] || self.grid[ni] || obs[ni] { continue; }
                    vis[ni] = true;
                    queue.push_back(ni);
                }
            }
            count
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
        assert_eq!(s.snakes[0].body.get(2).unwrap(), Pos::new(5, 9));
        assert_eq!(s.snakes[0].body.head(), Pos::new(5, 7));
    }

    #[test]
    fn test_gravity_rests_on_platform() {
        // Build grid with platform at row y=5, then construct state
        let mut grid = vec![false; 100];
        for x in 0..10usize { grid[5 * 10 + x] = true; }
        let mut s = GameState::new(10, 10, grid);
        s.snakes.push(snake(0, 0, &[(5, 2), (5, 3), (5, 4)]));
        s.apply_gravity();
        assert_eq!(s.snakes[0].body.get(2).unwrap(), Pos::new(5, 4));
    }

    #[test]
    fn test_gravity_snake_stacks_on_snake() {
        let mut s = open(10, 5);
        s.snakes.push(snake(1, 1, &[(5, 4), (4, 4), (3, 4)]));
        s.snakes.push(snake(0, 0, &[(5, 1), (5, 2), (5, 3)]));
        s.apply_gravity();
        let b = s.snakes.iter().find(|sn| sn.id == 1).unwrap();
        let a = s.snakes.iter().find(|sn| sn.id == 0).unwrap();
        assert_eq!(b.body.head(), Pos::new(5, 4));
        assert_eq!(a.body.get(2).unwrap(), Pos::new(5, 3));
    }

    #[test]
    fn test_gravity_snake_falls_off_bottom() {
        let mut s = open(5, 3);
        s.snakes.push(snake(0, 0, &[(2, 0)]));
        s.apply_gravity();
        assert_eq!(s.snakes[0].body.head(), Pos::new(2, 2));
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

    // ── real game maps from CG logs ───────────────────────────────────

    /// Build a GameState from a raw flat grid string (# = platform, . = empty),
    /// food list, and snake list (id, player, body).
    fn from_cg_map(
        w: i32,
        h: i32,
        rows: &[&str],
        food: &[(i32, i32)],
        snakes: &[(u8, u8, &[(i32, i32)])],
    ) -> GameState {
        let mut grid = vec![false; (w * h) as usize];
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    grid[y * w as usize + x] = true;
                }
            }
        }
        let mut s = GameState::new(w, h, grid);
        for &(x, y) in food {
            s.add_food(Pos::new(x, y));
        }
        for &(id, player, body) in snakes {
            s.snakes.push(snake(id, player, body));
        }
        s
    }

    #[test]
    fn test_map_877959752_initial_state() {
        // 32×17, game 877959752 — 8 snakes (4 per player), 30 food items
        let s = from_cg_map(
            32, 17,
            &[
                "..............#..#..............",
                "..............#..#..............",
                ".........#............#.........",
                ".........#............#.........",
                "................................",
                ".........#.....##.....#.........",
                ".........#............#.........",
                ".......#.......##.......#.......",
                ".....##........##........##.....",
                "........#..............#........",
                "..##...#................#...##..",
                ".#....#.......#..#.......#....#.",
                ".#..........##....##..........#.",
                "###..#.........##.........#..###",
                "######.....#..####..#.....######",
                "#######..##############..#######",
                "################################",
            ],
            &[
                (13,13),(18,13),(21,5),(10,5),(26,9),(5,9),(20,12),(11,12),
                (7,11),(24,11),(30,6),(1,6),(23,11),(8,11),(8,7),(23,7),
                (29,3),(2,3),(29,9),(2,9),(4,3),(27,3),(2,7),(29,7),
                (12,8),(19,8),(0,9),(31,9),(7,13),(24,13),
            ],
            &[
                (0, 0, &[(25,12),(25,13),(25,14)]),
                (1, 0, &[(14,8),(14,9),(14,10)]),
                (2, 0, &[(10,12),(10,13),(10,14)]),
                (3, 0, &[(6,5),(6,6),(6,7)]),
                (4, 1, &[(6,12),(6,13),(6,14)]),
                (5, 1, &[(17,8),(17,9),(17,10)]),
                (6, 1, &[(21,12),(21,13),(21,14)]),
                (7, 1, &[(25,5),(25,6),(25,7)]),
            ],
        );
        assert_eq!(s.width, 32);
        assert_eq!(s.height, 17);
        assert_eq!(s.snakes.len(), 8);
        assert_eq!(s.food_count, 30);
        // All snakes start with 3 segments and face Up (head.y < body[1].y)
        for sn in &s.snakes {
            assert_eq!(sn.len(), 3);
            assert_eq!(sn.dir, Dir::Up);
        }
        // Spot-check: snake 0 head on a non-platform cell
        assert!(!s.is_platform(Pos::new(25, 12)));
        // Bottom row is solid platform
        assert!(s.is_platform(Pos::new(0, 16)));
    }

    #[test]
    fn test_map_877959870_initial_state() {
        // 42×23, game 877959870 (timeout game) — 8 snakes, 62 food items
        let s = from_cg_map(
            42, 23,
            &[
                "....................##....................",
                "..........................................",
                "..........................................",
                ".......##........................##.......",
                "..........................................",
                "..........................................",
                "...##................................##...",
                "....#..........#..........#..........#....",
                "..........##....#........#....##..........",
                "................#...##...#................",
                "..........................................",
                ".#..................##..................#.",
                "#...........#.....#....#.....#...........#",
                "............##....#.##.#....##............",
                "..#..........#..............#..........#..",
                ".#......#..#..................#..#......#.",
                "...##..#..#....................#..#..##...",
                "..#............##.#....#.##............#..",
                "#.......#......#...#..#...#......#.......#",
                "####...#########...#..#...#########...####",
                "#####..##########........##########..#####",
                "#####..###########..##..###########..#####",
                "##########################################",
            ],
            &[
                (10,2),(31,2),(19,2),(22,2),(19,3),(22,3),(9,4),(32,4),
                (19,5),(22,5),(0,6),(41,6),(6,8),(35,8),(19,8),(22,8),
                (15,9),(26,9),(6,12),(35,12),(14,12),(27,12),(0,13),(41,13),
                (20,15),(21,15),(11,16),(30,16),(20,16),(21,16),(3,17),(38,17),
                (6,17),(35,17),(9,0),(32,0),(13,2),(28,2),(16,3),(25,3),
                (13,4),(28,4),(0,8),(41,8),(2,8),(39,8),(7,8),(34,8),
                (13,8),(28,8),(5,10),(36,10),(7,11),(34,11),(14,11),(27,11),
                (4,12),(37,12),(16,12),(25,12),(13,17),(28,17),
            ],
            &[
                (0, 0, &[(12,16),(12,17),(12,18)]),
                (1, 0, &[(11,5),(11,6),(11,7)]),
                (2, 0, &[(23,14),(23,15),(23,16)]),
                (3, 0, &[(0,15),(0,16),(0,17)]),
                (4, 1, &[(29,16),(29,17),(29,18)]),
                (5, 1, &[(30,5),(30,6),(30,7)]),
                (6, 1, &[(18,14),(18,15),(18,16)]),
                (7, 1, &[(41,15),(41,16),(41,17)]),
            ],
        );
        assert_eq!(s.width, 42);
        assert_eq!(s.height, 23);
        assert_eq!(s.snakes.len(), 8);
        assert_eq!(s.food_count, 62);
        for sn in &s.snakes {
            assert_eq!(sn.len(), 3);
            assert_eq!(sn.dir, Dir::Up);
        }
        // This is the largest map — verify corner platforms
        assert!(s.is_platform(Pos::new(0, 22)));
        assert!(s.is_platform(Pos::new(41, 22)));
    }

    #[test]
    fn test_map_877977892_initial_state() {
        // 40×22, game 877977892 — 8 snakes, 54 food items
        let s = from_cg_map(
            40, 22,
            &[
                ".......#....#...##....##...#....#.......",
                "........#....#............#....#........",
                ".........#....................#.........",
                "..#..................................#..",
                "...#................................#...",
                "#................#....#................#",
                ".##...........###......###...........##.",
                "...............#........#...............",
                ".##..................................##.",
                "..#......#....................#......#..",
                "..#.....#......................#.....#..",
                "........................................",
                "..##..........#.###..###.#..........##..",
                "..#.....#....#....#..#....#....#.....#..",
                "....#...#......................#...#....",
                ".....#.........#........#.........#.....",
                "...............#........#...............",
                "#...###.....#..............#.....###...#",
                "##..##......#....#....#....#......##..##",
                "###.......####..###..###..####.......###",
                "###......######################......###",
                "########################################",
            ],
            &[
                (14,1),(25,1),(3,2),(36,2),(19,2),(20,2),(15,3),(24,3),
                (17,4),(22,4),(18,4),(21,4),(8,6),(31,6),(16,9),(23,9),
                (6,11),(33,11),(10,11),(29,11),(10,13),(29,13),(19,13),(20,13),
                (9,14),(30,14),(18,14),(21,14),(0,15),(39,15),(3,16),(36,16),
                (6,19),(33,19),(15,2),(24,2),(5,4),(34,4),(5,6),(34,6),
                (11,8),(28,8),(6,10),(33,10),(0,11),(39,11),(11,11),(28,11),
                (5,12),(34,12),(0,13),(39,13),(11,15),(28,15),
            ],
            &[
                (0, 0, &[(0,2),(0,3),(0,4)]),
                (1, 0, &[(9,17),(9,18),(9,19)]),
                (2, 0, &[(9,6),(9,7),(9,8)]),
                (3, 0, &[(36,18),(36,19),(36,20)]),
                (4, 1, &[(39,2),(39,3),(39,4)]),
                (5, 1, &[(30,17),(30,18),(30,19)]),
                (6, 1, &[(30,6),(30,7),(30,8)]),
                (7, 1, &[(3,18),(3,19),(3,20)]),
            ],
        );
        assert_eq!(s.width, 40);
        assert_eq!(s.height, 22);
        assert_eq!(s.snakes.len(), 8);
        assert_eq!(s.food_count, 54);
        for sn in &s.snakes {
            assert_eq!(sn.len(), 3);
            assert_eq!(sn.dir, Dir::Up);
        }
        assert!(s.is_platform(Pos::new(0, 21)));
    }

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

    // ── game log replay: 877959752 ────────────────────────────────────

    /// Replay moves from game log 877959752 (32×17, yoannk (P0) vs m315 (P1)).
    /// Confirms the real-game head-on-food collision at turn 12: our two snakes
    /// (5 and 7, both P1/m315) both step onto food at (19,8) in the same turn.
    #[test]
    fn test_replay_877959752_food_collision() {
        let mut s = from_cg_map(
            32, 17,
            &[
                "..............#..#..............",
                "..............#..#..............",
                ".........#............#.........",
                ".........#............#.........",
                "................................",
                ".........#.....##.....#.........",
                ".........#............#.........",
                ".......#.......##.......#.......",
                ".....##........##........##.....",
                "........#..............#........",
                "..##...#................#...##..",
                ".#....#.......#..#.......#....#.",
                ".#..........##....##..........#.",
                "###..#.........##.........#..###",
                "######.....#..####..#.....######",
                "#######..##############..#######",
                "################################",
            ],
            &[
                (13,13),(18,13),(21,5),(10,5),(26,9),(5,9),(20,12),(11,12),
                (7,11),(24,11),(30,6),(1,6),(23,11),(8,11),(8,7),(23,7),
                (29,3),(2,3),(29,9),(2,9),(4,3),(27,3),(2,7),(29,7),
                (12,8),(19,8),(0,9),(31,9),(7,13),(24,13),
            ],
            &[
                (0, 0, &[(25,12),(25,13),(25,14)]),
                (1, 0, &[(14,8),(14,9),(14,10)]),
                (2, 0, &[(10,12),(10,13),(10,14)]),
                (3, 0, &[(6,5),(6,6),(6,7)]),
                (4, 1, &[(6,12),(6,13),(6,14)]),
                (5, 1, &[(17,8),(17,9),(17,10)]),
                (6, 1, &[(21,12),(21,13),(21,14)]),
                (7, 1, &[(25,5),(25,6),(25,7)]),
            ],
        );

        // moves[t] = (p0_actions, p1_actions) from the CG replay log
        // Format: (snake_id, Dir)
        let moves: &[&[(u8, Dir)]] = &[
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Right),(3,Dir::Right),(4,Dir::Right),(5,Dir::Right),(6,Dir::Right),(7,Dir::Left )], // t1
            &[(0,Dir::Up  ),(1,Dir::Up  ),(2,Dir::Down ),(3,Dir::Right),(4,Dir::Up   ),(5,Dir::Right),(6,Dir::Up   ),(7,Dir::Left )], // t2
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Right),(3,Dir::Down ),(4,Dir::Right),(5,Dir::Up   ),(6,Dir::Right),(7,Dir::Down )], // t3
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Right),(3,Dir::Down ),(4,Dir::Down ),(5,Dir::Right),(6,Dir::Right),(7,Dir::Left )], // t4
            &[(0,Dir::Left),(1,Dir::Left),(2,Dir::Down ),(3,Dir::Left ),(4,Dir::Down ),(5,Dir::Right),(6,Dir::Right),(7,Dir::Left )], // t5
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Left ),(3,Dir::Down ),(4,Dir::Left ),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )], // t6
            &[(0,Dir::Down),(1,Dir::Right),(2,Dir::Down),(3,Dir::Left ),(4,Dir::Up   ),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )], // t7
            &[(0,Dir::Left),(1,Dir::Right),(2,Dir::Up  ),(3,Dir::Left ),(4,Dir::Up   ),(5,Dir::Left ),(6,Dir::Right),(7,Dir::Up  )], // t8
            &[(0,Dir::Down),(1,Dir::Right),(2,Dir::Left),(3,Dir::Left ),(4,Dir::Right),(5,Dir::Left ),(6,Dir::Up   ),(7,Dir::Left )], // t9
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Up  ),(3,Dir::Left ),(4,Dir::Right),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )], // t10
            &[(0,Dir::Up  ),(1,Dir::Up  ),(2,Dir::Up  ),(3,Dir::Left ),(4,Dir::Up   ),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )], // t11
            &[(0,Dir::Left),(1,Dir::Right),(2,Dir::Up  ),(3,Dir::Up   ),(4,Dir::Right),(5,Dir::Up   ),(6,Dir::Right),(7,Dir::Left )], // t12
        ];

        let mut collision_turn = None;
        let mut collision_snakes: (u8, u8) = (0, 0);
        let mut collision_pos = Pos::new(0, 0);

        for (t, turn_moves) in moves.iter().enumerate() {
            let acts: HashMap<u8, Dir> = turn_moves.iter().cloned().collect();

            // Before stepping: check if two snakes' proposed heads land on the
            // same food cell — that's the real-game collision we're capturing.
            let proposed: Vec<(u8, Pos)> = s.snakes.iter()
                .filter_map(|sn| {
                    acts.get(&sn.id).map(|&d| {
                        let (dx, dy) = d.delta();
                        (sn.id, sn.head().translate(dx, dy))
                    })
                })
                .collect();

            for i in 0..proposed.len() {
                for j in (i+1)..proposed.len() {
                    let pos = proposed[i].1;
                    if pos == proposed[j].1 && s.food[s.cell_idx(pos)] {
                        collision_turn  = Some(t + 1);
                        collision_snakes = (proposed[i].0, proposed[j].0);
                        collision_pos   = pos;
                    }
                }
            }

            s.step(&acts);
        }

        // Verified from the real CG replay: our snakes 5 and 7 (both P1/m315) both
        // step onto food at (19,8) on turn 12 — friendly-fire food collision.
        assert_eq!(collision_turn,    Some(12),        "collision should be turn 12");
        assert_eq!(collision_snakes,  (5, 7),          "snakes 5 and 7 collide");
        assert_eq!(collision_pos,     Pos::new(19, 8), "food cell at (19,8)");
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
        for (i, p) in s.body.iter().enumerate() {
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
