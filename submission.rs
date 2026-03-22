#![allow(dead_code, unused_imports, unused_variables)]
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::io::{self, BufRead};

pub type DirArr = [Option<Dir>; 8];

struct BfsScratch {
    i32_buf: Vec<i32>,
    u8_buf:  Vec<u8>,
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

    static STEP_SCRATCH: RefCell<Vec<bool>> = RefCell::new(Vec::new());

    static GRAV_SCRATCH: RefCell<Vec<u8>>   = RefCell::new(Vec::new());

    static BFS_SCRATCH: RefCell<BfsScratch> = RefCell::new(BfsScratch {
        i32_buf: Vec::new(),
        u8_buf:  Vec::new(),
        queue:   VecDeque::new(),
    });

    static OBS_SCRATCH: RefCell<Vec<bool>> = RefCell::new(Vec::new());
    static SNG_SCRATCH: RefCell<Vec<u8>>   = RefCell::new(Vec::new());

    static LIB_SCRATCH: RefCell<(Vec<bool>, VecDeque<usize>)> =
        RefCell::new((Vec::new(), VecDeque::new()));

    static FOOD_DIST_CACHE: RefCell<Vec<i32>> = RefCell::new(Vec::new());
}

#[inline]
fn bb_shl_or(src: &[u64; 16], k: usize, dst: &mut [u64; 16], n: usize) {
    let rk = 64 - k;
    dst[0] |= src[0] << k;
    for i in 1..n { dst[i] |= (src[i] << k) | (src[i-1] >> rk); }
}

#[inline]
fn bb_shr_or(src: &[u64; 16], k: usize, dst: &mut [u64; 16], n: usize) {
    let lk = 64 - k;
    dst[n-1] |= src[n-1] >> k;
    for i in (0..n-1).rev() { dst[i] |= (src[i] >> k) | (src[i+1] << lk); }
}

#[derive(Clone)]
pub struct BfsBitsSetup {
    pub blocked: [u64; 16],
    pub tbits:   [u64; 16],
    pub rcol:    [u64; 16],
    pub lcol:    [u64; 16],
    pub n:       usize,
}

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

pub const MAX_SNAKE_LEN: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct SnakeBody {
    pub head:  Pos,
    pub tail:  Pos,
    pub len:   u16,
    start: u8,
    dirs:  [u8; MAX_SNAKE_LEN],
}

impl SnakeBody {

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

    #[inline] pub fn set_head(&mut self, p: Pos) { self.head = p; }

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

    #[inline]
    pub fn pop_back(&mut self) {
        if self.len <= 1 { self.len = 0; return; }

        let last = self.start.wrapping_add((self.len - 2) as u8) as usize;
        let (dx, dy) = Dir::from_u8(self.dirs[last]).delta();
        self.tail = Pos { x: self.tail.x - dx, y: self.tail.y - dy };
        self.len -= 1;
    }

    #[inline]
    pub fn pop_front(&mut self) {
        if self.len <= 1 { self.len = 0; return; }
        let (dx, dy) = Dir::from_u8(self.dirs[self.start as usize]).delta();
        self.head = Pos { x: self.head.x + dx, y: self.head.y + dy };
        self.start = self.start.wrapping_add(1);
        self.len -= 1;
        if self.len == 1 { self.tail = self.head; }
    }

    #[inline]
    pub fn apply_dy(&mut self, dy: i32) {
        self.head.y += dy;
        self.tail.y += dy;
    }
}

impl Default for SnakeBody {
    fn default() -> Self {
        SnakeBody { head: Pos::default(), tail: Pos::default(),
                    len: 0, start: 0, dirs: [0u8; MAX_SNAKE_LEN] }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Snake {
    pub id: u8,

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

pub fn infer_dir(parts: &[Pos]) -> Dir {
    if parts.len() < 2 { return Dir::Up; }
    match (parts[0].x - parts[1].x, parts[0].y - parts[1].y) {
        (0, -1) => Dir::Up,   (0, 1) => Dir::Down,
        (-1, 0) => Dir::Left, (1, 0) => Dir::Right,
        _ => Dir::Up,
    }
}

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

impl std::ops::Deref for SnakeVec {
    type Target = [Snake];
    #[inline] fn deref(&self) -> &[Snake] { &self.data[..self.len as usize] }
}
impl std::ops::DerefMut for SnakeVec {
    #[inline] fn deref_mut(&mut self) -> &mut [Snake] { &mut self.data[..self.len as usize] }
}

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

#[derive(Clone, Debug)]
pub struct GameState {
    pub width:  i32,
    pub height: i32,

    pub grid: Arc<Vec<bool>>,

    pub food: Vec<bool>,

    pub food_count: u32,

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

    pub fn add_food(&mut self, p: Pos) {
        if p.in_bounds(self.width, self.height) {
            let ci = self.cell_idx(p);
            if !self.food[ci] {
                self.food[ci] = true;
                self.food_count += 1;
            }
        }
    }

    pub fn clear_food(&mut self) {
        self.food.fill(false);
        self.food_count = 0;
    }

    #[inline]
    pub fn is_platform(&self, p: Pos) -> bool {
        p.in_bounds(self.width, self.height)
            && self.grid[(p.y * self.width + p.x) as usize]
    }

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

        self.snakes.iter_mut().for_each(|s| {
            if let Some(&d) = actions.get(&s.id) { s.dir = d; }
        });

        self.step_phases_2_to_11();
    }

    fn step_phases_2_to_11(&mut self) {
        let n = self.snakes.len();

        let mut proposed = [Pos::new(0, 0); 8];
        self.snakes.iter().enumerate().for_each(|(i, s)| {
            let (dx, dy) = s.dir.delta();
            proposed[i] = s.head().translate(dx, dy);
        });

        let w = self.width as usize;
        let mut eaters = [false; 32];
        for (i, &h) in proposed.iter().enumerate() {
            if h.in_bounds(self.width, self.height) {
                let ci = h.y as usize * w + h.x as usize;
                if self.food[ci] { eaters[i] = true; }
            }
        }

        let grid_size = (self.width * self.height) as usize;
        let mut head_destroyed = [false; 8];
        STEP_SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() < grid_size { g.resize(grid_size, false); }
            let body_cells = &mut g[..grid_size];
            body_cells.fill(false);

            for (idx, s) in self.snakes.iter().enumerate() {
                let end = if eaters[idx] { s.len() } else { s.len().saturating_sub(1) };
                s.body.iter().take(end).for_each(|p| {
                    if p.in_bounds(self.width, self.height) {
                        body_cells[p.y as usize * w + p.x as usize] = true;
                    }
                });
            }

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

        let mut ri = 0usize;
        self.snakes.retain(|_| { let keep = !should_remove[ri]; ri += 1; keep });

        self.apply_gravity();

        let (w_i, h_i) = (self.width, self.height);
        self.snakes.retain(|s| s.body.iter().all(|p| p.in_bounds(w_i, h_i)));

        self.turn += 1;
    }

    pub fn apply_gravity(&mut self) {
        let size = (self.width * self.height) as usize;
        let w    = self.width as usize;
        let pow  = &self.food;

        let mut to_fall     = [0usize; 8];
        let mut to_fall_len = 0usize;

        GRAV_SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();

            if g.len() < size { g.resize(size, u8::MAX); }
            let snake_at = &mut g[..size];

            loop {

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

    pub fn power_grid(&self) -> Vec<bool> {
        self.food.clone()
    }

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

            for (ci, &b) in self.food.iter().enumerate() {
                if b {
                    dist[ci] = 0;
                    queue.push_back(ci);
                }
            }

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

    #[inline]
    pub fn cached_food_dist(&self, pos: Pos) -> i32 {
        if !pos.in_bounds(self.width, self.height) { return -1; }
        let ci = pos.y as usize * self.width as usize + pos.x as usize;
        FOOD_DIST_CACHE.with(|cell| cell.borrow()[ci])
    }

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

    #[inline]
    fn is_grounded_cell_ci(&self, ci: usize, w: usize, pow: &[bool]) -> bool {
        let below_ci = ci + w;
        below_ci >= self.grid.len()
            || self.grid[below_ci]
            || pow[below_ci]
    }

    #[inline]
    fn is_grounded_cell_ci_sng(&self, ci: usize, w: usize, pow: &[bool], sng: &[u8]) -> bool {
        let below_ci = ci + w;
        below_ci >= self.grid.len()
            || self.grid[below_ci]
            || pow[below_ci]
            || sng[below_ci] != u8::MAX
    }

    #[inline]
    pub fn is_grounded_cell(&self, p: Pos) -> bool {
        let below_y = p.y + 1;
        if below_y >= self.height { return true; }
        if p.x < 0 || p.x >= self.width { return false; }
        let below_ci = below_y as usize * self.width as usize + p.x as usize;
        self.grid[below_ci] || self.food[below_ci]
    }

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

    pub fn prepare_bfs_bits(&self, targets: &[bool], obs: &[bool]) -> BfsBitsSetup {
        let w    = self.width as usize;
        let size = w * self.height as usize;
        let n    = (size + 63) >> 6;

        let mut blocked = [!0u64; 16];
        let mut tbits   = [0u64;  16];
        for i in 0..size {
            if !self.grid[i] && !obs[i] { blocked[i >> 6] &= !(1u64 << (i & 63)); }
            if targets[i]               { tbits  [i >> 6] |=   1u64 << (i & 63);  }
        }

        let mut rcol = [0u64; 16];
        let mut lcol = [0u64; 16];
        for y in 0..self.height as usize {
            let ri = y * w + (w - 1);
            let li = y * w;
            rcol[ri >> 6] |= 1u64 << (ri & 63);
            lcol[li >> 6] |= 1u64 << (li & 63);
        }

        BfsBitsSetup { blocked, tbits, rcol, lcol, n }
    }

    pub fn bfs_dist_bits_with(&self, start: Pos, s: &BfsBitsSetup) -> i32 {
        let w  = self.width as usize;
        let n  = s.n;
        let start_ci = start.y as usize * w + start.x as usize;
        if s.tbits[start_ci >> 6] & (1u64 << (start_ci & 63)) != 0 { return 0; }

        let size = w * self.height as usize;
        let mut front = [0u64; 16];
        front[start_ci >> 6] |= 1u64 << (start_ci & 63);
        let mut visited = front;

        for dist in 1..=size as i32 {
            let mut exp = [0u64; 16];

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
            let mut new_front = [0u64; 16];
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

    pub fn bfs_dist_bits(&self, start: Pos, targets: &[bool], obs: &[bool]) -> i32 {
        let setup = self.prepare_bfs_bits(targets, obs);
        self.bfs_dist_bits_with(start, &setup)
    }

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
            if dist[ci] != -1 { continue; }
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

                if !targets[ni] && !self.is_grounded_cell_ci(ni, w, targets) { continue; }
                dist[ni] = d + 1;
                q.push_back(ni);
            }
        }
        dist
    }

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
                    if !targets[ni] && !self.is_grounded_cell_ci(ni, w, targets) { continue; }
                    first_dir[ni] = fd;
                    if targets[ni] { return Some(dirs[fd as usize]); }
                    queue.push_back(ni);
                }
            }
            None
        })
    }

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

#[cfg(test)]
mod tests {

    fn open(w: i32, h: i32) -> GameState {
        GameState::new(w, h, vec![false; (w * h) as usize])
    }

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

        for sn in &s.snakes {
            assert_eq!(sn.len(), 3);
            assert_eq!(sn.dir, Dir::Up);
        }

        assert!(!s.is_platform(Pos::new(25, 12)));

        assert!(s.is_platform(Pos::new(0, 16)));
    }

    #[test]
    fn test_map_877959870_initial_state() {

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

        assert!(s.is_platform(Pos::new(0, 22)));
        assert!(s.is_platform(Pos::new(41, 22)));
    }

    #[test]
    fn test_map_877977892_initial_state() {

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

        let moves: &[&[(u8, Dir)]] = &[
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Right),(3,Dir::Right),(4,Dir::Right),(5,Dir::Right),(6,Dir::Right),(7,Dir::Left )],
            &[(0,Dir::Up  ),(1,Dir::Up  ),(2,Dir::Down ),(3,Dir::Right),(4,Dir::Up   ),(5,Dir::Right),(6,Dir::Up   ),(7,Dir::Left )],
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Right),(3,Dir::Down ),(4,Dir::Right),(5,Dir::Up   ),(6,Dir::Right),(7,Dir::Down )],
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Right),(3,Dir::Down ),(4,Dir::Down ),(5,Dir::Right),(6,Dir::Right),(7,Dir::Left )],
            &[(0,Dir::Left),(1,Dir::Left),(2,Dir::Down ),(3,Dir::Left ),(4,Dir::Down ),(5,Dir::Right),(6,Dir::Right),(7,Dir::Left )],
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Left ),(3,Dir::Down ),(4,Dir::Left ),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )],
            &[(0,Dir::Down),(1,Dir::Right),(2,Dir::Down),(3,Dir::Left ),(4,Dir::Up   ),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )],
            &[(0,Dir::Left),(1,Dir::Right),(2,Dir::Up  ),(3,Dir::Left ),(4,Dir::Up   ),(5,Dir::Left ),(6,Dir::Right),(7,Dir::Up  )],
            &[(0,Dir::Down),(1,Dir::Right),(2,Dir::Left),(3,Dir::Left ),(4,Dir::Right),(5,Dir::Left ),(6,Dir::Up   ),(7,Dir::Left )],
            &[(0,Dir::Left),(1,Dir::Up  ),(2,Dir::Up  ),(3,Dir::Left ),(4,Dir::Right),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )],
            &[(0,Dir::Up  ),(1,Dir::Up  ),(2,Dir::Up  ),(3,Dir::Left ),(4,Dir::Up   ),(5,Dir::Up   ),(6,Dir::Up   ),(7,Dir::Up  )],
            &[(0,Dir::Left),(1,Dir::Right),(2,Dir::Up  ),(3,Dir::Up   ),(4,Dir::Right),(5,Dir::Up   ),(6,Dir::Right),(7,Dir::Left )],
        ];

        let mut collision_turn = None;
        let mut collision_snakes: (u8, u8) = (0, 0);
        let mut collision_pos = Pos::new(0, 0);

        for (t, turn_moves) in moves.iter().enumerate() {
            let acts: HashMap<u8, Dir> = turn_moves.iter().cloned().collect();

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

        assert_eq!(collision_turn,    Some(12),        "collision should be turn 12");
        assert_eq!(collision_snakes,  (5, 7),          "snakes 5 and 7 collide");
        assert_eq!(collision_pos,     Pos::new(19, 8), "food cell at (19,8)");
    }
}

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

pub trait Bot {
    fn name(&self) -> &str;

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir>;
}

pub fn greedy_actions(state: &GameState, player: u8) -> HashMap<u8, Dir> {

    let obs = state.build_obstacles();

    state.snakes.iter()
        .filter(|s| s.player == player)
        .filter_map(|s| {
            state.bfs_first_step_grounded(s.head(), &state.food, &obs)
                .map(|d| (s.id, d))
        })
        .collect()
}

pub fn dirmap_to_hashmap(player: u8, arr: &DirArr) -> HashMap<u8, Dir> {

    let _ = player;
    arr.iter().enumerate()
        .filter_map(|(id, &opt)| opt.map(|d| (id as u8, d)))
        .collect()
}

pub fn greedy_dirmap(state: &GameState, player: u8) -> DirArr {
    let obs = state.build_obstacles();
    let mut result: DirArr = [None; 8];
    state.snakes.iter()
        .filter(|s| s.player == player)
        .for_each(|s| {
            if let Some(d) = state.bfs_first_step_grounded(s.head(), &state.food, &obs) {
                result[s.id as usize] = Some(d);
            }
        });
    result
}

pub fn old_greedy_dirmap(state: &GameState, player: u8) -> DirArr {
    state.with_obstacles(|obs| {
        let mut result: DirArr = [None; 8];
        state.snakes.iter()
            .filter(|s| s.player == player)
            .for_each(|s| {
                if let Some(d) = state.bfs_first_step(s.head(), &state.food, obs) {
                    result[s.id as usize] = Some(d);
                }
            });
        result
    })
}

pub fn greedy_dirmap_fast(state: &GameState, player: u8) -> DirArr {
    let w = state.width as usize;
    state.with_obstacles(|obs| {
        let mut result: DirArr = [None; 8];
        state.snakes.iter()
            .filter(|s| s.player == player)
            .for_each(|s| {
                let head = s.head();
                let best = Dir::all().iter()
                    .filter_map(|&dir| {
                        let (dx, dy) = dir.delta();
                        let (nx, ny) = (head.x + dx, head.y + dy);
                        if nx < 0 || ny < 0 || nx >= state.width || ny >= state.height {
                            return None;
                        }
                        let ni = ny as usize * w + nx as usize;
                        if state.grid[ni] || obs[ni] { return None; }
                        let d = state.cached_food_dist(Pos::new(nx as i32, ny as i32));

                        Some((dir, if d == -1 { i32::MAX } else { d }))
                    })
                    .min_by_key(|&(_, d)| d)
                    .map(|(dir, _)| dir);
                result[s.id as usize] = best;
            });
        result
    })
}

pub fn gen_combos(state: &GameState, player: u8) -> Vec<DirArr> {
    let ids: Vec<u8> = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| s.id)
        .collect();

    if ids.is_empty() { return vec![[None; 8]]; }

    let mut combos: Vec<DirArr> = vec![[None; 8]];
    for id in ids {
        let s = state.snakes.iter().find(|s| s.id == id).unwrap();
        let neck = (s.len() >= 2).then(|| s.body.get(1).unwrap());
        let dirs: Vec<Dir> = Dir::all().iter()
            .filter(|&&d| {
                let (dx, dy) = d.delta();
                Some(s.head().translate(dx, dy)) != neck
            })
            .copied()
            .collect();
        let mut next = Vec::with_capacity(combos.len() * dirs.len());
        for &existing in &combos {
            for &d in &dirs {
                let mut c = existing;
                c[id as usize] = Some(d);
                next.push(c);
            }
        }
        combos = next;
    }
    combos
}

pub fn gen_action_combos(state: &GameState, player: u8) -> Vec<HashMap<u8, Dir>> {
    let ids: Vec<u8> = state.snakes.iter()
        .filter(|s| s.player == player).map(|s| s.id).collect();

    if ids.is_empty() { return vec![HashMap::new()]; }

    let mut combos: Vec<HashMap<u8, Dir>> = vec![HashMap::new()];
    for id in ids {
        let s = state.snakes.iter().find(|s| s.id == id).unwrap();

        let neck = (s.len() >= 2).then(|| s.body.get(1).unwrap());
        let dirs: Vec<Dir> = Dir::all().iter()
            .filter(|&&d| {
                let (dx, dy) = d.delta();
                Some(s.head().translate(dx, dy)) != neck
            })
            .copied()
            .collect();
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

const COMBO_CAP: usize = 9;

fn rank_and_prune_combos(combos: &mut Vec<DirArr>, greedy: &DirArr, state: &GameState, player: u8) {
    if combos.len() <= COMBO_CAP { return; }
    let w = state.width as usize;

    let mut heads: [Option<Pos>; 8] = [None; 8];
    state.snakes.iter()
        .filter(|s| s.player == player)
        .for_each(|s| heads[s.id as usize] = Some(s.head()));

    combos.sort_by_cached_key(|c| {
        let mut score = 0i32;
        for id in 0..8usize {
            let Some(dir) = c[id] else { continue };
            if let Some(h) = heads[id] {
                let (dx, dy) = dir.delta();
                let (nx, ny) = (h.x + dx, h.y + dy);
                if nx >= 0 && ny >= 0 && nx < state.width && ny < state.height
                    && state.food[ny as usize * w + nx as usize] {
                    score += 100;
                }
            }
            if greedy[id] == Some(dir) { score += 2; }
        }
        -score
    });
    combos.truncate(COMBO_CAP);
}

pub struct BeamSearchBot {
    pub beam_width:   usize,
    pub horizon:      usize,
    pub time_limit:   Duration,
    pub heuristic_fn: fn(&GameState, u8) -> i32,
    pub dirmap_fn:    fn(&GameState, u8) -> DirArr,
}

impl BeamSearchBot {
    pub fn new(
        beam_width: usize,
        horizon: usize,
        time_limit_ms: u64,
        heuristic_fn: fn(&GameState, u8) -> i32,
    ) -> Self {
        BeamSearchBot {
            beam_width,
            horizon,
            time_limit: Duration::from_millis(time_limit_ms),
            heuristic_fn,
            dirmap_fn: old_greedy_dirmap,
        }
    }

    pub fn new_full(
        beam_width: usize,
        horizon: usize,
        time_limit_ms: u64,
        heuristic_fn: fn(&GameState, u8) -> i32,
        dirmap_fn: fn(&GameState, u8) -> DirArr,
    ) -> Self {
        BeamSearchBot {
            beam_width, horizon,
            time_limit: Duration::from_millis(time_limit_ms),
            heuristic_fn,
            dirmap_fn,
        }
    }
}

pub fn old_heuristic(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    let food_bonus: i32 = state.with_obstacles(|obs| {
        let setup = state.prepare_bfs_bits(&state.food, obs);
        state.snakes.iter()
            .filter(|s| s.player == player)
            .map(|s| {
                let d = state.bfs_dist_bits_with(s.head(), &setup);
                if d == i32::MAX { -30 } else { 20 - d.min(20) }
            })
            .sum()
    });

    my * 100 - opp * 80 + food_bonus
}

pub fn heuristic_v6(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    let food_bonus: i32 = state.with_obstacles(|obs| {
        let setup = state.prepare_bfs_bits(&state.food, obs);
        state.snakes.iter()
            .filter(|s| s.player == player)
            .map(|s| {
                let d = state.bfs_dist_bits_with(s.head(), &setup);
                if d == i32::MAX { -30 } else { 20 - d.min(20) }
            })
            .sum()
    });

    let danger_penalty: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|my_snake| {
            let h = my_snake.head();
            state.snakes.iter()
                .filter(|o| o.player != player)
                .map(|opp_snake| {
                    let oh = opp_snake.head();
                    let dist = (h.x - oh.x).abs() + (h.y - oh.y).abs();
                    if dist <= 1 { -8 } else if dist <= 2 { -4 } else { 0 }
                })
                .sum::<i32>()
        })
        .sum();

    my * 100 - opp * 80 + food_bonus + danger_penalty
}

pub fn heuristic_v7(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.cached_food_dist(s.head());
            if d == -1 { -30 } else { 20 - d.min(20) }
        })
        .sum();

    my * 100 - opp * 80 + food_bonus
}

pub fn heuristic_v1(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;
    let w   = state.width as usize;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();

    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {

            let d = state.bfs_dist_grounded_sng(s.head(), &state.food, &obs, &sng);
            if d == i32::MAX { -50 } else { 20 - d.min(20) }
        })
        .sum();

    let stability = stability_score(state, player, &sng, w);
    my * 100 - opp * 80 + food_bonus + stability
}

fn stability_score(state: &GameState, player: u8, sng: &[u8], w: usize) -> i32 {
    state.snakes.iter().enumerate()
        .filter(|(_, s)| s.player == player)
        .map(|(snake_idx, s)| {
            let grounded = s.body.iter().any(|p| {
                let below_y = p.y + 1;
                if below_y >= state.height { return true; }
                if p.x < 0 || p.x >= state.width { return false; }
                let below_ci = below_y as usize * w + p.x as usize;
                if state.grid[below_ci] { return true; }
                if state.food[below_ci] { return true; }
                let sat = sng[below_ci];
                sat != u8::MAX && sat as usize != snake_idx
            });
            if grounded { 0 } else { -120 }
        })
        .sum()
}

pub fn heuristic_v2(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my_score  = state.score(player) as i32;
    let opp_score = state.score(1 - player) as i32;
    let score_delta = my_score * 100 - opp_score * 80;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();
    let w   = state.width as usize;

    let stability = stability_score(state, player, &sng, w);

    let food_cis: Vec<usize> = state.food.iter().enumerate()
        .filter(|(_, &b)| b)
        .map(|(ci, _)| ci)
        .collect();

    if food_cis.is_empty() {
        return score_delta + stability;
    }

    let my_snakes: Vec<_> = state.snakes.iter().filter(|s| s.player == player).collect();
    let op_snakes: Vec<_> = state.snakes.iter().filter(|s| s.player != player).collect();
    let n_my   = my_snakes.len();
    let n_food = food_cis.len();

    let dist_my: Vec<Vec<i32>> = my_snakes.iter()
        .map(|s| state.bfs_dist_map_grounded(s.head(), &state.food, &obs))
        .collect();
    let dist_op: Vec<Vec<i32>> = op_snakes.iter()
        .map(|s| state.bfs_dist_map_grounded(s.head(), &state.food, &obs))
        .collect();

    let best_op_dist: Vec<i32> = food_cis.iter().map(|&ci| {
        dist_op.iter()
            .map(|d| d[ci])
            .filter(|&d| d >= 0)
            .min()
            .unwrap_or(i32::MAX)
    }).collect();

    let mut triples: Vec<(usize, usize, i32)> =
        Vec::with_capacity(n_food * n_my);
    for fi in 0..n_food {
        let ci = food_cis[fi];
        for si in 0..n_my {
            let d = dist_my[si][ci];
            if d >= 0 { triples.push((fi, si, d)); }
        }
    }
    triples.sort_unstable_by_key(|&(_, _, d)| d);

    let mut food_claimed  = vec![false; n_food];
    let mut snake_food: Vec<Option<(usize, i32)>> = vec![None; n_my];
    for (fi, si, d) in triples {
        if food_claimed[fi] || snake_food[si].is_some() { continue; }
        food_claimed[fi] = true;
        snake_food[si]   = Some((fi, d));
    }

    let food_score: i32 = (0..n_my).map(|si| {
        match snake_food[si] {

            None => -50,

            Some((fi, my_d)) => {
                let op_d      = best_op_dist[fi];
                let proximity = 20 - my_d.min(20);

                if op_d == i32::MAX {

                    proximity + 20
                } else if my_d < op_d {

                    proximity + 10
                } else if my_d == op_d {

                    proximity - 5
                } else {

                    -15
                }
            }
        }
    }).sum();

    score_delta + food_score + stability
}

pub fn heuristic_v3(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let score_delta = state.score(player) as i32 * 100
                    - state.score(1 - player) as i32 * 80;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();
    let w   = state.width as usize;

    let stability = stability_score(state, player, &sng, w);

    let food_cis: Vec<usize> = state.food.iter().enumerate()
        .filter(|(_, &b)| b)
        .map(|(ci, _)| ci)
        .collect();

    if food_cis.is_empty() {
        return score_delta + stability;
    }

    let my_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| s.head())
        .collect();
    let op_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player != player)
        .map(|s| s.head())
        .collect();

    let n_my = my_heads.len();

    let (my_dist, my_src) = state.bfs_multisource_dist_map(&my_heads, &state.food, &obs);
    let (op_dist, _)      = state.bfs_multisource_dist_map(&op_heads, &state.food, &obs);

    let mut snake_food: Vec<Option<(usize, i32)>> = vec![None; n_my];
    let mut food_assigned = vec![false; food_cis.len()];

    let mut sorted: Vec<(usize, usize, i32)> =
        food_cis.iter().enumerate()
            .filter_map(|(fi, &ci)| {
                let d = my_dist[ci];
                let s = my_src[ci];
                if d >= 0 && s != u8::MAX { Some((fi, s as usize, d)) } else { None }
            })
            .collect();
    sorted.sort_unstable_by_key(|&(_, _, d)| d);

    for (fi, si, d) in sorted {
        if food_assigned[fi] || snake_food[si].is_some() { continue; }
        food_assigned[fi]  = true;
        snake_food[si]     = Some((fi, d));
    }

    let food_score: i32 = (0..n_my).map(|si| {
        match snake_food[si] {
            None => -50,
            Some((fi, my_d)) => {
                let op_d = {
                    let d = op_dist[food_cis[fi]];
                    if d < 0 { i32::MAX } else { d }
                };
                let proximity = 20 - my_d.min(20);
                if      op_d == i32::MAX { proximity + 20 }
                else if my_d < op_d      { proximity + 10 }
                else if my_d == op_d     { proximity -  5 }
                else                     { -15             }
            }
        }
    }).sum();

    score_delta + food_score + stability
}

pub fn heuristic_v4(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let score_delta = state.score(player) as i32 * 100
                    - state.score(1 - player) as i32 * 80;

    let obs = state.build_obstacles();
    let sng = state.snake_grid();
    let w   = state.width as usize;

    let stability = stability_score(state, player, &sng, w);

    let food_bonus: i32 = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| {
            let d = state.bfs_dist_grounded(s.head(), &state.food, &obs);
            if d == i32::MAX { -50 } else { 20 - d.min(20) }
        })
        .sum();

    let my_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player == player)
        .map(|s| s.head())
        .collect();
    let op_heads: Vec<Pos> = state.snakes.iter()
        .filter(|s| s.player != player)
        .map(|s| s.head())
        .collect();

    let (my_dist, _) = state.bfs_multisource_dist_map(&my_heads, &state.food, &obs);
    let (op_dist, _) = state.bfs_multisource_dist_map(&op_heads, &state.food, &obs);

    let territory: i32 = my_dist.iter().zip(op_dist.iter())
        .map(|(&md, &od)| match (md, od) {
            (m, o) if m < 0 && o < 0 => 0,
            (m, _) if m < 0          => -1,
            (_, o) if o < 0          =>  1,
            (m, o) if m < o          =>  1,
            (m, o) if o < m          => -1,
            _                        =>  0,
        })
        .sum();

    score_delta + food_bonus + territory * 3 + stability
}

pub fn heuristic_v5(state: &GameState, player: u8) -> i32 {
    if !state.snakes_alive(player) { return i32::MIN / 2; }

    let my  = state.score(player) as i32;
    let opp = state.score(1 - player) as i32;

    const PENALTY_0_MOVES: i32 = 120;

    let (food_bonus, liberty_penalty): (i32, i32) = state.with_obstacles(|obs| {
        let mut food_sum = 0i32;
        let mut lib_pen  = 0i32;
        let w = state.width as usize;
        for s in state.snakes.iter().filter(|s| s.player == player) {
            let d = state.bfs_dist(s.head(), &state.food, obs);
            food_sum += if d == i32::MAX { -30 } else { 20 - d.min(20) };

            let open: usize = Dir::all().iter()
                .filter(|&&dir| {
                    let (dx, dy) = dir.delta();
                    let (nx, ny) = (s.head().x + dx, s.head().y + dy);
                    if nx < 0 || ny < 0 || nx >= state.width || ny >= state.height {
                        return false;
                    }
                    let ni = ny as usize * w + nx as usize;
                    !state.grid[ni] && !obs[ni]
                })
                .count();
            if open == 0 {
                lib_pen -= PENALTY_0_MOVES;
            }
        }
        (food_sum, lib_pen)
    });

    my * 100 - opp * 80 + food_bonus + liberty_penalty
}

impl Bot for BeamSearchBot {
    fn name(&self) -> &str { "BeamSearchBot" }

    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {

        let limit = if state.turn == 0 {
            Duration::from_millis(950)
        } else {
            self.time_limit
        };
        let t0 = Instant::now();

        state.cache_food_dist();

        type BeamItem = (DirArr, GameState, i32);

        #[inline]
        fn merge_dirs(a: &DirArr, b: &DirArr) -> DirArr {
            let mut out = *a;
            for i in 0..8 { if out[i].is_none() { out[i] = b[i]; } }
            out
        }

        let first_combos = gen_combos(state, player);
        if first_combos.is_empty() { return HashMap::new(); }

        let opp = (self.dirmap_fn)(state, 1 - player);
        let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
            let mut ns = state.clone();
            ns.step_arr(&merge_dirs(&first, &opp));
            let score = (self.heuristic_fn)(&ns, player);
            (first, ns, score)
        }).collect();
        beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        beam.truncate(self.beam_width);

        let mut result: HashMap<u8, Dir> = dirmap_to_hashmap(player, &beam[0].0);

        let mut total_states: u32 = beam.len() as u32;
        let mut max_depth: u32 = 0;

        for _depth in 1..self.horizon {
            if t0.elapsed() >= limit { break; }

            let cur_beam = std::mem::take(&mut beam);
            let mut next: Vec<BeamItem> = Vec::with_capacity(cur_beam.len() * 9);
            for (first_acts, cur, _) in cur_beam {
                if t0.elapsed() >= limit { break; }
                if cur.is_over() {
                    let score = (self.heuristic_fn)(&cur, player);
                    next.push((first_acts, cur, score));
                    continue;
                }
                let mut my_combos = gen_combos(&cur, player);

                if my_combos.len() > COMBO_CAP {
                    let my_pref = (self.dirmap_fn)(&cur, player);
                    rank_and_prune_combos(&mut my_combos, &my_pref, &cur, player);
                }
                let opp_acts  = (self.dirmap_fn)(&cur, 1 - player);
                for combo in my_combos {
                    let mut ns = cur.clone();
                    ns.step_arr(&merge_dirs(&combo, &opp_acts));
                    let score = (self.heuristic_fn)(&ns, player);
                    next.push((first_acts, ns, score));
                }
            }

            if next.is_empty() { break; }
            next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            next.truncate(self.beam_width);
            result = dirmap_to_hashmap(player, &next[0].0);
            total_states += next.len() as u32;
            max_depth = _depth as u32;
            beam = next;
        }

        eprintln!("BS={} MD={} T={}ms", total_states, max_depth, t0.elapsed().as_millis());
        result
    }
}

pub struct BeamHashMapBot {
    pub beam_width:   usize,
    pub horizon:      usize,
    pub time_limit:   Duration,
    pub heuristic_fn: fn(&GameState, u8) -> i32,
}

impl BeamHashMapBot {
    pub fn new(
        beam_width: usize,
        horizon: usize,
        time_limit_ms: u64,
        heuristic_fn: fn(&GameState, u8) -> i32,
    ) -> Self {
        BeamHashMapBot { beam_width, horizon, time_limit: Duration::from_millis(time_limit_ms), heuristic_fn }
    }
}

impl Bot for BeamHashMapBot {
    fn name(&self) -> &str { "BeamHashMapBot" }

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

        let opp = greedy_actions(state, 1 - player);
        let mut beam: Vec<BeamItem> = first_combos.into_iter().map(|first| {
            let mut combined = first.clone();
            for (&k, &v) in &opp { combined.entry(k).or_insert(v); }
            let mut ns = state.clone();
            ns.step(&combined);
            let score = (self.heuristic_fn)(&ns, player);
            (first, ns, score)
        }).collect();
        beam.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        beam.truncate(self.beam_width);

        let mut result: HashMap<u8, Dir> = beam.first()
            .map(|(a, _, _)| a.clone())
            .unwrap_or_default();

        for _depth in 1..self.horizon {
            if t0.elapsed() >= limit { break; }
            let cur_beam = std::mem::take(&mut beam);
            let mut next: Vec<BeamItem> = Vec::with_capacity(cur_beam.len() * 9);
            for (first_acts, cur, _) in cur_beam {
                if t0.elapsed() >= limit { break; }
                if cur.is_over() {
                    let score = (self.heuristic_fn)(&cur, player);
                    next.push((first_acts, cur, score));
                    continue;
                }
                let my_combos = gen_action_combos(&cur, player);
                let opp_acts  = greedy_actions(&cur, 1 - player);
                for combo in my_combos {
                    let mut combined = combo;
                    for (&k, &v) in &opp_acts { combined.entry(k).or_insert(v); }
                    let mut ns = cur.clone();
                    ns.step(&combined);
                    let score = (self.heuristic_fn)(&ns, player);
                    next.push((first_acts.clone(), ns, score));
                }
            }
            if next.is_empty() { break; }
            next.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            next.truncate(self.beam_width);
            result = next[0].0.clone();
            beam = next;
        }

        result
    }
}

macro_rules! read_line {
    ($lines:expr) => {{
        $lines.next().unwrap().unwrap()
    }};
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

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
    let mut bot: Box<dyn Bot> = Box::new(BeamSearchBot::new(160, 200, 40, old_heuristic));

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
        state.turn += 1;
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
    parts.sort();
    parts.join(";")
}
