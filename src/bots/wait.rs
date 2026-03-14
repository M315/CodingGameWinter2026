use std::collections::HashMap;
use super::{Bot, GameState, Dir};

pub struct WaitBot;

impl Bot for WaitBot {
    fn name(&self) -> &str { "WaitBot" }
    fn choose_actions(&mut self, _state: &GameState, _player: u8) -> HashMap<u8, Dir> {
        HashMap::new()
    }
}
