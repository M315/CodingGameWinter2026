use std::collections::HashMap;
use super::{Bot, GameState, Dir, greedy_actions};

pub struct GreedyBot;

impl Bot for GreedyBot {
    fn name(&self) -> &str { "GreedyBot" }
    fn choose_actions(&mut self, state: &GameState, player: u8) -> HashMap<u8, Dir> {
        greedy_actions(state, player)
    }
}
