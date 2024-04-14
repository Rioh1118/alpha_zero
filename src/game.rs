/// 方向
enum Direction {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

/// ゲーム状態
pub struct State {
    /// 連続パスによる終了
    pass_end: bool,

    /// 石の配置
    /// |0  1  2  3  4  5  6  7|
    /// |8  9 10 11 12 13 14 15|
    /// |16 17 18 19 20 21 22 23|
    /// |24 25 26 27 28 29 30 31|
    /// |32 33 34 35 36 37 38 39|
    /// |40 41 42 43 44 45 46 47|
    /// |48 49 50 51 52 53 54 55|
    /// |56 57 58 59 60 61 62 63|
    pieces: [u8; 64], // 自分の石配置
    enemy_pieces: [u8; 64], // 相手の石配置
    depth: u8, // ターン数
}

/// 行動
#[derive(PartialEq, Clone, Copy)]
enum Action {
    /// 石を置く
    Put(u8),
    /// パス
    Pass
}

impl Default for State {
    fn default() -> Self {
        State {
            pass_end: false,
            pieces: [0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,1,0,0,0,0,
                    0,0,0,0,1,0,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0],
            enemy_pieces: [0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,
                           0,0,0,0,1,0,0,0,
                           0,0,0,1,0,0,0,0,
                           0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0],
            depth: 0,
        }
    }
}

impl State {
    /// コンストラクタ
    fn new(pieces: [u8; 64], enemy_pieces: [u8; 64], depth: u8, pass_end: bool) -> Self {
        State {
            pieces,
            enemy_pieces,
            depth,
            pass_end,
        }  
    }
    /// 石の数取得
    fn piece_count(&self, pieces: [u8; 64]) -> u8 {
        let mut count: u8 = 0;
        for &piece in pieces.iter() {
            count += piece;
        }
        count
    }
    /// 負けかどうか
    fn is_lose(&self) -> bool {
        self.is_done() && self.piece_count(self.pieces) < self.piece_count(self.enemy_pieces)
    }
    /// 引き分けかどうか
    fn is_draw(&self) -> bool {
        self.is_done() && self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)
    }
    /// ゲーム終了かどうか
    fn is_done(&self) -> bool {
        self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 64 || self.pass_end
    }
    /// 次の状態を取得
    fn next(&self, action: Action) -> Self {
        let mut state = State::new(self.pieces, self.enemy_pieces, self.depth + 1, false);
        
        if let Action::Put(pos) = action {
                state.is_legal_action_xy(pos%8, pos/8, true);
        }

        std::mem::swap(&mut state.pieces, &mut state.enemy_pieces);

        if action == Action::Pass && state.legal_actions().len() == 1 && state.legal_actions().contains(&Action::Pass) { // 直す
            state.pass_end = true;
        }
        state
    }
    /// 合法手のリストを取得
    fn legal_actions(&mut self) -> Vec<Action>{
        let mut actions: Vec<Action> = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                if self.is_legal_action_xy(i, j, false) {
                    actions.push(Action::Put(i+j*8))
                }
            }
        }
        if actions.is_empty() {
            actions.push(Action::Pass);
        }
        actions
    }

    fn is_legal_action_xy_dxy(&mut self,x: u8, y: u8, direction: Direction, flip: bool) -> bool {
        let (dx,dy) = match direction {
            Direction::Up => (0, -1),
            Direction::Down => (0,1),
            Direction::Left => (-1,0),
            Direction::Right => (1,0),
            Direction::UpLeft => (-1,-1),
            Direction::UpRight => (1,-1),
            Direction::DownLeft => (-1,1),
            Direction::DownRight => (1,1),
        };
        let mut x = x as i8 + dx;
        let mut y = y as i8 + dy;

        // 一つ目は相手の石でないといけない
        if !(0..=7).contains(&y) || !(0..=7).contains(&x) || self.enemy_pieces[(x+y*8) as usize] != 1 {
            return false;
        }

        // 二つ目以降
        for _j in 0..8 {
            // 空の時
            if !(0..=7).contains(&y) || !(0..=7).contains(&x) || (self.enemy_pieces[(x+y*8) as usize] == 0 && self.pieces[(x+y*8) as usize] == 0) {
                return false;
            }
            // 自分の石の時
            if self.pieces[(x+y*8) as usize] == 1 {
                // 反転させる場合
                if flip {
                    for _i in 0..8 {
                        x -= dx;
                        y -= dy;
                        if self.pieces[(x+y*8) as usize] == 1 {
                            return true
                        }
                        self.pieces[(x+y*8) as usize] = 1;
                        self.enemy_pieces[(x+y*8) as usize] = 0;
                    }
                }
                return true;
            }
            // 相手の石の時
            x += dx;
            y += dy;
        }
        false
    }

    /// 任意のマスが合法手かどうか
    fn is_legal_action_xy(&mut self, x: u8, y: u8, flip: bool) -> bool {

        // 空きがない場合
        if self.enemy_pieces[(x+y*8) as usize] == 1 || self.pieces[(x+y*8) as usize] == 1 {
            return false
        }
        
        // 意思を置く
        if flip {
            self.pieces[(x+y*8) as usize] = 1;
        }

        let mut flag = false;
        for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight, Direction::DownLeft, Direction::DownRight] {
            if self.is_legal_action_xy_dxy(x, y, direction, flip) {
                flag = true;
            }
        }
        flag
    }
    /// 先手かどうか
    fn is_first_player(&self) -> bool {
        self.depth % 2 == 0
    }
    /// 文字列の表示
    fn ___str___(&self) -> String {
        let ox = if self.is_first_player() {("o", "x")} else {("x", "o")};
        let mut str = "".to_string();
        for i in 0..64 {
            if self.pieces[i] == 1 {
                str.push_str(ox.0);
            } else if self.enemy_pieces[i] == 1 {
                str.push_str(ox.1);
            } else {
                str.push('-');
            }
            if i % 8 == 7 {
                str.push('\n');
            }
        }
        str
    } 

}
// /// ランダムで行動を選択する
// fn random_action(state: State){}
// /// アルファベータ法で状態価値を計算する
// fn alpha_beta(state: State, alpha: f64, beta: f64){}
// /// プレイアウト
// fn playout(state: State){}
// /// 最大値のインデックスを返す
// fn argmax(collection: Vec<f64>){}
// /// モンテカルロ木探索の行動選択
// fn mcts_action(state: State){}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_action(state: &mut State) -> Action {
        let legal_actions = state.legal_actions();

        let mut rng = rand::thread_rng();
        let random_number: usize = rng.gen_range(0..legal_actions.len());
        legal_actions[random_number]
    }
        

    #[test]
    fn test_overall() {
        let mut state = State::default();

        loop {

            if state.is_done() {
                break;
            }
            
            let action = random_action(&mut state);

            state = state.next(action);

            println!("{}", state.___str___());
            println!();
        }
    }
}