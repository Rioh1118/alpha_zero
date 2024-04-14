
/// パラメータの準備
const DN_INPUT_SHAPE: (i32, i32, i32) = (8, 8, 2); // 入力シェイプ 8*8配列2つ
const DN_OUTPUT_SIZE: usize = 65; // 出力サイズ 8*8+1(パス)
const DN_FILTERS: i32 = 128; // 畳み込み層のカーネル数 
const DN_RESIDUAL_NUM: usize = 16; // 残差ブロックの数

