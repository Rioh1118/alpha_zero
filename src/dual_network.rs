
use tensorflow::{
    ops::{self, Add, Const, Dense, GlobalAvgPool, Mean, Placeholder, Relu, Save, Variable},
    Graph, Operation, Output, Scope, Session, SessionOptions, Tensor, DataType, Shape, TensorContent,
};
use std::{path::Path, fs};

/// パラメータの準備
const DN_INPUT_SHAPE: (i32, i32, i32) = (8, 8, 2); // 入力シェイプ 8*8配列2つ
const DN_OUTPUT_SIZE: usize = 65; // 出力サイズ 8*8+1(パス)
const DN_FILTERS: i32 = 128; // 畳み込み層のカーネル数 
const DN_RESIDUAL_NUM: usize = 16; // 残差ブロックの数

/// 畳み込み層の作成
fn conv<'a>(scope: &mut Scope, input: Output, filters: i32) -> Result<Output, Box<dyn std::error::Error>> {
    let kernel_shape = Shape::from_slice(&[3, 3, input.tensor_shape()?.dim(3).unwrap() as i32, filters]);
    let kernel = Variable::builder()
        .data_type(DataType::Float)
        .shape(kernel_shape)
        .initializer("he_normal")
        .build(scope, "kernel")?;

    let conv = Conv2D::builder()
        .input(input)
        .filter(kernel)
        .strides(&[1, 1, 1, 1])
        .padding("SAME")
        .build(scope, "conv")?;
    Ok(conv)
}

fn residual_block<'a>(scope: &mut Scope, input: Output) -> Result<Output, Box<dyn std::error::Error>> {
    let sc = input;
    let mut x = conv(scope, sc, DN_FILTERS)?;
    x = batch_norm(scope, x, 1e-3f32)?;
    x = Relu::new().build(scope, x)?;
    x = conv(scope, x, DN_FILTERS)?;
    x = batch_norm(scope, x, 1e-3f32)?;
    let add = Add::new().build(scope, &[x, sc])?;
    let output = Relu::new().build(scope, add)?;
    Ok(output)
}

/// バッチ正規化の実装
fn batch_norm<'a>(
    scope: &mut Scope,
    input: Output,
    epsilon: f32,
) -> Result<Output, Box<dyn std::error::Error>> {
    let mean = ops::Mean::new()
        .reduction_indices(&[0, 1, 2])
        .keep_dims(true)
        .build(scope, input)?;
    
    let variance = ops::SquaredDifference::new()
        .build(scope, input, mean)?;

    let variance_mean = ops::Mean::new()
        .reduction_indices(&[0, 1, 2])
        .keep_dims(true)
        .build(scope, variance)?;
    
    let norm = ops::Sub::new()
        .build(scope, input, mean)?;

    let adj_variance = ops::Add::new()
        .build(scope, variance_mean, ops::Const::new(scope, epsilon.into())?)?;

    let std = ops::Sqrt::new().build(scope, adj_variance)?;

    let output = ops::Div::new()
        .build(scope, norm, std)?;

    Ok(output)
}

/// デュアルネットワークを構築し、訓練し、保存する
fn dual_network() -> Result<(), Box<dyn std::error::Error>> {
    // モデルファイルの存在確認
    if Path::new("./model/best.ckpt").exists() {
        println!("Model already exists.");
        return Ok(());
    }

    let mut graph = Graph::new();
    let scope = &mut Scope::new_root_scope().with_graph(&mut graph);

    // Input layer
    let input = Placeholder::new()
        .dtype(DataType::Float)
        .shape(Shape::from_slice(&[1, DN_INPUT_SHAPE.0, DN_INPUT_SHAPE.1, DN_INPUT_SHAPE.2]))
        .build(scope, "input")?;

    let mut x = conv(scope, input, DN_FILTERS)?;
    x = batch_norm(scope, x, 1e-3)?;
    x = Relu::new().build(scope, x)?;

    for _ in 0..DN_RESIDUAL_NUM {
        x = residual_block(scope, x)?;
    }

    x = GlobalAvgPool::new().build(scope, x, "global_avg_pool")?;
    let p = Dense::new()
        .units(DN_OUTPUT_SIZE as i64)
        .activation("softmax")
        .build(scope, x, "policy_output")?;
    let v = Dense::new()
        .units(1)
        .build(scope, x, "value_output_raw")?;
    let v = Relu::new().build(scope, v, "value_output")?;

    // モデルディレクトリの作成
    fs::create_dir_all("./model/")?;

    // セッションの設定と実行
    let session = Session::new(&SessionOptions::new(), &graph)?;
    let mut run_args = SessionRunArgs::new();
    run_args.add_target(&p);
    run_args.add_target(&v);
    session.run(&mut run_args)?;

    // モデルの保存
    let mut save_builder = Save::new();
    save_builder.folder("./model/").build(scope, "save")?;

    Ok(())
}