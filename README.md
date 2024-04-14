# AlphaZeroによるオセロの強化学習要素

|強化学習の要素 |オセロ|
|-------------|----|
|目的|勝つ|
|エピソード|終局まで|
|状態|局面|
|行動|手を打つ|
|報酬|勝ったら+1、負けたら-1|
|学習手法|モンテカルロ木探索+ResNet+セルフプレイ|
|パラメータの更新|1エピソードごと|

# 強化学習サイクル

```mermaid
graph TD;
A[強化学習の開始] --> B[<span style="font-size: 20px">デュアルネットワークの作成</span><br/>重みがランダムのみ学習状態]
B --> C[<span style="font-size: 20px">セルフプレイ部</span><br/>セルフプレイを500回繰り返し、学習データを作成]
C --> D[パラメータ学習部]
D --> E[新パラメータ評価部]
```

# ファイル構成
|ファイル名|説明|
|--------|----|
|game.rs|ゲーム状態|
|dual_network.rs|デュアルネットワーク|
|pv_mcts.rs|モンテカルロ木探索|
|self_play.rs|セルフプレイ部|
|train_network.rs|パラメータ更新部|
|evaluate_network.rs|新パラメータ評価部|
|train_cycle.rs|学習サイクルの実行|
|human_play.rs|ゲームUI|

# デュアルネットワークの作成
ResNetをモデルのベースとする

```mermaid
graph TD;
A[現在の局面] --- B[畳み込み層]
B --- C[残差ブロック]
C -.合計で16個.- D[残差ブロック]
D --- E[ブーリング層]
E --- F[ポリシー出力]
E --- G[バリュー出力]
F --- H[方策]
G --- I[価値]


style A stroke-width:0px, fill:none;
style H stroke-width:0px, fill:none;
style I stroke-width:0px, fill:none;
```
<html>
<body>
    <div>
        <h2>デュアルネットワークの入力</h2>
        <ul>
            <li>自分の石の配置(8x8の2次元配列)</li>
            <li>相手の石の配置(8x8の2次元配列)</li>
        </ul>
        <h2>デュアルネットワークの出力</h2>
        <ul>
            <li>方策 (要素が65で、要素の値の合計が「1」の配列)</li>
            <li>価値 (0 ~ 1の値を持つ長さ「1」の配列)</li>
        </ul>
    </div>
</body>
</html>

```mermaid
graph TD;
A[残差ブロック] --- B[<span style="font-size: 20px">畳み込み層</span><br/>3x3のカーネル128枚]
B --- C[BatchNormalization]
C --- D[ReLU]
D --- E[<span style="font-size: 20px">畳み込み層</span><br/>3x3のカーネル128枚]
E --- F[畳み込み層]
F --- G[BatchNormalization]
G --- H(Add)
H --> A
H --- I[ReLU]
I --- J{ }

style A stroke-width:0px, fill: none;
style J stroke-width:0px, fill: none;
```