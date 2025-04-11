<div align="center">

# <span style="color: #2c7be5;">PyTorch CNN 画像分類プロジェクト 🚀🧠</span>

[简体中文](README_CN.md) / [繁体中文](README_TC.md) / [English](README.md) / [Deutsch](README_DE.md) / 日本語

</div>

これはPyTorchフレームワークを使用して実装されたCNN画像分類プロジェクトで、完全なトレーニングプロセスとデータ処理機能を提供します。プロジェクトにはアテンションメカニズムが統合され、複数のデータ拡張方法をサポートし、完全なトレーニングと評価プロセスを提供します。

## <span style="color: #228be6;">プロジェクト構造 📁🗂️</span>

```
├── models/             # モデル関連の定義
│   ├── cnn.py         # CNNの基本モデル
│   └── attention.py   # アテンションメカニズムモジュール
├── data_processing/    # データ処理関連
│   └── dataset.py     # データセットの読み込みと前処理
├── trainers/          # トレーニング関連
│   └── trainer.py     # トレーナーの実装
├── utils/             # ユーティリティ関数
│   ├── config.py      # 設定管理
│   └── visualization.py # 可視化ツール
├── tests/             # テストコード
│   └── test_model.py  # モデルテスト
├── static/            # 静的リソース
├── templates/         # ウェブテンプレート
├── predict.py         # 予測スクリプト
├── main.py           # メインプログラムのエントリーポイント
└── requirements.txt   # プロジェクトの依存関係
```

## <span style="color: #228be6;">主な機能 ✨🌟</span>

<span style="color: #38d9a9;">- アテンションメカニズムを統合し、モデルの性能を向上させる</span>
<span style="color: #38d9a9;">- 複数のデータ拡張方法をサポートする</span>
<span style="color: #38d9a9;">- オンライン予測用のWebインターフェースを提供する</span>
<span style="color: #38d9a9;">- モデルのトレーニングプロセスの可視化をサポートする</span>
<span style="color: #38d9a9;">- 完全なテストケース</span>

## <span style="color: #228be6;">環境構成 ⚙️🛠️</span>

1. 仮想環境の作成とアクティブ化（推奨）：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. 依存関係のインストール：
   ```bash
   pip install -r requirements.txt
   ```

3. CUDAサポート（推奨）：
   - NVIDIA GPUドライバがインストールされていることを確認する
   - PyTorchは自動的に利用可能なGPUを検出して使用する

## <span style="color: #228be6;">データの準備 📊📂</span>

1. `data`ディレクトリにデータセットを整理する：
   - 各カテゴリにサブフォルダを作成する
   - 対応するカテゴリの画像をそれぞれのフォルダに配置する

例の構造：
```
data/
  ├── 淡白舌灰黑苔/
  │   ├── example1.jpg
  │   └── example2.jpg
  ├── 淡白舌白苔/
  │   ├── example1.jpg
  │   └── example2.jpg
  └── ...
```

## <span style="color: #228be6;">モデルのトレーニング 🧠💪</span>

1. トレーニングパラメータの設定
   `utils/config.py`ファイルを編集する：
   ```python
   # トレーニングパラメータの設定
   num_classes = 15  # 分類カテゴリ数
   batch_size = 32   # バッチサイズ
   num_epochs = 100  # トレーニングエポック数
   learning_rate = 0.001  # 学習率
   ```

2. トレーニングの開始：
   ```bash
   python main.py --mode train
   ```

3. トレーニングプロセスの可視化：
   - 損失曲線と精度曲線がリアルタイムで更新される
   - モデルのチェックポイントが`checkpoints`ディレクトリに自動的に保存される

## <span style="color: #228be6;">モデルを使用した予測 🎯✅</span>

### Webインターフェースによる予測

1. Webサービスの起動：
   ```bash
   python app.py
   ```

2. `http://localhost:5000`にアクセスしてオンライン予測を行う

### コマンドラインによる予測

```python
from predict import ImagePredictor

# 予測器の初期化
predictor = ImagePredictor('checkpoints/best_model.pth')

# 単一画像の予測
result = predictor.predict_single('data/ak47/001_0001.jpg')
print(f'予測クラス: {result["class"]}')
print(f'信頼度: {result["probability"]}')

# バッチ予測
results = predictor.predict_batch('data/ak47')
for result in results:
    print(f'画像: {result["image"]}')
    print(f'予測結果: {result["prediction"]}')
```

## <span style="color: #228be6;">モデルアーキテクチャ 🏗️</span>

- 基本的なCNNアーキテクチャ：3つの畳み込み層ブロック（畳み込み + バッチ正規化 + ReLU + プーリング）
- アテンションメカニズム：自己アテンションモジュールで特徴抽出能力を強化する
- 全結合層：特徴次元削減と分類に使用する3層
- Dropout層：過学習を防止する
- 損失関数：交差エントロピー損失
- オプティマイザ：Adam

## <span style="color: #228be6;">注意事項 ⚠️</span>

- サポートされる画像形式：jpg、jpeg、png
- GPUを使用したトレーニングを推奨する
- 設定ファイルを編集してモデルの構造とハイパーパラメータを調整できる
- 学習済みモデルファイルを定期的にバックアップする
- 予測時にモデルファイルのパスが正しいことを確認する