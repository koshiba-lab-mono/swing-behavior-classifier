# swing-behavior-classifier
![image](https://github.com/koshiba-lab-mono/swing-behavior-classifier/blob/images/sample.gif)  
DeepSortのバウンティングボックスで切り抜いた画像を入力として，3ラベルの行動分類をします．

## 事前準備
* Python 3.6^ をインストールする必要があります．
* Dockerをインストールする必要があります．
* 10xx - 20xx系のNvidia製GPUを搭載している必要があります．

## Quick Start

### DeepSort出力の入手
* 以下のコマンドでリポジトリをクローンします．
```
git clone https://github.com/koshiba-lab-mono/swing-behavior-classifier.git
```
* プロジェクト直下に移動します．
* ./data/videos/ に動画を配置します．(複数可)
* 以下のコマンドを実行すると，Dockerイメージ及びコンテナを構築し，コンテナ内でDeepSortが起動されます．
```
docker compose run --rm deepsort
```
* しばらく待つと，出力が ./data/boxes/ に配置されています．

### 分類モデルのデモ実行
* ライブラリの依存関係をダウンロードします．
```
pip install -r requirements.txt
```
* デモの実行方法は以下のコマンドを実行してください．
```
python ./src/eval_model.py
```
* 必要に応じて，eval_model.py内で実行する関数の引数（ファイルのパス）を適宜変更してください．
* CNNの詳細なアーキテクチャは./src/train.pyを参照してください．