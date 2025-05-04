# hw11(此題用AI完成)
## 一、環境準備
建立虛擬環境並安裝套件：

```
cd ~/Documents/ML/_ml
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install scikit-learn matplotlib
```
切到 hw11 資料夾：
```
cd hw11
```
## 二、分類：1.py
```
python 1.py
```
* 資料集：load_wine()（Wine 資料集，3 個類別，共 178 筆）
* 模型：RandomForestClassifier
* 切分：70% 訓練、30% 測試
* 結果：
測試集分類準確率 (Accuracy) = 1.0000
特徵重要性列印如下：
```
Feature importances : [0.1252, 0.0456, 0.0159, …, 0.1611]
```

## 三、分群：2.py
```python 2.py```
* 資料：make_blobs(n_samples=300, centers=[(-5,0),(0,5),(5,0)])
* 演算法：
1. KMeans(n_clusters=3)
2. DBSCAN(eps=1.2, min_samples=5)
3. AgglomerativeClustering(n_clusters=3)

* 輸出：彈出一個畫布，內含三張子圖，分別顯示三種分群方法的標籤結果，各群用不同顏色區分。

示意圖：


## 四、回歸：3.py
```python 3.py```
* 資料集：load_diabetes()（Diabetes 資料集，442 筆）
* 模型：LinearRegression
* 切分：70% 訓練、30% 測試



執行後輸出類似：
```
Diabetes 資料集線性回歸 R²  : 0.4463
Mean Squared Error (MSE)   : 3065.27
Coefficients               : [  186.40, -233.71, ... ]
Intercept                  : 152.47
```
## 總結：
分類：對 Wine 資料集以隨機森林取得 100% 準確率。

分群：用三種演算法比對同一組人造資料，視覺化比較分群效果。

回歸：對 Diabetes 資料集進行線性回歸，並列印模型性能與係數。
