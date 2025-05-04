from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入 Wine 資料集
data = load_wine()
X, y = data.data, data.target

# 切分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 建立並訓練模型
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# 預測並評估
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Wine 資料集分類 Accuracy   : {acc:.4f}")
print(f"Feature importances         : {clf.feature_importances_}")
