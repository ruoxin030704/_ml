from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 載入 Diabetes 資料集
data = load_diabetes()
X, y = data.data, data.target

# 切分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 建立模型並訓練
model = LinearRegression()
model.fit(X_train, y_train)

# 預測並評估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Diabetes 資料集線性回歸 R²  : {r2:.4f}")
print(f"Mean Squared Error (MSE)   : {mse:.2f}")
print(f"Coefficients               : {model.coef_}")
print(f"Intercept                  : {model.intercept_:.4f}")
