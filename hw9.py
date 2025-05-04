# 來源：複製 ChatGPT
import torch

# 建立訓練資料(y = 2x + 1)
x_train = torch.linspace(0, 10, steps=100).unsqueeze(1)  # shape (100,1)
y_train = 2 * x_train + 1

# 定義模型參數
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 設定超參數與優化器
lr = 0.01
optimizer = torch.optim.SGD([w, b], lr=lr)
loss_fn = torch.nn.functional.mse_loss

# 訓練
epochs = 200
for epoch in range(epochs):
    # 前向計算
    y_pred = x_train * w + b
    loss = loss_fn(y_pred, y_train)

    # 反向傳播與更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1:3d}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

# 測試
x_test = torch.tensor([[4.0], [7.5]])
y_test = x_test * w + b
print("\n測試結果：")
for xi, yi in zip(x_test, y_test):
    print(f"x={xi.item():.1f} -> y_pred={yi.item():.4f}")
