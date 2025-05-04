import torch

x = torch.zeros(1, requires_grad=True)
y = torch.zeros(1, requires_grad=True)
z = torch.zeros(1, requires_grad=True)

lr = 0.1
epochs = 100

for epoch in range(epochs):
    f = x*x + y*y + z*z - 2*x - 4*y - 6*z + 8
    f.backward()
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        z -= lr * z.grad
        x.grad.zero_()
        y.grad.zero_()
        z.grad.zero_()
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}: x={x.item():.4f}, y={y.item():.4f}, z={z.item():.4f}, f={f.item():.4f}")

print("\n最終結果：")
print(f"x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")
print(f"f = {f.item():.4f}")
