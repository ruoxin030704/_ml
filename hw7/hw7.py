from micrograd.engine import Value

x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

lr = 0.1
epochs = 100

for epoch in range(epochs):
    f = x*x + y*y + z*z - x*2 - y*4 - z*6 + Value(8.0)
    x.grad = y.grad = z.grad = 0.0
    f.backward()
    x.data -= lr * x.grad
    y.data -= lr * y.grad
    z.data -= lr * z.grad
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}: x={x.data:.4f}, y={y.data:.4f}, z={z.data:.4f}, f={f.data:.4f}")

print("\n最終結果：")
print(f"x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")
print(f"f = {f.data:.4f}")
