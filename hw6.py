# 來源：複製 ChatGPT
import math

class Value:
    """儲存單一標量值及其梯度，並構建自動微分圖"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data            # 節點的數值
        self.grad = 0.0             # 節點的梯度
        self._backward = lambda: None  # 反向傳播函數占位
        self._prev = set(_children)    # 父節點集合（計算圖中上游）
        self._op = _op               # 本節點所屬運算（用於除錯或視覺化）

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # 反向：加法對任意輸入的偏導都是 1
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # 反向：d(x*y)/dx = y, d(x*y)/dy = x
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward

        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "只支援 int/float 次方"
        out = Value(self.data**exponent, (self,), f'**{exponent}')

        # 反向：d(x^n)/dx = n * x^(n-1)
        def _backward():
            self.grad += exponent * (self.data**(exponent-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        # 反向：只於 x>0 時傳遞梯度，否則阻斷
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        """指數函數 f(x) = e^x"""
        out = Value(math.exp(self.data), (self,), 'exp')

        # 反向：d(e^x)/dx = e^x
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """Sigmoid 函數 σ(x) = 1 / (1 + e^(-x))"""
        # 1. 計算 e^{-x}
        neg = (-self).exp()
        # 2. 計算 (1 + e^{-x})
        denom = neg + Value(1.0)
        # 3. 做倒數得到 sigmoid
        out = Value(1.0) / denom

        # 反向：dσ/dx = σ(x) * (1 - σ(x))
        def _backward():
            sig = out.data
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """執行反向傳播，計算所有節點的梯度"""
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # 最終節點梯度設為 1（∂f/∂f = 1）
        self.grad = 1.0
        # 按照拓撲排序反向執行
        for node in reversed(topo):
            node._backward()

    # 讓 Python 數值與 Value 互動
    def __neg__(self):             return self * -1
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
