import random

def f(x, y, z):
    return x*x + y*y + z*z - 2*x - 4*y - 6*z + 8

def hill_climbing(x, y, z, h=0.01, max_iter=1000):
    for i in range(max_iter):
        current = f(x, y, z)
        
        moves = [(x + h, y,z),(x - h, y,z),(x,y + h, z),(x,y - h, z),(x,y,z + h),(x,y,z - h),]
        values = [f(nx, ny, nz) for (nx, ny, nz) in moves]
        best_value = min(values)
        best_move  = moves[values.index(best_value)]
        if best_value < current:
            x, y, z = best_move
        else:
            break  

    return x, y, z, f(x, y, z)

if __name__ == "__main__":
    x0, y0, z0 = 0.0, 0.0, 0.0
    x_min, y_min, z_min, f_min = hill_climbing(x0, y0, z0, h=0.01)

    print(f"x={x_min:.4f}, y={y_min:.4f}, z={z_min:.4f}")
    print(f"f={f_min:.4f}")
