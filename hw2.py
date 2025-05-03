import random

citys = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def height(path):
    dist = 0.0
    n = len(path)
    for i in range(n):
        dist += distance(citys[path[i]], citys[path[(i+1) % n]])
    return -dist

def neighbor(path):
    new_path = path.copy()
    a, b = random.sample(range(len(path)), 2)
    new_path[a], new_path[b] = new_path[b], new_path[a]
    return new_path

def hillClimbing(x, height_func, neighbor_func, max_fail=10000):
    fail = 0
    current = x.copy()
    current_h = height_func(current)
    while True:
        nxt = neighbor_func(current)
        nxt_h = height_func(nxt)
        if nxt_h > current_h:
            current, current_h = nxt, nxt_h
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return current

if __name__ == "__main__":
    init_path = list(range(len(citys)))
    best_path = hillClimbing(init_path, height, neighbor, max_fail=10000)
    print(best_path, -height(best_path))