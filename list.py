import sys
import bisect

def count_factor(x, p):
    cnt = 0
    while x % p == 0:
        cnt += 1
        x //= p
    return cnt

def solve():
    input = sys.stdin.read().split()
    n, k = int(input[0]), int(input[1])
    arr = list(map(int, input[2:2+n]))

    pre2 = [0] * (n + 1)
    pre5 = [0] * (n + 1)
    
    for i in range(1, n + 1):
        pre2[i] = pre2[i-1] + count_factor(arr[i-1], 2)
        pre5[i] = pre5[i-1] + count_factor(arr[i-1], 5)

    # 创建点列表：每个点包含 (pre2[i], pre5[i], i)
    points = [(pre2[i], pre5[i], i) for i in range(n + 1)]
    
    # 按 pre2 排序
    points.sort()
    
    # 离散化 pre5 值
    all_pre5 = sorted(set(pre5))
    pre5_to_idx = {val: idx+1 for idx, val in enumerate(all_pre5)}
    m = len(all_pre5)
    
    # Fenwick 树
    BIT = [0] * (m + 2)
    
    def update(pos, delta):
        while pos <= m:
            BIT[pos] += delta
            pos += pos & -pos
    
    def query(pos):
        res = 0
        while pos > 0:
            res += BIT[pos]
            pos -= pos & -pos
        return res
    
    total = 0
    j = 0
    
    # 按 pre2 顺序处理每个点作为右端点
    for i in range(1, n + 1):
        A = pre2[i] - k
        B = pre5[i] - k
        
        if A < 0 or B < 0:
            continue
        
        # 将 pre2 <= A 的点加入 Fenwick 树
        while j < len(points) and points[j][0] <= A:
            if points[j][2] < i:  # 确保左端点在右端点之前
                update(pre5_to_idx[points[j][1]], 1)
            j += 1
        
        # 查找 pre5 <= B 的点的数量
        idx = bisect.bisect_right(all_pre5, B)
        if idx > 0:
            total += query(idx)
    
    print(total)

if __name__ == "__main__":
    solve()