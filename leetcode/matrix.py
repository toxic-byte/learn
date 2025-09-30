def solve():
    n, k = map(int, input().split())
    grid = [input().strip() for _ in range(n)]
    
    total_steps = 2 * n - 1
    
    # 初始化上一行（第0行）
    prev_row = [[0] * (total_steps + 1) for _ in range(n)]
    
    # 起点(0,0)
    if grid[0][0] == '0':
        prev_row[0][1] = 1
    else:
        prev_row[0][0] = 1
        
    # 初始化第0行的其他列
    for j in range(1, n):
        max_c = min(total_steps, j + 1)  # 当前最多0的个数
        if grid[0][j] == '0':
            for c in range(1, max_c + 1):
                prev_row[j][c] = prev_row[j - 1][c - 1]
        else:
            for c in range(0, max_c + 1):
                prev_row[j][c] = prev_row[j - 1][c]
    
    # 处理第1行到第n-1行
    for i in range(1, n):
        curr_row = [[0] * (total_steps + 1) for _ in range(n)]
        # 处理第0列
        max_c0 = min(total_steps, i + 1)
        if grid[i][0] == '0':
            for c in range(1, max_c0 + 1):
                curr_row[0][c] = prev_row[0][c - 1]
        else:
            for c in range(0, max_c0 + 1):
                curr_row[0][c] = prev_row[0][c]
                
        # 处理当前行的其他列
        for j in range(1, n):
            max_c = min(total_steps, i + j + 1)
            if grid[i][j] == '0':
                for c in range(1, max_c + 1):
                    curr_row[j][c] = prev_row[j][c - 1] + curr_row[j - 1][c - 1]
            else:
                for c in range(0, max_c + 1):
                    curr_row[j][c] = prev_row[j][c] + curr_row[j - 1][c]
                    
        prev_row = curr_row  # 更新上一行为当前行
    
    # 计算满足条件的路径总数
    total = 0
    for c in range(total_steps + 1):
        ones = total_steps - c
        if abs(c - ones) <= k:
            total += prev_row[n - 1][c]
            
    print(total)

if __name__ == "__main__":
    solve()