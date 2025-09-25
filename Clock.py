def count_ways_dp(n):
    if n == 0:
        return 1  # 0步时只有一种方案（不动）
    if n % 2 != 0:
        return 0  # 奇数步无法回到起点
    
    # dp[i][j]：走i步后停在j点的方案数
    dp = [[0] * 12 for _ in range(n + 1)]
    dp[0][0] = 1  # 初始状态
    
    for step in range(1, n + 1):
        for pos in range(12):
            # 上一步可能是 pos-1（顺时针）或 pos+1（逆时针）
            prev_clockwise = (pos - 1) % 12
            prev_counterclockwise = (pos + 1) % 12
            dp[step][pos] = dp[step - 1][prev_clockwise] + dp[step - 1][prev_counterclockwise]
    
    return dp[n][0]

# 测试
print(count_ways_dp(2))  # 输出：2（+1-1 或 -1+1）
print(count_ways_dp(4))  # 输出：8
