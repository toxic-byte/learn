def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    power = list(map(int, data[idx:idx + n]))
    idx += n
    mana = list(map(int, data[idx:idx + n]))
    idx += n
    bonus = list(map(int, data[idx:idx + m]))
    idx += m
    
    # Initialize DP table
    # dp[i][j][k] represents the max power after i-th course, at j-th floor, with total mana cost k
    # Initialize to -infinity
    INF = -1 << 60
    dp = [[[INF for _ in range(M + 1)] for __ in range(m)] for ___ in range(n + 1)]
    
    # Base case: 0 courses, 0 mana, any floor (but no power)
    for j in range(m):
        dp[0][j][0] = 0
    
    for i in range(1, n + 1):
        current_power = power[i - 1]
        current_mana = mana[i - 1]
        for prev_j in range(m):
            for prev_k in range(M + 1):
                if dp[i - 1][prev_j][prev_k] == INF:
                    continue
                for curr_j in range(m):
                    # Calculate the cost for current course in curr_j floor
                    cost = current_mana * bonus[curr_j]
                    total_cost = prev_k + cost
                    # Check if switching floors
                    if curr_j != prev_j and i != 1:
                        if curr_j > prev_j:
                            switch_cost = curr_j - prev_j
                            total_cost += switch_cost
                    if total_cost > M:
                        continue
                    # Update DP
                    added_power = current_power * bonus[curr_j]
                    if dp[i][curr_j][total_cost] < dp[i - 1][prev_j][prev_k] + added_power:
                        dp[i][curr_j][total_cost] = dp[i - 1][prev_j][prev_k] + added_power
    
    # Find the maximum power achievable
    max_power = 0
    for j in range(m):
        for k in range(M + 1):
            if dp[n][j][k] > max_power:
                max_power = dp[n][j][k]
    
    print(max_power if max_power != INF else 0)

solve()