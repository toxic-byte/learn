n, x = map(int, input().split())
a = list(map(int, input().split()))

INF = float('inf')
dp = [INF] * x
dp[0] = 0

for num in a:
    new_dp = [INF] * x
    for j in range(x):
        if dp[j] == INF:
            continue
        # Option 1: delete the number (operation +1)
        if new_dp[j] > dp[j] + 1:
            new_dp[j] = dp[j] + 1
        # Option 2: keep the number, possibly add 1 k times
        # Original contribution: num % x, no add (k=0)
        original_mod = num % x
        new_j = (j + original_mod) % x
        if new_dp[new_j] > dp[j]:
            new_dp[new_j] = dp[j]
        # Adding 1 k times: mod becomes (original_mod + k) % x, operation k
        for k in range(1, x):
            mod = (original_mod + k) % x
            new_j = (j + mod) % x
            cost = k
            if new_dp[new_j] > dp[j] + cost:
                new_dp[new_j] = dp[j] + cost
    dp = new_dp

print(dp[0])