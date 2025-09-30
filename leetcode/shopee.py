def min_deletions_to_contain_substring_shopee(s: str) -> int:
    target = "shopee"
    n = len(s)
    m = len(target)
    INF = 10**9
    ans = INF

    for start in range(n):
        i = start
        t = 0
        # 在从 start 开始的后缀里寻找 target 的子序列
        while i < n and t < m:
            if s[i] == target[t]:
                t += 1
            i += 1
        if t == m:
            end = i - 1  # 最后一个匹配字符的位置
            deletions = (end - start + 1) - m
            ans = min(ans, deletions)

    return -1 if ans == INF else ans

# 验证题中样例
print(min_deletions_to_contain_substring_shopee("shoipeeabcdefg"))  # 1（删掉 i，让 shopee 连成连续子串）
print(min_deletions_to_contain_substring_shopee("abcshopee"))       # 0（已是连续子串）
print(min_deletions_to_contain_substring_shopee("shope"))           # -1（无法得到 ee）