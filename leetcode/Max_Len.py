def length_of_lis_dp(nums):
    """
    动态规划解法：时间复杂度 O(n²)，空间复杂度 O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # 每个元素本身就是一个长度为1的子序列
    
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# 测试
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"最长递增子序列长度: {length_of_lis_dp(nums)}")  # 输出: 4
