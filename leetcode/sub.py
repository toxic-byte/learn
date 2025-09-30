def min_changes_to_arithmetic_array():
    import sys
    n = int(sys.stdin.readline())
    arr = list(map(int, sys.stdin.readline().split()))
    
    if n <= 2:
        print(0)
        return
    
    min_changes = float('inf')
    
    # 枚举前两个元素的所有可能修改情况（共 9 种）
    for delta1 in [-1, 0, 1]:
        for delta2 in [-1, 0, 1]:
            a1 = arr[0] + delta1
            a2 = arr[1] + delta2
            d = a2 - a1
            changes = abs(delta1) + abs(delta2)  # 前两个元素的修改次数
            
            # 检查后续元素
            valid = True
            for i in range(2, n):
                expected = a1 + i * d
                if arr[i] != expected:
                    changes += 1
                    if changes >= min_changes:
                        valid = False
                        break  # 提前终止，优化效率
            
            if valid and changes < min_changes:
                min_changes = changes
    
    print(min_changes)

min_changes_to_arithmetic_array()