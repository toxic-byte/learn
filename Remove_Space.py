def remove_extra_spaces(chars):
    """
    原地删除多余空格
    1. 删除首尾空格
    2. 单词间多个空格保留一个
    时间复杂度: O(n), 空间复杂度: O(1)
    """
    if not chars:
        return []
    
    n = len(chars)
    
    # 双指针: slow指向下一个有效字符的位置，fast遍历原数组
    slow, fast = 0, 0
    
    # 跳过开头的空格
    while fast < n and chars[fast] == ' ':
        fast += 1
    
    # 处理中间部分
    while fast < n:
        # 如果当前字符不是空格，直接复制
        if chars[fast] != ' ':
            chars[slow] = chars[fast]
            slow += 1
            fast += 1
        else:
            # 当前是空格，保留一个空格
            chars[slow] = ' '
            slow += 1
            fast += 1
            
            # 跳过后续的所有空格
            while fast < n and chars[fast] == ' ':
                fast += 1
    
    # 处理尾部可能多余的空格
    # 如果slow>0且最后一个字符是空格，需要删除
    if slow > 0 and chars[slow - 1] == ' ':
        slow -= 1
    
    # 截断数组到slow位置
    return chars[:slow]

# 测试用例
def test_remove_extra_spaces():
    # 测试用例1: 正常情况
    test1 = list("  hello   world  ")
    result1 = remove_extra_spaces(test1)
    print(f"输入: {list('  hello   world  ')}")
    print(f"输出: {result1}")
    print(f"字符串形式: {''.join(result1)}")
    print()
    
    # 测试用例2: 只有空格
    test2 = list("     ")
    result2 = remove_extra_spaces(test2)
    print(f"输入: {list('     ')}")
    print(f"输出: {result2}")
    print(f"字符串形式: {''.join(result2)}")
    print()
    
    # 测试用例3: 没有空格
    test3 = list("hello")
    result3 = remove_extra_spaces(test3)
    print(f"输入: {list('hello')}")
    print(f"输出: {result3}")
    print(f"字符串形式: {''.join(result3)}")
    print()
    
    # 测试用例4: 单词间单个空格
    test4 = list("a b c")
    result4 = remove_extra_spaces(test4)
    print(f"输入: {list('a b c')}")
    print(f"输出: {result4}")
    print(f"字符串形式: {''.join(result4)}")
    print()
    
    # 测试用例5: 空数组
    test5 = []
    result5 = remove_extra_spaces(test5)
    print(f"输入: {[]}")
    print(f"输出: {result5}")
    print(f"字符串形式: {''.join(result5)}")

if __name__ == "__main__":
    test_remove_extra_spaces()