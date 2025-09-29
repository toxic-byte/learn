def find_min_number():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        n = int(input[idx])
        k = int(input[idx + 1])
        idx += 2
        
        # 检查无解条件
        if (k > n) or (k == 1 and n > 1) or (n == 1 and k != 1):
            print(-1)
            continue
        
        # 初始化：第一位是1
        res = ['1']
        used = {'1'}
        need_new = k - 1  # 还需要引入的新数字数量（已用1）
        last_char = '1'
        
        # 构造剩余n-1位
        for _ in range(n - 1):
            # 按0-9顺序遍历候选字符，确保最小
            for c in map(str, range(10)):
                if c == last_char:
                    continue
                # 计算剩余位置和新的需引入新数字数量
                remaining_pos = n - len(res)
                new_remaining = remaining_pos - 1
                if c in used:
                    new_need = need_new
                else:
                    new_need = need_new - 1
                
                # 检查剩余位置是否足够引入所需新数字
                if new_remaining >= new_need:
                    res.append(c)
                    if c not in used:
                        used.add(c)
                        need_new -= 1
                    last_char = c
                    break  # 选到最小的合法字符，跳出循环
        
        print(''.join(res))

if __name__ == "__main__":
    find_min_number()