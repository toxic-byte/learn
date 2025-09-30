def to_base_str(n, base):
    if n == 0:
        return "0"
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    res = []
    while n > 0:
        rem = n % base
        res.append(digits[rem])
        n = n // base
    if not res:
        return "0"
    return ''.join(reversed(res))

def is_wave_number(s):
    if len(s) == 1:
        return True
    unique_chars = set(s)
    if len(unique_chars) != 2:
        return False
    a, b = s[0], s[1]
    if a == b:
        return False
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return False
    return True

def solve():
    import sys
    input = sys.stdin.read().strip()
    a, b, l, r, k = map(int, input.split())
    
    result = []

    for n in range(l, r + 1):
        count = 0
        for base in range(a, b + 1):
            s = to_base_str(n, base)
            if is_wave_number(s):
                count += 1
                if count >= k:
                    break  # 已经足够，提前退出循环优化
        if count >= k:
            result.append(n)
    
    for num in sorted(result):
        print(num)

if __name__ == "__main__":
    solve()