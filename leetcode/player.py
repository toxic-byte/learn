import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:1+n]))
    b = list(map(int, input[1+n:1+2*n]))
    
    players = [(a_i, b_i, idx) for idx, (a_i, b_i) in enumerate(zip(a, b))]
        
    from functools import cmp_to_key
    def compare(p1, p2):
        a1, b1, idx1 = p1
        a2, b2, idx2 = p2
        if b1 == 0 and b2 == 0:
            return 0
        if b1 == 0:
            return -1 if a1 > 0 else 1
        if b2 == 0:
            return 1 if a2 > 0 else -1
        diff = a1 * b2 - a2 * b1
        if diff > 0:
            return -1
        elif diff < 0:
            return 1
        else:
            return 0
    
    players.sort(key=cmp_to_key(compare))
        
    result = [0] * n
    for i in range(n):
        a_i, b_i, idx_i = players[i]
        for j in range(n):
            if i == j:
                continue
            a_j, b_j, idx_j = players[j]
            if a_i * b_j > a_j * b_i:
                result[idx_i] += 1
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()