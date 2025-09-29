import sys

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    res = []
    for i in range(n):
        current = a[i]
        left = False
        right = False
        # Check left neighbor
        if i > 0 and a[i-1] > current:
            left = True
        # Check right neighbor
        if i < n - 1 and a[i+1] > current:
            right = True
        if left or right:
            res.append(1)
        else:
            possible = False
            # Check if left neighbor can be made > current
            if i > 0:
                if i > 1 and a[i-2] > a[i-1]:
                    if a[i-1] + a[i-2] > current:
                        possible = True
            # Check if right neighbor can be made > current
            if i < n - 1:
                if i < n - 2 and a[i+2] > a[i+1]:
                    if a[i+1] + a[i+2] > current:
                        possible = True
            if possible:
                res.append(2)
            else:
                res.append(-1)
    print(' '.join(map(str, res)))

solve()