import sys

def dfs(grid, row, col, word, idx, visited):
    if idx == len(word):
        return True
    if (row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or
        visited[row][col] or grid[row][col] != word[idx]):
        return False
    visited[row][col] = True
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if dfs(grid, nr, nc, word, idx + 1, visited):
            return True
    visited[row][col] = False
    return False

data = sys.stdin.read().splitlines()
if not data:
    exit()

n, q = map(int, data[0].split())
grid = []
for i in range(1, 1 + n):
    grid.append(data[i].strip())

queries = data[1 + n].split()

results = []
for word in queries:
    found = False
    for i in range(n):
        for j in range(n):
            visited = [[False]*n for _ in range(n)]
            if dfs(grid, i, j, word, 0, visited):
                found = True
                break
        if found:
            break
    results.append("Yes" if found else "No")

print(" ".join(results))