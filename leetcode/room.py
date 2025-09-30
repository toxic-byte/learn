import sys
from collections import deque

def wallsAndGates(rooms):
    if not rooms:
        return rooms
    
    m = len(rooms)
    n = len(rooms[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    queue = deque()
    
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                queue.append((i, j))
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and rooms[nx][ny] == 2147483647:
                rooms[nx][ny] = rooms[x][y] + 1
                queue.append((nx, ny))
    
    return rooms

def main():
    input_lines = [line.strip() for line in sys.stdin if line.strip()]
    T = int(input_lines[0])  
    idx = 1
    for _ in range(T):
        first_row = list(map(int, input_lines[idx].split(',')))
        m = len(first_row)
        idx += 1
        rooms = [first_row]
        for _ in range(m - 1):
            row = list(map(int, input_lines[idx].split(',')))
            rooms.append(row)
            idx += 1
        result = wallsAndGates(rooms)
        print(result)

if __name__ == "__main__":
    main()