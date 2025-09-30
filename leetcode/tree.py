import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    
    for _ in range(t):
        n = int(data[index]); index += 1
        if n == 1:
            results.append("1")
            continue
        p_list = list(map(int, data[index:index+n-1]))
        index += n-1
        
        children = [[] for _ in range(n+1)]
        for i in range(2, n+1):
            p = p_list[i-2]
            children[p].append(i)
        
        visited = [False] * (n+1)
        time = 0
        
        while sum(visited) < n:
            time += 1
            new_known = []
            for node in range(1, n+1):
                if not visited[node]:
                    continue
                for child in children[node]:
                    if not visited[child]:
                        new_known.append(child)
            for node in new_known:
                visited[node] = True
            for i in range(1, n+1):
                if not visited[i]:
                    visited[i] = True
                    break
        
        results.append(str(time))
    
    print("\n".join(results))

if __name__ == "__main__":
    main()