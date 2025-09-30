# from operator import le
# import sys

# n,m=map(int,input().split())
# S=input()

# def is_huiwen(s):
#     left,right=0,len(s)-1
#     while left<=right :
#         if s[left]==s[right]:
#             left+=1
#             right-=1
#         else:
#             return False
#     return True

# count=0
# left=0
# right=m
# while right<n:
#     sub=S[left:right]
#     if is_huiwen(sub):
#         count+=1
#     left+=1
#     right+=1
# print(count)

def count_palindromic_substrings(s, m):
    n = len(s)
    count = 0
    
    for i in range(n - m + 1):
        left = i
        right = i + m - 1
        is_palindrome = True
        
        while left < right:
            if s[left] != s[right]:
                is_palindrome = False
                break
            left += 1
            right -= 1
        
        if is_palindrome:
            count += 1
    
    return count

n, m = map(int, input().split())
S = input()

print(count_palindromic_substrings(S, m))