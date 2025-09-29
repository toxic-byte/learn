def fun(n):
    if n>9:
        fun(n//10)
    print(n%10)
fun(1234)