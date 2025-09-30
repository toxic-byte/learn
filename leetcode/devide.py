dividend=1
divide=int(input("请输入除数:"))
try:
    result=dividend/divide
    print("结果是:",result)
except ZeroDivisionError:
    print("2")

except Exception as e:
    print("3",e)

else:
    print("4")