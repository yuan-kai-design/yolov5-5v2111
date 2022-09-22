def run(*input):
    *args,iter1=input if len(input)>1 else ("red","blue",input[0]) #
    print(*args)
    print(iter1)

if __name__=="__main__":
    run(1,2,3)