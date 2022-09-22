class try_next():
    def __init__(self,n):
        self.n=n

    def __iter__(self):
        self.count=0
        return self

    def __next__(self):
        if self.count==self.n:
            raise StopIteration
        self.count+=1
        return self.count

print(try_next(3))
for i in try_next(3):
    print(i)


