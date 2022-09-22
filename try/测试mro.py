class a(object):
    def __init__(self):
        pass


class b(a):
    def __init__(self):
        super(b, self).__init__()


class c(a):
    def __init__(self):
        super(c, self).__init__()


class d(c, b):
    def __init__(self):
        super(c, self).__init__()


print(d)
dd = d()
print(dd.__class__)
print(dd.__class__.mro())
print(d.mro())
print(d.__mro__)
