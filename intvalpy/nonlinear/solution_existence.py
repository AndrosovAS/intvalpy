from intvalpy.utils import zeros


def miranda(func, x):
    n = len(x)

    def l(i):
        result = zeros(n)
        for k in range(n):
            if k == i:
                result[k] = x[k].a
            else:
                result[k] = x[k]
        return result

    def u(i):
        result = zeros(n)
        for k in range(n):
            if k == i:
                result[k] = x[k].b
            else:
                result[k] = x[k]
        return result

    for k in range(n):
        if (func(l(k))[k].b <= 0 and func(u(k))[k].a >= 0) or \
           (func(l(k))[k].a >= 0 and func(u(k))[k].b <= 0):
            pass
        else:
            return False
    return True
