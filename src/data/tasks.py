import random

def task1(nmax, nmin, nchar):
    """
    Counting Task
    a^nb^n, a^nb^nc^n, a^nb^nc^nd^n...
    """
    n = random.randint(nmin, nmax-1 if nmax > nmin else nmin)
    p = []
    for c in range(nchar):
        p += [c] * n
        
    return ''.join(chr(ord('a') + v ) for v in p) 

def task2(nmax, nmin, nchar, nrep=2):
    """
    Counting Task
    a^nb^kn n>=1
    """
    n = random.randint(nmin, nmax-1 if nmax > nmin else nmin)
    c2 = random.randint(1, nchar-1)
    p = [0] * n + [c2] * (nrep * n)
    return ''.join(chr(ord('a') + v ) for v in p)

def task3(nmax, nmin, nchar=3):
    """
    Counting + Addition
    a^nb^mc^{n+m}
    3種類目の文字だけはランダムに選びたいのか？
    一応C++に忠実に → task.h
    """
    n = random.randint(nmin, nmax-1 if nmax > nmin else nmin) # a, b合計の個数
    m = random.randint(1, max(1, n-1)) # bの個数
    n = n - m
    p = [0] * n + [1] * m + [random.randint(2, min(2, nchar-1))] * (n + m)
    return ''.join(chr(ord('a') + v) for v in p)
    
def task4(nmax, nmin, nchar):
    """
    Memorization (Reverse Copy)
    c1c2...cn a cn...c2c1
    """
    n = random.randint(nmin, nmax-1) if nmax > nmin else nmin
    p = [0] * (2 * n + 1)
    
    # a 以外の文字で最初のn個を埋め, それを反転 真ん中にセパレータとしてa存在
    for i in range(n):
        p[i] = random.randint(1, nchar - 1)
        p[2 * n - i] = p[i]
    return ''.join(chr(ord('a') + v) for v in p)

def task5(nmax, nmin, nchar=3):
    """
    Counting + Multiplication
    a^nb^m c^{nm}
    task3に同じ
    """
    n = random.randint(nmin, nmax-1 if nmax > nmin else nmin) # a, b合計の個数
    k = random.randint(1, max(1, n-1)) # bの個数
    n = n - k
    p = [0] * n + [1] * k + [random.randint(2, min(2, nchar-1))] * (n * k)
    return ''.join(chr(ord('a') + v) for v in p)

def task6(nmax, nmin, nchar=4):
    """
    Counting
    a^nb^m^c^nd^m
    """
    if nchar != 4:
        raise ValueError("task6 requires exactly 4 characters (a,b,c,d)")
    n = random.randint(nmin, nmax-1) if nmax > nmin else nmin
    m = random.randint(1, max(1, n-1))
    n = n - m
    p = [0] * n + [1] * m + [2] * n + [3] * m
    return ''.join(chr(ord('a') + v) for v in p)

#C++ではbaseを入力で受けてる, なんで？
def task7(nmax, nmin, base=2):
    """
    Binary Addition Task
    Generate addition sequence like n+m=result.
    """
    if base != 2:
        raise ValueError("base must be = 2")
    
    tln = random.randint(nmin, nmax-1) if nmax > nmin else nmin
    ln = random.randint(0, tln)
    lm = tln - ln
    
    # Generate number n
    if ln == 0:
        n_str = "0"
    else:
        n_str = str(random.randint(1, base-1))
        for _ in range(1, ln):
            n_str += str(random.randint(0, base-1))
    
    # Generate number m
    if lm == 0:
        m_str = "0"
    else:
        m_str = str(random.randint(1, base-1))
        for _ in range(1, lm):
            m_str += str(random.randint(0, base-1))
    
    # Calculate sum
    n_val = int(n_str, base)
    m_val = int(m_str, base)
    result = n_val + m_val
    
    # Convert result to base
    if result == 0:
        result_str = "0"
    else:
        result_str = ""
        temp = result
        while temp > 0:
            result_str = str(temp % base) + result_str
            temp //= base
    
    return n_str + "+" + m_str + "=" + result_str + "."


def generate_next_sequence(nmax, nmin, nchar, nrep, ntask):
    """Generate sequence based on task number."""
    if ntask == 1:
        return task1(nmax, nmin, nchar)
    elif ntask == 2:
        return task2(nmax, nmin, nchar, nrep)
    elif ntask == 3:
        return task3(nmax, nmin, 3)
    elif ntask == 4:
        return task4(nmax, nmin, nchar)
    elif ntask == 5:
        return task5(nmax, nmin, nchar)
    elif ntask == 6:
        return task6(nmax, nmin, nchar)
    elif ntask == 7:
        return task7(nmax, nmin, base=2)  # Task 7 always uses base 2
    else:
        return task1(nmax, nmin, nchar)