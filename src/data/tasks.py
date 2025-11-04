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


def task8(nmax, nmin, nchar=4):
    """
    Reverse Polish Notation (RPN) Evaluation
    逆ポーランド記法の評価タスク
    
    例: "ab+c*" → (a+b)*c を評価
    文字: a,b,c,...は値、+は加算演算子、=は結果の開始マーカー
    
    生成形式: operands operators = result
    例: "abc++=" で a+b+c の結果
    
    簡略化版: 単純な加算のみ
    nchar=4 の場合: a,b,c が値、+ が演算子
    """
    if nchar < 3:
        raise ValueError("task8 requires at least 3 characters (2 operands + 1 operator)")
    
    # オペランドの数（2〜nmax）
    n_operands = random.randint(max(2, nmin), max(2, nmax))
    
    # nchar-1 個の値を使用（最後の1文字は演算子用）
    operand_chars = min(nchar - 1, n_operands)
    
    sequence = []
    
    # オペランドを追加（a, b, c, ...）
    for i in range(n_operands):
        sequence.append(chr(ord('a') + (i % operand_chars)))
    
    # 演算子を追加（n_operands - 1 個の+）
    # RPN: a b + は a+b を意味する
    for _ in range(n_operands - 1):
        sequence.append('+')
    
    # 結果マーカー
    sequence.append('=')
    
    # 結果を追加（簡略化: 同じオペランドの繰り返し）
    # 実際の計算結果ではなく、パターンとして
    for i in range(n_operands):
        sequence.append(chr(ord('a') + (i % operand_chars)))
    
    return ''.join(sequence)


def task9(nmax, nmin, nchar=3):
    """
    Balanced Parentheses (Dyck Language) - 3 character version
    かっこ合わせタスク
    
    形式: '(' 中身 ')'
    例: "(.)" や "((.)(.))"
    
    生成: 正しくネストされたかっこの列
    nchar=3: '(' (a), '.' (b), ')' (c) の3文字
    - '(' は開きかっこ
    - '.' は中身・セパレータ
    - ')' は閉じかっこ
    
    深さ nmin〜nmax-1 のランダムなDyck word + 中身
    """
    if nchar != 3:
        raise ValueError("task9 requires exactly 3 characters: '(', '.', ')'")
    
    # 深さ（ペアの数）
    n = random.randint(nmin, nmax - 1) if nmax > nmin else nmin
    
    if n == 0:
        return ""
    
    # Dyck wordをランダム生成（括弧と中身を持つ構造）
    # アルゴリズム: 開きかっこ、中身(.)、閉じかっこをバランスを保ちながら配置
    
    def generate_balanced(depth):
        """再帰的にバランスした括弧列を生成"""
        if depth == 0:
            return []
        
        # ランダムに分割: depth個のペアをどう配置するか
        if depth == 1:
            # 単一ペア: (.)
            return ['(', '.', ')']
        
        # 複数ペアの場合
        result = []
        remaining = depth
        
        while remaining > 0:
            # この括弧ペアに何個のペアをネストするか
            nest_depth = random.randint(0, remaining - 1)
            result.append('(')
            
            if nest_depth > 0:
                # ネストされた構造
                result.extend(generate_balanced(nest_depth))
            else:
                # 中身だけ
                result.append('.')
            
            result.append(')')
            remaining -= (nest_depth + 1)
        
        return result
    
    sequence = generate_balanced(n)
    
    # 文字を 'a', 'b', 'c' に変換
    # '(' -> 'a', '.' -> 'b', ')' -> 'c'
    char_map = {'(': 'a', '.': 'b', ')': 'c'}
    result = ''.join(char_map[c] for c in sequence)
    
    return result


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
    elif ntask == 8:
        return task8(nmax, nmin, nchar)  # Reverse Polish Notation
    elif ntask == 9:
        return task9(nmax, nmin, nchar)  # Balanced Parentheses
    else:
        return task1(nmax, nmin, nchar)