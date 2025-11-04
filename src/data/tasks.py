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
    Parentheses Matching Task (Dyck Language Classification)
    括弧整合判定タスク
    
    形式: "括弧列 + ラベル"
    
    nchar=3: Dyck-1 (1種類の括弧のみ)
        文字: '(' (a), ')' (b), ラベル (c: 正解 or 不正)
        例: "(())" + 'c' → "aabbc" (正しい場合、ラベルは 'c'=正解)
        例: "(()"  + 'a' → "aaba"  (不正な場合、ラベルは 'a'=不正)
        
    nchar=4: Dyck-1 with content (1種類の括弧 + 中身文字)
        文字: '(' (a), 中身 (b), ')' (c), ラベル (d: 正解 or 不正)
        例: "(x(x)x)" + 'd' → "ababcbd" (正しい場合)
        例: "((x)x"   + 'a' → "abcba"   (不正な場合)
    
    nchar=5: Dyck-2 (2種類の括弧)
        文字: '(' (a), ')' (b), '[' (c), ']' (d), ラベル (e: 正解 or 不正)
        例: "([()])" + 'e' → "acadbde" (正しい場合)
        例: "([)]"   + 'a' → "acbda"   (不正な場合、交差している)
    
    タスク: 括弧列が正しくバランスしているかを判定
    - 正しい → 最後の文字が nchar-1 のインデックス
    - 不正   → 最後の文字が 0 のインデックス
    
    評価: 最後の1文字の予測精度
    """
    if nchar not in [3, 4, 5]:
        raise ValueError("task9 requires 3, 4, or 5 characters")
    
    # 系列長（括弧列の長さ）
    n = random.randint(nmin, nmax - 1) if nmax > nmin else nmin
    
    if n == 0:
        n = 1
    
    # 50%の確率で正しい括弧列、50%で不正な括弧列を生成
    is_valid = random.random() < 0.5
    
    if nchar == 3:
        # Dyck-1: 1種類の括弧のみ (括弧 + ラベル)
        if is_valid:
            # 正しい括弧列を生成
            sequence = []
            open_count = 0
            close_count = 0
            target_pairs = max(1, n // 2)
            
            for _ in range(2 * target_pairs):
                if open_count < target_pairs and (open_count == close_count or random.random() < 0.5):
                    sequence.append('(')
                    open_count += 1
                else:
                    sequence.append(')')
                    close_count += 1
            
            while close_count < target_pairs:
                sequence.append(')')
                close_count += 1
        else:
            # 不正な括弧列を生成
            sequence = []
            for _ in range(n):
                sequence.append(random.choice(['(', ')']))
            
            # 確実に不正にする：バランスチェック
            depth = 0
            is_actually_valid = True
            for char in sequence:
                if char == '(':
                    depth += 1
                else:
                    depth -= 1
                    if depth < 0:
                        is_actually_valid = False
                        break
            if depth != 0:
                is_actually_valid = False
            
            # もし偶然正しくなってしまったら、1文字変更して不正にする
            if is_actually_valid and len(sequence) > 0:
                idx = random.randint(0, len(sequence) - 1)
                sequence[idx] = ')' if sequence[idx] == '(' else '('
        
        # 文字を 'a', 'b' に変換
        result = ''.join('a' if c == '(' else 'b' for c in sequence)
        
        # ラベルを追加: 正解なら 'c' (index 2), 不正なら 'a' (index 0)
        label_char = chr(ord('a') + 2) if is_valid else 'a'
        result += label_char
        
    elif nchar == 4:
        # Dyck-1 with content: 括弧 + 中身文字 + ラベル
        if is_valid:
            # 正しい括弧列を生成
            def generate_balanced(depth):
                if depth == 0:
                    return []
                
                if depth == 1:
                    if random.random() < 0.5:
                        return ['(', 'x', ')']
                    else:
                        return ['(', ')']
                
                result = []
                remaining = depth
                
                while remaining > 0:
                    nest_depth = random.randint(0, remaining - 1)
                    result.append('(')
                    
                    if nest_depth > 0:
                        result.extend(generate_balanced(nest_depth))
                    else:
                        if random.random() < 0.7:
                            result.append('x')
                    
                    result.append(')')
                    remaining -= (nest_depth + 1)
                
                return result
            
            target_pairs = max(1, n // 3)
            sequence = generate_balanced(target_pairs)
        else:
            # 不正な括弧列を生成
            sequence = []
            for _ in range(n):
                sequence.append(random.choice(['(', ')', 'x']))
            
            # 確実に不正にする
            depth = 0
            is_actually_valid = True
            for char in sequence:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth < 0:
                        is_actually_valid = False
                        break
            if depth != 0:
                is_actually_valid = False
            
            if is_actually_valid and len(sequence) > 0:
                idx = random.randint(0, len(sequence) - 1)
                if sequence[idx] != 'x':
                    sequence[idx] = ')' if sequence[idx] == '(' else '('
        
        # 文字を 'a', 'b', 'c' に変換
        # '(' -> 'a', 'x' -> 'b', ')' -> 'c'
        char_map = {'(': 'a', 'x': 'b', ')': 'c'}
        result = ''.join(char_map.get(c, 'b') for c in sequence)
        
        # ラベルを追加: 正解なら 'd' (index 3), 不正なら 'a' (index 0)
        label_char = chr(ord('a') + 3) if is_valid else 'a'
        result += label_char
    
    elif nchar == 5:
        # Dyck-2: 2種類の括弧 () と []
        if is_valid:
            # 正しい括弧列を生成（ネストと順序が正しい）
            def generate_dyck2_balanced(depth):
                if depth == 0:
                    return []
                
                if depth == 1:
                    # ランダムに () か [] を選択
                    if random.random() < 0.5:
                        return ['(', ')']
                    else:
                        return ['[', ']']
                
                result = []
                remaining = depth
                
                while remaining > 0:
                    # ネストする深さを決定
                    nest_depth = random.randint(0, remaining - 1)
                    
                    # ランダムに () か [] を選択
                    if random.random() < 0.5:
                        result.append('(')
                        if nest_depth > 0:
                            result.extend(generate_dyck2_balanced(nest_depth))
                        result.append(')')
                    else:
                        result.append('[')
                        if nest_depth > 0:
                            result.extend(generate_dyck2_balanced(nest_depth))
                        result.append(']')
                    
                    remaining -= (nest_depth + 1)
                
                return result
            
            target_pairs = max(1, n // 2)
            sequence = generate_dyck2_balanced(target_pairs)
        else:
            # 不正な括弧列を生成
            # パターン1: ランダム生成（バランスが崩れる）
            # パターン2: 交差する括弧（例: ([)]）
            
            def check_dyck2_valid(seq):
                """Dyck-2の括弧列が正しいかチェック"""
                stack = []
                pairs = {'(': ')', '[': ']'}
                
                for char in seq:
                    if char in pairs:  # 開き括弧
                        stack.append(char)
                    else:  # 閉じ括弧
                        if not stack:
                            return False
                        open_char = stack.pop()
                        if pairs.get(open_char) != char:
                            return False
                
                return len(stack) == 0
            
            if random.random() < 0.5:
                # パターン1: ランダム生成
                sequence = []
                for _ in range(n):
                    sequence.append(random.choice(['(', ')', '[', ']']))
                
                # 確実に不正にする
                is_actually_valid = check_dyck2_valid(sequence)
                
                if is_actually_valid and len(sequence) > 0:
                    # 無理やり不正にする
                    idx = random.randint(0, len(sequence) - 1)
                    sequence[idx] = random.choice(['(', ')', '[', ']'])
            else:
                # パターン2: 交差する括弧を生成（より難しい）
                # 例: ([)], [(]], etc.
                base_length = max(4, n)
                sequence = []
                
                # 意図的に交差させる
                paren_types = [('(', ')'), ('[', ']')]
                for _ in range(base_length // 4):
                    type1, type2 = random.sample(paren_types, 2)
                    # 交差パターン: open1, open2, close1, close2
                    sequence.extend([type1[0], type2[0], type1[1], type2[1]])
                
                # 残りをランダムに埋める
                while len(sequence) < n:
                    sequence.append(random.choice(['(', ')', '[', ']']))
        
        # 文字を 'a', 'b', 'c', 'd' に変換
        # '(' -> 'a', ')' -> 'b', '[' -> 'c', ']' -> 'd'
        char_map = {'(': 'a', ')': 'b', '[': 'c', ']': 'd'}
        result = ''.join(char_map.get(c, 'a') for c in sequence)
        
        # ラベルを追加: 正解なら 'e' (index 4), 不正なら 'a' (index 0)
        label_char = chr(ord('a') + 4) if is_valid else 'a'
        result += label_char
    
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