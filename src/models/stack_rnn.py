import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackRNN(nn.Module):
    """
    In: 
         my_int si,       // _IN: 入力次元(語彙数など)入力は整数IDで渡し, one-hot相当で扱う. _INと_OUTを分ける意味がないのでncharで統一.
         my_int sh,       // _HIDDEN: 隠れ層の次元
         my_int nstack,   // _NB_STACK: スタック本数(並列スタックの数)
         my_int stack_capacity, // _STACK_SIZE: 各スタックの容量(固定長デック)
         my_int so,       // _OUT: 出力次元(クラス数)
         my_int sm,       // _BPTT: BPTTの最大展開長(循環バッファ長)
         my_int bptt_step,// _BPTT_STEP: 何ステップごとに出力勾配を流すか(TBPTTの粒度)
         my_int mod = 1,  // 再帰の種類 0: なし, 1: スタック経由のみ, 2: フルRNN(隠れ直結+スタック)
         bool isnoop=false,// noop 行動を使うなら true(push/pop に加えて noop で3値)
         my_int depth=1,  // _DEPTH: 次の隠れ状態を作るとき参照する「スタックの上から何段分」
         my_real reg=0)   // _reg: 行動分布エントロピーの正則化係数(ここでは使っていない)
         
         デフォルトでスタック数10, スタックサイズ200(<-?), BPTT長50, depth2, mod1, noopなし, 正則化なし
    """
    def __init__(self, nchar, nhid, nstack=10, stack_size=200, depth=2, 
                 mod=1, use_noop=False, bptt=50, reg=0.0, init_method='xavier'):
        super().__init__()
        
        self.nchar = nchar
        self.n_in = nchar
        self.n_out = nchar
        self.nhid = nhid
        self.nstack = nstack
        self.n_stack = nstack
        self.stack_size = stack_size
        self.depth = depth
        self.mod = mod
        self.use_noop = use_noop
        self.n_action = 3 if use_noop else 2
        self.top_of_stack = 0
        self.bptt = bptt
        self.reg = reg
        self.count = 0
        
        # initialization method. ref 論文の初期化は 呼び方がわからないので 'cpp'
        valid_methods = ['xavier', 'cpp', 'he', 'orthogonal', 'uniform']
        if init_method not in valid_methods:
            raise ValueError(f"init_method must be one of {valid_methods}, got '{init_method}'")
        
        self._create_layers()
        self._init_weights()
        
        self._create_shift_matrices()
        
        self.reset_state()
        
    def _create_layers(self):
        self.in2hid = nn.Linear(self.n_in, self.nhid, bias=False)
        self.hid2hid = nn.Linear(self.nhid, self.nhid, bias=False)

        self.hid2act = nn.ModuleList([
            nn.Linear(self.n_hid, self.n_action, bias=False) 
            for _ in range(self.n_stack)
        ])
        self.hid2stack = nn.ModuleList([
            nn.Linear(self.n_hid, self.stack_size, bias=False)
            for _ in range(self.n_stack)
        ])
        self.stack2hid = nn.ModuleList([
            nn.Linear(self.stack_size, self.n_hid, bias=False)
            for _ in range(self.n_stack)
        ])
        
        self.hid2out = nn.Linear(self.n_hid, self.n_out, bias=False)
        
    def _initialize_weights(self):
        """Initialize weights using specified initialization method."""
        
        # Define initialization functions
        def cpp_init(layer):
            """C++ original: sum of 3 uniform(-0.1, 0.1) distributions."""
            with torch.no_grad():
                w = layer.weight.data
                u1 = torch.empty_like(w).uniform_(-0.1, 0.1)
                u2 = torch.empty_like(w).uniform_(-0.1, 0.1)
                u3 = torch.empty_like(w).uniform_(-0.1, 0.1)
                w.copy_(u1 + u2 + u3)
        
        def xavier_init(layer):
            """Xavier/Glorot initialization."""
            fan_in = layer.weight.size(1)
            fan_out = layer.weight.size(0)
            std = math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
        
        def he_init(layer):
            """He initialization (good for ReLU)."""
            fan_in = layer.weight.size(1)
            std = math.sqrt(2.0 / fan_in)
            nn.init.normal_(layer.weight, mean=0.0, std=std)
        
        def orthogonal_init(layer):
            """Orthogonal initialization (good for RNN recurrence)."""
            nn.init.orthogonal_(layer.weight)
        
        def uniform_init(layer):
            """Simple uniform distribution [-0.1, 0.1]."""
            nn.init.uniform_(layer.weight, -0.1, 0.1)
        
        # Select initialization function
        init_functions = {
            'cpp': cpp_init,
            'xavier': xavier_init,
            'he': he_init,
            'orthogonal': orthogonal_init,
            'uniform': uniform_init,
        }
        init_func = init_functions[self.init_method]
        
        init_func(self.in2hid)
        
        if self.mod != 2:
            nn.init.zeros_(self.hid2hid.weight)
        else:
            # For recurrent connections, orthogonal is often betterらしい
            if self.init_method == 'orthogonal':
                orthogonal_init(self.hid2hid)
            else:
                init_func(self.hid2hid)
        
        for s in range(self.n_stack):
            init_func(self.hid2act[s])
            init_func(self.hid2stack[s])
            init_func(self.stack2hid[s])
            
            with torch.no_grad():
                self.stack2hid[s].weight[:, self.top_of_stack + self.depth:] = 0
                self.hid2stack[s].weight[self.top_of_stack + 1:, :] = 0
        
        init_func(self.hid2out)
        
    def _create_shift_matrices(self):
        """
        スタックの更新を行列積で表現できるようにする.
        New Stack = Shift Mat @ Old Stack の形. 
        
        C++実装 (参考)
        
        push演算:
            スタックを下方向に1段シフトし, TOP位置に新しい値 v_t を挿入する.
            下記のように実装される：
                for i = STACK_SIZE-1 → TOP+1:
                    stack[i] = stack[i-1];
                stack[TOP] = v_t;
            → 古い要素は1段下に潜り, DEPTH以降の領域に保存され続ける.

        pop演算:
            スタックを上方向に1段シフトし, 最上段を削除する（末尾を0埋め）.
                for i = TOP → STACK_SIZE-2:
                    stack[i] = stack[i+1];
                stack[STACK_SIZE-1] = 0;
            → 深い層にあった値が浮上し, 再び読み出し可能になる.
        
        以下のような連続化が行われる: 
        new_stack[i] = p_push * push_buf[i]
                 + p_pop  * pop_buf[i]
                 + p_noop * stack[i];
                 
        行列演算として表現するには, まずは depth以降の0マスクが必要.
        その上で適切に要素をシフトし上位の段を連続的に計算する.
        """
        
        