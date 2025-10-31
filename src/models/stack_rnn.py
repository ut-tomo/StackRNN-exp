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
        self.init_method = init_method

        
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
        #in2hid: 論文のU, hid2hid: 論文のR, stack2hid: 論文のP
        self.in2hid = nn.Linear(self.n_in, self.nhid, bias=False)
        self.hid2hid = nn.Linear(self.nhid, self.nhid, bias=False)

        self.hid2act = nn.ModuleList([
            nn.Linear(self.nhid, self.n_action, bias=False) 
            for _ in range(self.n_stack)
        ])
        self.hid2stack = nn.ModuleList([
            nn.Linear(self.nhid, self.stack_size, bias=False)
            for _ in range(self.n_stack)
        ])
        self.stack2hid = nn.ModuleList([
            nn.Linear(self.stack_size, self.nhid, bias=False)
            for _ in range(self.n_stack)
        ])
        
        self.hid2out = nn.Linear(self.nhid, self.n_out, bias=False)
        
    def _init_weights(self):
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
                 
        行列演算として表現するには, depth以降の0マスクが必要.
        その上で適切に要素をシフトし上位の段を連続的に計算する.
        """
        
        push_matrix = torch.zeros(self.stack_size, self.stack_size)
        for i in range(self.top_of_stack + 1, self.stack_size):
            push_matrix[i, i-1] = 1.0 #対角線の1つ下に1 -> 元ベクトルを一段下にシフト
        self.register_buffer('push_matrix', push_matrix)
        
        pop_matrix = torch.zeros(self.stack_size, self.stack_size)
        for i in range(self.top_of_stack, self.stack_size - 1):
            pop_matrix[i, i+1] = 1.0 #対角線の1つ上に1 -> 元ベクトルを一段上にシフト
        self.register_buffer('pop_matrix', pop_matrix)
        
        noop_matrix = torch.eye(self.stack_size)
        noop_matrix[:self.top_of_stack, :] = 0
        self.register_buffer('noop_matrix', noop_matrix)
        
    def reset_state(self):
        """Reset internal states for new seq"""
        self.count = 0
        
        self.stacks = []
        for s in range(self.n_stack):
            stack = torch.full((self.stack_size,), -1.0, dtype=torch.float32)
            self.stacks.append(stack)
        
        self.hidden = torch.zeros(self.nhid, dtype=torch.float32)
        self.is_emptied = True
        self.step_count = 0
    
    def empty_stacks(self):
        """Empty all stacks (reset to -1.0)."""
        self.count = 0
        self.is_emptied = True
        self.step_count = 0
        for s in range(self.n_stack):
            self.stacks[s].fill_(-1.0)
            
    def forward_step(self, cur_input, target, is_hard=False, training=True):
        """
        Single forward step with gradient control for online learning.
        
        Args:
            cur_input: Current input token (integer)
            target: Target output token (integer)
            is_hard: Whether to use hard (discretized) actions
            training: If True, detach for single-step gradients; if False, detach completely
            
        Returns:
            output_probs: Output probability distribution
            loss: Cross-entropy loss for this step
            
        h_t = sigmoid( U x_t + R h_t-1 + P s_t-1 ^k )    を計算したい
        """
        device = next(self.parameters()).device
        
        if training:
            # TBPTT 用：一つ前の状態を計算グラフから切り離しつつ「現在の変数」は生かす
            old_hidden = self.hidden.detach().clone()
            old_stacks = [s.detach().clone() for s in self.stacks]
        else:
            # 完全切断：内部状態も detach してから使用
            self.hidden = self.hidden.to(device).detach()
            self.stacks = [stk.to(device).detach() for stk in self.stacks]
            old_hidden = self.hidden.clone()
            old_stacks = [stk.clone() for stk in self.stacks]
            
        for s in range(self.n_stack):
            self.stacks[s] = self.stacks[s].to(device).detach()
        self.hidden = self.hidden.to(device).detach()
        
        old_hidden = self.hidden.clone()
        old_stacks = [stack.clone() for stack in self.stacks]
        
        self.count += 1
        self.is_emptied = False
        self.step_count += 1
        
        # Input の one-hot エンコーディング -> hidden state
        input_one_hot = torch.zeros(self.nchar, device=device)
        input_one_hot[cur_input] = 1.0
        self.hidden = self.in2hid(input_one_hot)
        
        if self.mod != 0:
            for s in range(self.n_stack):
                # P s_t-1 ^k 部分
                stack_slice = old_stacks[s][self.top_of_stack:self.top_of_stack + self.depth]
                weight_slice = self.stack2hid[s].weight[:, self.top_of_stack:self.top_of_stack + self.depth]
                hidden_contrib = weight_slice @ stack_slice
                self.hidden += hidden_contrib
        
        if self.mod == 2:
            self.hidden += self.hid2hid(old_hidden)
        
        self.hidden = torch.sigmoid(self.hidden)
        
        new_stacks = []
        
        # 連続的な書き込み 計算済みのシフト行列を使用
        # s_t[i] = a_t[PUSH] s_t-1[i-1] + a_t[POP]s_t-1[i+1] (+ a_t[NOOP] s_t-1[i] )
        
        for s in range(self.n_stack):
            action_logits = self.hid2act[s](self.hidden)
            actions = F.softmax(action_logits, dim=0)
            
            if is_hard:
                # argmax 離散化
                im = torch.argmax(actions).item()
                actions = torch.zeros_like(actions)
                actions[im] = 1.0
                
            # Action 確信度
            pop_weight = actions[0]
            push_weight = actions[1]
            noop_weight = actions[2] if self.n_action == 3 else torch.tensor(0.0, device=device)
            
            old_stack = old_stacks[s]
            
            pushed_stack = self.push_matrix @ old_stack
            
            # new top element
            top_input = self.hid2stack[s].weight[self.top_of_stack, :] @ self.hidden
            top_input = torch.clamp(top_input, -50.0, 50.0)  # Numerical stability
            top_value = torch.sigmoid(top_input)
            pushed_stack[self.top_of_stack] = top_value
            
            popped_stack = self.pop_matrix @ old_stack
            popped_stack[self.stack_size - 1] = -1.0  # Empty value at bottom
            
            # Noop: 
            noop_stack = self.noop_matrix @ old_stack
            
            # soft operation の統合
            new_stack = (push_weight * pushed_stack + 
                        pop_weight * popped_stack + 
                        noop_weight * noop_stack)
            
            new_stacks.append(new_stack)
            
        for s in range(self.n_stack):
            self.stacks[s] = new_stacks[s]
        

        output_logits = self.hid2out(self.hidden)
        output_probs = F.softmax(output_logits, dim=0)
        

        loss = -torch.log(output_probs[target] + 1e-10)
        
        # Regularization (Not used in ref paper)
        if self.reg > 0 and training:
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.sum(param ** 2)
            loss = loss + self.reg * l2_reg
        
        return output_probs, loss
    
    def detach_hidden_states(self):
        self.hidden = self.hidden.detach()
        for s in range(self.n_stack):
            self.stacks[s] = self.stacks[s].detach()
        self.step_count = 0
        
    def forward(self, input_sequence, is_hard=False):
        output_probs_list = []
        
        for i in range(len(input_sequence) - 1):
            cur_input = input_sequence[i]
            target = input_sequence[i + 1]
            
            output_probs, _ = self.forward_step(cur_input, target, is_hard=is_hard, training=False)
            output_probs_list.append(output_probs)
        
        return output_probs_list
    
    def get_model_info(self):
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Stack-RNN',
            'nchar': self.nchar,
            'nhid': self.nhid,
            'nstack': self.nstack,
            'stack_size': self.stack_size,
            'depth': self.depth,
            'mod': self.mod,
            'use_noop': self.use_noop,
            'n_action': self.n_action,
            'init_method': self.init_method,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }

