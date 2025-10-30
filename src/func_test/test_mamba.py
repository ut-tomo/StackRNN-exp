import pytest
import torch
import torch.nn.functional as F
from models.baselines.mamba_model import MambaModel, MambaBlock


class TestMambaBlock:
    """MambaBlockのテスト"""
    
    @pytest.fixture
    def mamba_block(self):
        return MambaBlock(d_model=32, d_state=16, d_conv=4, expand_factor=2)
    
    def test_initialization(self, mamba_block):
        """ブロックの初期化テスト"""
        assert mamba_block.d_model == 32
        assert mamba_block.d_state == 16
        assert mamba_block.d_conv == 4
        assert mamba_block.d_inner == 64  # 32 * 2
        
        # パラメータが存在することを確認
        assert hasattr(mamba_block, 'A_log')
        assert hasattr(mamba_block, 'D')
        assert mamba_block.A_log.shape == (64, 16)
        assert mamba_block.D.shape == (64,)
    
    def test_forward_shape(self, mamba_block):
        """順伝播の形状テスト"""
        batch_size, seq_len, d_model = 2, 10, 32
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mamba_block(x)
        
        # 入力と同じ形状を保持
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_no_nan_inf(self, mamba_block):
        """NaN/Infが発生しないことを確認"""
        x = torch.randn(2, 10, 32)
        output = mamba_block(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_selective_scan(self, mamba_block):
        """selective_scanメソッドのテスト"""
        batch_size, seq_len, d_inner = 2, 5, 64
        d_state = 16
        
        x = torch.randn(batch_size, seq_len, d_inner)
        delta = F.softplus(torch.randn(batch_size, seq_len, d_inner))
        A = -torch.exp(torch.randn(d_inner, d_state))
        B = torch.randn(batch_size, seq_len, d_state)
        C = torch.randn(batch_size, seq_len, d_state)
        
        y = mamba_block.selective_scan(x, delta, A, B, C)
        
        assert y.shape == (batch_size, seq_len, d_inner)
        assert not torch.isnan(y).any()


class TestMambaModel:
    """MambaModelの包括的テストスイート"""
    
    @pytest.fixture
    def model(self):
        """標準的なテスト用モデル"""
        return MambaModel(nchar=10, nhid=32, nlayers=2)
    
    @pytest.fixture
    def small_model(self):
        """軽量テスト用モデル"""
        return MambaModel(nchar=5, nhid=16, nlayers=1)
    
    # === 基本的な初期化とパラメータテスト ===
    
    def test_initialization(self, model):
        """モデルの初期化テスト"""
        assert model.nchar == 10
        assert model.nhid == 32
        assert model.nlayers == 2
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'output_proj')
        assert len(model.layers) == 2
    
    def test_initialization_various_configs(self):
        """様々な設定での初期化"""
        configs = [
            {'nchar': 5, 'nhid': 16, 'nlayers': 1},
            {'nchar': 20, 'nhid': 64, 'nlayers': 4},
            {'nchar': 10, 'nhid': 32, 'nlayers': 2, 'd_state': 8},
        ]
        
        for config in configs:
            model = MambaModel(**config)
            assert model.nchar == config['nchar']
            assert model.nhid == config['nhid']
            assert model.nlayers == config['nlayers']
    
    def test_parameter_count(self, model):
        """パラメータ数の確認"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        print(f"Total parameters: {total_params:,}")
    
    def test_get_model_info(self, model):
        """モデル情報取得テスト"""
        info = model.get_model_info()
        assert info['model_type'] == 'Mamba'
        assert info['nchar'] == 10
        assert info['nhid'] == 32
        assert info['nlayers'] == 2
        assert info['total_params'] > 0
        assert info['trainable_params'] == info['total_params']
    
    # === 形状テスト（最重要） ===
    
    def test_forward_shape_batch(self, model):
        """バッチ入力の形状テスト"""
        batch_size, seq_len = 4, 10
        input_seq = torch.randint(0, 10, (batch_size, seq_len))
        output = model(input_seq)
        
        assert output.shape == (batch_size, seq_len, 10), \
            f"Expected shape ({batch_size}, {seq_len}, 10), got {output.shape}"
    
    def test_forward_shape_single(self, model):
        """単一シーケンスの形状テスト"""
        seq_len = 5
        input_seq = torch.randint(0, 10, (1, seq_len))
        output = model(input_seq)
        
        assert output.shape == (1, seq_len, 10)
    
    def test_forward_shape_1d_input(self, model):
        """1次元入力（バッチなし）の形状テスト"""
        seq_len = 7
        input_seq = torch.randint(0, 10, (seq_len,))
        output = model(input_seq)
        
        # 1次元入力の場合、バッチ次元が削除される
        assert output.shape == (seq_len, 10), \
            f"Expected shape ({seq_len}, 10), got {output.shape}"
    
    def test_forward_shape_various_lengths(self, model):
        """様々なシーケンス長でのテスト"""
        for seq_len in [1, 5, 20, 50]:
            input_seq = torch.randint(0, 10, (2, seq_len))
            output = model(input_seq)
            assert output.shape == (2, seq_len, 10), \
                f"Failed for seq_len={seq_len}"
    
    def test_forward_step_shape(self, model):
        """forward_stepの形状テスト"""
        seq_len = 7
        input_seq = torch.randint(0, 10, (seq_len,))
        
        output = model.forward_step(input_seq)
        
        # forward_stepは最後のトークンの出力を返す
        assert output.shape == (10,), \
            f"Expected shape (10,), got {output.shape}"
    
    def test_forward_step_with_scalar_input(self, model):
        """forward_stepにスカラーを入力"""
        input_char = torch.tensor(5)
        output = model.forward_step(input_char)
        
        assert output.shape == (10,)
    
    # === 数値安定性テスト ===
    
    def test_no_nan_or_inf(self, model):
        """NaN/Infが発生しないことを確認"""
        input_seq = torch.randint(0, 10, (4, 15))
        output = model(input_seq)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_numerical_stability_long_sequence(self, small_model):
        """長いシーケンスでの数値安定性"""
        input_seq = torch.randint(0, 5, (2, 200))
        output = small_model(input_seq)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        # 出力の統計量が妥当な範囲内
        assert output.abs().mean() < 100, "Output values too large"
    
    def test_output_range(self, small_model):
        """出力値が妥当な範囲にあるか"""
        input_seq = torch.randint(0, 5, (2, 10))
        output = small_model(input_seq)
        
        # Logitsなので、一般的に[-100, 100]程度の範囲
        assert output.min() > -1000, "Output too negative"
        assert output.max() < 1000, "Output too positive"
    
    # === 勾配フローテスト ===
    
    def test_backward_pass(self, small_model):
        """勾配計算テスト"""
        input_seq = torch.randint(0, 5, (2, 10))
        target = torch.randint(0, 5, (2, 10))
        
        output = small_model(input_seq)
        loss = F.cross_entropy(output.view(-1, 5), target.view(-1))
        loss.backward()
        
        # 全パラメータに勾配があることを確認
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} gradient contains NaN"
    
    def test_gradient_flow(self, small_model):
        """勾配が適切に伝播することを確認"""
        optimizer = torch.optim.Adam(small_model.parameters())
        input_seq = torch.randint(0, 5, (2, 8))
        target = torch.randint(0, 5, (2, 8))
        
        # パラメータのコピー
        params_before = {name: param.clone() for name, param in small_model.named_parameters()}
        
        # 1ステップの最適化
        optimizer.zero_grad()
        output = small_model(input_seq)
        loss = F.cross_entropy(output.view(-1, 5), target.view(-1))
        loss.backward()
        optimizer.step()
        
        # パラメータが更新されていることを確認
        params_changed = False
        for name, param in small_model.named_parameters():
            if not torch.allclose(param, params_before[name]):
                params_changed = True
                break
        
        assert params_changed, "No parameters were updated"
    
    def test_gradient_norm(self, small_model):
        """勾配のノルムが妥当な範囲にあるか"""
        input_seq = torch.randint(0, 5, (2, 10))
        target = torch.randint(0, 5, (2, 10))
        
        output = small_model(input_seq)
        loss = F.cross_entropy(output.view(-1, 5), target.view(-1))
        loss.backward()
        
        total_norm = 0
        for param in small_model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm > 0, "Gradient norm is zero"
        assert total_norm < 10000, f"Gradient norm too large: {total_norm}"
        print(f"Gradient norm: {total_norm:.4f}")
    
    # === 動作の一貫性テスト ===
    
    def test_determinism(self):
        """同じシードで同じ結果が得られるか"""
        torch.manual_seed(42)
        model1 = MambaModel(nchar=10, nhid=32, nlayers=1)
        
        torch.manual_seed(42)
        model2 = MambaModel(nchar=10, nhid=32, nlayers=1)
        
        torch.manual_seed(123)
        input_seq = torch.randint(0, 10, (2, 5))
        
        output1 = model1(input_seq)
        output2 = model2(input_seq)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Same initialization should produce same output"
    
    def test_batch_independence(self, model):
        """バッチ内の各サンプルが独立に処理されるか"""
        input_seq1 = torch.randint(0, 10, (1, 8))
        input_seq2 = torch.randint(0, 10, (1, 8))
        input_batch = torch.cat([input_seq1, input_seq2], dim=0)
        
        output1 = model(input_seq1)
        output2 = model(input_seq2)
        output_batch = model(input_batch)
        
        assert torch.allclose(output1, output_batch[0:1], atol=1e-5), \
            "Batch processing should be independent"
        assert torch.allclose(output2, output_batch[1:2], atol=1e-5), \
            "Batch processing should be independent"
    
    # === エッジケーステスト ===
    
    def test_minimum_sequence_length(self, model):
        """最小シーケンス長（1）のテスト"""
        input_seq = torch.randint(0, 10, (2, 1))
        output = model(input_seq)
        assert output.shape == (2, 1, 10)
    
    def test_large_batch_size(self, small_model):
        """大きなバッチサイズでのテスト"""
        input_seq = torch.randint(0, 5, (64, 10))
        output = small_model(input_seq)
        assert output.shape == (64, 10, 5)
    
    def test_different_batch_sizes(self, small_model):
        """様々なバッチサイズでのテスト"""
        for batch_size in [1, 2, 8, 16]:
            input_seq = torch.randint(0, 5, (batch_size, 10))
            output = small_model(input_seq)
            assert output.shape == (batch_size, 10, 5), \
                f"Failed for batch_size={batch_size}"
    
    # === log_probsオプションのテスト ===
    
    def test_forward_step_with_log_probs(self, model):
        """log_probsオプションのテスト"""
        input_seq = torch.randint(0, 10, (5,))
        
        output = model.forward_step(input_seq, log_probs=True)
        
        # log_softmaxの出力は負の値
        assert (output <= 0).all(), "Log probabilities should be <= 0"
        
        # exp(log_prob)の和が1に近い
        probs = torch.exp(output)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_forward_step_without_log_probs(self, model):
        """log_probs=Falseの動作確認"""
        input_seq = torch.randint(0, 10, (5,))
        
        output = model.forward_step(input_seq, log_probs=False)
        
        # Logitsなので正負両方あり得る
        assert output.shape == (10,)
    
    # === 学習可能性テスト ===
    
    @pytest.mark.slow
    def test_can_overfit_small_dataset(self, small_model):
        """小さなデータセットでオーバーフィット可能か"""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        
        # 固定された小さなデータセット
        torch.manual_seed(42)
        input_seq = torch.randint(0, 5, (4, 10))
        target = torch.randint(0, 5, (4, 10))
        
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            output = small_model(input_seq)
            loss = F.cross_entropy(output.view(-1, 5), target.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # 損失が減少していることを確認（Mambaは学習が難しいので緩い条件）
        assert losses[-1] < losses[0] * 0.7, \
            f"Model failed to overfit: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        # 損失が実際に減少していることも確認
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"✓ Overfit test passed: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # === 特殊ケーステスト ===
    
    def test_init_hidden(self, model):
        """init_hiddenメソッドの動作確認"""
        # MambaはstatelessなのでNoneを返す
        hidden = model.init_hidden(batch_size=4)
        assert hidden is None
    
    # === デバイステスト ===
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, small_model):
        """GPU上での実行テスト"""
        small_model = small_model.cuda()
        input_seq = torch.randint(0, 5, (2, 10)).cuda()
        
        output = small_model(input_seq)
        
        assert output.device.type == 'cuda'
        assert not torch.isnan(output).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_consistency(self):
        """CPU/GPU間での結果の一貫性"""
        torch.manual_seed(42)
        model_cpu = MambaModel(nchar=5, nhid=16, nlayers=1)
        
        torch.manual_seed(42)
        model_gpu = MambaModel(nchar=5, nhid=16, nlayers=1).cuda()
        
        input_seq = torch.randint(0, 5, (2, 5))
        
        output_cpu = model_cpu(input_seq)
        output_gpu = model_gpu(input_seq.cuda())
        
        assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-4), \
            "CPU and GPU outputs should be similar"


class TestMambaIntegration:
    """統合テスト"""
    
    def test_full_training_step(self):
        """完全な学習ステップのテスト"""
        model = MambaModel(nchar=5, nhid=32, nlayers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # ダミーデータ
        input_seq = torch.randint(0, 5, (4, 10))
        target = torch.randint(0, 5, (4, 10))
        
        # Forward
        output = model(input_seq)
        assert output.shape == (4, 10, 5)
        
        # Loss calculation
        loss = F.cross_entropy(output.view(-1, 5), target.view(-1))
        assert loss.item() >= 0
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        print(f"✓ Full training step completed with loss={loss.item():.4f}")
    
    def test_inference_mode(self):
        """推論モードでのテスト"""
        model = MambaModel(nchar=5, nhid=32, nlayers=1)
        model.eval()
        
        with torch.no_grad():
            input_seq = torch.randint(0, 5, (2, 10))
            output = model(input_seq)
            
            assert output.shape == (2, 10, 5)
            assert not torch.isnan(output).any()
        
        print("✓ Inference mode test passed")
    
    def test_multi_step_generation(self):
        """複数ステップの生成テスト（自己回帰的生成）"""
        model = MambaModel(nchar=5, nhid=16, nlayers=1)
        model.eval()
        
        # 初期入力
        current_seq = torch.tensor([0])
        
        generated = [0]
        with torch.no_grad():
            for _ in range(9):
                output = model.forward_step(current_seq)
                # 次の入力として最も確率の高いトークンを選択
                next_token = output.argmax(dim=-1)
                generated.append(next_token.item())
                # シーケンスを更新
                current_seq = torch.tensor(generated)
        
        assert len(generated) == 10
        assert all(0 <= token < 5 for token in generated)
        print(f"Generated sequence: {generated}")
    
    def test_different_sequence_positions(self):
        """シーケンスの異なる位置での予測"""
        model = MambaModel(nchar=5, nhid=16, nlayers=1)
        model.eval()
        
        # 同じシーケンスの異なる長さでの予測
        full_seq = torch.randint(0, 5, (10,))
        
        with torch.no_grad():
            # 長さ5までの予測
            out_5 = model.forward_step(full_seq[:5])
            # 長さ10での最後の予測
            out_10 = model(full_seq.unsqueeze(0))[0, -1, :]
            
            # 形状の確認
            assert out_5.shape == (5,)
            assert out_10.shape == (5,)


class TestMambaSpecialCases:
    """特殊ケースのテスト"""
    
    def test_very_short_sequence(self):
        """非常に短いシーケンスのテスト"""
        model = MambaModel(nchar=5, nhid=16, nlayers=1)
        
        # 長さ1のシーケンス
        input_seq = torch.randint(0, 5, (1,))
        output = model(input_seq)
        
        assert output.shape == (1, 5)
        assert not torch.isnan(output).any()
    
    def test_model_with_many_layers(self):
        """多層Mambaモデルのテスト"""
        model = MambaModel(nchar=5, nhid=16, nlayers=6)
        
        input_seq = torch.randint(0, 5, (2, 10))
        output = model(input_seq)
        
        assert output.shape == (2, 10, 5)
        assert not torch.isnan(output).any()
    
    def test_zero_input_sequence(self):
        """すべてゼロの入力シーケンス"""
        model = MambaModel(nchar=5, nhid=16, nlayers=1)
        
        input_seq = torch.zeros(2, 10, dtype=torch.long)
        output = model(input_seq)
        
        assert output.shape == (2, 10, 5)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    # 簡易実行用
    print("Running basic Mamba tests...")
    
    model = MambaModel(nchar=10, nhid=32, nlayers=2)
    print(f"✓ Model created: {model.get_model_info()}")
    
    input_seq = torch.randint(0, 10, (2, 5))
    output = model(input_seq)
    print(f"✓ Forward pass: input {input_seq.shape} -> output {output.shape}")
    
    print("\nRun full tests with: pytest func_test/test_mamba.py -v")
