import pytest
import torch
import torch.nn.functional as F
from models.baselines.transformer_model import TransformerModel, PositionalEncoding


class TestPositionalEncoding:
    """PositionalEncodingのテスト"""
    
    @pytest.fixture
    def pos_encoder(self):
        return PositionalEncoding(d_model=64, max_len=100)
    
    def test_initialization(self, pos_encoder):
        """初期化テスト"""
        assert pos_encoder.pe.shape == (100, 64)
        # PEは登録されたバッファであることを確認
        assert not pos_encoder.pe.requires_grad
    
    def test_forward_shape(self, pos_encoder):
        """順伝播の形状テスト"""
        x = torch.randn(10, 2, 64)  # (seq_len, batch_size, d_model)
        output = pos_encoder(x)
        assert output.shape == (10, 2, 64)
    
    def test_positional_encoding_values(self):
        """位置エンコーディングの値が正しいか"""
        pos_encoder = PositionalEncoding(d_model=4, max_len=10)
        # 最初の位置エンコーディングは0でない
        assert pos_encoder.pe[0].abs().sum() > 0
        # 異なる位置は異なる値を持つ
        assert not torch.allclose(pos_encoder.pe[0], pos_encoder.pe[1])


class TestTransformerModel:
    """TransformerModelの包括的テストスイート"""
    
    @pytest.fixture
    def model(self):
        """標準的なテスト用モデル"""
        return TransformerModel(nchar=10, nhid=64, nhead=8, nlayers=2)
    
    @pytest.fixture
    def small_model(self):
        """軽量テスト用モデル"""
        return TransformerModel(nchar=5, nhid=32, nhead=4, nlayers=1)
    
    # === 基本的な初期化とパラメータテスト ===
    
    def test_initialization(self, model):
        """モデルの初期化テスト"""
        assert model.nchar == 10
        assert model.nhid == 64
        assert model.nhead == 8
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'pos_encoder')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'output_proj')
    
    def test_parameter_count(self, model):
        """パラメータ数の確認"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        print(f"Total parameters: {total_params:,}")
    
    def test_get_model_info(self, model):
        """モデル情報取得テスト"""
        info = model.get_model_info()
        assert info['model_type'] == 'Transformer'
        assert info['nchar'] == 10
        assert info['nhid'] == 64
        assert info['nhead'] == 8
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
    
    def test_forward_shape_various_lengths(self, model):
        """様々なシーケンス長でのテスト"""
        for seq_len in [1, 5, 20, 50]:
            input_seq = torch.randint(0, 10, (2, seq_len))
            output = model(input_seq)
            assert output.shape == (2, seq_len, 10), \
                f"Failed for seq_len={seq_len}"
    
    def test_forward_step_shape(self, model):
        """forward_stepの形状テスト"""
        batch_size = 3
        seq_len = 7
        input_seq = torch.randint(0, 10, (batch_size, seq_len))
        output = model.forward_step(input_seq)
        
        # forward_stepは最後のトークンの出力を返す
        assert output.shape == (10,) or output.shape == (batch_size, 10), \
            f"Expected shape (10,) or ({batch_size}, 10), got {output.shape}"
    
    # === マスク生成テスト ===
    
    def test_causal_mask_generation(self, model):
        """因果マスクの生成テスト"""
        seq_len = 5
        mask = model.generate_square_subsequent_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
        
        # 下三角行列のチェック（対角含む）
        # mask[i, j] が 0.0 なら i >= j (過去を見れる)
        # mask[i, j] が -inf なら i < j (未来を見れない)
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0.0"
                else:
                    assert mask[i, j] == float('-inf'), f"mask[{i},{j}] should be -inf"
    
    def test_forward_with_causal_mask(self, model):
        """因果マスクありでのforward"""
        input_seq = torch.randint(0, 10, (2, 8))
        output = model(input_seq, use_causal_mask=True)
        
        assert output.shape == (2, 8, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_without_mask(self, model):
        """マスクなしでのforward"""
        input_seq = torch.randint(0, 10, (2, 8))
        output = model(input_seq, use_causal_mask=False)
        
        assert output.shape == (2, 8, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    # === 数値安定性テスト ===
    
    def test_no_nan_or_inf(self, model):
        """NaN/Infが発生しないことを確認"""
        input_seq = torch.randint(0, 10, (4, 15))
        output = model(input_seq)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_numerical_stability_long_sequence(self, small_model):
        """長いシーケンスでの数値安定性"""
        input_seq = torch.randint(0, 5, (2, 100))
        output = small_model(input_seq)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        # 出力の統計量が妥当な範囲内
        assert output.abs().mean() < 100, "Output values too large"
    
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
    
    # === 動作の一貫性テスト ===
    
    def test_determinism(self):
        """同じシードで同じ結果が得られるか（evalモードで）"""
        torch.manual_seed(42)
        model1 = TransformerModel(nchar=10, nhid=32, nhead=4, nlayers=1)
        model1.eval()  # Dropoutを無効化
        
        torch.manual_seed(42)
        model2 = TransformerModel(nchar=10, nhid=32, nhead=4, nlayers=1)
        model2.eval()  # Dropoutを無効化
        
        torch.manual_seed(123)  # 入力用のシード
        input_seq = torch.randint(0, 10, (2, 5))
        
        with torch.no_grad():
            output1 = model1(input_seq)
            output2 = model2(input_seq)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Same initialization should produce same output in eval mode"
    
    def test_batch_independence(self, model):
        """バッチ内の各サンプルが独立に処理されるか（evalモードで確認）"""
        model.eval()  # Dropoutを無効化して決定的な動作にする
        
        input_seq1 = torch.randint(0, 10, (1, 8))
        input_seq2 = torch.randint(0, 10, (1, 8))
        input_batch = torch.cat([input_seq1, input_seq2], dim=0)
        
        # マスクなしで処理（全トークンが見える）
        with torch.no_grad():
            output1 = model(input_seq1, use_causal_mask=False)
            output2 = model(input_seq2, use_causal_mask=False)
            output_batch = model(input_batch, use_causal_mask=False)
        
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
    
    def test_maximum_sequence_length(self, small_model):
        """最大シーケンス長付近のテスト"""
        # max_len=1000がデフォルトなので、それより短い長さでテスト
        seq_len = 200
        input_seq = torch.randint(0, 5, (2, seq_len))
        output = small_model(input_seq)
        assert output.shape == (2, seq_len, 5)
    
    def test_different_batch_sizes(self, small_model):
        """様々なバッチサイズでのテスト"""
        for batch_size in [1, 2, 8, 16]:
            input_seq = torch.randint(0, 5, (batch_size, 10))
            output = small_model(input_seq)
            assert output.shape == (batch_size, 10, 5), \
                f"Failed for batch_size={batch_size}"
    
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
        
        # 損失が減少していることを確認（0.5倍以下になればOK）
        assert losses[-1] < losses[0] * 0.5, \
            f"Model failed to overfit: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        # 損失が実際に減少していることも確認
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"✓ Overfit test passed: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # === 特殊ケーステスト ===
    
    def test_init_hidden(self, model):
        """init_hiddenメソッドの動作確認"""
        # Transformerはhidden stateを持たないのでNoneを返す
        hidden = model.init_hidden(batch_size=4)
        assert hidden is None
    
    def test_forward_step_with_log_probs(self, model):
        """log_probsオプションのテスト"""
        input_seq = torch.randint(0, 10, (2, 5))
        output = model.forward_step(input_seq, log_probs=True)
        
        # log_softmaxの出力は負の値
        assert (output <= 0).all(), "Log probabilities should be <= 0"
        
        # 各サンプルについて、exp(log_prob)の和が1に近い
        if output.dim() == 2:  # (batch, nchar)
            probs = torch.exp(output)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    
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
        model_cpu = TransformerModel(nchar=5, nhid=32, nhead=4, nlayers=1)
        
        torch.manual_seed(42)
        model_gpu = TransformerModel(nchar=5, nhid=32, nhead=4, nlayers=1).cuda()
        
        input_seq = torch.randint(0, 5, (2, 5))
        
        output_cpu = model_cpu(input_seq)
        output_gpu = model_gpu(input_seq.cuda())
        
        assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-4), \
            "CPU and GPU outputs should be similar"


class TestTransformerIntegration:
    """統合テスト"""
    
    def test_full_training_step(self):
        """完全な学習ステップのテスト"""
        model = TransformerModel(nchar=5, nhid=32, nhead=4, nlayers=1)
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
        model = TransformerModel(nchar=5, nhid=32, nhead=4, nlayers=1)
        model.eval()
        
        with torch.no_grad():
            input_seq = torch.randint(0, 5, (2, 10))
            output = model(input_seq)
            
            assert output.shape == (2, 10, 5)
            assert not torch.isnan(output).any()
        
        print("✓ Inference mode test passed")


if __name__ == "__main__":
    # 簡易実行用
    print("Running basic Transformer tests...")
    
    model = TransformerModel(nchar=10, nhid=64, nhead=8, nlayers=2)
    print(f"✓ Model created: {model.get_model_info()}")
    
    input_seq = torch.randint(0, 10, (2, 5))
    output = model(input_seq)
    print(f"✓ Forward pass: input {input_seq.shape} -> output {output.shape}")
    
    print("\nRun full tests with: pytest func_test/test_transformer.py -v")
