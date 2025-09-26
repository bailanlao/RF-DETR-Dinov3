import torch
import torch.nn as nn
from typing import Optional, Tuple

class MockPatchEmbeddings(nn.Module):
    def __init__(self, patch_size: int = 1):
        super().__init__()
        self.projection = nn.Identity()
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1x1 Patch嵌入：[B,C,H,W] → [B,H*W,C]（行优先展平）"""
        B, C, H, W = x.shape
        x = x.flatten(2)  # [B,C,H*W]
        x = x.transpose(1, 2)  # [B,H*W,C]
        return x

class TestModel(nn.Module):
    def __init__(self, num_windows: int = 2, patch_size: int = 1, num_register_tokens: int = 0):
        super().__init__()
        self.patch_embeddings = MockPatchEmbeddings(patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1))
        self.dropout = nn.Identity()
        self.config = type('', (), {})()
        self.config.num_windows = num_windows
        self.config.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, 1)) if num_register_tokens > 0 else None
        self.patch_size = patch_size

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        bool_masked_pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, int, int, int, torch.Tensor]:
        B, _, H, W = pixel_values.shape
        # 1. Patch嵌入
        x = self.patch_embeddings(pixel_values.to(dtype=torch.float32))  # [1,16,1]
        # 2. 拼接初始CLS
        cls_init = self.cls_token.expand(B, -1, -1)  # [1,1,1]
        x_pre_window = torch.cat([cls_init, x], dim=1)  # [1,17,1]（1CLS+16Patch）
        # 3. 窗口划分（严格按用户提供的逻辑）
        num_wins = self.config.num_windows
        win_grid_h, win_grid_w = 0, 0
        x_windowed = x_pre_window.clone()

        if num_wins > 1:
            win_grid_h = (H//self.patch_size) // num_wins  # 2
            win_grid_w = (W//self.patch_size) // num_wins  # 2
            # 分离CLS和Pixel Token
            cls_with_pos = x_pre_window[:, :1]  # [1,1,1]
            pix = x_pre_window[:, 1:]  # [1,16,1]
            # 维度变换：按窗口拆分
            pix = pix.view(B, H//self.patch_size, W//self.patch_size, -1)  # [1,4,4,1]
            pix = pix.view(B, num_wins, win_grid_h, num_wins, win_grid_w, -1)  # [1,2,2,2,2,1]
            pix = pix.permute(0, 1, 3, 2, 4, 5).contiguous()  # [1,2,2,2,2,1]
            pix = pix.view(B * num_wins**2, win_grid_h*win_grid_w, -1)  # [4,4,1]
            # 为每个窗口分配独立CLS（0,1,2,3）
            cls_values = torch.arange(num_wins**2, device=pix.device).float().view(-1,1,1)  # [4,1,1]
            cls_with_pos = cls_values.repeat(B,1,1)  # [4,1,1]
            # 重组窗口特征
            x_windowed = torch.cat([cls_with_pos, pix], dim=1)  # [4,5,1]（1CLS+4Patch/窗口）

        x_windowed = self.dropout(x_windowed)
        return x_windowed, H//self.patch_size, W//self.patch_size, win_grid_h, win_grid_w, x_pre_window

class MockRopeEmbed(nn.Module):
    def forward(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成与图像区域匹配的ROPE：左上角10、右上角11、左下角12、右下角13"""
        rope = torch.zeros(H, W, 1)
        half_h, half_w = H//2, W//2
        rope[:half_h, :half_w] = 10.0  # 窗口0
        rope[:half_h, half_w:] = 11.0  # 窗口1
        rope[half_h:, :half_w] = 12.0  # 窗口2
        rope[half_h:, half_w:] = 13.0  # 窗口3
        return rope.flatten(0,1), rope.flatten(0,1)*0.5  # [16,1], [16,1]

def create_test_image(H: int = 4, W: int = 4) -> torch.Tensor:
    """4x4测试图像：2x2区域分别为10、11、12、13"""
    img = torch.zeros(1, 1, H, W)
    half_h, half_w = H//2, W//2
    img[0,0,:half_h,:half_w] = 10.0
    img[0,0,:half_h,half_w:] = 11.0
    img[0,0,half_h:,:half_w] = 12.0
    img[0,0,half_h:,half_w:] = 13.0
    return img

def test_dimension_and_reshape():
    # 固定参数：4x4图像、2x2窗口、1x1 Patch
    H, W = 4, 4
    num_windows = 2
    patch_size = 1

    # 1. 创建测试图像并打印
    test_image = create_test_image(H, W)
    print("="*50)
    print("1. 原始图像信息")
    print(f"图像形状: {test_image.shape} → [B, C, H, W]")
    print("图像像素值（2x2区域划分）:")
    print(test_image[0,0])  # 打印4x4矩阵
    print("="*50)

    # 2. 模型推理（窗口划分）
    model = TestModel(num_windows=num_windows, patch_size=patch_size)
    x_windowed, grid_h, grid_w, win_grid_h, win_grid_w, x_pre_window = model(test_image)

    # 3. 打印窗口划分前的特征（维度+值）
    print("\n2. 窗口划分前特征（含CLS）")
    print(f"特征形状: {x_pre_window.shape} → [B, CLS+Patch数, C]")
    print(f"CLS数量: 1, Patch总数: {grid_h*grid_w}（{grid_h}x{grid_w}）")
    print("特征值（第1个样本，格式：CLS → Patch1 → Patch2 → ... → Patch16）:")
    print(x_pre_window[0].squeeze(-1))  # 展平最后一维，便于查看
    print("="*50)

    # 4. 打印窗口划分后的特征（维度+每个窗口的值）
    print("\n3. 窗口划分后特征（2x2窗口）")
    print(f"特征形状: {x_windowed.shape} → [窗口总数, CLS+窗口内Patch数, C]")
    print(f"窗口总数: {num_windows**2}（{num_windows}x{num_windows}）")
    print(f"每个窗口含Patch数: {win_grid_h*win_grid_w}（{win_grid_h}x{win_grid_w}）")
    print("各窗口特征值（格式：CLS → 窗口内Patch1 → Patch2 → Patch3 → Patch4）:")
    for win_idx in range(num_windows**2):
        win_val = x_windowed[win_idx].squeeze(-1)
        print(f"窗口{win_idx}: {win_val}")
    print("="*50)

    # 5. 打印ROPE编码（与窗口的对应关系，只打印值，不做标量转换）
    rope_embed = MockRopeEmbed()
    sin_rope, cos_rope = rope_embed(H, W)
    print("\n4. ROPE编码与窗口对应关系")
    print(f"ROPE形状: {sin_rope.shape} → [Patch总数, C]")
    # 修正ROPE索引：4x4图像按2x2窗口划分的正确Patch索引（行优先）
    win_rope_idx_map = {
        0: [0,1,4,5],    # 窗口0（左上角）：图像(0,0)-(1,1) → 展平后索引
        1: [2,3,6,7],    # 窗口1（右上角）：图像(0,2)-(1,3)
        2: [8,9,12,13],  # 窗口2（左下角）：图像(2,0)-(3,1)
        3: [10,11,14,15] # 窗口3（右下角）：图像(2,2)-(3,3)
    }
    print("各窗口对应的ROPE sin值:")
    for win_idx, rope_idx in win_rope_idx_map.items():
        win_sin = sin_rope[rope_idx].squeeze(-1)
        print(f"窗口{win_idx} → ROPE索引{rope_idx} → sin值: {win_sin}")
    print("="*50)

def test_view_4x4():
    # 1. 初始化测试数据：0-15顺序值（模拟4x4 patch展平序列）
    # 对应场景：B=1（单样本）、N=16（4x4 patch总数）、C=1（通道数）
    flat_sequence = torch.arange(16).view(1, 16, 1)
    print("="*60)
    print("1. 原始展平序列（行优先，对应4x4 patch展平后）")
    print(f"形状: {flat_sequence.shape} → [B, N=grid_h*grid_w, C]")
    print(f"数值（展平顺序）: {flat_sequence.squeeze().tolist()}")
    print("注：数值0-15对应4x4 patch的空间位置（行优先）：")
    print("行0: 0,1,2,3 | 行1:4,5,6,7 | 行2:8,9,10,11 | 行3:12,13,14,15")
    print("="*60)

    # 2. 第一步：恢复4x4 2D patch网格（对应代码中pix.view(B, grid_h, grid_w, -1)）
    # 模拟场景：grid_h=4（高度方向patch数）、grid_w=4（宽度方向patch数）
    reshaped_2d = flat_sequence.view(1, 4, 4, 1)
    print("\n2. 恢复2D patch网格（view(1,4,4,1)）")
    print(f"形状: {reshaped_2d.shape} → [B, grid_h, grid_w, C]")
    print("数值（4x4网格，对应原始空间位置）：")
    for h in range(4):  # 遍历高度方向（行）
        print(f"行{h}: {reshaped_2d[0, h, :, 0].tolist()}")
    print("="*60)

    # 3. 核心测试：view→permute→view 三步窗口划分（num_windows=2）
    # 配置参数：num_wins=2（2x2窗口划分）、win_grid_h=2（每个窗口高度patch数）、win_grid_w=2（每个窗口宽度patch数）
    B = 1
    num_wins = 2
    win_grid_h = 4 // num_wins  # 2
    win_grid_w = 4 // num_wins  # 2
    pix = reshaped_2d  # 此时pix是[1,4,4,1]，对应代码中"pix = pix.view(B, grid_h, grid_w, -1)"后的结果

    # 3.1 第一步view：拆分「窗口维度」与「窗口内patch维度」
    # 维度变化：[B, grid_h, grid_w, C] → [B, num_wins, win_grid_h, num_wins, win_grid_w, C]
    pix_view1 = pix.view(B, num_wins, win_grid_h, num_wins, win_grid_w, 1)
    print("\n3.1 第一步view：拆分窗口与窗口内patch（view(1,2,2,2,2,1)）")
    print(f"形状: {pix_view1.shape} → [B, 高度窗口数, 窗口内高度, 宽度窗口数, 窗口内宽度, C]")
    print("维度含义：")
    print(f"- 高度窗口数={num_wins}：高度方向分2个窗口（窗口0：行0-1，窗口1：行2-3）")
    print(f"- 窗口内高度={win_grid_h}：每个窗口含2行patch")
    print(f"- 宽度窗口数={num_wins}：宽度方向分2个窗口（窗口0：列0-1，窗口1：列2-3）")
    print(f"- 窗口内宽度={win_grid_w}：每个窗口含2列patch")
    print("数值（按维度展开，展示窗口拆分逻辑）：")
    for h_win in range(num_wins):  # 遍历高度窗口（0：上半部分，1：下半部分）
        for win_h in range(win_grid_h):  # 遍历窗口内高度（0：窗口内第1行，1：窗口内第2行）
            for w_win in range(num_wins):  # 遍历宽度窗口（0：左半部分，1：右半部分）
                for win_w in range(win_grid_w):  # 遍历窗口内宽度（0：窗口内第1列，1：窗口内第2列）
                    val = pix_view1[0, h_win, win_h, w_win, win_w, 0].item()
                    print(f"高度窗口{h_win}→窗口内行{win_h}→宽度窗口{w_win}→窗口内列{win_w}：{val}")
    print("="*60)

    # 3.2 第二步permute：调整维度顺序，对齐「窗口空间顺序」
    # 维度变化：[B, num_wins, win_grid_h, num_wins, win_grid_w, C] → [B, num_wins, num_wins, win_grid_h, win_grid_w, C]
    pix_permute = pix_view1.permute(0, 1, 3, 2, 4, 5).contiguous()
    print("\n3.2 第二步permute：调整窗口顺序（permute(0,1,3,2,4,5)）")
    print(f"形状: {pix_permute.shape} → [B, 高度窗口数, 宽度窗口数, 窗口内高度, 窗口内宽度, C]")
    print("核心作用：将「窗口内高度」与「宽度窗口数」交换，让维度优先体现「窗口的空间位置」（而非窗口内细节）")
    print("数值（按窗口分组，展示左上→右上→左下→右下的顺序）：")
    # 遍历所有窗口（高度窗口x宽度窗口，共4个窗口）
    window_order = [(0,0), (0,1), (1,0), (1,1)]  # 左上→右上→左下→右下
    window_names = ["左上窗口", "右上窗口", "左下窗口", "右下窗口"]
    for idx, (h_win, w_win) in enumerate(window_order):
        print(f"\n{window_names[idx]}（高度窗口{h_win}，宽度窗口{w_win}）：")
        # 打印该窗口内的2x2 patch值
        win_vals = []
        for win_h in range(win_grid_h):
            row_val = [pix_permute[0, h_win, w_win, win_h, win_w, 0].item() for win_w in range(win_grid_w)]
            win_vals.append(row_val)
        for row in win_vals:
            print(f"  窗口内行: {row}")
    print("="*60)

    # 3.3 第三步view：展平窗口，生成最终窗口特征
    # 维度变化：[B, num_wins, num_wins, win_grid_h, win_grid_w, C] → [B*num_wins², win_grid_h*win_grid_w, C]
    pix_view2 = pix_permute.view(B * num_wins * num_wins, win_grid_h * win_grid_w, 1)
    print("\n3.3 第三步view：展平窗口（view(4,4,1)）")
    print(f"形状: {pix_view2.shape} → [窗口总数, 每个窗口patch数, C]")
    print(f"- 窗口总数={B*num_wins*num_wins}：2x2划分共4个窗口")
    print(f"- 每个窗口patch数={win_grid_h*win_grid_w}：每个窗口含2x2=4个patch")
    print("数值（每个窗口的展平patch序列，顺序：左上→右上→左下→右下）：")
    for win_idx in range(4):
        win_val = pix_view2[win_idx].squeeze().tolist()
        print(f"{window_names[win_idx]}（索引{win_idx}）: {win_val}")
    print("="*60)

    # 4. 最终结论验证
    print("\n4. 结论验证：")
    print("三步操作后，窗口划分结果与「左上→右上→左下→右下」的空间顺序完全一致：")
    expected = {
        "左上窗口": [0,1,4,5],
        "右上窗口": [2,3,6,7],
        "左下窗口": [8,9,12,13],
        "右下窗口": [10,11,14,15]
    }
    for win_name in expected:
        idx = window_names.index(win_name)
        actual = pix_view2[idx].squeeze().tolist()
        status = "✅ 一致" if actual == expected[win_name] else "❌ 不一致"
        print(f"- {win_name}：预期{expected[win_name]}，实际{actual} → {status}")

if __name__ == "__main__":
    # test_dimension_and_reshape()
    test_view_4x4()
