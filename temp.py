import torch

# 模拟模型类（仅用于提供配置参数）
class MockModel:
    def __init__(self, num_windows, num_register_tokens):
        self.num_windows = num_windows  # 窗口数（如2）
        self.num_register_tokens = num_register_tokens  # 每个窗口的reg数量（如2）

def char_to_tensor(char_list, C=1):
    """将字符列表转换为张量（用字符的ASCII码作为数值，便于反向解析）"""
    # 为每个字符分配唯一数值（如cls1→101, reg1-1→201, patch1-1→301）
    char_to_val = {}
    val = 100  # 起始数值，避免与0冲突
    for char in char_list:
        if char not in char_to_val:
            char_to_val[char] = val
            val += 1
    # 转换为张量（形状：(len(char_list), C)）
    tensor = torch.tensor([[char_to_val[char]] for char in char_list], dtype=torch.float32)
    return tensor, char_to_val

def tensor_to_char(tensor, char_to_val):
    """将张量反向解析为字符列表"""
    val_to_char = {v: k for k, v in char_to_val.items()}
    return [val_to_char[round(val.item())] for val in tensor[:, 0]]

def test_rearrange_and_restore():
    # -------------------------- 1. 配置参数 --------------------------
    model = MockModel(num_windows=2, num_register_tokens=2)  # 2窗口→4个窗口（2²），每个窗口2个reg
    num_windows_squared = model.num_windows ** 2  # 4个窗口
    num_patches_per_window = 3  # 每个窗口3个patch
    C = 1  # 特征维度
    B_original = num_windows_squared  # 原始批次大小（4，每个窗口对应1个样本）
    
    # 每个窗口的字符结构：[clsX, regX-1, regX-2, patchX-1, patchX-2, patchX-3]
    window_chars = []
    for win_idx in range(1, num_windows_squared + 1):
        win_chars = [
            f"cls{win_idx}",  # 1个cls
            f"reg{win_idx}-1", f"reg{win_idx}-2",  # 2个reg（对应num_register_tokens=2）
            f"patch{win_idx}-1", f"patch{win_idx}-2", f"patch{win_idx}-3"  # 3个patch
        ]
        window_chars.append(win_chars)
    
    # 打印原始窗口结构（字符可视化）
    print("="*50)
    print("原始窗口结构（每个窗口的字符序列）：")
    for i, chars in enumerate(window_chars):
        print(f"窗口{i+1}: {chars}")
    
    # -------------------------- 2. 生成原始张量 --------------------------
    # 拼接所有窗口的字符，生成原始hidden_states（形状：(B_original, HW, C)）
    HW = len(window_chars[0])  # 每个窗口的长度：1+2+3=6
    all_chars = [char for win_chars in window_chars for char in win_chars]
    hidden_states_original, char_to_val = char_to_tensor(all_chars, C=C)
    hidden_states_original = hidden_states_original.view(B_original, HW, C)  # (4, 6, 1)
    attention_output_original = hidden_states_original.clone()  # 模拟attention_output（与hidden_states结构一致）
    
    print("\n" + "="*50)
    print(f"原始hidden_states形状: {hidden_states_original.shape}")
    print(f"原始attention_output形状: {attention_output_original.shape}")
    
    # -------------------------- 3. 执行重排列（remove windows） --------------------------
    run_full_attention = True
    if run_full_attention:
        B, HW, C = hidden_states_original.shape
        cls_per_window = 1
        reg_per_window = model.num_register_tokens
        patch_per_window = HW - cls_per_window - reg_per_window
        num_windows_squared = model.num_windows ** 2
        B_new = B // num_windows_squared  # 4//4=1
        
        # 重排列核心逻辑
        hidden_states = hidden_states_original.view(B_new, num_windows_squared, HW, C)
        all_cls = hidden_states[:, :, 0:cls_per_window, :]
        all_reg = hidden_states[:, :, cls_per_window:cls_per_window + reg_per_window, :]
        all_patch = hidden_states[:, :, cls_per_window + reg_per_window:, :]
        
        flattened_cls = all_cls.reshape(B_new, num_windows_squared * cls_per_window, C)
        flattened_reg = all_reg.reshape(B_new, num_windows_squared * reg_per_window, C)
        flattened_patch = all_patch.reshape(B_new, num_windows_squared * patch_per_window, C)
        
        hidden_states_rearranged = torch.cat([flattened_cls, flattened_reg, flattened_patch], dim=1)
        attention_output_rearranged = attention_output_original.view(B_new, num_windows_squared, HW, C)
        attention_output_rearranged = torch.cat([
            attention_output_rearranged[:, :, 0:cls_per_window, :].reshape(B_new, num_windows_squared * cls_per_window, C),
            attention_output_rearranged[:, :, cls_per_window:cls_per_window + reg_per_window, :].reshape(B_new, num_windows_squared * reg_per_window, C),
            attention_output_rearranged[:, :, cls_per_window + reg_per_window:, :].reshape(B_new, num_windows_squared * patch_per_window, C)
        ], dim=1)
    
    # 解析重排列后的字符序列（可视化）
    rearranged_chars = tensor_to_char(hidden_states_rearranged[0], char_to_val)
    print("\n" + "="*50)
    print(f"重排列后hidden_states形状: {hidden_states_rearranged.shape}")
    print(f"重排列后字符序列（cls→reg→patch）：")
    print(rearranged_chars)
    
    # -------------------------- 4. 执行恢复（add windows back） --------------------------
    if run_full_attention:
        B, HW, C = hidden_states_rearranged.shape
        num_windows_squared = model.num_windows ** 2
        cls_per_window = 1
        reg_per_window = model.num_register_tokens
        window_seq_length = HW // num_windows_squared  # 24//4=6（每个窗口原始长度）
        patch_per_window = window_seq_length - cls_per_window - reg_per_window
        total_cls = num_windows_squared * cls_per_window
        total_reg = num_windows_squared * reg_per_window
        
        # 恢复核心逻辑
        all_cls = hidden_states_rearranged[:, :total_cls, :].reshape(B, num_windows_squared, cls_per_window, C)
        all_reg = hidden_states_rearranged[:, total_cls:total_cls+total_reg, :].reshape(B, num_windows_squared, reg_per_window, C)
        all_patch = hidden_states_rearranged[:, total_cls+total_reg:, :].reshape(B, num_windows_squared, patch_per_window, C)
        
        windows = torch.cat([all_cls, all_reg, all_patch], dim=2)
        hidden_states_restored = windows.reshape(B * num_windows_squared, window_seq_length, C)
        
        # 恢复attention_output
        all_cls_attn = attention_output_rearranged[:, :total_cls, :].reshape(B, num_windows_squared, cls_per_window, C)
        all_reg_attn = attention_output_rearranged[:, total_cls:total_cls+total_reg, :].reshape(B, num_windows_squared, reg_per_window, C)
        all_patch_attn = attention_output_rearranged[:, total_cls+total_reg:, :].reshape(B, num_windows_squared, patch_per_window, C)
        
        windows_attn = torch.cat([all_cls_attn, all_reg_attn, all_patch_attn], dim=2)
        attention_output_restored = windows_attn.reshape(B * num_windows_squared, window_seq_length, C)
    
    # 解析恢复后的字符序列（可视化）
    restored_window_chars = []
    for i in range(num_windows_squared):
        win_tensor = hidden_states_restored[i]
        win_chars = tensor_to_char(win_tensor, char_to_val)
        restored_window_chars.append(win_chars)
    
    print("\n" + "="*50)
    print(f"恢复后hidden_states形状: {hidden_states_restored.shape}")
    print(f"恢复后窗口结构（每个窗口的字符序列）：")
    for i, chars in enumerate(restored_window_chars):
        print(f"窗口{i+1}: {chars}")
    
    # -------------------------- 5. 验证一致性 --------------------------
    print("\n" + "="*50)
    # 数值验证
    hidden_states_match = torch.allclose(hidden_states_restored, hidden_states_original)
    attention_output_match = torch.allclose(attention_output_restored, attention_output_original)
    
    # 字符序列验证
    char_match = True
    for orig_chars, restored_chars in zip(window_chars, restored_window_chars):
        if orig_chars != restored_chars:
            char_match = False
            break
    
    print(f"数值一致性验证：hidden_states {'通过' if hidden_states_match else '失败'}")
    print(f"数值一致性验证：attention_output {'通过' if attention_output_match else '失败'}")
    print(f"字符序列一致性验证：{'通过' if char_match else '失败'}")
    
    if hidden_states_match and attention_output_match and char_match:
        print("\n✅ 所有测试通过！重排列和恢复逻辑完全可逆，窗口结构无混乱。")
    else:
        print("\n❌ 测试失败！存在结构或顺序不一致。")

if __name__ == "__main__":
    test_rearrange_and_restore()