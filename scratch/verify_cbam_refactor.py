import sys
import os
import copy
import torch
import torch.nn as nn

# Ensure project ultralytics is prioritized in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Drone-Detection-1"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
from ultralytics.nn.modules.cbam import apply_attention_to_yolo, AttentionLayerWrapper, ResidualAttentionBlock

def test_1_wrap_order_and_numerical_equivalence():
    print("==================================================")
    print("  TEST 1: Pretrained Load Order & Gamma=0 Identity")
    print("==================================================")
    
    # 1. Load baseline yolov8x.pt FIRST
    weights_path = os.path.join(project_root, "yolov8x.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8x.pt"
        
    print(f"Loading baseline weights from: {weights_path}...")
    baseline_model = YOLO(weights_path)
    baseline_net = baseline_model.model
    baseline_net.eval()
    
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        base_out = baseline_net(dummy_input)
        if isinstance(base_out, tuple):
            base_pred = base_out[0]
        else:
            base_pred = base_out

    modes = ['none', 'channel', 'spatial', 'cbam']
    
    for mode in modes:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        model = YOLO(weights_path)
        
        # Wrap AFTER loading weights
        if mode != 'none':
            apply_attention_to_yolo(model, mode=mode, target_layers=(12, 15))
            
        net = model.model
        net.eval()
        
        # Verify wrapped modules
        if mode != 'none':
            l12_wrapped = isinstance(net.model[12], AttentionLayerWrapper)
            l15_wrapped = isinstance(net.model[15], AttentionLayerWrapper)
            print(f"Layer 12 wrapped: {l12_wrapped}")
            print(f"Layer 15 wrapped: {l15_wrapped}")
            assert l12_wrapped and l15_wrapped, "Layer wrapping failed!"
            
        # Run forward pass
        with torch.no_grad():
            out = net(dummy_input)
            pred = out[0] if isinstance(out, tuple) else out
            
        # Numerical diff check
        max_diff = (pred - base_pred).abs().max().item()
        print(f"Max abs diff from baseline output: {max_diff:.8e}")
        assert max_diff < 1e-5, f"Numerical identity check failed for mode {mode}! Diff: {max_diff}"
        print(f"SUCCESS: Mode {mode.upper()} matches baseline output identically at gamma=0.")

def test_2_optimizer_grouping():
    print("\n==================================================")
    print("  TEST 2: Optimizer Parameter Grouping for Gamma")
    print("==================================================")
    
    weights_path = os.path.join(project_root, "yolov8x.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8x.pt"
        
    model = YOLO(weights_path)
    apply_attention_to_yolo(model, mode='cbam', target_layers=(12, 15))
    net = model.model
    
    # Simulate standard PyTorch / Ultralytics optimizer setup
    g_decay, g_no_decay = [], []
    for v in net.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g_no_decay.append(v.bias)
        if isinstance(v, (nn.BatchNorm2d, nn.GroupNorm)):
            g_no_decay.extend([p for p in v.parameters() if p.requires_grad])
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g_decay.append(v.weight)
            
    # Check where gamma parameters are placed
    gammas = [p for name, p in net.named_parameters() if "gamma" in name]
    print(f"Found {len(gammas)} gamma parameters in CBAM model.")
    
    for name, p in net.named_parameters():
        if "gamma" in name:
            in_decay = any(p is item for item in g_decay)
            in_no_decay = any(p is item for item in g_no_decay)
            print(f"Parameter '{name}': in_decay={in_decay}, in_no_decay={in_no_decay}")

def test_3_ema_copy_behavior():
    print("\n==================================================")
    print("  TEST 3: EMA Model Parameter Isolation")
    print("==================================================")
    
    weights_path = os.path.join(project_root, "yolov8x.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8x.pt"
        
    model = YOLO(weights_path)
    apply_attention_to_yolo(model, mode='cbam', target_layers=(12, 15))
    
    # Build EMA copy as Ultralytics ModelEMA does via deepcopy
    ema_model = copy.deepcopy(model.model)
    
    # Mutate live model's gamma
    live_gamma12 = model.model.model[12].attn_block.gamma
    ema_gamma12 = ema_model.model[12].attn_block.gamma
    
    with torch.no_grad():
        live_gamma12.add_(1.5)
        
    print(f"Live model layer 12 gamma value: {live_gamma12.item():.4f}")
    print(f"EMA  model layer 12 gamma value: {ema_gamma12.item():.4f}")
    
    assert live_gamma12.item() != ema_gamma12.item(), "EMA gamma is tied to live model gamma!"
    print("SUCCESS: EMA model maintains separate, isolated CBAM and gamma parameters.")

def test_4_checkpoint_save_reload():
    print("\n==================================================")
    print("  TEST 4: Checkpoint Save & Fresh Reload Test")
    print("==================================================")
    
    weights_path = os.path.join(project_root, "yolov8x.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8x.pt"
        
    model = YOLO(weights_path)
    apply_attention_to_yolo(model, mode='cbam', target_layers=(12, 15))
    
    # Set a non-zero gamma value
    with torch.no_grad():
        model.model.model[12].attn_block.gamma.add_(0.75)
        model.model.model[15].attn_block.gamma.add_(1.25)
        
    temp_ckpt_path = os.path.join(project_root, "temp_cbam_test.pt")
    print(f"Saving temporary checkpoint to {temp_ckpt_path}...")
    torch.save({"model": model.model}, temp_ckpt_path)
    
    # Reload checkpoint in plain process
    print("Reloading checkpoint with plain YOLO('temp_cbam_test.pt')...")
    reloaded_model = YOLO(temp_ckpt_path)
    r_net = reloaded_model.model
    
    l12_is_wrapped = isinstance(r_net.model[12], AttentionLayerWrapper)
    l15_is_wrapped = isinstance(r_net.model[15], AttentionLayerWrapper)
    
    g12 = r_net.model[12].attn_block.gamma.item()
    g15 = r_net.model[15].attn_block.gamma.item()
    
    print(f"Reloaded Layer 12 wrapped: {l12_is_wrapped}, gamma = {g12:.4f}")
    print(f"Reloaded Layer 15 wrapped: {l15_is_wrapped}, gamma = {g15:.4f}")
    
    assert l12_is_wrapped and l15_is_wrapped, "Reloaded model lost AttentionLayerWrapper!"
    assert abs(g12 - 0.75) < 1e-4 and abs(g15 - 1.25) < 1e-4, "Reloaded gamma values mismatch!"
    
    # Clean up temp checkpoint file
    if os.path.exists(temp_ckpt_path):
        os.remove(temp_ckpt_path)
        
    print("SUCCESS: Checkpoint save and reload verified without environment variables.")

if __name__ == "__main__":
    test_1_wrap_order_and_numerical_equivalence()
    test_2_optimizer_grouping()
    test_3_ema_copy_behavior()
    test_4_checkpoint_save_reload()
    print("\n==================================================")
    print(" ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! ")
    print("==================================================")
