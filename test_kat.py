import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
import katransformer  # Import để register KAT models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from tqdm import tqdm
import argparse
import json
import os

def load_model(model_name, checkpoint_path, num_classes=25):
    print(f"Creating model: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully!")
    return model

def create_test_loader(data_path, model, batch_size=64, num_workers=4):
    data_config = resolve_data_config({}, model=model)
    print(f"Data config: {data_config}")
    
    dataset = create_dataset(
        name='',
        root=data_path,
        split='validation', 
        download=False,
        class_map=''
    )
    
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=num_workers,
        crop_pct=data_config['crop_pct'], # Có thể thử chỉnh lên 1.0 nếu muốn test không crop
        pin_memory=True,
    )
    
    return loader, dataset

def get_label_priors(data_loader, num_classes=25):
    """
    Tính tần suất xuất hiện của các class trong tập test hiện tại.
    Dùng để cân bằng Logit Adjustment.
    """
    print("Computing class priors for Logit Adjustment...")
    label_counts = {}
    total_samples = 0
    
    # Duyệt nhanh để đếm (không forward model)
    for _, targets in data_loader:
        targets = targets.numpy()
        for t in targets:
            label_counts[t] = label_counts.get(t, 0) + 1
            total_samples += 1
            
    priors = np.zeros(num_classes)
    for cls, count in label_counts.items():
        if cls < num_classes:
            priors[cls] = count / total_samples
            
    # Tránh log(0) bằng cách gán giá trị rất nhỏ cho class không xuất hiện
    priors = np.maximum(priors, 1e-6)
    
    return torch.from_numpy(priors).float()

def evaluate_model(model, data_loader, device='cuda', use_tta=True, tau=1.0, num_classes=25):

    model.eval()
    model = model.to(device)
    
    # 1. Chuẩn bị Priors cho Logit Adjustment
    if tau > 0:
        priors = get_label_priors(data_loader, num_classes).to(device)
        log_priors = torch.log(priors)
    
    all_preds = []
    all_targets = []
    
    mode_msg = []
    if use_tta: mode_msg.append("10-Crop TTA")
    if tau > 0: mode_msg.append(f"Logit Adjustment (Tau={tau})")
    print(f"\nEvaluating with: {' + '.join(mode_msg) if mode_msg else 'Standard Mode'}")
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            
            # --- PHASE 1: PREDICTION (TTA or STANDARD) ---
            if use_tta:
                # Kỹ thuật 10-Crop
                b, c, h, w = images.shape
                pad = 4 
                padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')
                
                crops = [images] # Center
                crops.append(padded[:, :, 0:h, 0:w])       # Top-Left
                crops.append(padded[:, :, 0:h, -w:])       # Top-Right
                crops.append(padded[:, :, -h:, 0:w])       # Bot-Left
                crops.append(padded[:, :, -h:, -w:])       # Bot-Right
                
                crops_flipped = [torch.flip(c, [3]) for c in crops]
                
                # Stack thành batch lớn: (B*10, C, H, W)
                inputs = torch.cat(crops + crops_flipped, dim=0)
                
                logits = model(inputs)
                logits = logits.view(10, b, -1)
                outputs = logits.mean(dim=0) # Average Pooling
            else:
                outputs = model(images)
            
            # --- PHASE 2: LOGIT ADJUSTMENT (Cân bằng Macro) ---
            if tau > 0:
                # Trừ đi bias của các class phổ biến
                outputs = outputs - (tau * log_priors)

            # Lấy kết quả cuối cùng
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'num_samples': len(all_targets),
        'config': {'tta': use_tta, 'tau': tau}
    }
    
    return results, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          save_path='confusion_matrix.png',
                          figsize=(12, 10), 
                          normalize=False,
                          top_k_classes=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_raw = cm.copy()
    
    if top_k_classes is not None:
        class_counts = np.sum(cm, axis=1)
        top_k_indices = np.argsort(class_counts)[-top_k_classes:]
        cm = cm[top_k_indices][:, top_k_indices]
        cm_raw = cm_raw[top_k_indices][:, top_k_indices]
        if class_names is not None:
            class_names = [class_names[i] for i in top_k_indices]
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype('float') / row_sums
    else:
        cm_normalized = cm.astype('float')
    
    plt.figure(figsize=figsize)
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm_raw[i, j]
            if count > 0:
                annotations[i, j] = f'{count}'
            else:
                annotations[i, j] = ''
    
    sns.heatmap(cm_normalized if normalize else cm, 
                annot=annotations if cm.shape[0] <= 30 else False,
                fmt='', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                square=True, linewidths=0.5, linecolor='gray')
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if class_names:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate KAT model with TTA and Logit Adjustment')
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/dataset')
    parser.add_argument('--model', type=str, default='kat_base_patch16_224')
    parser.add_argument('--checkpoint', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/kat/output/kat_focal_loss_weighted/20260108-231219-kat_base_patch16_224-224/model_best.pth.tar')
    
    # Batch size nên giảm khi dùng TTA (vì 1 ảnh -> 10 ảnh)
    parser.add_argument('--batch-size', type=int, default=16) 
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--top-k-classes', type=int, default=25)
    parser.add_argument('--output-dir', type=str, default='./results')
    
    # --- CÁC THAM SỐ QUAN TRỌNG ĐỂ CÂN BẰNG METRICS ---
    parser.add_argument('--no-tta', action='store_true', help='Disable Test-Time Augmentation (10-Crop)')
    parser.add_argument('--tau', type=float, default=0.8, 
                        help='Logit Adjustment Factor. 0=Off, 1.0=Standard Balance. Increase to boost Macro, Decrease to save Accuracy.')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model = load_model(args.model, args.checkpoint)
    
    test_loader, test_dataset = create_test_loader(
        args.data_path, model, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    results, all_preds, all_targets = evaluate_model(
        model, test_loader, device, 
        use_tta=not args.no_tta,  
        tau=args.tau
    )
    
    # Lưu kết quả
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_path}")
    
    class_names = [
        'Lim Xẹt', 'Lát Hoa', 'Lim Xanh', 'Thông Mã Vĩ', 'Keo Lá Tràm',
        'Sữa', 'Gội Trắng', 'Dẻ Cau', 'Re Hương', 'Đa',
        'Bằng Lăng', 'Phượng Vĩ', 'Vả', 'Sấu', 'Sếu',
        'Sồi Phảng', 'Lòng Mang', 'Bời Lời Lá Tròn', 'Thôi Ba', 'Gội',
        'Mé Cò Ke', 'Bời Lời', 'Sồi Xanh', 'Nanh Chuột', 'Chẹo'
    ]
    
    # Vẽ biểu đồ
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_targets, all_preds, class_names=class_names, save_path=cm_path, top_k_classes=args.top_k_classes)
    
    cm_norm_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(all_targets, all_preds, class_names=class_names, save_path=cm_norm_path, normalize=True, top_k_classes=args.top_k_classes)
    
    if args.top_k_classes:
        print(f"\n=== Classification Report (Top {args.top_k_classes} classes) ===")
        class_counts = np.bincount(all_targets)
        top_k_indices = np.argsort(class_counts)[-args.top_k_classes:]
        mask = np.isin(all_targets, top_k_indices)
        print(classification_report(all_targets[mask], all_preds[mask], digits=4))

if __name__ == '__main__':
    main()