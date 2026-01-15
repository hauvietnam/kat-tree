import torch
import torch.nn as nn
import torch.nn.functional as F  # Import thêm cho padding
from torch.utils.data import DataLoader
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
import katransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
        crop_pct=1.0,
        pin_memory=True,
    )
    
    return loader, dataset

def evaluate_model(model, data_loader, device='cuda', use_tta=True):

    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    print(f"Evaluating model ...")
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            
            if use_tta:
                # --- KỸ THUẬT 10-CROP TTA ---
                # 1. Tạo 5 biến thể vị trí: Center, Top-Left, Top-Right, Bot-Left, Bot-Right
                # Bằng cách pad ảnh lên rồi crop lại
                b, c, h, w = images.shape
                pad = 4  # Dịch chuyển 4 pixel (phù hợp với ảnh 224)
                padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')
                
                crops = []

                crops.append(images)
                # 4 Góc
                crops.append(padded[:, :, 0:h, 0:w])       # Top-Left
                crops.append(padded[:, :, 0:h, -w:])       # Top-Right
                crops.append(padded[:, :, -h:, 0:w])       # Bot-Left
                crops.append(padded[:, :, -h:, -w:])       # Bot-Right
                

                crops_flipped = [torch.flip(c, [3]) for c in crops]
                
                # Tổng cộng 10 biến thể
                all_variants = crops + crops_flipped
                
                # Stack lại thành batch lớn: (B*10, C, H, W)
                # Lưu ý: Nếu batch size ban đầu quá lớn (ví dụ 64), 
                # thì batch này sẽ thành 640 -> có thể OOM. 
                # Code này an toàn vì default batch-size của bạn là 1.
                inputs = torch.cat(all_variants, dim=0)
                
                # Forward pass
                logits = model(inputs)
                
                # Reshape lại để tính trung bình: (10, B, Num_Classes)
                logits = logits.view(10, b, -1)
                
                # Lấy trung bình cộng các dự đoán (Ensemble logic)
                outputs = logits.mean(dim=0)
            else:
                # Đánh giá thường
                outputs = model(images)
            
            # Lấy predictions
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    
    print(f"\nTest Accuracy: {accuracy * 100:.4f}%")
    
    results = {
        'accuracy': accuracy,
        'num_samples': len(all_targets),
        'mode': 'tta_10_crop' if use_tta else 'standard'
    }
    
    return results, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          save_path='confusion_matrix.png',
                          figsize=(12, 10), 
                          normalize=False,
                          top_k_classes=None):
    # (Giữ nguyên code cũ của hàm này)
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
                fmt='',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                square=True,
                linewidths=0.5,
                linecolor='gray')
    
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
    parser = argparse.ArgumentParser(description='Evaluate KAT model with TTA')
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/dataset')
    parser.add_argument('--model', type=str, default='kat_base_patch16_224')
    parser.add_argument('--checkpoint', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/kat/output/kat_focal_loss_weighted/20260108-231219-kat_base_patch16_224-224/model_best.pth.tar')
    
    # Để batch-size nhỏ khi dùng TTA để tránh tràn VRAM (vì mỗi ảnh sẽ nhân 10 lên)
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size should be smaller when using TTA') 
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--top-k-classes', type=int, default=25)
    parser.add_argument('--output-dir', type=str, default='./results')
    
    parser.add_argument('--aug-test', action='store_true', default=True, 
                        help='')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model = load_model(args.model, args.checkpoint)
    
    test_loader, test_dataset = create_test_loader(
        args.data_path, 
        model, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    

    results, all_preds, all_targets = evaluate_model(model, test_loader, device, use_tta=args.aug_test)
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    class_names = [
        'Lim Xẹt', 'Lát Hoa', 'Lim Xanh', 'Thông Mã Vĩ', 'Keo Lá Tràm',
        'Sữa', 'Gội Trắng', 'Dẻ Cau', 'Re Hương', 'Đa',
        'Bằng Lăng', 'Phượng Vĩ', 'Vả', 'Sấu', 'Sếu',
        'Sồi Phảng', 'Lòng Mang', 'Bời Lời Lá Tròn', 'Thôi Ba', 'Gội',
        'Mé Cò Ke', 'Bời Lời', 'Sồi Xanh', 'Nanh Chuột', 'Chẹo'
    ]
    
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