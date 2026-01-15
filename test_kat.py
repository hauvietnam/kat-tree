import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import argparse
import json
import os
import sys

# Import module katransformer để register models
# Lưu ý: File katransformer.py phải nằm cùng thư mục hoặc trong PYTHONPATH
try:
    import katransformer
except ImportError:
    print("Warning: Could not import 'katransformer'. Make sure the module is in your path if using custom KAT models.")

def load_model(model_name, checkpoint_path, num_classes=25):
    """
    Load KAT model từ checkpoint
    """
    print(f"Creating model: {model_name}")
    
    # Tạo model
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Please ensure 'katransformer' is imported and model name is correct.")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Xử lý checkpoint structure
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict vào model
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Checkpoint loaded successfully! ({msg})")
    
    return model


def create_test_loader(data_path, model, batch_size=64, num_workers=4):
    """
    Tạo DataLoader cho tập test
    """
    # Lấy data config từ model
    data_config = resolve_data_config({}, model=model)
    print(f"Data config: {data_config}")
    
    # Tạo dataset
    dataset = create_dataset(
        name='',
        root=data_path,
        split='validation', 
        download=False,
        class_map=''
    )
    
    # Tạo loader
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=num_workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=True,
    )
    
    return loader, dataset


def evaluate_model(model, data_loader, device='cuda'):
    """
    Đánh giá model trên tập test
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Lấy predictions
            _, preds = torch.max(outputs, 1)
            
            # Lưu predictions và targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Chuyển về numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Tính toán metrics
    accuracy = accuracy_score(all_targets, all_preds)
    
    print(f"\nTest Accuracy: {accuracy * 100:.4f}%")
    
    results = {
        'accuracy': accuracy,
        'num_samples': len(all_targets),
    }
    
    return results, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          save_path='confusion_matrix.png',
                          figsize=(12, 10), 
                          normalize=False,
                          top_k_classes=None,
                          select_smallest=False):
    """
    Vẽ confusion matrix với tùy chọn lọc top-k hoặc bottom-k classes.
    """
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_raw = cm.copy() 
    
    # Logic lọc classes
    if top_k_classes is not None and top_k_classes < len(cm):
        # Đếm số lượng mẫu thực tế của từng class (True Label Distribution)
        class_counts = np.sum(cm, axis=1) 
        
        # Sắp xếp index dựa trên số lượng (tăng dần: ít -> nhiều)
        sorted_indices = np.argsort(class_counts) 

        if select_smallest:
            # Lấy k class đầu tiên (ít nhất)
            target_indices = sorted_indices[:top_k_classes]
            filter_type = "Smallest"
        else:
            # Lấy k class cuối cùng (nhiều nhất)
            target_indices = sorted_indices[-top_k_classes:]
            filter_type = "Largest"
        
        print(f"Filtering {top_k_classes} classes with {filter_type} instances.")
        
        # Lọc confusion matrix
        # Cắt cả hàng và cột theo indices đã chọn
        cm = cm[target_indices][:, target_indices]
        cm_raw = cm_raw[target_indices][:, target_indices]
        
        if class_names is not None:
            class_names = [class_names[i] for i in target_indices]
    
    # Normalize theo hàng (mỗi hàng tổng = 1)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # Tránh chia cho 0
        cm_normalized = cm.astype('float') / row_sums
    else:
        cm_normalized = cm.astype('float')
    
    # Vẽ confusion matrix
    plt.figure(figsize=figsize)
    
    # Tạo annotations - chỉ hiển thị số lượng > 0
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm_raw[i, j]
            if count > 0:
                annotations[i, j] = f'{count}'
            else:
                annotations[i, j] = ''
    
    # Xác định tiêu đề
    title_suffix = ""
    if top_k_classes:
        title_suffix = f" ({'Bottom' if select_smallest else 'Top'} {top_k_classes} Classes)"

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
    
    plt.title(f"Confusion Matrix{' (Normalized)' if normalize else ''}{title_suffix}", 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if class_names:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate KAT model and plot confusion matrices')
    
    # Paths
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/dataset', 
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='kat_base_patch16_224',
                        help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, 
                        default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/kat/output/kat_focal_loss_weighted/20260108-231219-kat_base_patch16_224-224/model_best.pth.tar',
                        help='Path to checkpoint file')
    
    # Configs
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')
    
    # Plotting Options
    parser.add_argument('--top-k-classes', type=int, default=25,
                        help='Number of classes to plot (Default is mostly frequent ones)')
    parser.add_argument('--bottom-k-classes', type=int, default=None,
                        help='If set, also plot confusion matrix for K classes with FEWEST instances')
    
    args = parser.parse_args()
    
    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # 1. Load Model
    model = load_model(args.model, args.checkpoint)
    
    # 2. Create Data Loader
    test_loader, test_dataset = create_test_loader(
        args.data_path, 
        model, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 3. Evaluate Model
    results, all_preds, all_targets = evaluate_model(model, test_loader, device)
    
    # Lưu kết quả JSON
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nNumeric results saved to: {results_path}")
    
    # Tên các class (25 loại cây)
    class_names = [
        'Lim Xẹt', 'Lát Hoa', 'Lim Xanh', 'Thông Mã Vĩ', 'Keo Lá Tràm',
        'Sữa', 'Gội Trắng', 'Dẻ Cau', 'Re Hương', 'Đa',
        'Bằng Lăng', 'Phượng Vĩ', 'Vả', 'Sấu', 'Sếu',
        'Sồi Phảng', 'Lòng Mang', 'Bời Lời Lá Tròn', 'Thôi Ba', 'Gội',
        'Mé Cò Ke', 'Bời Lời', 'Sồi Xanh', 'Nanh Chuột', 'Chẹo'
    ]
    
    # --- VẼ BIỂU ĐỒ ---
    
    # 1. Vẽ Top K (Nhiều nhất - Mặc định)
    print("\n--- Generating Plots for Top Frequent Classes ---")
    plot_confusion_matrix(
        all_targets, all_preds, class_names=class_names,
        save_path=os.path.join(args.output_dir, 'cm_top_k.png'),
        normalize=False, top_k_classes=args.top_k_classes, select_smallest=False
    )
    plot_confusion_matrix(
        all_targets, all_preds, class_names=class_names,
        save_path=os.path.join(args.output_dir, 'cm_top_k_norm.png'),
        normalize=True, top_k_classes=args.top_k_classes, select_smallest=False
    )
    
    # 2. Vẽ Bottom K (Ít nhất - Nếu được yêu cầu)
    if args.bottom_k_classes is not None:
        print(f"\n--- Generating Plots for Bottom {args.bottom_k_classes} Least Frequent Classes ---")
        plot_confusion_matrix(
            all_targets, all_preds, class_names=class_names,
            save_path=os.path.join(args.output_dir, 'cm_bottom_k.png'),
            normalize=False, top_k_classes=args.bottom_k_classes, select_smallest=True
        )
        plot_confusion_matrix(
            all_targets, all_preds, class_names=class_names,
            save_path=os.path.join(args.output_dir, 'cm_bottom_k_norm.png'),
            normalize=True, top_k_classes=args.bottom_k_classes, select_smallest=True
        )

    # --- IN REPORT ---
    
    # Report cho Top K
    if args.top_k_classes:
        print(f"\n=== Classification Report (Top {args.top_k_classes} Frequent) ===")
        class_counts = np.bincount(all_targets)
        # Lấy index của các class có số lượng nhiều nhất
        if args.top_k_classes >= len(class_counts):
             top_indices = np.arange(len(class_counts)) # Lấy tất cả
        else:
             top_indices = np.argsort(class_counts)[-args.top_k_classes:]
        
        mask = np.isin(all_targets, top_indices)
        if np.sum(mask) > 0:
            print(classification_report(all_targets[mask], all_preds[mask], digits=4, zero_division=0))
        else:
            print("No samples found for these classes.")

    # Report cho Bottom K
    if args.bottom_k_classes:
        print(f"\n=== Classification Report (Bottom {args.bottom_k_classes} Rare) ===")
        class_counts = np.bincount(all_targets)
        # Lấy index của các class có số lượng ít nhất
        bottom_indices = np.argsort(class_counts)[:args.bottom_k_classes]
        
        mask = np.isin(all_targets, bottom_indices)
        if np.sum(mask) > 0:
            print(classification_report(all_targets[mask], all_preds[mask], digits=4, zero_division=0))
        else:
            print("No samples found for these classes.")

    print("\nEvaluation Script Completed!")

if __name__ == '__main__':
    main()