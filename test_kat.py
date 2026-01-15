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
# Đảm bảo file katransformer.py nằm cùng thư mục hoặc trong PYTHONPATH
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
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Xử lý các định dạng checkpoint khác nhau
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Checkpoint loaded successfully! ({msg})")
    
    return model

def create_test_loader(data_path, model, batch_size=64, num_workers=4):
    """
    Tạo DataLoader chuẩn theo config của model
    """
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
        crop_pct=data_config['crop_pct'],
        pin_memory=True,
    )
    
    return loader, dataset

def evaluate_model(model, data_loader, device='cuda'):
    """
    Chạy inference và trả về kết quả
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
            
            outputs = model(images)
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
    }
    
    return results, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          save_path='confusion_matrix.png',
                          figsize=(12, 10), 
                          normalize=False,
                          top_k_classes=None,
                          select_smallest=False):
    """
    Vẽ Confusion Matrix.
    
    Logic quan trọng:
    - Luôn tính toán dựa trên MA TRẬN ĐẦY ĐỦ (Full Matrix) trước.
    - Nếu normalize=True: Chia cho tổng hàng của Full Matrix (Global Normalization).
    - Sau đó mới cắt (slice) lấy k dòng/cột cần vẽ.
    
    Điều này đảm bảo khi vẽ Bottom-K, nếu mẫu bị nhận diện nhầm sang các lớp lớn (không được vẽ),
    tổng hàng sẽ < 1, phản ánh đúng hiệu suất thấp.
    """
    # 1. Tính Full Confusion Matrix
    cm_full = confusion_matrix(y_true, y_pred)
    
    # 2. Xử lý Normalize (Global)
    if normalize:
        # Tổng số mẫu thực tế của mỗi class
        row_sums = cm_full.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # Tránh chia cho 0
        
        # Ma trận tỷ lệ (dựa trên toàn bộ dữ liệu)
        cm_to_plot = cm_full.astype('float') / row_sums
    else:
        cm_to_plot = cm_full.astype('float')

    # Ma trận số lượng gốc (để hiển thị con số annotation)
    cm_raw_subset = cm_full.copy()

    # 3. Lọc Classes (Slicing)
    if top_k_classes is not None and top_k_classes < len(cm_full):
        # Đếm số lượng dựa trên Full Matrix để sort
        class_counts = np.sum(cm_full, axis=1)
        sorted_indices = np.argsort(class_counts)

        if select_smallest:
            # Lấy k class ít nhất
            target_indices = sorted_indices[:top_k_classes]
            filter_type = "Smallest"
        else:
            # Lấy k class nhiều nhất
            target_indices = sorted_indices[-top_k_classes:]
            filter_type = "Largest"

        print(f"Plotting logic: Filtering {top_k_classes} classes ({filter_type}).")
        
        # Cắt ma trận theo indices
        cm_to_plot = cm_to_plot[target_indices][:, target_indices]
        cm_raw_subset = cm_raw_subset[target_indices][:, target_indices]
        
        if class_names is not None:
            class_names = [class_names[i] for i in target_indices]

    # 4. Vẽ biểu đồ
    plt.figure(figsize=figsize)
    
    # Tạo annotations (chỉ hiện số lượng nguyên gốc)
    annotations = np.empty_like(cm_to_plot, dtype=object)
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            count = int(cm_raw_subset[i, j])
            if count > 0:
                annotations[i, j] = f'{count}'
            else:
                annotations[i, j] = ''
    
    # Tiêu đề biểu đồ
    title_suffix = ""
    if top_k_classes:
        title_suffix = f" ({'Bottom' if select_smallest else 'Top'} {top_k_classes} Classes)"
        if normalize and select_smallest:
            # Note cho người xem hiểu tại sao tổng hàng < 1
            title_suffix += "\n(Normalized by Global count: missing % indicates misclassification into excluded classes)"

    sns.heatmap(cm_to_plot, 
                annot=annotations if cm_to_plot.shape[0] <= 30 else False,
                fmt='',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                # Cố định thang màu từ 0-1 nếu normalize để dễ so sánh
                vmin=0.0 if normalize else None,
                vmax=1.0 if normalize else None,
                cbar_kws={'label': 'Proportion (Recall)' if normalize else 'Count'},
                square=True,
                linewidths=0.5,
                linecolor='gray')
    
    plt.title(f"Confusion Matrix{' (Normalized)' if normalize else ''}{title_suffix}", 
              fontsize=12, fontweight='bold')
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
    parser = argparse.ArgumentParser(description='Evaluate KAT model and plot Global Normalized Confusion Matrices')
    
    # Arguments cấu hình
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/dataset', 
                        help='Path to dataset root')
    parser.add_argument('--model', type=str, default='kat_base_patch16_224',
                        help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, 
                        default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/kat/output/kat_focal_loss_weighted/20260108-231219-kat_base_patch16_224-224/model_best.pth.tar',
                        help='Path to checkpoint file')
    
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='./results')
    
    # Arguments vẽ biểu đồ
    parser.add_argument('--top-k-classes', type=int, default=25,
                        help='Number of MOST frequent classes to plot')
    parser.add_argument('--bottom-k-classes', type=int, default=None,
                        help='Number of LEAST frequent classes to plot (Optional)')
    
    args = parser.parse_args()
    
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Thiết lập device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # 1. Load Model & Data
    model = load_model(args.model, args.checkpoint)
    test_loader, _ = create_test_loader(args.data_path, model, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # 2. Evaluate
    results, all_preds, all_targets = evaluate_model(model, test_loader, device)
    
    # Lưu kết quả JSON
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Danh sách tên class (Hardcoded theo yêu cầu)
    class_names = [
        'Lim Xẹt', 'Lát Hoa', 'Lim Xanh', 'Thông Mã Vĩ', 'Keo Lá Tràm',
        'Sữa', 'Gội Trắng', 'Dẻ Cau', 'Re Hương', 'Đa',
        'Bằng Lăng', 'Phượng Vĩ', 'Vả', 'Sấu', 'Sếu',
        'Sồi Phảng', 'Lòng Mang', 'Bời Lời Lá Tròn', 'Thôi Ba', 'Gội',
        'Mé Cò Ke', 'Bời Lời', 'Sồi Xanh', 'Nanh Chuột', 'Chẹo'
    ]
    
    print("\n--- Generating Visualization ---")
    
    # === TRƯỜNG HỢP 1: TOP K (Mặc định hoặc số lượng lớn nhất) ===
    print(f"\n[1/2] Processing Top {args.top_k_classes} classes...")
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
    
    # === TRƯỜNG HỢP 2: BOTTOM K (Các lớp dữ liệu ít nhất) ===
    if args.bottom_k_classes is not None:
        print(f"\n[2/2] Processing Bottom {args.bottom_k_classes} classes...")
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

    # === IN BÁO CÁO TEXT ===
    # Report cho Bottom K (thường quan trọng để soi lỗi)
    if args.bottom_k_classes:
        print(f"\n=== Classification Report (Bottom {args.bottom_k_classes} Rare Classes) ===")
        class_counts = np.bincount(all_targets)
        # Lấy index các class ít nhất
        bottom_indices = np.argsort(class_counts)[:args.bottom_k_classes]
        
        mask = np.isin(all_targets, bottom_indices)
        if np.sum(mask) > 0:
            print(classification_report(all_targets[mask], all_preds[mask], digits=4, zero_division=0))
        else:
            print("No samples found for these classes.")

    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()