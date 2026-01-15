import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
import katransformer  # Import để register KAT models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import argparse
import json


def load_model(model_name, checkpoint_path, num_classes=25):
    """
    Load KAT model từ checkpoint
    
    Args:
        model_name: Tên model (ví dụ: 'kat_tiny_patch16_224')
        checkpoint_path: Đường dẫn đến checkpoint file
        num_classes: Số lượng classes (mặc định 1000 cho ImageNet)
    
    Returns:
        model: PyTorch model đã load weights
    """
    print(f"Creating model: {model_name}")
    
    # Tạo model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Xử lý checkpoint (có thể có key 'state_dict' hoặc 'model')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict vào model
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully!")
    
    return model


def create_test_loader(data_path, model, batch_size=64, num_workers=4):
    """
    Tạo DataLoader cho tập test
    
    Args:
        data_path: Đường dẫn đến thư mục dataset (ImageNet)
        model: Model để lấy data config
        batch_size: Batch size
        num_workers: Số workers cho DataLoader
    
    Returns:
        loader: DataLoader cho tập test
        dataset: Dataset object
    """
    # Lấy data config từ model
    data_config = resolve_data_config({}, model=model)
    print(f"Data config: {data_config}")
    
    # Tạo dataset
    dataset = create_dataset(
        name='',
        root=data_path,
        split='validation',  # hoặc 'val'
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
    
    Args:
        model: PyTorch model
        data_loader: DataLoader cho tập test
        device: Device để chạy ('cuda' hoặc 'cpu')
    
    Returns:
        results: Dictionary chứa kết quả đánh giá
        all_preds: Tất cả predictions
        all_targets: Tất cả ground truth labels
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
                          top_k_classes=None):
    """
    Vẽ confusion matrix - hiển thì số lượng ở mỗi ô, chuẩn hóa theo hàng
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Tên các classes (optional)
        save_path: Đường dẫn để lưu hình
        figsize: Kích thước figure
        normalize: Có normalize confusion matrix không (theo hàng)
        top_k_classes: Chỉ vẽ top k classes có nhiều samples nhất
    """
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_raw = cm.copy()  # Giữ bản gốc để hiển thị số lượng
    
    # Nếu chỉ muốn vẽ top k classes
    if top_k_classes is not None:
        # Tìm top k classes có nhiều samples nhất
        class_counts = np.sum(cm, axis=1)
        top_k_indices = np.argsort(class_counts)[-top_k_classes:]
        
        # Lọc confusion matrix
        cm = cm[top_k_indices][:, top_k_indices]
        cm_raw = cm_raw[top_k_indices][:, top_k_indices]
        
        if class_names is not None:
            class_names = [class_names[i] for i in top_k_indices]
    
    # Normalize theo hàng (mỗi hàng tổng = 1)
    if normalize:
        # Tránh chia cho 0
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype('float') / row_sums
    else:
        cm_normalized = cm.astype('float')
    
    # Vẽ confusion matrix
    plt.figure(figsize=figsize)
    
    # Tạo annotations - chỉ hiển thị số lượng ở ô có kết quả
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm_raw[i, j]
            # Chỉ hiển thị số nếu count > 0
            if count > 0:
                annotations[i, j] = f'{count}'
            else:
                annotations[i, j] = ''
    
    # Sử dụng seaborn để vẽ heatmap đẹp hơn
    sns.heatmap(cm_normalized if normalize else cm, 
                annot=annotations if cm.shape[0] <= 30 else False,  # Chỉ hiện số nếu matrix không quá lớn
                fmt='',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                square=True,
                linewidths=0.5,
                linecolor='gray')
    
    plt.title('Confusion Matrix' + (' (Normalized by Row)' if normalize else ''), 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Xoay labels nếu là tên classes
    if class_names:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate KAT model and plot confusion matrix')
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/dataset', help='Path to ImageNet dataset')
    parser.add_argument('--model', type=str, default='kat_base_patch16_224',
                       help='Model name')
    parser.add_argument('--checkpoint', type=str, default='/mnt/disk2/home/vlir_hoang/Domain_Adaption/kat/output/kat_focal_loss_weighted/20260108-231219-kat_base_patch16_224-224/model_best.pth.tar',
                       help='Path to checkpoint file')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--top-k-classes', type=int, default=25,
                       help='Only plot top k classes in confusion matrix')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Tạo output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    model = load_model(args.model, args.checkpoint)
    
    # Tạo data loader
    test_loader, test_dataset = create_test_loader(
        args.data_path, 
        model, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate model
    results, all_preds, all_targets = evaluate_model(model, test_loader, device)
    
    # Lưu results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_path}")
    
    # Class names mapping
    class_names = [
        'Lim Xẹt', 'Lát Hoa', 'Lim Xanh', 'Thông Mã Vĩ', 'Keo Lá Tràm',
        'Sữa', 'Gội Trắng', 'Dẻ Cau', 'Re Hương', 'Đa',
        'Bằng Lăng', 'Phượng Vĩ', 'Vả', 'Sấu', 'Sếu',
        'Sồi Phảng', 'Lòng Mang', 'Bời Lời Lá Tròn', 'Thôi Ba', 'Gội',
        'Mé Cò Ke', 'Bời Lời', 'Sồi Xanh', 'Nanh Chuột', 'Chẹo'
    ]
    
    # Vẽ confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        all_targets, 
        all_preds,
        class_names=class_names,
        save_path=cm_path,
        normalize=False,
        top_k_classes=args.top_k_classes
    )
    
    # Vẽ normalized confusion matrix (chuẩn hóa theo hàng)
    cm_norm_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(
        all_targets, 
        all_preds,
        class_names=class_names,
        save_path=cm_norm_path,
        normalize=True,
        top_k_classes=args.top_k_classes
    )
    
    # In classification report cho top k classes
    if args.top_k_classes:
        print(f"\n=== Classification Report (Top {args.top_k_classes} classes) ===")
        class_counts = np.bincount(all_targets)
        top_k_indices = np.argsort(class_counts)[-args.top_k_classes:]
        
        # Lọc predictions và targets
        mask = np.isin(all_targets, top_k_indices)
        filtered_targets = all_targets[mask]
        filtered_preds = all_preds[mask]
        
        print(classification_report(filtered_targets, filtered_preds, digits=4))
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()