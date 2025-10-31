import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import torch as th

TEN = th.Tensor


class Evaluator:
    def __init__(self, out_dir: str = './output'):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Training metrics
        self.train_loss_list: List[float] = []
        self.valid_loss_list: List[float] = []
        
        # Current batch metrics
        self.obj_train_list: List[float] = []
        self.obj_valid_list: List[float] = []
        
        # Best validation loss tracking
        self.best_valid_loss = float('inf')
        self.patience = 0
        
    def update_obj_train(self, obj: Optional[TEN]):
        """Update training objective values"""
        if obj is not None:
            if isinstance(obj, th.Tensor):
                self.obj_train_list.extend(obj.detach().cpu().numpy().flatten())
            else:
                self.obj_train_list.append(obj)
        else:
            # End of batch, calculate average
            if self.obj_train_list:
                avg_loss = np.mean(self.obj_train_list)
                self.train_loss_list.append(avg_loss)
                self.obj_train_list = []
    
    def update_obj_valid(self, obj: Optional[TEN]):
        """Update validation objective values"""
        if obj is not None:
            if isinstance(obj, th.Tensor):
                self.obj_valid_list.extend(obj.detach().cpu().numpy().flatten())
            else:
                self.obj_valid_list.append(obj)
        else:
            # End of batch, calculate average
            if self.obj_valid_list:
                avg_loss = np.mean(self.obj_valid_list)
                self.valid_loss_list.append(avg_loss)
                
                # Update best validation loss and patience
                if avg_loss < self.best_valid_loss:
                    self.best_valid_loss = avg_loss
                    self.patience = 0
                else:
                    self.patience += 1
                
                self.obj_valid_list = []
    
    def log_print(self, step_idx: int):
        """Print current training status"""
        train_loss = self.train_loss_list[-1] if self.train_loss_list else 0.0
        valid_loss = self.valid_loss_list[-1] if self.valid_loss_list else 0.0
        
        print(f"| Step {step_idx:6d} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | "
              f"Best Valid: {self.best_valid_loss:.6f} | Patience: {self.patience}")
    
    def draw_train_valid_loss_curve(self, gpu_id: int):
        """Draw and save training/validation loss curves"""
        if len(self.train_loss_list) < 2 or len(self.valid_loss_list) < 2:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_list, label='Train Loss', color='blue', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.valid_loss_list, label='Valid Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.out_dir, f'loss_curves_gpu_{gpu_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"| Loss curves saved to {plot_path}")


class Validator:
    def __init__(self, out_dir: str = './output', if_report: bool = True):
        self.out_dir = out_dir
        self.if_report = if_report
        os.makedirs(out_dir, exist_ok=True)
        
        # Accuracy metrics
        self.accuracy_list: List[float] = []
        self.tpr_list: List[float] = []  # True Positive Rate
        self.fpr_list: List[float] = []  # False Positive Rate
        
        # Current batch predictions and labels
        self.pred_list: List[np.ndarray] = []
        self.label_list: List[np.ndarray] = []
    
    def reset_list(self):
        """Reset current batch lists"""
        self.pred_list = []
        self.label_list = []
    
    def record_accuracy_tpr_fpr(self, out: TEN, lab: TEN):
        """Record predictions and labels for accuracy calculation"""
        if isinstance(out, th.Tensor):
            out_np = out.detach().cpu().numpy()
        else:
            out_np = out
            
        if isinstance(lab, th.Tensor):
            lab_np = lab.detach().cpu().numpy()
        else:
            lab_np = lab
            
        self.pred_list.append(out_np)
        self.label_list.append(lab_np)
    
    def calculate_metrics(self):
        """Calculate accuracy, TPR, FPR from recorded predictions and labels"""
        if not self.pred_list or not self.label_list:
            return
            
        # Concatenate all predictions and labels
        all_preds = np.concatenate(self.pred_list, axis=0)
        all_labels = np.concatenate(self.label_list, axis=0)
        
        # Convert to binary predictions (assuming regression to classification)
        pred_binary = (all_preds > 0.5).astype(int)
        label_binary = (all_labels > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(pred_binary == label_binary)
        self.accuracy_list.append(accuracy)
        
        # Calculate TPR and FPR
        tp = np.sum((pred_binary == 1) & (label_binary == 1))
        fp = np.sum((pred_binary == 1) & (label_binary == 0))
        fn = np.sum((pred_binary == 0) & (label_binary == 1))
        tn = np.sum((pred_binary == 0) & (label_binary == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        self.tpr_list.append(tpr)
        self.fpr_list.append(fpr)
    
    def draw_roc_curve_and_accuracy_curve(self, gpu_id: int, step_idx: int):
        """Draw ROC curve and accuracy curve"""
        if len(self.accuracy_list) < 2:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy curve
        plt.subplot(1, 2, 1)
        plt.plot(self.accuracy_list, label='Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot ROC curve
        plt.subplot(1, 2, 2)
        if len(self.tpr_list) > 1 and len(self.fpr_list) > 1:
            plt.plot(self.fpr_list, self.tpr_list, label='ROC Curve', color='red')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.out_dir, f'roc_accuracy_curves_gpu_{gpu_id}_step_{step_idx}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"| ROC and accuracy curves saved to {plot_path}")
    
    def validate_save(self, result_path: str):
        """Save validation results to CSV"""
        if not self.accuracy_list:
            return
            
        results = {
            'accuracy': self.accuracy_list,
            'tpr': self.tpr_list,
            'fpr': self.fpr_list
        }
        
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(result_path, index=False)
        print(f"| Validation results saved to {result_path}")