# 🖼️ CIFAR-10 Classification — Augmentation & Optimization Experiments

> A ResNet-18 based image classifier trained on CIFAR-10, exploring the impact of **data augmentation strategies** and **optimization techniques** on generalization performance.

---

## 📌 Overview

This project investigates two orthogonal paths to improving CIFAR-10 classification accuracy:

| Path                     | Strategies Compared                          |
| ------------------------ | -------------------------------------------- |
| 🔀 **Data Augmentation** | Baseline (Crop + Flip) vs. CutMix vs. Cutout |
| ⚙️ **Optimization**      | SGD + CosineAnnealing vs. Adam               |

Results, training curves, confusion matrices, and Grad-CAM visualizations are reported in the accompanying Colab notebook and IEEE-format report.

---

## 🏗️ Model Architecture

- **Backbone:** ResNet-18, adapted for CIFAR-10's 32×32 input
- **Modifications:**
  - First conv: `7×7 → 3×3`, stride 1, padding 1
  - MaxPool replaced with `nn.Identity()`
  - Final FC layer outputs **10 classes**
- **Parameters:** ~11 Million trainable

---

## 📂 Project Structure

```
cifar10-project/
│
├── CIFAR10_project_notebook.ipynb   # Main Colab notebook (run end-to-end)
├── IEEE_report.tex                  # IEEE-format LaTeX report
└── README.md
```

---

## 🚀 Getting Started

### 1. Open in Google Colab

Upload `CIFAR10_project_notebook.ipynb` to [Google Colab](https://colab.research.google.com/) and run cells **in order**. A GPU runtime is recommended.

### 2. Run an Experiment

```python
model, history, trainloader, testloader = run_experiment(
    augment='basic',       # 'basic' | 'cutout'
    optimizer_name='SGD',  # 'SGD'   | 'Adam'
    epochs=50,
    lr=0.1,
    use_cutmix=True        # Enable CutMix augmentation
)
```

### 3. Evaluate & Visualize

```python
# Accuracy + Confusion Matrix
acc, y_true, y_pred = evaluate(model, testloader, device)
per_acc, cm = per_class_accuracy(y_true, y_pred)
plot_confusion_matrix(cm, classes)

# Grad-CAM explainability
heatmap = generate_gradcam(model, input_tensor)
plt.imshow(heatmap, cmap='jet')
```

---

## 🧪 Experiments & Key Findings

### Experiment 1 — Data Augmentation

| Method        | Notes                                                                            |
| ------------- | -------------------------------------------------------------------------------- |
| Baseline      | Random crop + horizontal flip                                                    |
| **CutMix** ✅ | Mixes image patches between samples; improved val accuracy & reduced overfitting |
| Cutout        | Randomly masks square patches; acts as a dropout-like regularizer                |

### Experiment 2 — Optimization Strategy

| Optimizer                               | Behaviour                                        |
| --------------------------------------- | ------------------------------------------------ |
| Adam                                    | Faster convergence, slightly lower peak accuracy |
| **SGD + Momentum + CosineAnnealing** ✅ | Slower start, higher final validation accuracy   |

---

## 📊 Evaluation Suite

- ✅ Training & validation loss/accuracy curves
- ✅ Confusion matrix (10×10 heatmap)
- ✅ Per-class accuracy breakdown
- ✅ **Grad-CAM** visualizations for model interpretability

---

## 🛠️ Dependencies

```
torch
torchvision
numpy
matplotlib
scikit-learn
seaborn
```

Install all via:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

---

## 📖 References

1. H. Zhang et al., _"mixup: Beyond Empirical Risk Minimization"_, ICLR 2018
2. S. Yun et al., _"CutMix: Regularization Strategy to Train Strong Classifiers"_, ICCV 2019

---

## 👤 Author

**Must** — Software Engineering Student
