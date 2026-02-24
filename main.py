import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_brainweb_data, add_rician_noise, compute_metrics_foreground
from src.solvers import RicianSolver

def main():
    # --- 1. 参数设置 (参考论文实验) ---
    sigma = 25          # 噪声强度
    alpha = 0.01       # 噪声惩罚项权重
    beta = 0.045         # 自适应保真项权重
    r = 1.0             # ADMM 惩罚参数
    file_path = "data/t1_icbm_normal_1mm_pn0_rf0.raws"

    print("Step 1: 正在加载原始 BrainWeb 数据...")
    u_true = load_brainweb_data(file_path)

    print(f"Step 2: 正在模拟莱斯噪声 (sigma={sigma})...")
    f = add_rician_noise(u_true, sigma)

    # --- 2. 初始化求解器 ---
    solver = RicianSolver(alpha=alpha, beta=beta, r=r, sigma=sigma, max_iter=100)

    print("Step 3: 启动 Algorithm 4.1 (Simplified ADMM)...")
    u_denoised = solver.solve(f)

    # --- 结果评估 ---
    psnr_noisy, ssim_noisy = compute_metrics_foreground(u_true, f)
    psnr_denoised, ssim_denoised = compute_metrics_foreground(u_true, u_denoised)

    print(f"\n[前景区域评估 (Threshold > 5)]")
    print(f"含噪图像: PSNR = {psnr_noisy:.2f}dB, SSIM = {ssim_noisy:.4f}")
    print(f"去噪图像: PSNR = {psnr_denoised:.2f}dB, SSIM = {ssim_denoised:.4f}")
    print(f"指标提升: PSNR:{psnr_denoised - psnr_noisy:.2f}dB, SSIM:{ssim_denoised - ssim_noisy:.4f}")

    # --- 4. 可视化对比 ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(u_true, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(f, cmap='gray')
    plt.title(f"Noisy (sigma={sigma}), PSNR: {psnr_noisy:.2f}dB")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(u_denoised, cmap='gray')
    plt.title(f"Denoised (Our Implementation)\nPSNR: {psnr_denoised:.2f}dB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("result_comparison.png")
    print("结果图已保存至: result_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
    