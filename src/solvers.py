import numpy as np
from .operators import compute_adaptive_g, forward_gradient, backward_divergence

class RicianSolver:
    def __init__(self, alpha, beta, r=1.0, sigma=25, max_iter=500, tol=1e-4):
        self.alpha = alpha # 正则化参数
        self.beta = beta # 数据保真项权重
        self.r = r # ADMM 惩罚参数
        self.sigma = sigma # 噪声标准差
        self.max_iter = max_iter # 最大迭代次数
        self.tol = tol # 收敛容忍度
    
    def solve(self, f):
        # 1. 初始化 
        # u^0 = g, n1^0 = 0, n2^0 ~ N(0, sigma)
        g = compute_adaptive_g(f, self.sigma) 
        u = g.copy()
        n1 = np.zeros_like(f)
        n2 = np.random.normal(0, self.sigma, f.shape)
        
        for k in range(self.max_iter):
            u_old = u.copy()
            
            # --- 步骤 1: v-subproblem (球面投影)  ---
            # 计算中间变量 v_hat (公式 4.11 相关的投影前状态) 
            # v_hat = (Qu + (1 - alpha/r)n)
            # v = f * (v_hat / |v_hat|) 
            v1, v2 = self._update_v(u, n1, n2, f)

            # --- 步骤 2: u-subproblem (ROF模型)  ---
            # 这是一个标准的 TV-L2 问题，使用 Chambolle's 投影法
            # 目标：min TV(u) + (beta_hat/2) * ||u - u_hat||^2 
            u = self._update_u(v1, n1, g)

            # --- 步骤 3: n-subproblem (噪声更新) [cite: 289] ---
            # 使用公式 (4.10) 直接更新 
            n1, n2 = self._update_n(u, v1, v2, n1, n2)

            # 检查收敛性 
            rel_err = np.linalg.norm(u - u_old) / np.linalg.norm(u_old)
            if rel_err < self.tol:
                break
                
        return u

    def _update_v(self, u, n1, n2, f):
        """ 实现公式 4.11 : 将变量投影到球面K上 """
        # 计算中间变量 v_hat (对应公式 4.23)
        # v1_hat = u + (1 - alpha/r) * n1
        # v2_hat = (1 - alpha/r) * n2
        coeff = 1.0 - (self.alpha / self.r)
        v1_hat = u + coeff * n1
        v2_hat = coeff * n2
        
        # 计算 v_hat 的模长 (Euclidean norm)
        v_norm = np.sqrt(v1_hat**2 + v2_hat**2)
        
        # 闭式解投影: v = f * (v_hat / |v_hat|)
        # 加 1e-8 防止除零错误
        v1 = f * (v1_hat / np.maximum(v_norm, 1e-8))
        v2 = f * (v2_hat / np.maximum(v_norm, 1e-8))
    
        return v1, v2

    def _update_u(self, v1, n1, g):
        """
        ROF 模型的 u-subproblem 求解，使用 Chambolle's 投影法 
        """
        # 计算参数
        beta_hat = self.beta + self.r
        u_hat = (self.beta * g + self.r * (v1 - n1)) / beta_hat
        
        # 初始化
        p_x = np.zeros_like(u_hat)
        p_y = np.zeros_like(u_hat)
        tau = 0.1 # 步长(0, 1/8)
        T = 2 # 内部迭代次数

        # 迭代更新 p_x, p_y
        for _ in range(T):
            div_p = backward_divergence(p_x, p_y)
            
            # 计算 grad(div(p) - beta_hat * u_hat)
            term_to_grad = div_p - beta_hat * u_hat
            grad_x, grad_y = forward_gradient(term_to_grad)
            
            norm = np.sqrt(grad_x**2 + grad_y**2)
            
            # 因为原模型中 TV 前系数是 1，所以这里不需要除以 alpha
            # 投影半径固定为 1
    
            p_x = (p_x + tau * grad_x) / (1.0 + tau * norm)
            p_y = (p_y + tau * grad_y) / (1.0 + tau * norm)
            
        return u_hat - (1.0 / beta_hat) * backward_divergence(p_x, p_y)
        

    def _update_n(self, u, v1, v2, n1, n2):
        # 实现公式 4.10 
        n1 = (self.alpha * n1 + self.r * (v1 - u)) / (self.alpha + self.r)
        n2 = (self.alpha * n2 + self.r * v2) / (self.alpha + self.r)
        return n1, n2