## Heterogeneity distributed learning ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import savemat

# 基本參數設定

cols = 5000
p_sensing = 0.02
p_masked = 0.1
N = 5000
K = 25 # sparsity level
num_trials = 10 # 更新測試次數
num_devices = 10 # devices數量
relative_errors = [] #計算每次的s_global跟真實s差距
eta = 0.61
total_measurements = 10000
## ADMM參數---------------------------------
lb = 1 
ub = 0 
rho1 = 1e-7
rho2 = 1e-7
max_iters = 5000
tol = 1e-13
## -----------------------------------------

overall_success = 0 # 每個裝置都估對才計1

def prox_vectorized_real_updated(g, b, rho1):
    return np.where(g != 0, (b + rho1 * np.abs(g)) / (1 + rho1) * np.sign(g), b / (1 + rho1))

def admm(A, b, lb, ub, rho1, rho2, max_iters, tol):
    # Initialization
    m, n = A.shape
    # z = np.random.randn(m)  # Initializing z as a real vector
    # z = z.reshape(-1, 1)
    z = np.copy(b)
    z = z.reshape(-1, 1)
    d = np.zeros(m)
    d = d.reshape(-1, 1)
    y = np.zeros(n)
    y = y.reshape(-1, 1)
    w = np.zeros(n)
    w = w.reshape(-1, 1)
    A_inv = A.T  
    A_star_A_plus_rho2_I_inv = np.linalg.inv(rho1 * A_inv @ A + rho2 * np.eye(n))
    x_conv = []
    # Main loop
    for k in range(max_iters):
        # Update x
        x = A_star_A_plus_rho2_I_inv @ (rho1 * A_inv @ z + A_inv @ d + rho2 * y - w)
        if k > 0:
            rel_error = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
            x_conv.append(rel_error)
            if rel_error < tol:
                break
        x_old = np.copy(x)
        # Update y with clipping to the box [lb, ub]
        y = np.minimum(np.maximum(x + w / rho2, ub), lb)
        # print(y)
        # Update z using the updated vectorized proximal operator
        Ax_minus_d = A @ x - d / rho1
        z = prox_vectorized_real_updated(Ax_minus_d, b, rho1)
        # Update dual variables d and w
        d = d + rho1 * (z - A @ x)
        w = w + rho2 * (x - y)
    return x, k + 1, x_conv

# 藉由closed-form求解nonzero entries
def compute_s_hat(m1, m2, b, y, Φ, n):
    Φ_m1_n = Φ[m1][n]
    Φ_m2_n = Φ[m2][n]
    b_m1 = b[m1]
    b_m2 = b[m2]
    y_m1 = y[m1]
    y_m2 = y[m2]
    numerator =  (y_m1/Φ_m1_n**2) - (y_m2/Φ_m2_n**2) - (b_m1**2/Φ_m1_n**2) + (b_m2**2/Φ_m2_n**2)
    denominator = 2 * (b_m1/Φ_m1_n - b_m2/Φ_m2_n) 
    s_n = numerator / denominator
    
    return s_n

# 生成 masked matrix
def generate_masked_matrix(N, p):
    num_zeros = int(N * p)
    diag_elements = np.ones(N)
    zero_indices = np.random.choice(N, num_zeros, replace=False)
    diag_elements[zero_indices] = 0
    return np.diag(diag_elements)

# FSRA
def FSRA(y, Φ, b):
    N = Φ.shape[1]
    sum_values = []
    cnt_values = []
    for n in range(N):
        Cn = np.nonzero(Φ[:, n])[0]
        sum_value = np.sum([1 for k in Cn if y[k] != abs(b[k])**2])
        sum_values.append(sum_value)
        cnt_values.append(len(Cn))
    return sum_values, cnt_values

# 生成信號
def generate_signal(N, K):
    s = np.zeros(N)
    indices = np.random.choice(N, K, replace=False)
    s[indices] = np.random.randn(K)
    return s.reshape(-1, 1)

# 比較 T_hat_global 和 s 的非零位置
def compare_positions(signal, support):
    actual_positions = np.where(signal != 0)[0]
    correctly_identified = set(actual_positions) & set(support)
    false_identified = set(support) - set(actual_positions)
    return len(correctly_identified), len(actual_positions), len(false_identified)

# 模擬多個裝置
for n in range(num_trials):
    s = generate_signal(N, K)
    sum_value_normalized_list = []
    global_sum_values = np.zeros(N)
    global_cnt_values = np.zeros(N)
    sensing_matrices = []
    biases = []
    ys = []
    all_s_hats = []
    masked_matrixs = []
    s = generate_signal(N, K)
    global_sum_values = np.zeros(N)
    global_cnt_values = np.zeros(N)
    sum_value_normalized_list = []

    # 每個裝置基本200個測量值
    min_measurements_per_device = 200

    measurements_per_device = np.full(num_devices, min_measurements_per_device)

    # 計算剩餘可分配的測量次數
    remaining_measurements = total_measurements - np.sum(measurements_per_device)

    # 使用正態分布隨機生成額外測量次數
    # 注意：這裡的參數可以根據需要調整，以達到想要的不均勻分配效果
    additional_measurements = np.abs(np.random.normal(0, 50, num_devices))
    additional_measurements = additional_measurements / additional_measurements.sum() * remaining_measurements
    additional_measurements = np.round(additional_measurements).astype(int)

    # 確保額外測量值總和不超過剩餘可分配的測量次數
    while np.sum(additional_measurements) > remaining_measurements:
        additional_measurements[np.argmax(additional_measurements)] -= 1

    # 結合基本測量值和額外測量值
    measurements_per_device += additional_measurements

    print(measurements_per_device)


    for device, num_measurements in enumerate(measurements_per_device):
        masked_matrix = generate_masked_matrix(N, p_masked)
        local_bias = 2 * np.random.randn(num_measurements, 1)
        mask = np.random.choice([0, 1], size=(num_measurements, cols), p=[1-p_sensing, p_sensing])
        gaussian_values = np.random.normal(0, 1, size=(num_measurements, cols))
        local_sensing_matrix = mask * gaussian_values
        local_y = abs(local_sensing_matrix @ masked_matrix @ s + local_bias)**2
        local_sum_values, local_cnt_values = FSRA(local_y, local_sensing_matrix, local_bias)
        global_sum_values += np.array(local_sum_values)
        global_cnt_values += np.array(local_cnt_values)
        sensing_matrices.append(local_sensing_matrix)
        biases.append(local_bias)
        ys.append(local_y)
        masked_matrixs.append(masked_matrix)
    T_hat_global = []
    for n in range(N):
        if global_cnt_values[n] > 0:
            sum_value_normalized = global_sum_values[n] / global_cnt_values[n]
            sum_value_normalized_list.append(sum_value_normalized)
            if sum_value_normalized > eta:
                T_hat_global.append(n)
                
    for device in range(num_devices):
        local_sensing_matrix = sensing_matrices[device]
        local_bias = biases[device]
        local_y = ys[device]
        
        s_hat = np.zeros(N)
    
        for n in T_hat_global:
            Cn = np.nonzero(local_sensing_matrix[:, n])[0]
            Cn_diff = list(set(Cn) - set().union(*[np.nonzero(local_sensing_matrix[:, j])[0] for j in T_hat_global if j != n]))
            
            if len(Cn_diff) >= 2:
                m1, m2 = np.random.choice(Cn_diff, 2, replace=False)
                s_hat_current = compute_s_hat(m1, m2, local_bias, local_y, local_sensing_matrix, n)
                s_hat[n] = s_hat_current
            else:
                s_hat[n] = 0
            
        all_s_hats.append(s_hat)
    
    non_zero_counts = []
    s_global = np.zeros(N)
    non_zero_device_counts = np.zeros(N)

    # 對於每個裝置的 s_hat
    for s_hat in all_s_hats:
        # 對於 T_hat_global 中的每個索引
        for n in T_hat_global:
            # 如果當前裝置在索引位置的值不為0
            if abs(s_hat[n]) > 1e-5 :
                # 累加到 s_global
                s_global[n] += s_hat[n]
                # 增加計數器
                non_zero_device_counts[n] += 1

    # 計算平均值，只有當非零裝置數大於0時才進行
    for n in T_hat_global:
        if non_zero_device_counts[n] > 0:
            s_global[n] = s_global[n] / non_zero_device_counts[n]
    
    s_global = s_global.reshape(-1, 1)
    relative_error = np.linalg.norm(s_global - s) / np.linalg.norm(s)
    relative_errors.append(relative_error)
    
    device_success = 0
    s_local_list = []
    signal_relative_errors = []
    for device in range(num_devices):
        local_sensing_matrix = sensing_matrices[device]
        local_bias = biases[device]
        local_sensing_matrix_reduced = local_sensing_matrix[:, T_hat_global]
        local_masked_matrix = masked_matrixs[device]
        s_global_value = s_global[T_hat_global, :].reshape(-1, 1)
        s_global_value_bias = np.append(s_global_value, 1)
        s_global_bias = s_global_value_bias.reshape(-1, 1)
        s_global_diagonal = np.diag(s_global_value_bias)
        masked_matrix_diagonal = np.diag(local_masked_matrix)
        masked_matrix_hat = masked_matrix_diagonal[T_hat_global].reshape(-1, 1)
        masked_matrix_bias = np.append(masked_matrix_hat, 1).reshape(-1, 1)
        combine_sensing_bias = np.hstack([local_sensing_matrix_reduced, local_bias])
        combine_sensing_sglobal = combine_sensing_bias @ s_global_diagonal
        
        column_sums = np.sum(np.abs(combine_sensing_sglobal), axis=0)
        zero_columns = column_sums == 0
        combine_sensing_sglobal_adjusted = combine_sensing_sglobal[:, ~zero_columns]
        masked_matrix_bias_adjusted = masked_matrix_bias[~zero_columns]
        
        
        y_t = abs(combine_sensing_sglobal_adjusted @ masked_matrix_bias_adjusted)**2
        y_t_sqrt = np.sqrt(y_t)
        y_t_sqrt = y_t_sqrt.reshape(-1, 1)
        masked_matrix_ADMM, num_iters_real_updated, x_conv= admm(combine_sensing_sglobal_adjusted, y_t_sqrt, lb, ub , rho1, rho2, max_iters, tol)
        masked_matrix_ADMM_round = np.round(masked_matrix_ADMM)
        if np.array_equal(masked_matrix_ADMM_round, masked_matrix_bias_adjusted):
            device_success += 1 
            s_local = s_global_diagonal @ masked_matrix_ADMM_round
            for i in range(len(s_local)):
                if s_local[i] != 0:  # 只處理非零元素
                    error = np.linalg.norm(s_local[i] - s_global_bias[i]) / np.linalg.norm(s_global_bias[i])
                    signal_relative_errors.append(error)
                
        else:
            print(f'false device is:{device}, iteration time is :{num_iters_real_updated}')
            plt.semilogy(x_conv, label='Relative Error of x')
            break
        
    if device_success == 10:
        overall_success += 1
    print(f'總共{device_success}個device成功恢復')

print(f'總共{num_trials}次，成功全部還原{overall_success}次')
avg_signal_relative_error = np.mean(signal_relative_errors)
print(f'Avearge Signal Relative Error is {avg_signal_relative_error}')
avg_relative_error = np.mean(relative_errors)
print(f'Average Relative Error over {num_trials} trials: {avg_relative_error} for K={K}, I={num_devices}')
    

