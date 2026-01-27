import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 物理常数与参数设置
# ==========================================
MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854e-12
C_LIGHT = 1 / np.sqrt(MU_0 * EPS_0)

# LWD 典型参数
FREQ = 2e6              # 2 MHz
OMEGA = 2 * np.pi * FREQ
LAMBDA = C_LIGHT / FREQ # 空气中波长 ~150m

# 网格设置
# 注意：Python中通常 row-major (C-style)，为了方便物理理解，
# 我们定义 shape = (NZ, NY, NX)
NX, NY, NZ = 20, 20, 20 
DL = 0.5                # 空间步长 0.5m
NPML = 8                # PML 层数

direct_solver = False  #True  #

print(f"--- Simulation Setup ---")
print(f"Frequency: {FREQ/1e6} MHz")
print(f"Grid: {NX} x {NY} x {NZ} (Total Unknowns: {3*NX*NY*NZ})")
print(f"Grid Spacing: {DL} m")

# ==========================================
# 2. 构建 PML 拉伸因子 (Stretched Coordinate)
# ==========================================
def create_pml_s_factor(N, N_pml, omega, dl):
    """
    创建 1D PML 复数拉伸因子 s
    """
    s = np.ones(N, dtype=np.complex128)
    
    # 理论反射系数 R(0) 设为 1e-6 对应的 sigma_max
    # 经验公式：sigma_max = -(m+1)*ln(R)/ (2*eta*d)
    # 这里简化处理，取一个经验值确保吸收
    sigma_max = 4.0 / (MU_0 * C_LIGHT * dl)
    
    for i in range(N_pml):
        # 归一化距离 (0 到 1)
        dist = (N_pml - i) / N_pml 
        # 多项式分布
        sigma = sigma_max * (dist ** 3)
        kappa = 1 + 5 * (dist ** 3) # 实部拉伸，减缓波速
        
        val = kappa + sigma / (-1j * omega * EPS_0)
        
        # 左边界 (i) 和 右边界 (N-1-i)
        s[i] = val
        s[N - 1 - i] = val
        
    return s

# 生成三个方向的 s 向量
sx = create_pml_s_factor(NX, NPML, OMEGA, DL)
sy = create_pml_s_factor(NY, NPML, OMEGA, DL)
sz = create_pml_s_factor(NZ, NPML, OMEGA, DL)

# ==========================================
# 3. 矩阵组装 (Assembly)
# ==========================================
def build_curl_matrix(nx, ny, nz, dl, sx, sy, sz):
    """
    利用 Kronecker 积构建 3D Curl 算子
    """
    # 1D 导数算子 D (包含 1/s 因子)
    def build_derivative_op(n, s_factor):
        # 前向差分: (u[i+1] - u[i]) / (s[i] * dl)
        data = []
        rows = []
        cols = []
        for i in range(n):
            # 处理周期性边界（PML 会处理掉边界值，所以这里用循环索引没问题）
            coeff = 1.0 / (s_factor[i] * dl)
            
            # u[i+1]
            rows.append(i); cols.append((i + 1) % n); data.append(coeff)
            # -u[i]
            rows.append(i); cols.append(i); data.append(-coeff)
            
        return sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    Dx = build_derivative_op(nx, sx)
    Dy = build_derivative_op(ny, sy)
    Dz = build_derivative_op(nz, sz)
    
    Ix = sp.eye(nx)
    Iy = sp.eye(ny)
    Iz = sp.eye(nz)

    # 注意 scipy.kron 的顺序与 reshape((nz, ny, nx)) 的关系
    # 假设 flatten() 是 C-style (最后一维 x 变化最快)
    # Px 对应 d/dx，应该作用在最内层 (Inner-most)
    Px = sp.kron(Iz, sp.kron(Iy, Dx))
    Py = sp.kron(Iz, sp.kron(Dy, Ix))
    Pz = sp.kron(Dz, sp.kron(Iy, Ix))
    
    # 构造 Curl 矩阵块
    #      [ 0  -Pz  Py]
    #  C = [ Pz  0  -Px]
    #      [-Py  Px  0 ]
    
    Z = sp.csr_matrix((nx*ny*nz, nx*ny*nz))
    C = sp.bmat([
        [Z, -Pz, Py],
        [Pz, Z, -Px],
        [-Py, Px, Z]
    ])
    
    return C

print("Building System Matrix...")
t0 = time.time()
Curl = build_curl_matrix(NX, NY, NZ, DL, sx, sy, sz)

# 系统方程: (Curl^T * Curl - k0^2 * I) E = -i * w * J
# 这里假设背景是真空/均匀介质 (k0)
k0_sq = OMEGA**2 * MU_0 * EPS_0
I_total = sp.eye(3 * NX * NY * NZ)
A = Curl.T @ Curl - k0_sq * I_total

print(f"Matrix built in {time.time()-t0:.2f}s")

# ==========================================
# 4. 设置磁偶极子源 (RHS)
# ==========================================
b = np.zeros(3 * NX * NY * NZ, dtype=np.complex128)

# 网格中心索引
cx, cy, cz = NX // 2, NY // 2, NZ // 2
grid_size = NX * NY * NZ

def get_idx(i, j, k, component):
    # component: 0=Ex, 1=Ey, 2=Ez
    # 索引顺序: Component -> Z -> Y -> X
    return component * grid_size + k * (NX * NY) + j * NX + i

# 磁源强度 (模拟 Faraday 定律右端的 -curl(M))
# Mz 源会驱动其周围的 Ex 和 Ey 形成涡旋
src_val = 1.0  # 任意单位强度

# 类似于 Stokes 定理，在中心点周围一圈加电动势
#   ^ y
#   |
#   --> Ex(y+1)
#   ^          ^
# Ey(x)  M   Ey(x+1)
#   |          |
#   <-- Ex(y)
#
# 这里的坐标 i,j,k 对应 Yee 网格的 E 节点位置，需要仔细对齐
# 简单起见，我们在中心点注入一个旋转场
b[get_idx(cx, cy, cz, 0)] -= src_val / DL      # Ex 下边
b[get_idx(cx, cy+1, cz, 0)] += src_val / DL    # Ex 上边
b[get_idx(cx, cy, cz, 1)] += src_val / DL      # Ey 左边
b[get_idx(cx+1, cy, cz, 1)] -= src_val / DL    # Ey 右边

# ==========================================
# 5. 求解与可视化
# ==========================================

if direct_solver:
    print("Solving Linear System (Direct Solver)...")
    t0 = time.time()
    # 使用 SuperLU 直接求解器 (CPU)
    solve_op = scipy.sparse.linalg.splu(A.tocsc())
    x = solve_op.solve(b)
    print(f"Solved in {time.time()-t0:.2f}s")
    
else:
    # ==========================================
    # 使用迭代求解器 (BiCGSTAB)
    # ==========================================
    print("Solving Linear System (Iterative Solver - BiCGSTAB)...")
    t0 = time.time()

    # 使用 BiCGSTAB，它是处理非对称矩阵(由于PML存在)的标准迭代法
    # tol 是收敛残差容差
    x, exit_code = scipy.sparse.linalg.bicgstab(A, b, tol=1e-5, atol=1e-5)

    if exit_code == 0:
        print(f"Converged in {time.time()-t0:.2f}s")
    else:
        print(f"Convergence failed with code {exit_code}")
    
# 提取 Ex 分量并 reshape
Ex_flat = x[0:grid_size]
Ex = Ex_flat.reshape((NZ, NY, NX))

# 绘图
plt.figure(figsize=(10, 8))

# 取中心 Z 切面
mid_z = NZ // 2
field_slice = np.real(Ex[mid_z, :, :])

plt.imshow(field_slice, cmap='RdBu', origin='lower')
plt.colorbar(label='Ex Field (Real)')
plt.title(f'Ex Field Distribution @ 2MHz (z={mid_z})')
plt.xlabel('X grid index')
plt.ylabel('Y grid index')

# 标记 PML 区域
plt.axvline(NPML, color='k', linestyle='--', alpha=0.5)
plt.axvline(NX-NPML, color='k', linestyle='--', alpha=0.5)
plt.axhline(NPML, color='k', linestyle='--', alpha=0.5)
plt.axhline(NY-NPML, color='k', linestyle='--', alpha=0.5)
plt.text(NPML/2, NY/2, "PML", ha='center', color='black')

plt.show()