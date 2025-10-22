# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:50:53 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy import stats

# 设置页面
st.set_page_config(
    page_title="统计分析工具",
    page_icon="📊",
    layout="wide"
)

# 标题和说明
st.title("📊 统计分析工具")
st.markdown("""
提供多种稳健统计分析方法，用于处理包含异常值的数据集。
支持迭代稳健统计法、四分位稳健统计法和Q/Hampel法。
""")

# 侧边栏 - 参数设置和方法选择
st.sidebar.header("⚙️ 分析设置")

# 方法选择
method = st.sidebar.selectbox(
    "选择统计方法:",
    ["迭代稳健统计法", "四分位稳健统计法", "Q/Hampel法"],
    help="选择适合数据特征的稳健统计方法"
)

# 根据选择的方法显示相应参数
if method == "迭代稳健统计法":
    k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
    max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)
elif method == "四分位稳健统计法":
    st.sidebar.info("四分位法使用固定参数计算")
elif method == "Q/Hampel法":
    st.sidebar.info("Q/Hampel法使用标准参数计算")

# 数据输入方式选择
input_method = st.radio("数据输入方式:", 
                       ["手动输入", "文件上传", "示例数据"])

data = None

if input_method == "手动输入":
    st.subheader("📝 手动输入数据")
    data_input = st.text_area("请输入数据（每行一个数值或用逗号分隔）:", 
                             "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6")
    
    if st.button("分析数据"):
        try:
            # 解析输入数据
            if "\n" in data_input:
                data_list = [float(x.strip()) for x in data_input.split("\n") if x.strip()]
            else:
                data_list = [float(x.strip()) for x in data_input.split(",") if x.strip()]
            
            data = np.array(data_list)
            st.success(f"成功解析 {len(data)} 个数据点")
            
        except ValueError as e:
            st.error("数据格式错误！请确保输入的是数字")

elif input_method == "文件上传":
    st.subheader("📁 上传数据文件")
    uploaded_file = st.file_uploader("选择CSV或TXT文件", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # 假设第一列是数据
                data = df.iloc[:, 0].values
            else:
                # 文本文件，每行一个数字
                content = uploaded_file.read().decode()
                data_list = [float(x.strip()) for x in content.split() if x.strip()]
                data = np.array(data_list)
            
            st.success(f"成功加载 {len(data)} 个数据点")
            st.write("前10个数据:", data[:10])
            
        except Exception as e:
            st.error(f"文件读取错误: {e}")

else:  # 示例数据
    st.subheader("🎯 示例数据分析")
    example_data = np.array([
        54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 
        54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 
        54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 
        54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 
        54.5, 55.9, 53.2, 54.6
    ])
    data = example_data
    st.write("示例数据已加载，包含44个测量值")

# 方法描述
st.sidebar.header("📚 方法说明")
if method == "迭代稳健统计法":
    st.sidebar.info("""
    **迭代稳健统计法**（算法A）通过迭代过程逐步修正异常值影响，
    收敛后得到稳健的统计估计。
    """)
elif method == "四分位稳健统计法":
    st.sidebar.info("""
    **四分位稳健统计法**以数据排序为基础，使用数据集中段50%的数据，
    崩溃点为25%，具有易于计算、操作简单的特点。
    """)
else:  # Q/Hampel法
    st.sidebar.info("""
    **Q/Hampel法**结合Q方法计算的稳健标准差和Hampel方法计算的
    稳健平均值，具有较好的抗异常值干扰能力。
    """)

# 统计方法实现
def iterative_robust_algorithm(data, max_iterations=50, k=1.5):
    """
    迭代稳健统计法（原算法A）
    """
    n = len(data)
    
    # 初始值
    X_star = np.median(data)
    abs_deviations = np.abs(data - X_star)
    median_abs_deviation = np.median(abs_deviations)
    S_star = 1.483 * median_abs_deviation
    
    # 迭代过程
    converged = False
    iteration = 0
    history = []
    
    while iteration < max_iterations and not converged:
        iteration += 1
        prev_X_star = X_star
        prev_S_star = S_star
        
        # 计算δ并修正数据点
        delta = k * S_star
        Xj_star = np.where(data < X_star - delta, X_star - delta, 
                          np.where(data > X_star + delta, X_star + delta, data))
        
        # 重新计算
        X_star = np.mean(Xj_star)
        sum_squared_deviations = np.sum((Xj_star - X_star)**2)
        S_star = 1.134 * np.sqrt(sum_squared_deviations / (n-1))
        
        # 记录历史
        history.append({
            'iteration': iteration,
            'X_star': X_star,
            'S_star': S_star,
            'delta': delta
        })
        
        # 检查收敛
        if (int(prev_X_star * 1000) == int(X_star * 1000) and 
            int(prev_S_star * 1000) == int(S_star * 1000)):
            converged = True
    
    # 最终结果
    final_delta = k * S_star
    lower_limit = X_star - final_delta
    upper_limit = X_star + final_delta
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    Z_scores = (data - X_star) / S_star
    
    return {
        'robust_mean': X_star,
        'robust_std': S_star,
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'iterations': iteration,
        'converged': converged,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'history': history,
        'method_name': '迭代稳健统计法'
    }

def quartile_robust_algorithm(data):
    """
    四分位稳健统计法
    """
    # 数据排序
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # 计算中位值
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    # 计算四分位数
    q1 = np.percentile(data, 25)  # 下四分位数
    q3 = np.percentile(data, 75)  # 上四分位数
    
    # 计算四分位距和标准化四分位距
    iqr = q3 - q1
    niqr = 0.7413 * iqr  # 标准化四分位距
    
    # 计算正常值范围（基于四分位数）
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    
    # 识别离群值
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    
    # 计算Z比分数（使用中位值和NIQR）
    Z_scores = (data - median) / niqr
    
    return {
        'robust_mean': median,      # 使用中位值作为稳健平均值
        'robust_std': niqr,         # 使用NIQR作为稳健标准差
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'niqr': niqr,
        'method_name': '四分位稳健统计法',
        'lower_limit': lower_limit,
        'upper_limit': upper_limit
    }

def q_hampel_robust_algorithm(data):
    """
    Q/Hampel稳健统计方法
    """
    # 简化版的Q/Hampel实现
    # 注意：完整的Q/Hampel方法需要多个实验室数据，这里提供简化版本
    
    n = len(data)
    
    # 计算中位值（作为Hampel方法的初始估计）
    median = np.median(data)
    
    # 计算Q方法的稳健标准差（简化版）
    # 基于成对绝对差的中位数
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append(abs(data[i] - data[j]))
    
    if len(pairs) > 0:
        q_std = np.median(pairs) / 1.0484  # 调整系数
    else:
        q_std = np.std(data, ddof=1)
    
    # Hampel方法的稳健平均值（迭代加权法简化版）
    # 使用中位值作为初始估计
    current_mean = median
    max_iterations = 10
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        # 计算残差
        residuals = data - current_mean
        mad = np.median(np.abs(residuals))
        
        if mad == 0:
            break
            
        # 标准化残差
        standardized_residuals = residuals / (1.4826 * mad)
        
        # Hampel权重函数
        weights = np.ones_like(data)
        mask1 = np.abs(standardized_residuals) > 1.5
        mask2 = np.abs(standardized_residuals) > 3
        mask3 = np.abs(standardized_residuals) > 4.5
        
        weights[mask1] = 1.5 / np.abs(standardized_residuals[mask1])
        weights[mask2] = 0
        weights[mask3] = 0
        
        # 更新均值
        new_mean = np.sum(weights * data) / np.sum(weights)
        
        # 检查收敛
        if abs(new_mean - current_mean) < tolerance:
            break
            
        current_mean = new_mean
    
    # 计算正常值范围
    lower_limit = current_mean - 3 * q_std
    upper_limit = current_mean + 3 * q_std
    
    # 识别离群值
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    
    # 计算Z比分数
    Z_scores = (data - current_mean) / q_std
    
    return {
        'robust_mean': current_mean,  # Hampel稳健平均值
        'robust_std': q_std,          # Q方法稳健标准差
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'method_name': 'Q/Hampel法',
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'weights': weights if 'weights' in locals() else np.ones_like(data)
    }

# 执行分析
if data is not None and len(data) > 0:
    st.markdown("---")
    st.subheader(f"📈 {method}分析结果")
    
    with st.spinner(f"正在执行{method}分析..."):
        if method == "迭代稳健统计法":
            results = iterative_robust_algorithm(data, max_iterations=max_iter, k=k_value)
        elif method == "四分位稳健统计法":
            results = quartile_robust_algorithm(data)
        else:  # Q/Hampel法
            results = q_hampel_robust_algorithm(data)
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("稳健平均值", f"{results['robust_mean']:.6f}")
        st.metric("稳健标准差", f"{results['robust_std']:.6f}")
        
    with col2:
        if 'iterations' in results:
            st.metric("迭代次数", results['iterations'])
        st.metric("离群值数量", len(results['outliers']))
    
    # 方法特定结果显示
    if method == "四分位稳健统计法":
        st.info("📊 **四分位统计量:**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            st.metric("下四分位数(Q1)", f"{results['q1']:.6f}")
        with col4:
            st.metric("上四分位数(Q3)", f"{results['q3']:.6f}")
        with col5:
            st.metric("四分位距(IQR)", f"{results['iqr']:.6f}")
        with col6:
            st.metric("标准化四分位距(NIQR)", f"{results['niqr']:.6f}")
    
    # 详细结果
    st.subheader("📋 详细结果")
    
    st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
    if 'converged' in results:
        st.write(f"**收敛状态**: {'是' if results['converged'] else '否'}")
    
    if len(results['outliers']) > 0:
        # 将np.float64转换为Python原生float类型
        outliers_list = [float(x) for x in sorted(results['outliers'])]
        st.write(f"**离群值**: {outliers_list}")
    else:
        st.write("**离群值**: 无")
    
    # Z比分数统计
    z_scores_abs = np.abs(results['Z_scores'])
    satisfactory = np.sum(z_scores_abs <= 2)
    questionable = np.sum((z_scores_abs > 2) & (z_scores_abs <= 3))
    unsatisfactory = np.sum(z_scores_abs > 3)
    
    st.write("**Z比分数分类**:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("满意 (|Z| ≤ 2)", f"{satisfactory} 个")
    with col2:
        st.metric("可疑 (2 < |Z| ≤ 3)", f"{questionable} 个")
    with col3:
        st.metric("不满意 (|Z| > 3)", f"{unsatisfactory} 个")
    
    # 可视化 - 使用新的Z值柱状图
    st.subheader("📊 Data Visualization")

   # 创建数据框用于可视化
    df_clean = pd.DataFrame({
        'Original_Data': data,
        'Z_Score': results['Z_scores']  # 确保列名一致
})

    # 根据Z值进行分类
    def classify_data(row):
        if abs(row['Z_Score']) <= 2:
            return 'Satisfactory'
        elif 2 < abs(row['Z_Score']) <= 3:
            return 'Questionable'
        else:
            return 'Unsatisfactory'

    df_clean['Category'] = df_clean.apply(classify_data, axis=1)

    # 确保列名正确，然后按照Z值从大到小排序
    # 首先检查列名是否存在
    if 'Z_Score' in df_clean.columns:
        df_sorted = df_clean.sort_values('Z_Score', ascending=False)
    else:
    # 如果列名不是 'Z_Score'，尝试其他可能的列名
        st.error(f"列名 'Z_Score' 不存在。可用的列名: {list(df_clean.columns)}")
    # 使用第一个数值列进行排序
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_sorted = df_clean.sort_values(numeric_cols[0], ascending=False)
        else:
            df_sorted = df_clean

    # 创建Z值柱状图
    fig, ax = plt.subplots(figsize=(14, 10))

    # 设置类别对应的颜色
    color_map = {
        'Satisfactory': '#2E8B57',    # 绿色
        'Questionable': '#FFA500',    # 橙色
        'Unsatisfactory': '#DC143C'    # 红色
    }

    # 为每个类别创建柱状图
    for category, color in color_map.items():
        category_data = df_sorted[df_sorted['Category'] == category]
        if not category_data.empty:
        # 使用排序后的索引作为Y轴标签
            bars = ax.barh([str(idx) for idx in category_data.index], 
                      category_data['Z_Score'], 
                      color=color, alpha=0.7, label=category, height=0.8)
        
        # 在柱状图上标注Z值
            for bar, z_value in zip(bars, category_data['Z_Score']):
                plt.text(bar.get_width() + 0.05 * (1 if bar.get_width() >= 0 else -1), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{z_value:.2f}', 
                    ha='left' if bar.get_width() >= 0 else 'right', 
                    va='center', fontsize=9, fontweight='bold')

    # 设置图形属性
    ax.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Data ID (Sorted by Z-Score)', fontsize=14, fontweight='bold')
    ax.set_title(f'{method} - Z-Score Distribution (Sorted)', fontsize=16, fontweight='bold')

    # 添加零线参考线
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # 添加阈值线
    ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)

    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加图例
    ax.legend(title='Category', title_fontsize=12, fontsize=11, loc='upper right')

    # 调整布局
    plt.tight_layout()

    # 显示图表
    st.pyplot(fig)
    
    # 导出功能
    st.subheader("💾 导出结果")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        '原始数据': data,
        'Z比分数': results['Z_scores'],
        '分类': np.where(np.abs(results['Z_scores']) <= 2, '满意',
                       np.where(np.abs(results['Z_scores']) <= 3, '可疑', '不满意'))
    })
    
    # 下载CSV
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="下载完整结果CSV",
        data=csv,
        file_name=f"{method}_分析结果.csv",
        mime="text/csv"
    )
    
    # 下载报告
    report = f"""
{method}分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
数据点数: {len(data)}
使用方法: {method}

关键结果:
--------
稳健平均值: {results['robust_mean']:.6f}
稳健标准差: {results['robust_std']:.6f}
正常值范围: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]
离群值数量: {len(results['outliers'])}

"""
    
    if method == "四分位稳健统计法":
        report += f"""
四分位统计量:
-----------
下四分位数(Q1): {results['q1']:.6f}
上四分位数(Q3): {results['q3']:.6f}
四分位距(IQR): {results['iqr']:.6f}
标准化四分位距(NIQR): {results['niqr']:.6f}

"""
    
    if 'iterations' in results:
        report += f"迭代次数: {results['iterations']}\n"
    
    report += f"""
数据质量分类:
-----------
满意 (|Z| ≤ 2): {satisfactory} 个数据点
可疑 (2 < |Z| ≤ 3): {questionable} 个数据点  
不满意 (|Z| > 3): {unsatisfactory} 个数据点

离群值列表:
----------
{', '.join([str(float(x)) for x in results['outliers']])}
"""
    
    st.download_button(
        label="下载分析报告",
        data=report,
        file_name=f"{method}_分析报告.txt",
        mime="text/plain"
    )

else:
    st.info("👆 请先输入或上传数据以开始分析")

# 页脚
st.markdown("---")
st.markdown("""
**方法说明:**
- **迭代稳健统计法**: 通过迭代过程逐步修正异常值影响
- **四分位稳健统计法**: 基于数据排序，使用中段50%数据，崩溃点25%
- **Q/Hampel法**: 结合Q方法稳健标准差和Hampel方法稳健平均值
""")