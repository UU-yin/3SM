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
    
    # 使用session_state来存储输入数据和历史记录
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = "54.4, 54.6, 54.2, 54.3, 53.9, 54.4, 54.3, 54.6, 54.5, 54.3, 54.5, 54.1, 54.2, 54.3, 54.8, 54.8, 54.8, 54.3, 54.4, 54.3, 54.3, 54.7, 54.4, 54.5, 54.4, 55.0, 55.0, 55.1, 54.1, 54.8, 54.5, 55.5, 55.6, 55.0, 54.3, 55.3, 54.3, 54.4, 54.3, 54.4, 54.5, 55.9, 53.2, 54.6"
    
    if 'data_history' not in st.session_state:
        st.session_state.data_history = [st.session_state.manual_data]
    
    # 创建两列布局，数据输入框在左侧，按钮在右侧上下排列
    col1, col2 = st.columns([4, 1])
    
    with col1:
        data_input = st.text_area(
            "请输入数据（每行一个数值或用逗号分隔）:", 
            value=st.session_state.manual_data,
            height=150,
            key="manual_input"
        )
    
    with col2:
        st.write("")  # 垂直间距
        
        # 撤销按钮 - 只有当有历史记录时才启用
        undo_disabled = len(st.session_state.data_history) <= 1
        if st.button("↶ 撤销", 
                    use_container_width=True, 
                    disabled=undo_disabled,
                    help="恢复到上一次的数据状态"):
            if len(st.session_state.data_history) > 1:
                # 移除当前状态
                st.session_state.data_history.pop()
                # 恢复到上一个状态
                st.session_state.manual_data = st.session_state.data_history[-1]
                st.rerun()
        
        # 清除按钮
        if st.button("一键清除", 
                    use_container_width=True, 
                    type="secondary",
                    help="清空所有数据"):
            # 保存当前状态到历史记录
            st.session_state.data_history.append(st.session_state.manual_data)
            st.session_state.manual_data = ""
            st.rerun()
    
    # 更新session_state中的数据
    if data_input != st.session_state.manual_data:
        # 如果数据有变化，保存到历史记录
        st.session_state.data_history.append(st.session_state.manual_data)
        # 限制历史记录长度，避免内存问题
        if len(st.session_state.data_history) > 10:
            st.session_state.data_history = st.session_state.data_history[-10:]
        st.session_state.manual_data = data_input
    
    # 在数据输入框下方放置分析按钮（左侧）
    if st.button("分析数据", type="primary"):
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
    
    # 首先放置文件上传器
    uploaded_file = st.file_uploader("选择CSV或TXT文件", type=['csv', 'txt'])
    
    # 然后在下方显示格式说明和示例
    with st.expander("📝 查看文件格式说明和示例", expanded=False):
        st.markdown("""
        **TXT文件格式要求：**
        - 每行一个数值
        - 支持整数和小数
        - 空行会自动忽略
        
        **CSV文件格式要求：**
        - 第一列包含数值数据
        - 可以有表头，也可以没有
        
        **示例文件内容：**
        ```
        54.4
        54.6
        54.2
        54.3
        53.9
        ```
        """)
        
        # 提供示例文件下载
        example_content = "54.4\n54.6\n54.2\n54.3\n53.9"
        st.download_button(
            label="下载示例TXT文件",
            data=example_content,
            file_name="example_data.txt",
            mime="text/plain",
            help="点击下载示例TXT文件，了解正确的数据格式"
        )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # 假设第一列是数据
                data = df.iloc[:, 0].values
                st.success(f"成功加载 {len(data)} 个数据点")
                st.write("前10个数据:", data[:10])
            else:
                # 文本文件，每行一个数字
                content = uploaded_file.read().decode()
                data_list = [float(x.strip()) for x in content.split() if x.strip()]
                data = np.array(data_list)
                st.success(f"成功加载 {len(data)} 个数据点")
                st.write("前10个数据:", data[:10])
            
        except Exception as e:
            st.error(f"文件读取错误: {e}")
            st.info("请确保文件格式正确：每行一个数值，且均为有效数字")         
    
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
        'Z_Score': results['Z_scores']
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

    # 按照Z值从大到小排序，但保留原始索引
    df_sorted = df_clean.sort_values('Z_Score', ascending=False)

    # 创建Z值柱状图
    fig, ax = plt.subplots(figsize=(14, 10))

    # 设置类别对应的颜色
    color_map = {
        'Satisfactory': '#2E8B57',    # 绿色
        'Questionable': '#FFA500',    # 橙色
        'Unsatisfactory': '#DC143C'    # 红色
    }

    # 创建一个统一颜色的列表
    colors = [color_map[cat] for cat in df_sorted['Category']]

    # 绘制所有数据点的柱状图，按Z值排序
    # 使用排序后的索引位置作为Y轴位置
    y_positions = range(len(df_sorted))
    bars = ax.barh(y_positions, 
                   df_sorted['Z_Score'], 
                   color=colors, 
                   alpha=0.7, 
                   height=0.8)

    # 在柱状图上标注Z值
    for i, (bar, z_value) in enumerate(zip(bars, df_sorted['Z_Score'])):
        ax.text(bar.get_width() + 0.05 * (1 if bar.get_width() >= 0 else -1), 
                bar.get_y() + bar.get_height()/2, 
                f'{z_value:.2f}', 
                ha='left' if bar.get_width() >= 0 else 'right', 
                va='center', fontsize=9, fontweight='bold')

    # 设置图形属性
    ax.set_xlabel('Z-Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Original Data ID', fontsize=14, fontweight='bold')
    ax.set_title('Z-Score Distribution (Sorted)', fontsize=16, fontweight='bold')

    # 设置Y轴刻度 - 使用原始数据编号作为标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{idx}" for idx in df_sorted.index])

    # 添加零线参考线
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # 添加阈值线
    ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=0.8)

    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加图例 - 使用颜色映射创建图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['Satisfactory'], label='Satisfactory (|Z| ≤ 2)'),
        Patch(facecolor=color_map['Questionable'], label='Questionable (2 < |Z| ≤ 3)'),
        Patch(facecolor=color_map['Unsatisfactory'], label='Unsatisfactory (|Z| > 3)')
    ]
    ax.legend(handles=legend_elements, title='Category', title_fontsize=12, fontsize=11, loc='upper right')

    # 反转Y轴，使最大的Z值在顶部
    ax.invert_yaxis()

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

# 在页面底部添加简化的反馈功能
st.markdown("---")
st.subheader("💬 用户反馈")

# 使用扩展器形式
with st.expander("💬 有问题或建议？点击这里联系我们", expanded=False):
    st.markdown("""
    **技术支持与反馈**
    
    我们重视每一位用户的反馈，如果您遇到以下情况：
    - 使用过程中遇到问题
    - 有功能改进建议
    - 发现数据计算异常
    - 其他任何疑问
    
    请通过以下方式联系我们：
    
    📩 **ypan1104@163.com**
    
    **联系人**：印博士
       
    感谢您帮助我们变得更好！
    """)