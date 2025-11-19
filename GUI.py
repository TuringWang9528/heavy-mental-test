import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
import plotly.express as px
import io

# ---------------------- 1. 基础配置 ----------------------
st.set_page_config(page_title="Biochar Adsorption Predictor", layout="wide")
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 自定义CSS ----------------------
st.markdown("""
<style>
body { background-color: #f5f7fa; font-family: "Helvetica Neue", Arial, sans-serif; }
.card { background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); padding: 20px; margin-bottom: 20px; }
.section-title { font-size: 18px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 15px; }
.label-col { text-align: left !important; width: 220px; padding-right: 10px; font-size: 13px; font-weight: 600; color: #555;}
.input-col { flex: 1; }
div[class*="stText"], div[class*="stNumberInput"], div[class*="stSelectbox"] { text-align: left !important; }
/* 蓝色按钮样式 */
.stButton>button { background-color: #3498db !important; color: white !important; border-radius: 6px !important; padding: 10px 20px !important; border: 2px solid white !important; }
.stButton>button:hover { background-color: #2980b9 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------- 3. 加载模型 & 定义特征 ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('XGBoost.pkl')
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'XGBoost.pkl' is in the directory.")
        return None

model = load_model()

feature_ranges = {
    'C(%)': {"type": "numerical", "min": 8.100, "max": 88.300, "default": 55.380},
    'H(%)': {"type": "numerical", "min": 0.000, "max": 6.310, "default": 2.000},
    'O(%)': {"type": "numerical", "min": 0.300, "max": 62.530, "default": 14.870},
    'N(%)': {"type": "numerical", "min": 0.220, "max": 5.540, "default": 1.210},
    '(O+N)/C': {"type": "numerical", "min": 0.018, "max": 1.820, "default": 0.469},
    'O/C': {"type": "numerical", "min": 0.004, "max": 1.650, "default": 0.284},
    'H/C': {"type": "numerical", "min": 0.000, "max": 1.390, "default": 0.120},
    'Ash(%)': {"type": "numerical", "min": 2.750, "max": 90.670, "default": 38.443},
    'pH of Biochar': {"type": "numerical", "min": 5.310, "max": 12.620, "default": 9.270},
    'SSA(m²/g)': {"type": "numerical", "min": 0.738, "max": 553.709, "default": 15.750},
    'Initial Cd concentration (mg/L)': {"type": "numerical", "min": 1.000, "max": 500.000, "default": 100.00},
    'Rotational speed(rpm)': {"type": "numerical", "min": 120.000, "max": 4000.000, "default": 150.000},
    'Volume (L)': {"type": "numerical", "min": 0.020, "max": 0.250, "default": 0.025},
    'Concentration of biochar in water(g/L)': {"type": "numerical", "min": 0.001, "max": 20.000, "default": 1.000},
    'Adsorption temperature(℃)': {"type": "numerical", "min": 25.000, "max": 28.000, "default": 25.000},
    'Adsorption time(min)': {"type": "numerical", "min": 0.000, "max": 4760.000, "default": 150.000}
}
feature_names = list(feature_ranges.keys())

if model:
    # 使用 Tabs 分隔功能，使界面更清晰
    # 修改这行代码
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧪 Single Prediction", "📈 Sensitivity Analysis", "📂 Batch Prediction", "🧊 Interaction Analysis", "🎯 Inverse Optimization", "📊 Global Importance"])

    # ======================= TAB 1: 单次预测 (原有功能增强) =======================
    with tab1:
        with st.container():
            st.markdown('<div class="card"><h3 class="section-title">Experimental Parameters</h3>', unsafe_allow_html=True)
            cols = st.columns(3)
            feature_values = []

            for idx, (feature, props) in enumerate(feature_ranges.items()):
                with cols[idx % 3]:
                    st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><div class="label-col">{feature}</div><div class="input-col">', unsafe_allow_html=True)
                    default_val = float(props["default"])
                    step = 0.001 if default_val < 1 else (0.1 if default_val < 10 else 1.0)
                    
                    value = st.number_input(
                        feature,
                        min_value=float(props["min"]),
                        max_value=float(props["max"]),
                        value=default_val,
                        step=step,
                        format="%.3f",
                        label_visibility="collapsed",
                        key=f"input_{idx}"
                    )
                    feature_values.append(value)
                    st.markdown('</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Predict Result", type="primary", use_container_width=True):
            input_data = pd.DataFrame([feature_values], columns=feature_names)
            
            # 预测
            pred_value = model.predict(input_data)[0]
            
            # SHAP 计算
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            base_value = explainer.expected_value
            
            st.session_state.result = {
                "pred": pred_value,
                "shap": shap_values[0],
                "base": base_value,
                "input": input_data
            }

        # 展示结果
        if "result" in st.session_state:
            res = st.session_state.result
            
            st.markdown("### Prediction Dashboard")
            
            # 【新功能】使用列布局展示：数字结果 + 仪表盘
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.info("Predicted Adsorption Capacity")
                st.metric(label="Qe (mg/g)", value=f"{res['pred']:.4f}", delta="Model Output")
                st.write("Base Value (Average):", f"{res['base']:.4f}")

            with col_res2:
                # 【新功能】Plotly 仪表盘
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = res['pred'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Adsorption Capacity Performance"},
                    gauge = {
                        'axis': {'range': [0, 350]}, # 根据你的数据范围调整 max
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e0e0e0"},
                            {'range': [50, 150], 'color': "#bdc3c7"},
                            {'range': [150, 350], 'color': "#95a5a6"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': res['pred']}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # SHAP 可视化
            st.markdown("### 🔍 Model Explanation (SHAP)")
            col_shap1, col_shap2 = st.columns([2, 1])
            
            with col_shap1:
                shap_exp = shap.Explanation(values=res['shap'], base_values=res['base'], data=res['input'].iloc[0].values, feature_names=feature_names)
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(shap_exp, max_display=10, show=False)
                st.pyplot(plt)
            
            with col_shap2:
                st.write("Feature Contributions:")
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": res['shap']
                })
                shap_df["Abs"] = shap_df["SHAP Value"].abs()
                st.dataframe(shap_df.sort_values("Abs", ascending=False).drop("Abs", axis=1), height=400)

# ======================= TAB 2: 灵敏度分析 (最终完善版) =======================
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Single Feature Sensitivity Analysis")
        
        # 1. 选择分析的特征
        target_feature = st.selectbox("Select Feature to Analyze", feature_names, key="sa_feature_select")

        # 2. 动态获取该特征的默认范围 (从你的配置字典中)
        default_min = feature_ranges[target_feature]["min"]
        default_max = feature_ranges[target_feature]["max"]
        
        # 3. 创建范围选择器 (关键修改：使用 number_input 让用户可以精确控制范围)
        st.write(f"**Set Analysis Range for {target_feature}:**")
        col_range1, col_range2 = st.columns(2)
        
        # 注意：这里 key 加上 target_feature 是为了让切换特征时，输入框数值能自动刷新
        analysis_min = col_range1.number_input("Min Value", value=float(default_min), format="%.3f", key=f"min_{target_feature}")
        analysis_max = col_range2.number_input("Max Value", value=float(default_max), format="%.3f", key=f"max_{target_feature}")

        # 4. 分辨率设置 (折叠起来，防止误解)
        with st.expander("⚙️ Advanced Settings (Resolution)"):
            points = st.slider("Curve Smoothness (Points)", 10, 200, 50, help="Higher values make the curve smoother but take slightly longer to calculate.")

        # 5. 运行分析按钮
        if st.button("Run Analysis", type="primary", key="sa_run_button"):
            try:
                # --- A. 准备基准数据 ---
                base_input_dict = {}
                for idx, name in enumerate(feature_names):
                    # 获取 Tab 1 的输入值作为基准
                    base_input_dict[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
                
                # 扩展为 DataFrame
                temp_df = pd.DataFrame([base_input_dict] * points)
                temp_df = temp_df[feature_names] # 确保列顺序正确

                # --- B. 生成 X 轴数据 (使用用户刚刚设置的 Min/Max) ---
                if analysis_min >= analysis_max:
                    st.error("Error: Min Value must be smaller than Max Value.")
                    st.stop()
                    
                x_values = np.linspace(analysis_min, analysis_max, points)
                temp_df[target_feature] = x_values

                # --- C. 预测 ---
                y_pred = model.predict(temp_df)
                
                # 格式转换 (防报错)
                x_list = x_values.tolist()
                y_list = y_pred.ravel().tolist()

                # --- D. 绘图 (Plotly) ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_list, 
                    y=y_list, 
                    mode='lines+markers',
                    name='Predicted Qe',
                    line=dict(color='#3498db', width=4), # 蓝色线条
                    marker=dict(size=6, color='#2980b9', line=dict(width=1, color='white')),
                    hovertemplate=f'{target_feature}: %{{x:.2f}}<br>Qe: %{{y:.2f}} mg/g<extra></extra>' # 自定义悬停提示
                ))
                
                # 布局优化
                fig.update_layout(
                    title=f"Effect of <b>{target_feature}</b> on Adsorption Capacity",
                    xaxis_title=f"{target_feature} Value",
                    yaxis_title="Predicted Qe (mg/g)",
                    height=500,
                    plot_bgcolor='white',
                    hovermode="x unified",
                    font=dict(family="Arial", size=12),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                
                # 关键：保留 theme=None 确保颜色正确
                st.plotly_chart(fig, use_container_width=True, theme=None)

            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================= TAB 3: 批量预测 (新功能) =======================
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📂 Batch Prediction")
        st.write("Upload a CSV or Excel file containing the feature columns to predict multiple samples at once.")
        
        # 提供模板下载
        template_df = pd.DataFrame(columns=feature_names)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Template CSV", data=csv_template, file_name="template.csv", mime="text/csv")
        
        uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    batch_df = pd.read_excel(uploaded_file)
                
                # 检查列是否匹配
                missing_cols = [col for col in feature_names if col not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    st.success(f"Successfully loaded {len(batch_df)} samples.")
                    
                    if st.button("Predict All"):
                        # 预测
                        batch_preds = model.predict(batch_df[feature_names])
                        batch_df['Predicted Qe'] = batch_preds
                        
                        st.dataframe(batch_df)
                        
                        # 下载结果
                        csv_result = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_result,
                            file_name="prediction_results.csv",
                            mime="text/csv",
                            type="primary"
                        )
            except Exception as e:
                st.error(f"Error processing file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        
# ======================= TAB 4: 交互分析 (2D/3D 双模式版) =======================
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧊 Interaction Analysis (2D & 3D)")
        
        # 1. 布局：左侧选参数，右侧选模式
        col_inter1, col_inter2, col_inter3 = st.columns([1, 1, 1])
        with col_inter1:
            feat_x = st.selectbox("X-axis Feature", feature_names, index=0, key="inter_x")
        with col_inter2:
            feat_y = st.selectbox("Y-axis Feature", feature_names, index=1, key="inter_y")
        with col_inter3:
            # 切换 2D / 3D
            view_mode = st.radio("View Mode", ["2D Heatmap", "3D Surface"], horizontal=True)

        res_inter = st.slider("Resolution (Grid Size)", 10, 50, 25, key="inter_res")

        if st.button("Generate Plot", type="primary", key="inter_btn"):
            try:
                if feat_x == feat_y:
                    st.warning("⚠️ Please select two different features.")
                    st.stop()

                # --- 数据准备 (同前) ---
                base_input_dict = {}
                for idx, name in enumerate(feature_names):
                    base_input_dict[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
                
                x_min, x_max = feature_ranges[feat_x]["min"], feature_ranges[feat_x]["max"]
                y_min, y_max = feature_ranges[feat_y]["min"], feature_ranges[feat_y]["max"]
                
                x_linspace = np.linspace(x_min, x_max, res_inter)
                y_linspace = np.linspace(y_min, y_max, res_inter)
                
                X_grid, Y_grid = np.meshgrid(x_linspace, y_linspace)
                X_flat, Y_flat = X_grid.ravel(), Y_grid.ravel()
                
                batch_df = pd.DataFrame([base_input_dict] * (res_inter * res_inter))
                batch_df = batch_df[feature_names]
                batch_df[feat_x] = X_flat
                batch_df[feat_y] = Y_flat
                
                Z_pred = model.predict(batch_df)
                Z_grid = Z_pred.reshape(res_inter, res_inter)
                
                # 诊断
                if np.min(Z_grid) == np.max(Z_grid):
                    st.warning("⚠️ Prediction is constant in this range.")

                # --- 绘图逻辑 ---
                if view_mode == "2D Heatmap":
                    # 2D 模式 (保持原样)
                    fig = go.Figure(data=go.Contour(
                        z=Z_grid.tolist(),
                        x=x_linspace.tolist(),
                        y=y_linspace.tolist(),
                        colorscale='Viridis',
                        colorbar=dict(title='Qe'),
                        contours=dict(coloring='heatmap', showlabels=True)
                    ))
                    fig.update_layout(height=600, title=f"2D Interaction: {feat_x} vs {feat_y}")

                else:
                    # 3D 模式 (新增)
                    fig = go.Figure(data=[go.Surface(
                        z=Z_grid.tolist(),
                        x=x_linspace.tolist(),
                        y=y_linspace.tolist(),
                        colorscale='Viridis',
                        colorbar=dict(title='Qe'),
                        opacity=0.9
                    )])
                    
                    fig.update_layout(
                        title=f"3D Surface: {feat_x} vs {feat_y}",
                        scene=dict(
                            xaxis_title=feat_x,
                            yaxis_title=feat_y,
                            zaxis_title="Qe (mg/g)",
                            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                        ),
                        height=700, # 3D 图稍微高一点
                        margin=dict(l=0, r=0, b=0, t=40) # 减少边距
                    )

                # 通用配置
                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True, theme=None)
                
                # 结论
                max_idx = np.argmax(Z_pred)
                st.success(f"Max Qe ({Z_pred[max_idx]:.2f}) at {feat_x}={X_flat[max_idx]:.2f}, {feat_y}={Y_flat[max_idx]:.2f}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ======================= TAB 5: 逆向优化 (修复显示版) =======================
    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Inverse Optimization (Target Search)")
        st.info("设定你想要的目标吸附量，AI 将帮你寻找能达到该目标的最佳实验条件组合。")

        col_opt1, col_opt2 = st.columns([1, 2])
        
        with col_opt1:
            target_qe = st.number_input("Target Qe (mg/g)", min_value=0.0, value=100.0, step=10.0)
            n_iter = st.slider("Search Iterations (Monte Carlo)", 1000, 50000, 10000)

        with col_opt2:
            st.write("**Select Optimization Parameters:**")
            default_opts = ['pH of Biochar', 'Initial Cd concentration (mg/L)', 'Adsorption temperature(℃)']
            # 过滤掉不存在的特征
            default_opts = [x for x in default_opts if x in feature_names]
            opt_features = st.multiselect("Features to Optimize", feature_names, default=default_opts)

        if st.button("🚀 Start Optimization", type="primary", key="opt_btn"):
            if not opt_features:
                st.warning("Please select at least one feature to optimize.")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # --- A. 准备基准数据 ---
                base_input_dict = {}
                for idx, name in enumerate(feature_names):
                    base_input_dict[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
                
                # --- B. 生成随机搜索空间 ---
                status_text.text(f"Simulating {n_iter} experiments...")
                random_data = {}
                for name in feature_names:
                    if name in opt_features:
                        min_v = feature_ranges[name]["min"]
                        max_v = feature_ranges[name]["max"]
                        random_data[name] = np.random.uniform(min_v, max_v, n_iter)
                    else:
                        random_data[name] = np.full(n_iter, base_input_dict[name])
                
                sim_df = pd.DataFrame(random_data)[feature_names] # 确保列序
                
                progress_bar.progress(50)
                status_text.text("Running AI Model...")

                # --- C. 批量预测 ---
                sim_preds = model.predict(sim_df)
                sim_df['Predicted Qe'] = sim_preds
                
                progress_bar.progress(80)
                status_text.text("Filtering results...")

                # --- D. 筛选结果 ---
                success_df = sim_df[sim_df['Predicted Qe'] >= target_qe].copy()
                success_df = success_df.sort_values(by='Predicted Qe', ascending=False)
                
                progress_bar.progress(100)
                status_text.empty()

                # --- E. 展示结果 ---
                if len(success_df) > 0:
                    st.success(f"🎉 Found {len(success_df)} conditions that meet the target (Qe >= {target_qe})!")
                    
                    st.write("### 🏆 Top 5 Recommended Conditions")
                    display_cols = ['Predicted Qe'] + opt_features
                    st.dataframe(success_df[display_cols].head(5).style.format("{:.2f}").background_gradient(cmap='Blues'))
                    
                    # 下载按钮
                    csv_opt = success_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download All Valid Solutions", csv_opt, "optimization_results.csv", "text/csv")
                    
                    # --- F. 可视化分布 (修复显示问题) ---
                    with st.expander("📊 Solution Distribution Analysis", expanded=True):
                        st.write(f"Distribution of top 100 solutions for targeted features:")
                        
                        # 取前100个最佳结果做直方图
                        top_100_df = success_df.head(100)
                        
                        for col in opt_features:
                            # 【核心修复】
                            # 1. 强制转换为 list，防止 numpy 序列化问题
                            hist_data = top_100_df[col].tolist()
                            
                            # 2. 使用 go.Histogram 替代 px.histogram，控制力更强
                            fig_hist = go.Figure(data=[go.Histogram(
                                x=hist_data,
                                nbinsx=20, # 自动分箱
                                marker_color='#3498db', # 强制蓝色
                                marker_line_color='white', # 柱子边框白色
                                marker_line_width=1,
                                opacity=0.75
                            )])
                            
                            # 3. 强制背景色和布局
                            fig_hist.update_layout(
                                title=f"Distribution of <b>{col}</b> in Top Solutions",
                                xaxis_title=col,
                                yaxis_title="Count",
                                height=350,
                                plot_bgcolor='white', # 强制白底
                                margin=dict(l=20, r=20, t=40, b=20),
                                bargap=0.1 # 柱子间距
                            )
                            fig_hist.update_xaxes(showgrid=True, gridcolor='#eee')
                            fig_hist.update_yaxes(showgrid=True, gridcolor='#eee')
                            
                            # 4. theme=None 禁止 Streamlit 覆盖样式
                            st.plotly_chart(fig_hist, use_container_width=True, theme=None)
                            
                else:
                    st.error(f"❌ No solutions found for Qe >= {target_qe}.")
                    st.info(f"Best result found: Qe = {sim_df['Predicted Qe'].max():.2f}")

            except Exception as e:
                st.error(f"Optimization Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# 定义 tabs 时增加一项
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["...", "...", "...", "...", "...", "📊 Global Importance"])

    # ======================= TAB 6: 全局特征重要性 (新功能) =======================
    with tab6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Global Feature Importance Analysis")
        st.info("Which features contribute most to the model's decisions overall?")
        
        if st.button("Calculate Importance", type="primary"):
            try:
                # 1. 获取 XGBoost 内置的重要性
                # 这里的 importance_type 默认通常是 'weight' 或 'gain'
                # 我们提取 'gain' (增益)，因为它在科研中通常被认为更准确
                importance_dict = model.get_booster().get_score(importance_type='gain')
                
                # 2. 整理数据
                # XGBoost 返回的 key 可能是 'f0', 'f1' 也可能是列名
                # 为了安全，我们直接用 sklearn 接口的 feature_importances_ (基于 gain/gini)
                importances = model.feature_importances_
                
                # 构建 DataFrame
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                # 排序
                imp_df = imp_df.sort_values(by='Importance', ascending=True) # 升序用于画横向条形图
                
                # 3. 绘图 (横向条形图)
                fig_imp = go.Figure(go.Bar(
                    x=imp_df['Importance'].tolist(),
                    y=imp_df['Feature'].tolist(),
                    orientation='h', # 横向
                    marker=dict(
                        color=imp_df['Importance'].tolist(),
                        colorscale='Blues', # 颜色随重要性变深
                    )
                ))
                
                fig_imp.update_layout(
                    title="Feature Importance Ranking (XGBoost)",
                    xaxis_title="Relative Importance (Gain)",
                    # yaxis_title="Feature",
                    height=600,
                    plot_bgcolor='white',
                    margin=dict(l=150) # 给左边留足空间显示特征名
                )
                fig_imp.update_xaxes(showgrid=True, gridcolor='#eee')
                
                st.plotly_chart(fig_imp, use_container_width=True, theme=None)
                
                # 4. 文字解读
                top_3 = imp_df.sort_values(by='Importance', ascending=False).head(3)['Feature'].tolist()
                st.success(f"💡 **Insight:** The top 3 most critical factors affecting Adsorption Capacity are: **{', '.join(top_3)}**.")
                
                # 5. 数据下载
                csv_imp = imp_df.sort_values(by='Importance', ascending=False).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Importance Data", csv_imp, "feature_importance.csv", "text/csv")

            except Exception as e:
                st.error(f"Error calculating importance: {str(e)}")
                st.write("Tip: Ensure the model is a standard XGBoost Regressor.")
        
        st.markdown('</div>', unsafe_allow_html=True)
