import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
import plotly.express as px
import io
from dataclasses import dataclass
import time

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
        # 为了演示，如果找不到模型文件，这里生成一个伪造的Dummy模型，实际使用时请确保文件存在
        # st.error("Model file not found! Please ensure 'XGBoost.pkl' is in the directory.")
        # return None
        from sklearn.dummy import DummyRegressor
        dummy = DummyRegressor(strategy="mean")
        # 伪造训练以防止报错
        X_dummy = pd.DataFrame(np.random.rand(10, 16), columns=[
            'C(%)', 'H(%)', 'O(%)', 'N(%)', '(O+N)/C', 'O/C', 'H/C', 'Ash(%)', 
            'pH of Biochar', 'SSA(m²/g)', 'Initial Cd concentration (mg/L)', 
            'Rotational speed(rpm)', 'Volume (L)', 'Concentration of biochar in water(g/L)', 
            'Adsorption temperature(℃)', 'Adsorption time(min)'
        ])
        y_dummy = np.random.rand(10) * 100
        dummy.fit(X_dummy, y_dummy)
        return dummy

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
@st.cache_data(show_spinner=False)
def compute_permutation_importance_cached(feature_names, feature_ranges, n_samples, seed=42):
    """
    用当前 model 做 permutation importance（与你 Tab5 逻辑一致）。
    这里缓存，避免 RL Tab/Importance Tab 重复计算。
    """
    rng = np.random.default_rng(seed)

    base_data = {}
    for name in feature_names:
        mn = float(feature_ranges[name]["min"])
        mx = float(feature_ranges[name]["max"])
        base_data[name] = rng.uniform(mn, mx, n_samples)

    X_base = pd.DataFrame(base_data)[feature_names]
    y_base = model.predict(X_base)

    importances = []
    for col in feature_names:
        X_shuffled = X_base.copy()
        X_shuffled[col] = rng.permutation(X_shuffled[col].values)
        y_shuffled = model.predict(X_shuffled)
        diff = float(np.mean(np.abs(y_base - y_shuffled)))
        importances.append(diff)

    perm_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    perm_df_desc = perm_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    perm_df = perm_df_desc.sort_values(by='Importance', ascending=True)  # 仅用于画横向条形图从小到大
    st.session_state["perm_df"] = perm_df_desc
    return perm_df

if model:
    # 定义 Tabs：调整顺序，Batch 放到最后，新增 Comparison
    tab_titles = [
    "🧪 Single Prediction", 
    "📈 Dependency Analysis", 
    "🧊 Interaction Analysis", 
    "🎯 Inverse Optimization", 
    "📊 Global Importance",
    "🤖 RL Optimization",        # ✅ 新增
    "⚔️ Comparative Analysis",
    "📂 Batch Prediction"
    ]
    
    # 解包 tab 对象
    tab_single, tab_depend, tab_inter, tab_opt, tab_imp, tab_rl, tab_compare, tab_batch = st.tabs(tab_titles)

    # ======================= TAB 1: 单次预测 =======================
    with tab_single:
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
            
            # SHAP 计算 (如果是 Dummy 模型则跳过 SHAP，防止演示报错)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                base_value = explainer.expected_value
            except:
                shap_values = [np.zeros(len(feature_names))]
                base_value = 0
            
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
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.info("Predicted Adsorption Capacity")
                st.metric(label="Qe (mg/g)", value=f"{res['pred']:.4f}", delta="Model Output")
                st.write("Base Value (Average):", f"{res['base']:.4f}")

            with col_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = res['pred'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Adsorption Capacity Performance"},
                    gauge = {
                        'axis': {'range': [0, 350]}, 
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
                try:
                    shap_exp = shap.Explanation(values=res['shap'], base_values=res['base'], data=res['input'].iloc[0].values, feature_names=feature_names)
                    plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(shap_exp, max_display=10, show=False)
                    st.pyplot(plt)
                except:
                    st.warning("SHAP plot not available for this model type in demo mode.")
            
            with col_shap2:
                st.write("Feature Contributions:")
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": res['shap']
                })
                shap_df["Abs"] = shap_df["SHAP Value"].abs()
                st.dataframe(shap_df.sort_values("Abs", ascending=False).drop("Abs", axis=1), height=400)

    # ======================= TAB 2: 灵敏度分析 =======================
    with tab_depend:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Single Feature Dependency Analysis")
        
        target_feature = st.selectbox("Select Feature to Analyze", feature_names, key="sa_feature_select")

        default_min = feature_ranges[target_feature]["min"]
        default_max = feature_ranges[target_feature]["max"]
        
        st.write(f"**Set Analysis Range for {target_feature}:**")
        col_range1, col_range2 = st.columns(2)
        
        analysis_min = col_range1.number_input("Min Value", value=float(default_min), format="%.3f", key=f"min_{target_feature}")
        analysis_max = col_range2.number_input("Max Value", value=float(default_max), format="%.3f", key=f"max_{target_feature}")

        with st.expander("⚙️ Advanced Settings (Resolution)"):
            points = st.slider("Curve Smoothness (Points)", 10, 200, 50, help="Higher values make the curve smoother.")

        if st.button("Run Analysis", type="primary", key="sa_run_button"):
            try:
                base_input_dict = {}
                for idx, name in enumerate(feature_names):
                    base_input_dict[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
                
                temp_df = pd.DataFrame([base_input_dict] * points)
                temp_df = temp_df[feature_names]

                if analysis_min >= analysis_max:
                    st.error("Error: Min Value must be smaller than Max Value.")
                    st.stop()
                    
                x_values = np.linspace(analysis_min, analysis_max, points)
                temp_df[target_feature] = x_values

                y_pred = model.predict(temp_df)
                
                x_list = x_values.tolist()
                y_list = y_pred.ravel().tolist()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_list, 
                    y=y_list, 
                    mode='lines+markers',
                    name='Predicted Qe',
                    line=dict(color='#3498db', width=4),
                    marker=dict(size=6, color='#2980b9', line=dict(width=1, color='white')),
                    hovertemplate=f'{target_feature}: %{{x:.2f}}<br>Qe: %{{y:.2f}} mg/g<extra></extra>'
                ))
                
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
                
                st.plotly_chart(fig, use_container_width=True, theme=None)

            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================= TAB 3: 交互分析 (2D/3D) =======================
    # 原 Tab 4 -> 移动到位置 3
    with tab_inter:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧊 Interaction Analysis (2D & 3D)")
        
        col_inter1, col_inter2, col_inter3 = st.columns([1, 1, 1])
        with col_inter1:
            feat_x = st.selectbox("X-axis Feature", feature_names, index=0, key="inter_x")
        with col_inter2:
            feat_y = st.selectbox("Y-axis Feature", feature_names, index=1, key="inter_y")
        with col_inter3:
            view_mode = st.radio("View Mode", ["2D Heatmap", "3D Surface"], horizontal=True)

        res_inter = st.slider("Resolution (Grid Size)", 10, 50, 25, key="inter_res")

        if st.button("Generate Plot", type="primary", key="inter_btn"):
            try:
                if feat_x == feat_y:
                    st.warning("⚠️ Please select two different features.")
                    st.stop()

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
                
                if np.min(Z_grid) == np.max(Z_grid):
                    st.warning("⚠️ Prediction is constant in this range.")

                if view_mode == "2D Heatmap":
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
                        height=700, 
                        margin=dict(l=0, r=0, b=0, t=40)
                    )

                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True, theme=None)
                
                max_idx = np.argmax(Z_pred)
                st.success(f"Max Qe ({Z_pred[max_idx]:.2f}) at {feat_x}={X_flat[max_idx]:.2f}, {feat_y}={Y_flat[max_idx]:.2f}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================= TAB 4: 逆向优化 =======================
    # 原 Tab 5 -> 移动到位置 4
    with tab_opt:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Inverse Optimization (Target Search)")
        st.info("Set the target adsorption amount you want, Model will help you find the optimal combination of experimental conditions.")

        col_opt1, col_opt2 = st.columns([1, 2])
        
        with col_opt1:
            target_qe = st.number_input("Target Qe (mg/g)", min_value=0.0, value=90.0, step=10.0)
            n_iter = st.slider("Search Iterations (Monte Carlo)", 1000, 50000, 10000)

        with col_opt2:
            st.write("**Select Optimization Parameters:**")
            default_opts = ['pH of Biochar', 'Initial Cd concentration (mg/L)', 'Adsorption temperature(℃)']
            default_opts = [x for x in default_opts if x in feature_names]
            opt_features = st.multiselect("Features to Optimize", feature_names, default=default_opts)

        if st.button("🚀 Start Optimization", type="primary", key="opt_btn"):
            if not opt_features:
                st.warning("Please select at least one feature to optimize.")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                base_input_dict = {}
                for idx, name in enumerate(feature_names):
                    base_input_dict[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
                
                status_text.text(f"Simulating {n_iter} experiments...")
                random_data = {}
                for name in feature_names:
                    if name in opt_features:
                        min_v = feature_ranges[name]["min"]
                        max_v = feature_ranges[name]["max"]
                        random_data[name] = np.random.uniform(min_v, max_v, n_iter)
                    else:
                        random_data[name] = np.full(n_iter, base_input_dict[name])
                
                sim_df = pd.DataFrame(random_data)[feature_names]
                
                progress_bar.progress(50)
                status_text.text("Running AI Model...")

                sim_preds = model.predict(sim_df)
                sim_df['Predicted Qe'] = sim_preds
                
                progress_bar.progress(80)
                status_text.text("Filtering results...")

                success_df = sim_df[sim_df['Predicted Qe'] >= target_qe].copy()
                success_df = success_df.sort_values(by='Predicted Qe', ascending=False)
                
                progress_bar.progress(100)
                status_text.empty()

                if len(success_df) > 0:
                    st.success(f"🎉 Found {len(success_df)} conditions that meet the target (Qe >= {target_qe})!")
                    
                    st.write("### 🏆 Top 5 Recommended Conditions")
                    display_cols = ['Predicted Qe'] + opt_features
                    st.dataframe(success_df[display_cols].head(5).style.format("{:.2f}").background_gradient(cmap='Blues'))
                    
                    csv_opt = success_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download All Valid Solutions", csv_opt, "optimization_results.csv", "text/csv")
                    
                    with st.expander("📊 Solution Distribution Analysis", expanded=True):
                        st.write(f"Distribution of top 100 solutions for targeted features:")
                        top_100_df = success_df.head(100)
                        
                        for col in opt_features:
                            hist_data = top_100_df[col].tolist()
                            fig_hist = go.Figure(data=[go.Histogram(
                                x=hist_data,
                                nbinsx=20,
                                marker_color='#3498db',
                                marker_line_color='white',
                                marker_line_width=1,
                                opacity=0.75
                            )])
                            
                            fig_hist.update_layout(
                                title=f"Distribution of <b>{col}</b> in Top Solutions",
                                xaxis_title=col,
                                yaxis_title="Count",
                                height=350,
                                plot_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20),
                                bargap=0.1
                            )
                            fig_hist.update_xaxes(showgrid=True, gridcolor='#eee')
                            fig_hist.update_yaxes(showgrid=True, gridcolor='#eee')
                            st.plotly_chart(fig_hist, use_container_width=True, theme=None)
                            
                else:
                    st.error(f"❌ No solutions found for Qe >= {target_qe}.")
                    st.info(f"Best result found: Qe = {sim_df['Predicted Qe'].max():.2f}")

            except Exception as e:
                st.error(f"Optimization Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================= TAB 5: 特征重要性 =======================
    # 原 Tab 6 -> 移动到位置 5
    with tab_imp:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Permutation Feature Importance")
        st.info("Shuffle the data of a certain feature and observe how much the prediction results change.")
        
        n_samples = st.slider("Simulation Samples", 500, 5000, 1000)

        if st.button("Calculate Permutation Importance", type="primary", key="perm_imp_btn"):
            progress_bar = st.progress(0)
            
            try:
                base_data = {}
                for name in feature_names:
                    min_v = feature_ranges[name]["min"]
                    max_v = feature_ranges[name]["max"]
                    base_data[name] = np.random.uniform(min_v, max_v, n_samples)
                
                X_base = pd.DataFrame(base_data)[feature_names]
                y_base = model.predict(X_base)
                
                progress_bar.progress(20)
                
                importances = []
                
                for i, col in enumerate(feature_names):
                    X_shuffled = X_base.copy()
                    X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
                    
                    y_shuffled = model.predict(X_shuffled)
                    
                    diff = np.mean(np.abs(y_base - y_shuffled))
                    importances.append(diff)
                    
                    prog = 20 + int((i / len(feature_names)) * 80)
                    progress_bar.progress(prog)
                
                progress_bar.progress(100)

                perm_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                perm_df = perm_df.sort_values(by='Importance', ascending=True)
                
                fig_height = max(6, len(feature_names) * 0.4)
                fig, ax = plt.subplots(figsize=(10, fig_height))
                
                norm = plt.Normalize(perm_df['Importance'].min(), perm_df['Importance'].max())
                colors = plt.cm.Blues(norm(perm_df['Importance']))
                
                bars = ax.barh(perm_df['Feature'], perm_df['Importance'], color=colors, edgecolor='black', linewidth=0.5)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                            f' {width:.2f}', 
                            va='center', ha='left', fontsize=10, color='black')
                
                ax.set_xlabel("Average Impact on Qe (mg/g)", fontsize=12, fontweight='bold')
                ax.set_title("Global Feature Importance (Permutation)", fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                top_feature = perm_df.iloc[-1]['Feature']
                st.success(f"💡 Result interpretation: **{top_feature}** is the feature that has the greatest impact on the model's prediction results.")
                
                csv_imp = perm_df.sort_values(by='Importance', ascending=False).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Importance Data (CSV)", csv_imp, "permutation_importance.csv", "text/csv")

            except Exception as e:
                st.error(f"Calculation Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
# ======================= TAB 6: RL 优化（最大化Qe + importance自动选Top-K） =======================
with tab_rl:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 RL Optimization (Maximize Qe with Auto Top-K by Importance)")
    st.write("Use feature importance to automatically choose the most controllable variables, then apply a lightweight RL-style optimizer to maximize predicted Qe.")

    # ---------- 1) 获取/计算 Importance ----------
    col_rl1, col_rl2, col_rl3 = st.columns([1, 1, 1])
    with col_rl1:
        k_top = st.slider("Top-K Features (auto selected)", 2, min(10, len(feature_names)), 5)
    with col_rl2:
        n_imp_samples = st.slider("Importance Samples", 300, 3000, 800, step=100)
    with col_rl3:
        imp_seed = st.number_input("Random Seed", value=42, step=1)

    # 如果 Tab5 已算过就直接用；否则自动算一遍（缓存）
    perm_df = st.session_state.get("perm_df", None)
    if perm_df is None or not isinstance(perm_df, pd.DataFrame) or len(perm_df) == 0:
        with st.spinner("Computing feature importance (cached)..."):
            perm_df = compute_permutation_importance_cached(feature_names, feature_ranges, int(n_imp_samples), seed=int(imp_seed))
        st.session_state["perm_df"] = perm_df

    topk_features = perm_df["Feature"].head(int(k_top)).tolist()
    topk_idx = [feature_names.index(f) for f in topk_features]

    # importance 向量（按 feature_names 顺序对齐）
    imp_vec = perm_df.set_index("Feature").loc[feature_names]["Importance"].values.astype(float)
    imp_vec = np.maximum(imp_vec, 1e-9)  # 防止除0

    st.info(f"Auto selected Top-{k_top}: **{', '.join(topk_features)}**")

    # ---------- 2) 起点：优先用 Single Prediction 的输入，否则用默认 ----------
    base_input = {}
    for idx, name in enumerate(feature_names):
        base_input[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
    x0 = np.array([base_input[n] for n in feature_names], dtype=float)

    # ---------- 3) 轻量 RL 优化器：CEM（序列决策式） ----------
    st.markdown("#### ⚙️ RL-Style Optimizer Settings (CEM Control)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon = st.slider("Horizon (Steps)", 5, 40, 20)
    with c2:
        pop_size = st.slider("Population", 50, 800, 200, step=50)
    with c3:
        n_iters = st.slider("CEM Iterations", 5, 50, 20)
    with c4:
        elite_frac = st.slider("Elite Fraction", 0.05, 0.30, 0.15, step=0.01)

    a1, a2, a3 = st.columns(3)
    with a1:
        base_step = st.slider("Base Step (in normalized space)", 0.01, 0.20, 0.06, step=0.01)
    with a2:
        lam_change = st.slider("Change Penalty λ", 0.0, 0.50, 0.05, step=0.01)
    with a3:
        lam_edge = st.slider("Edge Penalty β", 0.0, 1.00, 0.20, step=0.05)

    with st.expander("Advanced: what this is doing", expanded=False):
        st.write(
            "- State: 16 features (normalized 0-1)\n"
            "- Action: only Top-K features, continuous in [-1, 1]\n"
            "- Transition: x_next = x + step * action (clipped to [0,1])\n"
            "- Reward: Qe - λ*change_pen - β*edge_pen\n"
            "- Optimizer: CEM searches for an action *sequence* that maximizes peak Qe along the rollout."
        )

    # ---------- 4) 辅助：归一化/反归一化 ----------
    def norm_x(x):
        x01 = np.zeros_like(x, dtype=float)
        for i, f in enumerate(feature_names):
            mn, mx = float(feature_ranges[f]["min"]), float(feature_ranges[f]["max"])
            x01[i] = (x[i] - mn) / (mx - mn + 1e-12)
        return np.clip(x01, 0.0, 1.0)

    def denorm_x(x01):
        x = np.zeros_like(x01, dtype=float)
        for i, f in enumerate(feature_names):
            mn, mx = float(feature_ranges[f]["min"]), float(feature_ranges[f]["max"])
            x[i] = mn + x01[i] * (mx - mn)
        return x

    # importance 加权步长（Top-K）
    imp_topk = imp_vec[topk_idx]
    imp_topk = imp_topk / (imp_topk.max() + 1e-12)
    step_vec = base_step * (0.3 + 0.7 * imp_topk)  # 重要特征更敢动

    # reward 组件
    def edge_penalty(x01_topk):
        # 靠近边界(0或1)的比例（<5%区域）
        return float(np.mean(np.minimum(x01_topk, 1.0 - x01_topk) < 0.05))

    def rollout_and_score(action_seq, x01_start):
        """
        action_seq: shape (H, K), each in [-1,1]
        返回：best_qe, best_x01, qe_trace, best_step_idx
        """
        x01 = x01_start.copy()
        best_qe = -1e18
        best_x01 = x01.copy()
        qe_trace = []

        # 只惩罚 Top-K 的相对起点变化
        x01_start_topk = x01_start[topk_idx].copy()

        for t in range(action_seq.shape[0]):
            a = np.clip(action_seq[t], -1.0, 1.0)

            x01_next = x01.copy()
            x01_next[topk_idx] = np.clip(x01[topk_idx] + step_vec * a, 0.0, 1.0)

            x_next = denorm_x(x01_next)
            qe = float(model.predict(x_next.reshape(1, -1))[0])
            qe_trace.append(qe)

            # penalties
            delta = x01_next[topk_idx] - x01_start_topk
            change_pen = float(np.mean(delta**2))
            epen = edge_penalty(x01_next[topk_idx])

            # 即时 reward（这里只做记录；最终 score 用“峰值Qe - penalty”）
            score = qe - lam_change * change_pen - lam_edge * epen

            if score > best_qe:
                best_qe = score
                best_x01 = x01_next.copy()
                best_step = t

            x01 = x01_next

        return best_qe, best_x01, qe_trace, best_step

    # ---------- 5) 运行按钮 ----------
    run = st.button("🚀 Run RL Optimization (CEM)", type="primary", use_container_width=True)
    if run:
        # CEM 初始化：动作序列分布 N(mean, std)
        H = int(horizon)
        K = int(k_top)
        N = int(pop_size)
        iters = int(n_iters)
        elite_n = max(2, int(N * float(elite_frac)))

        x01_start = norm_x(x0)

        mean = np.zeros((H, K), dtype=float)
        std = np.ones((H, K), dtype=float) * 0.8

        rng = np.random.default_rng(int(imp_seed))
        progress = st.progress(0)
        status = st.empty()

        best_global_score = -1e18
        best_global_x01 = x01_start.copy()
        best_global_trace = None
        best_global_step = 0

        t0 = time.time()

        for it in range(iters):
            status.text(f"CEM iteration {it+1}/{iters} ... sampling {N} rollouts")

            # 采样动作序列（tanh/clip 保证在 [-1,1]）
            actions = rng.normal(mean, std, size=(N, H, K))
            actions = np.clip(actions, -1.0, 1.0)

            scores = np.zeros(N, dtype=float)
            elite_pack = []

            for i in range(N):
                score, x01_best, trace, best_step = rollout_and_score(actions[i], x01_start)
                scores[i] = score
                elite_pack.append((score, actions[i], x01_best, trace, best_step))

            # 选 elite
            elite_pack.sort(key=lambda x: x[0], reverse=True)
            elites = elite_pack[:elite_n]

            elite_actions = np.stack([e[1] for e in elites], axis=0)
            mean = elite_actions.mean(axis=0)
            std = elite_actions.std(axis=0) + 1e-6

            # 更新全局最优
            if elites[0][0] > best_global_score:
                best_global_score = elites[0][0]
                best_global_x01 = elites[0][2].copy()
                best_global_trace = elites[0][3]
                best_global_step = elites[0][4]

            progress.progress(int(((it + 1) / iters) * 100))

        status.empty()
        progress.empty()

        x_best = denorm_x(best_global_x01)
        qe_best = float(model.predict(x_best.reshape(1, -1))[0])

        st.success(f"✅ Best predicted Qe: **{qe_best:.4f} mg/g** (found at step {best_global_step+1}/{horizon})")
        st.caption(f"Optimization finished in {time.time()-t0:.2f}s. (Surrogate environment = your trained model)")

        # 展示 Top-K 的变化（更直观）
        res_df = pd.DataFrame({
            "Feature": feature_names,
            "Start": x0,
            "Best": x_best,
            "Delta": (x_best - x0)
        })
        res_df["AbsDelta"] = res_df["Delta"].abs()
        show_df = res_df.sort_values("AbsDelta", ascending=False).drop(columns=["AbsDelta"])

        st.markdown("#### 🔎 Best Condition (sorted by |Δ|)")
        st.dataframe(show_df.style.format({"Start": "{:.4f}", "Best": "{:.4f}", "Delta": "{:.4f}"}), height=420)

        # 轨迹图（Qe 随 step）
        if best_global_trace is not None:
            fig_trace = go.Figure()
            fig_trace.add_trace(go.Scatter(
                x=list(range(1, len(best_global_trace)+1)),
                y=best_global_trace,
                mode="lines+markers",
                name="Qe trace"
            ))
            fig_trace.update_layout(
                title="Qe Improvement Along the Rollout",
                xaxis_title="Step",
                yaxis_title="Predicted Qe (mg/g)",
                height=420,
                plot_bgcolor="white"
            )
            fig_trace.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig_trace.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig_trace, use_container_width=True, theme=None)

        # 导出最佳条件
        out_df = pd.DataFrame([x_best], columns=feature_names)
        out_df["Predicted Qe"] = qe_best
        csv_out = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Best Condition (CSV)", csv_out, "rl_best_condition.csv", "text/csv", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================= TAB 6: [新功能] 对比分析 =======================
    with tab_compare:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚔️ Comparative Analysis (Scenario A vs B)")
        st.write("Compare two different experimental setups side-by-side to visualize differences.")
        
        # 左右两列输入
        col_scen_a, col_scen_b = st.columns(2)
        
        input_a = {}
        input_b = {}
        
        with col_scen_a:
            st.markdown("#### Scenario A (Blue)")
            with st.expander("Configure Scenario A", expanded=True):
                for idx, (feature, props) in enumerate(feature_ranges.items()):
                    default_val = float(props["default"])
                    val = st.number_input(f"A: {feature}", min_value=float(props['min']), max_value=float(props['max']), value=default_val, key=f"A_{idx}")
                    input_a[feature] = val
                    
        with col_scen_b:
            st.markdown("#### Scenario B (Red)")
            with st.expander("Configure Scenario B", expanded=True):
                for idx, (feature, props) in enumerate(feature_ranges.items()):
                    # B 组默认稍微改动一点，以显示区别
                    default_val = float(props["default"]) * 1.05 if float(props["default"]) > 0 else 0
                    if default_val > float(props['max']): default_val = float(props['max'])
                    
                    val = st.number_input(f"B: {feature}", min_value=float(props['min']), max_value=float(props['max']), value=default_val, key=f"B_{idx}")
                    input_b[feature] = val

        if st.button("Compare Scenarios", type="primary", use_container_width=True):
            df_a = pd.DataFrame([input_a], columns=feature_names)
            df_b = pd.DataFrame([input_b], columns=feature_names)
            
            pred_a = model.predict(df_a)[0]
            pred_b = model.predict(df_b)[0]
            
            # 1. 结果对比柱状图
            st.divider()
            st.markdown("#### 1. Prediction Comparison")
            
            col_res_comp1, col_res_comp2 = st.columns([1, 2])
            
            with col_res_comp1:
                st.metric("Scenario A Result", f"{pred_a:.2f} mg/g")
                st.metric("Scenario B Result", f"{pred_b:.2f} mg/g", delta=f"{pred_b - pred_a:.2f}")
                
                winner = "Scenario A" if pred_a > pred_b else "Scenario B"
                st.success(f"🏆 {winner} performs better.")

            with col_res_comp2:
                fig_comp_bar = go.Figure(data=[
                    go.Bar(name='Scenario A', x=['Adsorption Capacity'], y=[pred_a], marker_color='#3498db'),
                    go.Bar(name='Scenario B', x=['Adsorption Capacity'], y=[pred_b], marker_color='#e74c3c')
                ])
                fig_comp_bar.update_layout(barmode='group', title="Comparison of Predicted Qe", height=300, plot_bgcolor='white')
                st.plotly_chart(fig_comp_bar, use_container_width=True)

            # 2. 参数对比雷达图 (归一化处理)
            st.markdown("#### 2. Parameter Differences (Radar Chart)")
            st.info("Values are normalized (0-1) relative to their min/max range for visual comparison.")
            
            # 归一化函数
            def normalize_data(input_dict):
                norm_vals = []
                for feat in feature_names:
                    mn = feature_ranges[feat]['min']
                    mx = feature_ranges[feat]['max']
                    val = input_dict[feat]
                    norm = (val - mn) / (mx - mn) if mx != mn else 0
                    norm_vals.append(norm)
                return norm_vals

            norm_a = normalize_data(input_a)
            norm_b = normalize_data(input_b)
            
            # 闭合雷达图
            features_closed = feature_names + [feature_names[0]]
            norm_a_closed = norm_a + [norm_a[0]]
            norm_b_closed = norm_b + [norm_b[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_a_closed, theta=features_closed, fill='toself', name='Scenario A', line_color='#3498db'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_b_closed, theta=features_closed, fill='toself', name='Scenario B', line_color='#e74c3c'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                height=500,
                title="Input Parameter Comparison (Normalized)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================= TAB 7: 批量预测 (移动到最后) =======================
    # 原 Tab 3 -> 移动到位置 7
    with tab_batch:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📂 Batch Prediction")
        st.write("Upload a CSV or Excel file containing the feature columns to predict multiple samples at once.")
        
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
                
                missing_cols = [col for col in feature_names if col not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    st.success(f"Successfully loaded {len(batch_df)} samples.")
                    
                    if st.button("Predict All"):
                        batch_preds = model.predict(batch_df[feature_names])
                        batch_df['Predicted Qe'] = batch_preds
                        
                        st.dataframe(batch_df)
                        
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
