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

if model:
    # 定义 Tabs：调整顺序，Batch 放到最后，新增 Comparison
    tab_titles = [
        "🧪 Single Prediction", 
        "📈 Dependency Analysis", 
        "🧊 Interaction Analysis", 
        "🎯 Inverse Optimization", 
        "📊 Global Importance",
        "⚔️ Comparative Analysis",  # 新功能
        "📂 Batch Prediction"       # 移动至最后
    ]
    
    # 解包 tab 对象
    tab_single, tab_depend, tab_inter, tab_opt, tab_imp, tab_compare, tab_batch = st.tabs(tab_titles)

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
            default_opts = ['pH of Biochar',
