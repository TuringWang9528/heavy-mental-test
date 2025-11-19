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
    tab1, tab2, tab3 = st.tabs(["🧪 Single Prediction", "📈 Sensitivity Analysis", "📂 Batch Prediction"])

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

# ======================= TAB 2: 灵敏度分析 (深度修复版) =======================
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Single Feature Sensitivity Analysis")
        
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            target_feature = st.selectbox("Select Feature", feature_names)
        with col_sel2:
            points = st.slider("Resolution", 10, 100, 40)

        # 改回按钮触发，因为这更符合你的习惯，且我们要排查是不是按钮逻辑的问题
        if st.button("Run Sensitivity Analysis", type="primary"):
            try:
                # --- 1. 准备输入数据 ---
                # 获取当前所有特征的默认值/输入值
                base_input_dict = {}
                for idx, name in enumerate(feature_names):
                    # 尝试从 session 获取，获取不到就用默认值
                    base_input_dict[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
                
                # 构造基础 DataFrame
                temp_df = pd.DataFrame([base_input_dict] * points)
                
                # 确保列顺序与训练时一致
                temp_df = temp_df[feature_names]

                # --- 2. 修改目标特征列 ---
                min_val = feature_ranges[target_feature]["min"]
                max_val = feature_ranges[target_feature]["max"]
                x_values = np.linspace(min_val, max_val, points)
                temp_df[target_feature] = x_values

                # --- 3. 预测 (关键步骤) ---
                y_pred = model.predict(temp_df)
                
                # 【诊断步骤 A】强制扁平化数据，防止 (N,1) 维度问题
                y_pred = y_pred.ravel() 
                
                # 【诊断步骤 B】检查是否有 NaN (空值)
                if np.isnan(y_pred).any():
                    st.error("⚠️ 错误：模型预测结果包含无效值 (NaN)。请检查输入特征范围是否合理。")
                    st.write("前5个预测值:", y_pred[:5])
                else:
                    # --- 4. 构建专门用于画图的 DataFrame ---
                    # Plotly 最喜欢这种格式，最不容易出错
                    plot_df = pd.DataFrame({
                        "x_axis": x_values,
                        "y_axis": y_pred
                    })

                    # --- 5. 打印数据预览 (Debug) ---
                    # 如果图还没出来，看这里有没有数字！
                    with st.expander("查看底层数据 (Debug Data)", expanded=False):
                        st.write(f"正在绘制 {target_feature} 的曲线，数据前5行：")
                        st.dataframe(plot_df.head())

                    # --- 6. 绘图 ---
                    fig = px.line(
                        plot_df, 
                        x="x_axis", 
                        y="y_axis", 
                        title=f"Sensitivity: {target_feature}",
                        labels={"x_axis": target_feature, "y_axis": "Predicted Qe (mg/g)"}
                    )
                    
                    # 强制设置线条颜色和粗细，防止“隐形”
                    fig.update_traces(line=dict(color='#3498db', width=4), mode='lines+markers')
                    
                    # 设置背景色，防止白线画在白底上
                    fig.update_layout(plot_bgcolor='#f4f4f4', height=450)
                    
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"运行出错: {str(e)}")
                st.write("详情:", e)
        
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
