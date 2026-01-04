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

# SAC deps (safe import)
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    gym = None
    spaces = None
    SAC = None
    DummyVecEnv = None

class BiocharSACEnv(gym.Env):
    """
    用训练好的回归模型 model.predict() 作为环境动力学（surrogate env）。
    状态：16维特征(归一化0-1)
    动作：Top-K特征的连续增量（[-1,1]）
    目标：最大化 Qe
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        model,
        feature_names,
        feature_ranges,
        topk_idx,
        imp_vec,
        base_step=0.06,
        lam_change=0.05,
        lam_edge=0.20,
        max_steps=25,
        random_start=True,
        seed=42,
    ):
        super().__init__()
        self.model = model
        self.feature_names = feature_names
        self.ranges = feature_ranges
        self.topk_idx = np.array(topk_idx, dtype=int)
        self.imp = np.array(imp_vec, dtype=float)
        self.base_step = float(base_step)
        self.lam_change = float(lam_change)
        self.lam_edge = float(lam_edge)
        self.max_steps = int(max_steps)
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(int(seed))

        self.n = len(feature_names)
        self.k = len(topk_idx)

        # SB3 + gymnasium
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.k,), dtype=np.float32)

        # 重要性加权步长（Top-K）
        imp_topk = self.imp[self.topk_idx]
        imp_topk = np.maximum(imp_topk, 1e-12)
        self.imp_topk = imp_topk / (imp_topk.max() + 1e-12)
        self.step_vec = self.base_step * (0.3 + 0.7 * self.imp_topk)  # 重要特征更敢动

        self.t = 0
        self.x = None
        self.x_start = None

    def _norm(self, x):
        x01 = np.zeros_like(x, dtype=np.float32)
        for i, f in enumerate(self.feature_names):
            mn, mx = float(self.ranges[f]["min"]), float(self.ranges[f]["max"])
            x01[i] = (x[i] - mn) / (mx - mn + 1e-12)
        return np.clip(x01, 0.0, 1.0)

    def _denorm(self, x01):
        x = np.zeros_like(x01, dtype=np.float32)
        for i, f in enumerate(self.feature_names):
            mn, mx = float(self.ranges[f]["min"]), float(self.ranges[f]["max"])
            x[i] = mn + x01[i] * (mx - mn)
        return x

    @staticmethod
    def _edge_penalty(x01_topk):
        # 靠近0或1(5%以内)就算“贴边”
        return float(np.mean(np.minimum(x01_topk, 1.0 - x01_topk) < 0.05))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # 支持从外部指定起点（用于评估：从用户输入出发）
        x0 = None
        if options is not None:
            x0 = options.get("x0", None)

        if x0 is not None:
            self.x = np.array(x0, dtype=np.float32)
        else:
            if self.random_start:
                x_init = []
                for f in self.feature_names:
                    mn, mx = float(self.ranges[f]["min"]), float(self.ranges[f]["max"])
                    x_init.append(self.rng.uniform(mn, mx))
                self.x = np.array(x_init, dtype=np.float32)
            else:
                # 若不随机但没给 x0，就用默认值
                self.x = np.array([float(self.ranges[f]["default"]) for f in self.feature_names], dtype=np.float32)

        self.x_start = self.x.copy()
        obs = self._norm(self.x)
        return obs, {}

    def step(self, action):
        self.t += 1
        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

        x01 = self._norm(self.x)
        x01_next = x01.copy()

        # 只更新 Top-K
        x01_next[self.topk_idx] = np.clip(
            x01[self.topk_idx] + self.step_vec.astype(np.float32) * action,
            0.0,
            1.0,
        )

        x_next = self._denorm(x01_next)
        qe = float(self.model.predict(x_next.reshape(1, -1))[0])

        # 变化惩罚（相对起点）
        start01 = self._norm(self.x_start)
        delta = x01_next[self.topk_idx] - start01[self.topk_idx]
        change_pen = float(np.mean(delta**2))

        # 边界惩罚（避免一直冲 min/max）
        epen = self._edge_penalty(x01_next[self.topk_idx])

        reward = qe - self.lam_change * change_pen - self.lam_edge * epen

        self.x = x_next
        obs = self._norm(self.x)

        terminated = False
        truncated = (self.t >= self.max_steps)
        info = {"qe": qe, "change_pen": change_pen, "edge_pen": epen}
        return obs, reward, terminated, truncated, info

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
        
# ======================= TAB: RL Optimization (SAC) =======================
with tab_rl:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 RL Optimization (SAC) — Maximize Qe with Auto Top-K by Importance")

    if gym is None or SAC is None or DummyVecEnv is None:
        st.error("SAC dependencies not available. Please install: pip install gymnasium stable-baselines3 torch")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ---------- 1) Importance：优先用 Tab5 结果，否则自动算 ----------
    col_rl1, col_rl2, col_rl3 = st.columns([1, 1, 1])
    with col_rl1:
        k_top = st.slider("Top-K Features (auto selected)", 2, min(10, len(feature_names)), 5)
    with col_rl2:
        n_imp_samples = st.slider("Importance Samples", 300, 3000, 800, step=100)
    with col_rl3:
        imp_seed = st.number_input("Random Seed", value=42, step=1)

    perm_df = st.session_state.get("perm_df", None)
    if perm_df is None or not isinstance(perm_df, pd.DataFrame) or len(perm_df) == 0:
        with st.spinner("Computing feature importance (cached)..."):
            perm_df = compute_permutation_importance_cached(feature_names, feature_ranges, int(n_imp_samples), seed=int(imp_seed))
        st.session_state["perm_df"] = perm_df

    topk_features = perm_df["Feature"].head(int(k_top)).tolist()
    topk_idx = [feature_names.index(f) for f in topk_features]

    imp_vec = perm_df.set_index("Feature").loc[feature_names]["Importance"].values.astype(float)
    imp_vec = np.maximum(imp_vec, 1e-9)

    st.info(f"Auto selected Top-{k_top}: **{', '.join(topk_features)}**")

    # ---------- 2) 起点：优先用 Single Prediction 输入，否则用默认 ----------
    base_input = {}
    for idx, name in enumerate(feature_names):
        base_input[name] = st.session_state.get(f"input_{idx}", feature_ranges[name]["default"])
    x0 = np.array([base_input[n] for n in feature_names], dtype=float)

    # ---------- 3) SAC 训练参数 ----------
    st.markdown("#### ⚙️ SAC Settings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_steps = st.slider("Episode Length (max_steps)", 10, 60, 25)
    with c2:
        total_steps = st.slider("Training Timesteps", 5_000, 200_000, 50_000, step=5_000)
    with c3:
        base_step = st.slider("Base Step (normalized)", 0.01, 0.20, 0.06, step=0.01)
    with c4:
        lr = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3], value=3e-4)

    p1, p2, p3 = st.columns(3)
    with p1:
        lam_change = st.slider("Change Penalty λ", 0.0, 0.50, 0.05, step=0.01)
    with p2:
        lam_edge = st.slider("Edge Penalty β", 0.0, 1.00, 0.20, step=0.05)
    with p3:
        gamma = st.slider("Discount γ", 0.80, 0.999, 0.98, step=0.001)

    adv = st.expander("Advanced SAC Hyperparameters", expanded=False)
    with adv:
        buffer_size = st.slider("Replay Buffer Size", 50_000, 500_000, 200_000, step=50_000)
        batch_size = st.select_slider("Batch Size", options=[128, 256, 512], value=256)
        tau = st.slider("Target Smoothing τ", 0.001, 0.05, 0.02, step=0.001)

    # ---------- 4) 训练按钮 ----------
    run_train = st.button("🚀 Train SAC Agent", type="primary", use_container_width=True)

    if run_train:
        t0 = time.time()
        status = st.empty()
        progress = st.progress(0)

        # 训练环境：随机起点（让策略更鲁棒）
        def make_env():
            return BiocharSACEnv(
                model=model,
                feature_names=feature_names,
                feature_ranges=feature_ranges,
                topk_idx=topk_idx,
                imp_vec=imp_vec,
                base_step=base_step,
                lam_change=lam_change,
                lam_edge=lam_edge,
                max_steps=max_steps,
                random_start=True,
                seed=int(imp_seed),
            )

        vec_env = DummyVecEnv([make_env])

        agent = SAC(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=float(lr),
            buffer_size=int(buffer_size),
            batch_size=int(batch_size),
            gamma=float(gamma),
            tau=float(tau),
        )

        # 简单的进度显示：分段 learn
        chunks = 10
        per = max(1, int(total_steps // chunks))
        learned = 0
        for i in range(chunks):
            status.text(f"Training SAC... {learned}/{total_steps} timesteps")
            agent.learn(total_timesteps=per, reset_num_timesteps=False, progress_bar=False)
            learned += per
            progress.progress(int(((i + 1) / chunks) * 100))

        status.empty()
        progress.empty()

        # 把 agent 放到 session_state，避免 rerun 丢失
        st.session_state["sac_agent"] = agent
        st.session_state["sac_cfg"] = {
            "topk_features": topk_features,
            "topk_idx": topk_idx,
            "imp_seed": int(imp_seed),
            "base_step": float(base_step),
            "lam_change": float(lam_change),
            "lam_edge": float(lam_edge),
            "max_steps": int(max_steps),
        }

        st.success(f"✅ SAC training finished in {time.time()-t0:.2f}s")

    # ---------- 5) 评估/生成最优条件（从用户输入 x0 出发 rollout） ----------
    agent = st.session_state.get("sac_agent", None)
    cfg = st.session_state.get("sac_cfg", None)

    if agent is not None and cfg is not None:
        st.markdown("#### 🎯 Evaluate from Current Input (Rollout)")

        eval_runs = st.slider("Evaluation Rollouts (pick best)", 1, 30, 5)
        deterministic = st.checkbox("Deterministic Policy", value=True)

        if st.button("📈 Run Evaluation Rollouts", type="primary", use_container_width=True):
            env_eval = BiocharSACEnv(
                model=model,
                feature_names=feature_names,
                feature_ranges=feature_ranges,
                topk_idx=cfg["topk_idx"],
                imp_vec=imp_vec,
                base_step=cfg["base_step"],
                lam_change=cfg["lam_change"],
                lam_edge=cfg["lam_edge"],
                max_steps=cfg["max_steps"],
                random_start=False,
                seed=int(cfg["imp_seed"]),
            )

            best_qe = -1e18
            best_x = None
            best_trace = None
            best_step = 0

            for r in range(int(eval_runs)):
                obs, _ = env_eval.reset(options={"x0": x0})
                trace = []
                for t in range(cfg["max_steps"]):
                    action, _ = agent.predict(obs, deterministic=bool(deterministic))
                    obs, reward, term, trunc, info = env_eval.step(action)
                    trace.append(info["qe"])
                    if info["qe"] > best_qe:
                        best_qe = info["qe"]
                        best_x = env_eval.x.copy()
                        best_trace = trace.copy()
                        best_step = t
                    if trunc:
                        break

            st.success(f"✅ Best predicted Qe: **{best_qe:.4f} mg/g** (best step {best_step+1}/{cfg['max_steps']})")

            # 展示参数变化
            res_df = pd.DataFrame({
                "Feature": feature_names,
                "Start": x0,
                "Best": best_x,
                "Delta": (best_x - x0)
            })
            res_df["AbsDelta"] = res_df["Delta"].abs()
            show_df = res_df.sort_values("AbsDelta", ascending=False).drop(columns=["AbsDelta"])

            st.markdown("#### 🔎 Best Condition (sorted by |Δ|)")
            st.dataframe(show_df.style.format({"Start": "{:.4f}", "Best": "{:.4f}", "Delta": "{:.4f}"}), height=420)

            # 轨迹图
            if best_trace is not None:
                fig_trace = go.Figure()
                fig_trace.add_trace(go.Scatter(
                    x=list(range(1, len(best_trace)+1)),
                    y=best_trace,
                    mode="lines+markers",
                    name="Qe trace"
                ))
                fig_trace.update_layout(
                    title="Qe Improvement Along the Rollout (Best Trajectory)",
                    xaxis_title="Step",
                    yaxis_title="Predicted Qe (mg/g)",
                    height=420,
                    plot_bgcolor="white"
                )
                fig_trace.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
                fig_trace.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
                st.plotly_chart(fig_trace, use_container_width=True, theme=None)

            # 导出最佳条件
            out_df = pd.DataFrame([best_x], columns=feature_names)
            out_df["Predicted Qe"] = best_qe
            csv_out = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Best Condition (CSV)", csv_out, "sac_best_condition.csv", "text/csv", type="primary")
    else:
        st.warning("Train the SAC agent first, then evaluate rollouts to get the best condition.")

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
