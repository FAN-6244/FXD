"""
smart_warning_panel_deploy.py
水质净化厂智能预警与调控决策系统 v6.0
功能：四种数据输入模式（手动/文件/API/自动模拟）
优化：趋势图逻辑清晰、界面美观、适合领导汇报
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 北京时间时区
# ==========================================
BEIJING_TZ = timezone(timedelta(hours=8))

st.set_page_config(
    page_title="水质净化厂智能预警与调控决策系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 自定义CSS —— 专业、清爽
# ==========================================
st.markdown("""
<style>
    /* 全局字体 */
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #0A2E4A;
        padding: 8px 0 12px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 18px;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #0A2E4A;
        margin: 18px 0 12px 0;
        padding-left: 10px;
        border-left: 5px solid #2E86AB;
    }
    .status-metric {
        text-align: center;
        background: #F8F9FA;
        border-radius: 10px;
        padding: 8px 12px;
        border: 1px solid #E5E9F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .status-metric .label { font-size: 13px; color: #6C7A89; font-weight: 500; }
    .status-metric .value { font-size: 18px; font-weight: 700; color: #0A2E4A; }
    .status-metric .value-normal { color: #1B7A4A; }
    .status-metric .value-critical { color: #C0392B; }
    .water-card-in {
        background: #F0F6FC;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #D6E4F0;
        margin-bottom: 8px;
    }
    .water-card-out {
        background: #F0FCF5;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #C8E6D9;
        margin-bottom: 8px;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 8px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 4px;
        border-left: 4px solid #2E86AB;
    }
    .metric-card .label { font-size: 13px; color: #6C7A89; font-weight: 500; }
    .metric-card .value { font-size: 20px; font-weight: 700; color: #0A2E4A; }
    .metric-card .sub { font-size: 12px; color: #999; }
    .limit-ref {
        font-size: 12px;
        color: #888;
        background: #F0F0F0;
        padding: 1px 10px;
        border-radius: 12px;
        display: inline-block;
    }
    .channel-item {
        flex: 1;
        background: white;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 4px solid #ccc;
    }
    .channel-item .ch-name { font-weight: 600; font-size: 15px; }
    .channel-item .ch-value { font-size: 22px; font-weight: 700; margin: 4px 0; }
    .channel-item .ch-desc { font-size: 12px; color: #666; }
    .channel-fast { border-top-color: #27AE60; }
    .channel-slow { border-top-color: #F39C12; }
    .channel-special { border-top-color: #E74C3C; }
    .timeline-step {
        display: flex;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #F0F0F0;
    }
    .timeline-step:last-child { border-bottom: none; }
    .timeline-time { min-width: 60px; font-weight: 700; font-size: 15px; color: #0A2E4A; }
    .timeline-action { font-size: 14px; color: #333; padding-left: 10px; }
    .calibration-success {
        background: #E8F5E9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #27AE60;
    }
    .data-status-realtime {
        background: #E3F2FD;
        border-radius: 10px;
        padding: 8px 16px;
        border: 1px solid #90CAF9;
        display: inline-block;
        font-size: 14px;
        color: #1565C0;
    }
    /* 诊断卡片 */
    .diag-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #888;
    }
    .diag-critical { background: #FDEDEC; border-left-color: #C0392B; }
    .diag-warning { background: #FEF9E7; border-left-color: #F39C12; }
    .diag-info { background: #EBF5FB; border-left-color: #2E86AB; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 设计标准和记忆长度
# ==========================================
DESIGN_LIMITS = {
    'COD': {'value': 30, 'unit': 'mg/L'},
    'NH3-N': {'value': 1.5, 'unit': 'mg/L'},
    'TP': {'value': 0.3, 'unit': 'mg/L'},
    'SS': {'value': 10, 'unit': 'mg/L'}
}

MEMORY = {
    'COD': {'hours': 8, 'channel': '快速', 'freq': '3-4h'},
    'NH3-N': {'hours': 6, 'channel': '快速', 'freq': '3-4h'},
    'TP': {'hours': 22, 'channel': '慢速', 'freq': '8-12h'},
    'SS': {'hours': None, 'channel': '特殊', 'freq': '实时报警'}
}

# ==========================================
# 数据缓存管理
# ==========================================
class DataBuffer:
    def __init__(self, max_hours=48):
        self.max_hours = max_hours
        self.data = []  # 每个元素: {timestamp, inlet, outlet_real, outlet_pred}

    def add_data(self, timestamp, inlet, outlet_real, outlet_pred):
        self.data.append({
            'timestamp': timestamp,
            'inlet': inlet.copy(),
            'outlet_real': outlet_real.copy() if outlet_real is not None else None,
            'outlet_pred': outlet_pred.copy() if outlet_pred is not None else None
        })
        cutoff = datetime.now(BEIJING_TZ) - timedelta(hours=self.max_hours)
        self.data = [d for d in self.data if d['timestamp'] >= cutoff]

    def get_recent(self, hours=24):
        cutoff = datetime.now(BEIJING_TZ) - timedelta(hours=hours)
        return [d for d in self.data if d['timestamp'] >= cutoff]

# ==========================================
# 加载预训练模型
# ==========================================
@st.cache_resource
def load_base_models():
    status_placeholder = st.empty()
    status_placeholder.info("🔄 正在加载预训练模型...")
    try:
        with open('model_cache/models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('model_cache/nh3_optimized_model.pkl', 'rb') as f:
            nh3_model = pickle.load(f)
        with open('model_cache/nh3_optimized_scaler.pkl', 'rb') as f:
            nh3_scaler = pickle.load(f)
        with open('model_cache/nh3_optimized_feature_cols.pkl', 'rb') as f:
            nh3_feature_cols = pickle.load(f)
        with open('model_cache/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_cache/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        status_placeholder.success("✅ 模型加载成功")
        return models, feature_cols, scaler, nh3_model, nh3_scaler, nh3_feature_cols
    except FileNotFoundError as e:
        status_placeholder.error(f"❌ 模型文件不存在: {e}")
        st.stop()

models, feature_cols, scaler, nh3_model, nh3_scaler, nh3_feature_cols = load_base_models()

# ==========================================
# 初始化 session_state
# ==========================================
if 'model_cod_tunable' not in st.session_state:
    st.session_state.model_cod_tunable = models['COD_out']
if 'model_tp_tunable' not in st.session_state:
    st.session_state.model_tp_tunable = models['TP_out']
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = DataBuffer()
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0
if 'auto_mode_running' not in st.session_state:
    st.session_state.auto_mode_running = False
if 'simulation_counter' not in st.session_state:
    st.session_state.simulation_counter = 0

st.markdown('<div class="main-title">🏭 水质净化厂智能预警与调控决策系统</div>', unsafe_allow_html=True)

# ==========================================
# 状态栏
# ==========================================
col_s1, col_s2, col_s3 = st.columns(3)
status_placeholder = col_s1.empty()

with col_s2:
    beijing_now = datetime.now(BEIJING_TZ)
    st.markdown(f"""
    <div class="status-metric">
        <div class="label">⏱️ 当前时间</div>
        <div class="value">{beijing_now.strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="status-metric">
        <div class="label">📋 出水设计标准</div>
        <div class="value">准Ⅳ类</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 预测函数
# ==========================================
def predict_cod_tp(input_data, target):
    if input_data is None:
        return None
    try:
        vec = np.array([input_data[col].values[0] if col in input_data.columns else 0 for col in feature_cols]).reshape(1, -1)
        vec_scaled = scaler.transform(vec)
        if target == 'COD_out':
            return max(0, st.session_state.model_cod_tunable.predict(vec_scaled)[0])
        elif target == 'TP_out':
            return max(0, st.session_state.model_tp_tunable.predict(vec_scaled)[0])
    except Exception:
        return None

def predict_nh3_optimized(input_data):
    if input_data is None:
        return None
    try:
        vec = np.array([input_data[col].values[0] if col in input_data.columns else 0 for col in nh3_feature_cols]).reshape(1, -1)
        vec_scaled = nh3_scaler.transform(vec)
        pred_log = nh3_model.predict(vec_scaled)[0]
        return max(0, np.expm1(pred_log))
    except Exception:
        return None

def build_input_with_lags(cod, nh3, tp, ss, flow, pac, carbon, mlss, do):
    data = pd.DataFrame({
        'COD_load': [cod * flow / 1000],
        'NH3_load': [nh3 * flow / 1000],
        'TP_load': [tp * flow / 1000],
        '流量': [flow],
        'PAC': [pac],
        '碳源': [carbon],
        'MLSS': [mlss],
        'DO': [do]
    })
    for i in range(1, 49):
        decay = 1 - (i / 48) * 0.3
        data[f'COD_load_lag{i}'] = cod * flow / 1000 * decay
        data[f'NH3_load_lag{i}'] = nh3 * flow / 1000 * decay
        data[f'TP_load_lag{i}'] = tp * flow / 1000 * decay
        data[f'流量_lag{i}'] = flow * decay
    return data

# ==========================================
# 生成模拟实时数据
# ==========================================
def generate_simulated_data():
    base_cod = 200 + np.random.normal(0, 30)
    base_nh3 = 20 + np.random.normal(0, 3)
    base_tp = 3.0 + np.random.normal(0, 0.4)
    base_ss = 150 + np.random.normal(0, 20)
    base_flow = 10000 + np.random.normal(0, 500)
    return {
        'COD': max(0, base_cod),
        'NH3-N': max(0, base_nh3),
        'TP': max(0, base_tp),
        'SS': max(0, base_ss),
        '流量': max(0, base_flow),
        'PAC': 30 + np.random.normal(0, 2),
        '碳源': 50 + np.random.normal(0, 3),
        'MLSS': 4000 + np.random.normal(0, 200),
        'DO': 2.0 + np.random.normal(0, 0.2)
    }

def simulate_outlet(inlet):
    return {
        'COD': 8 + np.random.normal(0, 0.8) + inlet['COD'] * 0.01,
        'NH3-N': 0.05 + np.random.normal(0, 0.01) + inlet['NH3-N'] * 0.003,
        'TP': 0.10 + np.random.normal(0, 0.01) + inlet['TP'] * 0.01,
        'SS': 3 + np.random.normal(0, 0.5) + inlet['SS'] * 0.005
    }

# ==========================================
# 校准函数
# ==========================================
def calibrate_model(target, X_new, y_real):
    if target == 'COD_out':
        current_model = st.session_state.model_cod_tunable
    elif target == 'TP_out':
        current_model = st.session_state.model_tp_tunable
    else:
        return None, "不支持的指标"
    try:
        X_scaled = scaler.transform(X_new.reshape(1, -1))
        dtrain = xgb.DMatrix(X_scaled, label=np.array([y_real]))
        params = {'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42}
        new_model = xgb.train(
            params=params,
            dtrain=dtrain,
            xgb_model=current_model,
            num_boost_round=10,
            verbose_eval=False
        )
        if target == 'COD_out':
            st.session_state.model_cod_tunable = new_model
        elif target == 'TP_out':
            st.session_state.model_tp_tunable = new_model
        st.session_state.calibration_count += 1
        return new_model, f"✅ {target} 微调成功"
    except Exception as e:
        return None, f"❌ 失败: {str(e)}"

# ==========================================
# 诊断函数（精简版，保留核心）
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []
    # 进水COD异常
    if inlet['COD'] > 500:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500）',
            'reasons': ['工业废水偷排', '管网沉积物冲刷', '消化液回流'],
            'actions': ['增加碳源30-40%', '提高DO至3.0-3.5', '降低进水量15-20%']
        })
    elif inlet['COD'] > 400:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500）',
            'reasons': ['工业废水间歇排放', '管网沉积物释放'],
            'actions': ['增加碳源20%', '提高DO至2.5-3.0']
        })
    # 进水NH3-N
    if inlet['NH3-N'] > 45:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45）',
            'reasons': ['工业废水偷排', '消化液回流'],
            'actions': ['提高DO至3.5-4.0', '补充NaHCO₃ 80-100', '延长SRT']
        })
    elif inlet['NH3-N'] > 35:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45）',
            'reasons': ['上游氨氮升高', '硝化菌受抑制'],
            'actions': ['提高DO至3.0-3.5', '补充碱度50-80']
        })
    # 进水TP
    if inlet['TP'] > 7.0:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7）',
            'reasons': ['工业废水偷排', '污泥厌氧释磷'],
            'actions': ['增加PAC 40-50%', '调节pH 6.5-7.5', '增加排泥']
        })
    elif inlet['TP'] > 5.0:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5-7）',
            'reasons': ['上游含磷波动', 'PAC不足'],
            'actions': ['增加PAC 20-30%', '检查pH']
        })
    # 出水超标
    if outlet['COD'] > DESIGN_LIMITS['COD']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['COD'] > 45 else 'warning',
            'indicator': '出水COD',
            'current': f"{outlet['COD']:.1f} mg/L",
            'title': f"{'🚨' if outlet['COD'] > 45 else '⚠️'} 出水COD超标",
            'reasons': [f'进水负荷过高', f'DO不足（{do:.1f}）', '污泥老化'],
            'actions': [f'增加碳源→{int(carbon*1.25)}', f'提高DO至2.5-3.0']
        })
    if outlet['NH3-N'] > DESIGN_LIMITS['NH3-N']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['NH3-N'] > 3.0 else 'warning',
            'indicator': '出水NH₃-N',
            'current': f"{outlet['NH3-N']:.2f} mg/L",
            'title': f"{'🚨' if outlet['NH3-N'] > 3.0 else '⚠️'} 出水NH₃-N超标",
            'reasons': [f'DO不足（{do:.1f}）', '碱度不足', 'SRT太短'],
            'actions': ['提高DO至3.0-3.5', '补充NaHCO₃', '延长SRT']
        })
    if outlet['TP'] > DESIGN_LIMITS['TP']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['TP'] > 0.6 else 'warning',
            'indicator': '出水TP',
            'current': f"{outlet['TP']:.3f} mg/L",
            'title': f"{'🚨' if outlet['TP'] > 0.6 else '⚠️'} 出水TP超标",
            'reasons': [f'PAC不足（{pac:.0f}）', 'pH不适宜'],
            'actions': [f'增加PAC→{int(pac*1.4)}', '调节pH 6.5-7.5']
        })
    if outlet['SS'] > DESIGN_LIMITS['SS']['value']:
        diagnoses.append({
            'level': 'warning',
            'indicator': '出水SS',
            'current': f"{outlet['SS']:.1f} mg/L",
            'title': '⚠️ 出水SS超标',
            'reasons': ['表面负荷过高', 'SVI升高', '排泥不足'],
            'actions': ['增加排泥20%', '投加PAM', '降低进水量10%']
        })
    # 运行参数
    if do < 0.8:
        diagnoses.append({
            'level': 'critical',
            'indicator': 'DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 DO严重不足（<0.8）',
            'reasons': ['曝气故障', '负荷突增'],
            'actions': ['检查曝气设备', '加大风量20-30%']
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning',
            'indicator': 'DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ DO偏低（<1.5）',
            'reasons': ['曝气量不足', '负荷增加'],
            'actions': ['增加曝气10-20%', '监测DO趋势']
        })
    if mlss < 2500:
        diagnoses.append({
            'level': 'warning',
            'indicator': 'MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ MLSS偏低（<2500）',
            'reasons': ['污泥流失', '负荷过低'],
            'actions': ['减少排泥', '增加回流量']
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info',
            'indicator': 'MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ MLSS偏高（>6000）',
            'reasons': ['排泥不足', '泥层厚'],
            'actions': ['增加排泥', '检查泥位']
        })
    return diagnoses

# ==========================================
# 侧边栏：四种数据输入模式
# ==========================================
st.sidebar.markdown("## 📊 数据输入模式")
input_mode_global = st.sidebar.radio(
    "选择数据模式",
    ["✏️ 手动输入", "📁 文件上传", "📡 API接入", "🔄 自动实时（模拟）"],
    index=0
)

# 定义固定表头
REQUIRED_COLS = ['COD', 'NH3-N', 'TP', 'SS', '流量', 'PAC', '碳源', 'MLSS', 'DO']

# ---- 手动输入 ----
if input_mode_global == "✏️ 手动输入":
    st.sidebar.markdown("### 进水实测")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        cod_in = st.number_input("COD (mg/L)", min_value=0.0, value=200.0, key="manual_cod")
        nh3_in = st.number_input("NH₃-N (mg/L)", min_value=0.0, value=20.0, key="manual_nh3")
    with c2:
        tp_in = st.number_input("TP (mg/L)", min_value=0.0, value=3.0, key="manual_tp")
        ss_in = st.number_input("SS (mg/L)", min_value=0.0, value=150.0, key="manual_ss")
    flow_in = st.sidebar.number_input("流量 (m³/h)", min_value=0.0, value=10000.0, key="manual_flow")
    st.sidebar.markdown("### 运行参数")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        pac = st.number_input("PAC (mg/L)", min_value=0.0, value=30.0, key="manual_pac")
        carbon = st.number_input("碳源 (mg/L)", min_value=0.0, value=50.0, key="manual_carbon")
    with c4:
        mlss = st.number_input("MLSS (mg/L)", min_value=0.0, value=4000.0, key="manual_mlss")
        do = st.number_input("DO (mg/L)", min_value=0.0, value=2.0, key="manual_do")
    input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
    st.sidebar.info("💡 手动模式：修改参数后自动更新预测")

# ---- 文件上传 ----
elif input_mode_global == "📁 文件上传":
    st.sidebar.markdown("### 📁 上传数据文件")
    st.sidebar.caption("请上传包含以下列的 Excel/CSV 文件：")
    st.sidebar.code("COD, NH3-N, TP, SS, 流量, PAC, 碳源, MLSS, DO", language='text')
    if st.sidebar.button("📥 下载空模板 (Excel)"):
        template_df = pd.DataFrame(columns=REQUIRED_COLS)
        template_df.loc[0] = [200, 20, 3.0, 150, 10000, 30, 50, 4000, 2.0]
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='模板')
        st.sidebar.download_button(
            label="📥 下载模板.xlsx",
            data=output.getvalue(),
            file_name="进水数据模板.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    uploaded_file = st.sidebar.file_uploader("选择文件", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            missing = set(REQUIRED_COLS) - set(df_upload.columns)
            if missing:
                st.sidebar.error(f"❌ 缺少列：{missing}")
                input_data = None
            else:
                row = df_upload.iloc[0]
                cod_in = row['COD']; nh3_in = row['NH3-N']; tp_in = row['TP']; ss_in = row['SS']
                flow_in = row['流量']; pac = row['PAC']; carbon = row['碳源']; mlss = row['MLSS']; do = row['DO']
                input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
                st.sidebar.success(f"✅ 加载成功 (共 {len(df_upload)} 行)")
        except Exception as e:
            st.sidebar.error(f"❌ 解析失败：{e}")
            input_data = None
    else:
        input_data = None

# ---- API接入 ----
elif input_mode_global == "📡 API接入":
    st.sidebar.markdown("### 📡 API 实时数据")
    api_url = st.sidebar.text_input("API地址", value="http://localhost:8080/api/data", key="api_url")
    api_key = st.sidebar.text_input("API Key", type="password", key="api_key")
    if st.sidebar.button("🔄 获取数据"):
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                cod_in = data.get('COD', 0); nh3_in = data.get('NH3-N', 0); tp_in = data.get('TP', 0)
                ss_in = data.get('SS', 0); flow_in = data.get('流量', 0); pac = data.get('PAC', 0)
                carbon = data.get('碳源', 0); mlss = data.get('MLSS', 0); do = data.get('DO', 0)
                input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
                st.sidebar.success("✅ 数据获取成功")
            else:
                st.sidebar.error(f"❌ {resp.status_code}")
                input_data = None
        except Exception as e:
            st.sidebar.error(f"❌ 连接失败：{e}")
            input_data = None
    else:
        input_data = None

# ---- 自动实时（模拟） ----
else:
    st.sidebar.markdown("### 🔄 自动实时数据")
    st.sidebar.info("🔄 每5秒生成一组模拟数据")
    if st.sidebar.button("▶️ 启动实时流"):
        st.session_state.auto_mode_running = True
        st.sidebar.success("✅ 已启动")
    if st.sidebar.button("⏹️ 停止实时流"):
        st.session_state.auto_mode_running = False
        st.sidebar.info("⏹️ 已停止")
    simulated_inlet = generate_simulated_data()
    cod_in = simulated_inlet['COD']; nh3_in = simulated_inlet['NH3-N']; tp_in = simulated_inlet['TP']
    ss_in = simulated_inlet['SS']; flow_in = simulated_inlet['流量']; pac = simulated_inlet['PAC']
    carbon = simulated_inlet['碳源']; mlss = simulated_inlet['MLSS']; do = simulated_inlet['DO']
    input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
    simulated_outlet = simulate_outlet(simulated_inlet)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="data-status-realtime">
        📊 数据组：{st.session_state.simulation_counter + 1}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 主界面
# ==========================================
if input_data is not None:
    # ---- 预测 ----
    pred_cod = predict_cod_tp(input_data, 'COD_out')
    pred_tp = predict_cod_tp(input_data, 'TP_out')
    pred_nh3 = predict_nh3_optimized(input_data)
    pred_ss = max(0, 5 + np.random.normal(0, 0.5))

    inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'SS': ss_in, '流量': flow_in}
    outlet_pred = {'COD': pred_cod if pred_cod else 0,
                   'NH3-N': pred_nh3 if pred_nh3 else 0,
                   'TP': pred_tp if pred_tp else 0,
                   'SS': pred_ss}

    # ---- 自动模式处理：生成实测值并缓存 ----
    if input_mode_global == "🔄 自动实时（模拟）" and st.session_state.auto_mode_running:
        # 模拟实测值（假设当前时刻有实测结果）
        real_outlet = {
            'COD': max(0, simulated_outlet['COD'] + np.random.normal(0, 0.3)),
            'NH3-N': max(0, simulated_outlet['NH3-N'] + np.random.normal(0, 0.005)),
            'TP': max(0, simulated_outlet['TP'] + np.random.normal(0, 0.005)),
            'SS': max(0, simulated_outlet['SS'] + np.random.normal(0, 0.2))
        }
        # 自动微调
        vec = np.array([input_data[col].values[0] if col in input_data.columns else 0 for col in feature_cols])
        if st.session_state.simulation_counter % 5 == 0 and st.session_state.simulation_counter > 0:
            if real_outlet['COD'] > 0:
                calibrate_model('COD_out', vec, real_outlet['COD'])
            if real_outlet['TP'] > 0:
                calibrate_model('TP_out', vec, real_outlet['TP'])
        # 存入缓存
        st.session_state.data_buffer.add_data(
            timestamp=datetime.now(BEIJING_TZ),
            inlet=inlet,
            outlet_real=real_outlet,
            outlet_pred=outlet_pred
        )
        st.session_state.simulation_counter += 1
        outlet_display = real_outlet
        outlet_label = "实测"
        st.info(f"🔄 已接收 {st.session_state.simulation_counter} 组 | 微调 {st.session_state.calibration_count} 次")
    else:
        # 手动/文件/API模式：只存预测值，不存实测值
        st.session_state.data_buffer.add_data(
            timestamp=datetime.now(BEIJING_TZ),
            inlet=inlet,
            outlet_real=None,
            outlet_pred=outlet_pred
        )
        outlet_display = outlet_pred
        outlet_label = "预测"

    # ---- 更新状态 ----
    has_abnormal = any(outlet_pred.get(k, 0) > DESIGN_LIMITS[k]['value'] for k in ['COD', 'NH3-N', 'TP', 'SS'])
    has_abnormal = has_abnormal or inlet['COD'] > 400 or inlet['NH3-N'] > 35 or inlet['TP'] > 5
    status_text = "异常" if has_abnormal else "正常"
    status_color = "value-critical" if has_abnormal else "value-normal"
    with status_placeholder:
        st.markdown(f"""
        <div class="status-metric">
            <div class="label">📊 系统状态</div>
            <div class="value {status_color}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 进出水水质卡片 ----
    st.markdown('<div class="section-header">📊 进出水水质实时监测</div>', unsafe_allow_html=True)
    st.caption(f"📌 设计标准：COD≤{DESIGN_LIMITS['COD']['value']} | NH₃-N≤{DESIGN_LIMITS['NH3-N']['value']} | TP≤{DESIGN_LIMITS['TP']['value']} | SS≤{DESIGN_LIMITS['SS']['value']} mg/L")

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.markdown('<div class="water-card-in"><div style="font-size:16px; font-weight:600; color:#0A2E4A;">🔵 进水水质（实测）</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("COD", f"{inlet['COD']:.0f} mg/L")
            st.metric("NH₃-N", f"{inlet['NH3-N']:.1f} mg/L")
            st.metric("TP", f"{inlet['TP']:.2f} mg/L")
        with c2:
            st.metric("SS", f"{inlet['SS']:.0f} mg/L")
            st.metric("流量", f"{inlet['流量']:.0f} m³/h")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<div class="water-card-out"><div style="font-size:16px; font-weight:600; color:#1A5C3A;">🟢 出水水质（{outlet_label}）</div>', unsafe_allow_html=True)
        cod_ok = outlet_display['COD'] <= DESIGN_LIMITS['COD']['value']
        nh3_ok = outlet_display['NH3-N'] <= DESIGN_LIMITS['NH3-N']['value']
        tp_ok = outlet_display['TP'] <= DESIGN_LIMITS['TP']['value']
        ss_ok = outlet_display['SS'] <= DESIGN_LIMITS['SS']['value']
        c3, c4 = st.columns(2)
        with c3:
            st.metric("COD", f"{outlet_display['COD']:.1f} mg/L", delta="达标" if cod_ok else f"超标{outlet_display['COD']-DESIGN_LIMITS['COD']['value']:.1f}", delta_color="normal" if cod_ok else "inverse")
            st.metric("NH₃-N", f"{outlet_display['NH3-N']:.2f} mg/L", delta="达标" if nh3_ok else f"超标{outlet_display['NH3-N']-DESIGN_LIMITS['NH3-N']['value']:.2f}", delta_color="normal" if nh3_ok else "inverse")
        with c4:
            st.metric("TP", f"{outlet_display['TP']:.3f} mg/L", delta="达标" if tp_ok else f"超标{outlet_display['TP']-DESIGN_LIMITS['TP']['value']:.3f}", delta_color="normal" if tp_ok else "inverse")
            st.metric("SS", f"{outlet_display['SS']:.1f} mg/L", delta="达标" if ss_ok else f"超标{outlet_display['SS']-DESIGN_LIMITS['SS']['value']:.1f}", delta_color="normal" if ss_ok else "inverse")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 趋势图（优化逻辑） ----
    st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
    st.caption("🟦 实线 = 实测值 | 虚线 = 预测值 | 🟥进水COD 🟧进水NH₃-N 🟪进水TP")

    recent = st.session_state.data_buffer.get_recent(24)
    if len(recent) > 1:
        df_trend = pd.DataFrame([{
            'timestamp': d['timestamp'],
            'inlet_COD': d['inlet']['COD'],
            'inlet_NH3': d['inlet']['NH3-N'],
            'inlet_TP': d['inlet']['TP'],
            'outlet_COD_real': d['outlet_real']['COD'] if d['outlet_real'] else None,
            'outlet_COD_pred': d['outlet_pred']['COD'] if d['outlet_pred'] else None,
            'outlet_NH3_real': d['outlet_real']['NH3-N'] if d['outlet_real'] else None,
            'outlet_NH3_pred': d['outlet_pred']['NH3-N'] if d['outlet_pred'] else None,
            'outlet_TP_real': d['outlet_real']['TP'] if d['outlet_real'] else None,
            'outlet_TP_pred': d['outlet_pred']['TP'] if d['outlet_pred'] else None,
        } for d in recent])

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=('COD', 'NH₃-N', 'TP'))

        # COD
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_COD'],
                                name='进水COD', line=dict(color='#E74C3C', width=2, dash='solid')), row=1, col=1)
        mask_real = df_trend['outlet_COD_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real]['outlet_COD_real'],
                                    name='出水COD_实测', line=dict(color='#2E86AB', width=2.5, dash='solid')), row=1, col=1)
        mask_pred = df_trend['outlet_COD_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred]['outlet_COD_pred'],
                                    name='出水COD_预测', line=dict(color='#2E86AB', width=2, dash='dot')), row=1, col=1)
        fig.add_hline(y=DESIGN_LIMITS['COD']['value'], line_dash="dash", line_color="red", row=1, col=1)

        # NH3-N
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_NH3'],
                                name='进水NH₃-N', line=dict(color='#F39C12', width=2, dash='solid')), row=2, col=1)
        mask_real = df_trend['outlet_NH3_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real]['outlet_NH3_real'],
                                    name='出水NH₃-N_实测', line=dict(color='#27AE60', width=2.5, dash='solid')), row=2, col=1)
        mask_pred = df_trend['outlet_NH3_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred]['outlet_NH3_pred'],
                                    name='出水NH₃-N_预测', line=dict(color='#27AE60', width=2, dash='dot')), row=2, col=1)
        fig.add_hline(y=DESIGN_LIMITS['NH3-N']['value'], line_dash="dash", line_color="red", row=2, col=1)

        # TP
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_TP'],
                                name='进水TP', line=dict(color='#8E44AD', width=2, dash='solid')), row=3, col=1)
        mask_real = df_trend['outlet_TP_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real]['outlet_TP_real'],
                                    name='出水TP_实测', line=dict(color='#F39C12', width=2.5, dash='solid')), row=3, col=1)
        mask_pred = df_trend['outlet_TP_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred]['outlet_TP_pred'],
                                    name='出水TP_预测', line=dict(color='#F39C12', width=2, dash='dot')), row=3, col=1)
        fig.add_hline(y=DESIGN_LIMITS['TP']['value'], line_dash="dash", line_color="red", row=3, col=1)

        fig.update_layout(height=450, showlegend=True, hovermode='x unified')
        fig.update_xaxes(title_text="时间", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 数据收集中…… 请等待至少2个时间点")

    # ---- 记忆长度与分频调控 ----
    st.markdown('<div class="section-header">🧠 记忆长度与分频调控策略</div>', unsafe_allow_html=True)
    st.caption("💡 基于历史数据学习的各污染物响应时间，指导调控频率与等待窗口")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="channel-item channel-fast">
            <div class="ch-name">⚡ 快速通道</div>
            <div class="ch-value" style="color:#27AE60;">6-8h</div>
            <div class="ch-desc">NH₃-N (6h) · COD (8h)</div>
            <div class="ch-desc">更新频率: 3-4小时</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="channel-item channel-slow">
            <div class="ch-name">🐢 慢速通道</div>
            <div class="ch-value" style="color:#F39C12;">22h</div>
            <div class="ch-desc">TP (≈22h)</div>
            <div class="ch-desc">更新频率: 8-12小时</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="channel-item channel-special">
            <div class="ch-name">🔴 特殊通道</div>
            <div class="ch-value" style="color:#E74C3C;">不稳定</div>
            <div class="ch-desc">SS — 实时阈值报警</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 时序决策建议 ----
    st.markdown('<div class="section-header">⏱️ 时序决策建议（具体操作）</div>', unsafe_allow_html=True)
    indicator = st.selectbox("选择指标", ['COD', 'NH3-N', 'TP', 'SS'])
    mem = MEMORY[indicator]['hours']
    cur = outlet_display[indicator]
    lim = DESIGN_LIMITS[indicator]['value']

    if mem:
        steps = {
            'COD': [(0,"🚨 记录异常，启动应急"),(2,"📞 通知值班长"),(4,"⚙️ 增加碳源20%"),(6,"🔍 检查DO"),(8,"📊 评估效果"),(12,"✅ 确认达标")],
            'NH3-N': [(0,"🚨 启动应急"),(2,"📞 准备碱度"),(3,"⚙️ 提高DO至3.0"),(5,"🔍 补充碱度"),(6,"📊 评估"),(9,"✅ 确认达标")],
            'TP': [(0,"🚨 启动应急"),(4,"📞 确认PAC"),(8,"⚙️ 增加PAC30%"),(14,"🔍 检查pH"),(22,"📊 评估"),(33,"✅ 确认达标")],
            'SS': [(0,"🚨 SS超标"),(1,"📞 检查刮泥机"),(2,"⚙️ 增加排泥20%"),(3,"🔍 检查SVI"),(4,"📊 评估"),(6,"✅ 确认达标")]
        }
        st.markdown('<div style="background:#FAFBFC;border-radius:8px;padding:12px 16px;border:1px solid #E8ECF0;">', unsafe_allow_html=True)
        st.markdown(f"**📋 {indicator}：{cur:.2f} / {lim} mg/L**")
        st.markdown("---")
        for t, action in steps.get(indicator, []):
            st.markdown(f"""
            <div class="timeline-step">
                <div class="timeline-time">⏱️ {t}h</div>
                <div class="timeline-action">{action}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 异常诊断 ----
    st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
    st.caption("💡 基于当前工况和同类型A²/O工艺经验库分析")
    diagnoses = diagnose_system(inlet, outlet_display, pac, carbon, mlss, do)
    if diagnoses:
        for d in diagnoses:
            cls = "diag-critical" if d['level']=='critical' else ("diag-warning" if d['level']=='warning' else "diag-info")
            with st.expander(f"{d['title']}（当前值：{d['current']}）", expanded=(d['level']=='critical')):
                col_r, col_a = st.columns(2)
                with col_r:
                    st.markdown("**🔍 可能原因**")
                    for r in d['reasons']:
                        st.markdown(f"- {r}")
                with col_a:
                    st.markdown("**💡 优化措施**")
                    for a in d['actions']:
                        st.markdown(f"- {a}")
    else:
        st.success("✅ 系统运行正常，未检测到异常")
        st.info("📋 建议：保持当前运行参数，定期巡检")

else:
    with status_placeholder:
        st.markdown("""
        <div class="status-metric">
            <div class="label">📊 系统状态</div>
            <div class="value" style="color:#888;">等待数据</div>
        </div>
        """, unsafe_allow_html=True)
    st.info("👈 请从侧边栏选择数据输入方式并提交数据")

st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 v6.0 | 四种数据模式 | 实测/预测趋势区分 | 更新时间：{beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
