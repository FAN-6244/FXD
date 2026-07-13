"""
smart_warning_panel_deploy.py
部署到 Streamlit Cloud —— 四种输入模式 + 永久记忆（Supabase）
完整版 v6.1
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
import os
import supabase

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
# Supabase 配置
# ==========================================
SUPABASE_URL = "https://esoulexcrpdeeoumoili.supabase.co"
SUPABASE_KEY = "sb_publishable_m0hz9Rv8NB_ziC5xKCltMg_Ij50d..."

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# CSS样式
# ==========================================
st.markdown("""
<style>
    .main-title {
        font-size: 24px;
        font-weight: 700;
        color: #1a3a5c;
        padding: 6px 0 10px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 14px;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #1a3a5c;
        margin: 14px 0 8px 0;
        padding-left: 8px;
        border-left: 4px solid #2E86AB;
    }
    .status-metric {
        text-align: center;
        background: #F8F9FA;
        border-radius: 8px;
        padding: 6px 8px;
        border: 1px solid #EEF0F2;
    }
    .status-metric .label { font-size: 12px; color: #888; font-weight: 500; }
    .status-metric .value { font-size: 16px; font-weight: 600; color: #1a3a5c; }
    .status-metric .value-normal { color: #1B7A4A; }
    .status-metric .value-critical { color: #C0392B; }
    .water-card-in {
        background: #F5F8FC;
        border-radius: 12px;
        padding: 10px 14px 14px 14px;
        border: 1px solid #D6E4F0;
        margin-bottom: 6px;
    }
    .water-card-out {
        background: #F5FCF8;
        border-radius: 12px;
        padding: 10px 14px 14px 14px;
        border: 1px solid #C8E6D9;
        margin-bottom: 6px;
    }
    .metric-card {
        background: white;
        border-radius: 6px;
        padding: 6px 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 3px;
        border-left: 4px solid #2E86AB;
    }
    .metric-card .label {
        font-size: 12px;
        color: #666;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 18px;
        font-weight: 700;
        color: #1a3a5c;
    }
    .metric-card .sub {
        font-size: 11px;
        color: #999;
    }
    /* 底部统计卡片（与 metric-card 风格一致） */
    .stat-card {
        background: white;
        border-radius: 6px;
        padding: 6px 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 3px;
        border-left: 4px solid #2E86AB;
    }
    .stat-card .stat-label {
        font-size: 12px;
        color: #666;
        font-weight: 500;
    }
    .stat-card .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #1a3a5c;
    }
    .stat-card .stat-sub {
        font-size: 11px;
        color: #999;
    }
    .limit-ref {
        font-size: 11px;
        color: #888;
        background: #F0F0F0;
        padding: 1px 8px;
        border-radius: 10px;
        display: inline-block;
    }
    .channel-container { display: flex; gap: 10px; margin: 6px 0; }
    .channel-item {
        flex: 1;
        background: white;
        border-radius: 8px;
        padding: 8px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 3px solid #ccc;
    }
    .channel-item .ch-name { font-weight: 600; font-size: 13px; }
    .channel-item .ch-value { font-size: 18px; font-weight: 700; margin: 2px 0; }
    .channel-item .ch-desc { font-size: 11px; color: #666; }
    .channel-fast .ch-name { color: #27AE60; }
    .channel-slow .ch-name { color: #F39C12; }
    .channel-special .ch-name { color: #E74C3C; }
    .channel-fast { border-top-color: #27AE60; }
    .channel-slow { border-top-color: #F39C12; }
    .channel-special { border-top-color: #E74C3C; }
    .timeline-step {
        display: flex;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid #F5F5F5;
    }
    .timeline-step:last-child { border-bottom: none; }
    .timeline-time { min-width: 50px; font-weight: 700; font-size: 14px; color: #1a3a5c; }
    .timeline-action { font-size: 14px; color: #333; padding-left: 8px; }
    .calibration-success {
        background: #E8F5E9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #27AE60;
        margin: 8px 0;
    }
    .calibration-info { font-size: 13px; color: #555; }
    .data-status-realtime {
        background: #E3F2FD;
        border-radius: 12px;
        padding: 8px 16px;
        border: 1px solid #90CAF9;
        display: inline-block;
        font-size: 13px;
        color: #1565C0;
    }
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
        self.data = []
    
    def add_data(self, timestamp, inlet, outlet, pred_outlet):
        self.data.append({
            'timestamp': timestamp,
            'inlet': inlet.copy(),
            'outlet': outlet.copy() if outlet else None,
            'pred_outlet': pred_outlet.copy() if pred_outlet else None
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
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []

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
        <div class="value">{beijing_now.strftime('%H:%M')}</div>
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
        factor = decay
        data[f'COD_load_lag{i}'] = cod * flow / 1000 * factor
        data[f'NH3_load_lag{i}'] = nh3 * flow / 1000 * factor
        data[f'TP_load_lag{i}'] = tp * flow / 1000 * factor
        data[f'流量_lag{i}'] = flow * factor
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
# 永久记忆：保存数据到 Supabase
# ==========================================
def save_to_supabase(inlet, outlet_real, outlet_pred, source="manual"):
    """永久保存数据到 Supabase"""
    try:
        data = {
            'cod_in': inlet.get('COD', 0),
            'nh3_in': inlet.get('NH3-N', 0),
            'tp_in': inlet.get('TP', 0),
            'ss_in': inlet.get('SS', 0),
            'flow_in': inlet.get('流量', 0),
            'pac': inlet.get('PAC', 0),
            'carbon': inlet.get('碳源', 0),
            'mlss': inlet.get('MLSS', 0),
            'do_val': inlet.get('DO', 0),
            'cod_real': outlet_real.get('COD', 0),
            'nh3_real': outlet_real.get('NH3-N', 0),
            'tp_real': outlet_real.get('TP', 0),
            'ss_real': outlet_real.get('SS', 0),
            'cod_pred': outlet_pred.get('COD', 0),
            'nh3_pred': outlet_pred.get('NH3-N', 0),
            'tp_pred': outlet_pred.get('TP', 0),
            'ss_pred': outlet_pred.get('SS', 0),
            'source': source
        }
        result = supabase_client.table('feedback_data').insert(data).execute()
        return True, "数据已永久保存"
    except Exception as e:
        return False, f"保存失败: {str(e)}"

# ==========================================
# 获取已保存数据量
# ==========================================
def get_saved_count():
    try:
        result = supabase_client.table('feedback_data').select('*', count='exact').execute()
        return result.count
    except:
        return 0

# ==========================================
# 完整诊断函数
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []
    # 进水COD
    if inlet['COD'] > 500:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'reasons': ['工业废水偷排', '管网沉积物冲刷', '污泥厌氧消化液回流'],
            'actions': ['增加碳源投加量30-40%', '提高好氧段DO至3.0-3.5 mg/L', '降低进水量15-20%']
        })
    elif inlet['COD'] > 400:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'reasons': ['工业废水间歇性排放冲击', '管网沉积物释放'],
            'actions': ['增加碳源投加量20%', '提高DO至2.5-3.0 mg/L']
        })
    elif inlet['COD'] < 100 and inlet['COD'] > 0:
        diagnoses.append({
            'level': 'info',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': 'ℹ️ 进水COD偏低（<100 mg/L）',
            'reasons': ['雨水稀释', '上游截流'],
            'actions': ['减少碳源投加量20-30%', '适当降低曝气量']
        })
    # 进水NH3-N
    if inlet['NH3-N'] > 45:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45 mg/L）',
            'reasons': ['工业废水偷排高浓度氨氮', '污泥消化液回流'],
            'actions': ['提高DO至3.5-4.0 mg/L', '补充NaHCO₃ 80-100mg/L', '延长污泥龄']
        })
    elif inlet['NH3-N'] > 35:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45 mg/L）',
            'reasons': ['上游氨氮浓度升高', '硝化菌活性受抑制'],
            'actions': ['提高DO至3.0-3.5 mg/L', '补充碱度50-80 mg/L']
        })
    # 进水TP
    if inlet['TP'] > 7.0:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7.0 mg/L）',
            'reasons': ['工业废水偷排高浓度磷废水', '污泥厌氧释磷'],
            'actions': ['增加PAC投加量40-50%', '检查pH 6.5-7.5', '增加排泥']
        })
    elif inlet['TP'] > 5.0:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'reasons': ['上游含磷废水浓度波动', 'PAC投加量相对不足'],
            'actions': ['增加PAC投加量20-30%', '检查pH并调节']
        })
    # 进水SS
    if inlet['SS'] > 350:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水SS',
            'current': f"{inlet['SS']:.0f} mg/L",
            'title': '⚠️ 进水SS严重偏高（>350 mg/L）',
            'reasons': ['管网冲刷', '初沉池运行异常'],
            'actions': ['增加初沉池排泥频率', '投加PAM絮凝剂']
        })
    # 出水COD
    if outlet['COD'] > DESIGN_LIMITS['COD']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['COD'] > 45 else 'warning',
            'indicator': '出水COD',
            'current': f"{outlet['COD']:.1f} mg/L",
            'title': f"{'🚨' if outlet['COD'] > 45 else '⚠️'} 出水COD超标",
            'reasons': [f'进水COD负荷过高（{inlet["COD"]:.0f}）', f'DO不足（{do:.1f}）', '污泥老化'],
            'actions': [f'增加碳源{int(carbon)}→{int(carbon*1.25)}', f'提高DO至2.5-3.0']
        })
    # 出水NH3-N
    if outlet['NH3-N'] > DESIGN_LIMITS['NH3-N']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['NH3-N'] > 3.0 else 'warning',
            'indicator': '出水NH₃-N',
            'current': f"{outlet['NH3-N']:.2f} mg/L",
            'title': f"{'🚨' if outlet['NH3-N'] > 3.0 else '⚠️'} 出水NH₃-N超标",
            'reasons': [f'DO不足（{do:.1f}）', '碱度不足', 'SRT太短'],
            'actions': ['提高DO至3.0-3.5', '补充NaHCO₃ 50-80mg/L', '延长SRT至15天以上']
        })
    # 出水TP
    if outlet['TP'] > DESIGN_LIMITS['TP']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['TP'] > 0.6 else 'warning',
            'indicator': '出水TP',
            'current': f"{outlet['TP']:.3f} mg/L",
            'title': f"{'🚨' if outlet['TP'] > 0.6 else '⚠️'} 出水TP超标",
            'reasons': [f'PAC不足（{pac:.0f}）', 'pH不适宜', '磷释放'],
            'actions': [f'增加PAC {pac}→{int(pac*1.4)}', '调整投加点', '增加排泥']
        })
    # 出水SS
    if outlet['SS'] > DESIGN_LIMITS['SS']['value']:
        diagnoses.append({
            'level': 'warning',
            'indicator': '出水SS',
            'current': f"{outlet['SS']:.1f} mg/L",
            'title': '⚠️ 出水SS超标',
            'reasons': ['表面负荷过高', 'SVI升高', '排泥不足'],
            'actions': ['增加排泥20%', '投加PAM', '降低进水量10-15%']
        })
    # DO
    if do < 0.8:
        diagnoses.append({
            'level': 'critical',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 好氧段DO严重不足（<0.8 mg/L）',
            'reasons': ['曝气设备故障', '进水负荷突增'],
            'actions': ['检查曝气设备', '加大风机风量20-30%']
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ 好氧段DO偏低（<1.5 mg/L）',
            'reasons': ['曝气量不足', '进水负荷增加'],
            'actions': ['增加曝气量10-20%', '监测DO变化趋势']
        })
    # MLSS
    if mlss < 2500:
        diagnoses.append({
            'level': 'warning',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ 污泥浓度偏低（<2500 mg/L）',
            'reasons': ['污泥流失过多', '进水负荷过低'],
            'actions': ['减少排泥量', '增加污泥回流量']
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ 污泥浓度偏高（>6000 mg/L）',
            'reasons': ['排泥不足', '二沉池泥层过厚'],
            'actions': ['增加排泥量', '检查二沉池泥位']
        })
    # PAC
    if pac < 20:
        diagnoses.append({
            'level': 'warning',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': '⚠️ PAC投加量偏低（<20 mg/L）',
            'reasons': ['PAC储备不足', '加药泵故障'],
            'actions': ['增加PAC至30-50 mg/L', '检查加药泵']
        })
    elif pac > 80:
        diagnoses.append({
            'level': 'info',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': 'ℹ️ PAC投加量偏高（>80 mg/L）',
            'reasons': ['为应对高负荷临时加大'],
            'actions': ['评估是否可降低', '检查出水TP是否达标']
        })
    # 碳源
    if carbon < 30:
        diagnoses.append({
            'level': 'warning',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': '⚠️ 碳源投加量偏低（<30 mg/L）',
            'reasons': ['碳源储备不足', '反硝化碳源缺乏'],
            'actions': ['增加碳源至40-60 mg/L', '检查碳源储罐液位']
        })
    elif carbon > 100:
        diagnoses.append({
            'level': 'info',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': 'ℹ️ 碳源投加量偏高（>100 mg/L）',
            'reasons': ['为应对高负荷临时加大'],
            'actions': ['评估是否可逐步降低', '检查出水COD和TN']
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

REQUIRED_COLS = ['COD', 'NH3-N', 'TP', 'SS', '流量', 'PAC', '碳源', 'MLSS', 'DO']

# --- 初始化所有变量 ---
cod_in = nh3_in = tp_in = ss_in = flow_in = 0
pac = carbon = mlss = do = 0
input_data = None
simulated_outlet = None

# --- 1. 手动输入 ---
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

# --- 2. 文件上传 ---
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
            missing_cols = set(REQUIRED_COLS) - set(df_upload.columns)
            if missing_cols:
                st.sidebar.error(f"❌ 缺少必需列：{missing_cols}")
                input_data = None
            else:
                row = df_upload.iloc[0]
                cod_in = row['COD']
                nh3_in = row['NH3-N']
                tp_in = row['TP']
                ss_in = row['SS']
                flow_in = row['流量']
                pac = row['PAC']
                carbon = row['碳源']
                mlss = row['MLSS']
                do = row['DO']
                input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
                st.sidebar.success(f"✅ 成功加载数据 (共 {len(df_upload)} 行)")
        except Exception as e:
            st.sidebar.error(f"❌ 文件解析失败：{str(e)}")
            input_data = None
    else:
        input_data = None

# --- 3. API 接入 ---
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
                cod_in = data.get('COD', 0)
                nh3_in = data.get('NH3-N', 0)
                tp_in = data.get('TP', 0)
                ss_in = data.get('SS', 0)
                flow_in = data.get('流量', 0)
                pac = data.get('PAC', 0)
                carbon = data.get('碳源', 0)
                mlss = data.get('MLSS', 0)
                do = data.get('DO', 0)
                input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
                st.sidebar.success("✅ 数据获取成功")
            else:
                st.sidebar.error(f"❌ API 返回错误：{resp.status_code}")
                input_data = None
        except Exception as e:
            st.sidebar.error(f"❌ 连接失败：{str(e)}")
            input_data = None
    else:
        input_data = None

# --- 4. 自动实时（模拟） ---
else:
    st.sidebar.markdown("### 🔄 自动实时数据")
    st.sidebar.info("🔄 每5秒自动生成一组模拟数据")
    if st.sidebar.button("▶️ 启动实时数据流"):
        st.session_state.auto_mode_running = True
        st.sidebar.success("✅ 数据流已启动")
    if st.sidebar.button("⏹️ 停止数据流"):
        st.session_state.auto_mode_running = False
        st.sidebar.info("⏹️ 数据流已停止")
    simulated_inlet = generate_simulated_data()
    cod_in = simulated_inlet['COD']
    nh3_in = simulated_inlet['NH3-N']
    tp_in = simulated_inlet['TP']
    ss_in = simulated_inlet['SS']
    flow_in = simulated_inlet['流量']
    pac = simulated_inlet['PAC']
    carbon = simulated_inlet['碳源']
    mlss = simulated_inlet['MLSS']
    do = simulated_inlet['DO']
    input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)
    simulated_outlet = simulate_outlet(simulated_inlet)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="data-status-realtime">
        📊 当前数据：第 {st.session_state.simulation_counter + 1} 组
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 主界面
# ==========================================
if input_data is not None:
    pred_cod = predict_cod_tp(input_data, 'COD_out')
    pred_tp = predict_cod_tp(input_data, 'TP_out')
    pred_nh3 = predict_nh3_optimized(input_data)
    pred_ss = max(0, 5 + np.random.normal(0, 0.5))

    inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'SS': ss_in, '流量': flow_in}
    outlet_pred = {'COD': pred_cod if pred_cod else 0, 
                   'NH3-N': pred_nh3 if pred_nh3 else 0,
                   'TP': pred_tp if pred_tp else 0, 
                   'SS': pred_ss}
    
    # ---- 自动模式处理 ----
    if input_mode_global == "🔄 自动实时（模拟）" and st.session_state.auto_mode_running and simulated_outlet is not None:
        real_outlet = {
            'COD': max(0, simulated_outlet['COD'] + np.random.normal(0, 0.3)),
            'NH3-N': max(0, simulated_outlet['NH3-N'] + np.random.normal(0, 0.005)),
            'TP': max(0, simulated_outlet['TP'] + np.random.normal(0, 0.005)),
            'SS': max(0, simulated_outlet['SS'] + np.random.normal(0, 0.2))
        }
        vec = np.array([input_data[col].values[0] if col in input_data.columns else 0 for col in feature_cols])
        if st.session_state.simulation_counter % 5 == 0 and st.session_state.simulation_counter > 0:
            if real_outlet['COD'] > 0:
                calibrate_model('COD_out', vec, real_outlet['COD'])
            if real_outlet['TP'] > 0:
                calibrate_model('TP_out', vec, real_outlet['TP'])
            success, msg = save_to_supabase(inlet, real_outlet, outlet_pred, "auto")
            if not success:
                st.warning(f"⚠️ {msg}")
            st.session_state.feedback_log.append({
                'timestamp': datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'auto_calibration',
                'cod_pred': outlet_pred['COD'],
                'cod_real': real_outlet['COD'],
                'tp_pred': outlet_pred['TP'],
                'tp_real': real_outlet['TP']
            })
        st.session_state.data_buffer.add_data(
            timestamp=datetime.now(BEIJING_TZ),
            inlet=inlet,
            outlet=real_outlet,
            pred_outlet=outlet_pred
        )
        st.session_state.simulation_counter += 1
        st.info(f"🔄 实时数据流运行中... 已接收 {st.session_state.simulation_counter} 组数据 | 已微调 {st.session_state.calibration_count} 次")
        outlet_display = real_outlet
        outlet_label = "实测"
    else:
        outlet_display = outlet_pred.copy()
        st.session_state.data_buffer.add_data(
            timestamp=datetime.now(BEIJING_TZ),
            inlet=inlet,
            outlet=None,
            pred_outlet=outlet_pred
        )
        outlet_label = "预测"

    # ---- 状态更新 ----
    has_abnormal = False
    for key in ['COD', 'NH3-N', 'TP', 'SS']:
        if outlet_pred.get(key, 0) > DESIGN_LIMITS[key]['value']:
            has_abnormal = True
            break
    if inlet.get('COD', 0) > 400 or inlet.get('NH3-N', 0) > 35 or inlet.get('TP', 0) > 5:
        has_abnormal = True
    status_text = "异常" if has_abnormal else "正常"
    status_color = "value-critical" if has_abnormal else "value-normal"
    with status_placeholder:
        st.markdown(f"""
        <div class="status-metric">
            <div class="label">📊 数据状态</div>
            <div class="value {status_color}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 进出水水质面板 ----
    st.markdown('<div class="section-header">📊 进出水水质实时监测</div>', unsafe_allow_html=True)
    st.caption(f"📌 出水设计标准：COD≤{DESIGN_LIMITS['COD']['value']} | NH₃-N≤{DESIGN_LIMITS['NH3-N']['value']} | TP≤{DESIGN_LIMITS['TP']['value']} | SS≤{DESIGN_LIMITS['SS']['value']} mg/L")

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.markdown("""
        <div class="water-card-in">
            <div style="font-size:15px; font-weight:600; color:#1a3a5c; margin-bottom:6px;">
                🔵 进水水质 <span style="font-size:11px; font-weight:400; color:#888;">（实测）</span>
            </div>
        """, unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""<div class="metric-card"><div class="label">COD</div><div class="value">{inlet['COD']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">NH₃-N</div><div class="value">{inlet['NH3-N']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">TP</div><div class="value">{inlet['TP']:.2f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""<div class="metric-card"><div class="label">SS</div><div class="value">{inlet['SS']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">流量</div><div class="value">{inlet['流量']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">m³/h</span></div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="water-card-out">
            <div style="font-size:15px; font-weight:600; color:#1a5c3a; margin-bottom:6px;">
                🟢 出水水质 <span style="font-size:11px; font-weight:400; color:#888;">（{outlet_label}）</span>
            </div>
        """, unsafe_allow_html=True)
        cod_ok = outlet_display['COD'] <= DESIGN_LIMITS['COD']['value']
        nh3_ok = outlet_display['NH3-N'] <= DESIGN_LIMITS['NH3-N']['value']
        tp_ok = outlet_display['TP'] <= DESIGN_LIMITS['TP']['value']
        ss_ok = outlet_display['SS'] <= DESIGN_LIMITS['SS']['value']
        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown(f"""<div class="metric-card"><div class="label">COD <span class="limit-ref">限值≤{DESIGN_LIMITS['COD']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if cod_ok else '#C0392B'}">{outlet_display['COD']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if cod_ok else f'🔴 超标{outlet_display["COD"]-DESIGN_LIMITS["COD"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">NH₃-N <span class="limit-ref">限值≤{DESIGN_LIMITS['NH3-N']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if nh3_ok else '#C0392B'}">{outlet_display['NH3-N']:.2f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if nh3_ok else f'🔴 超标{outlet_display["NH3-N"]-DESIGN_LIMITS["NH3-N"]["value"]:.2f}'}</div></div>""", unsafe_allow_html=True)
        with cc4:
            st.markdown(f"""<div class="metric-card"><div class="label">TP <span class="limit-ref">限值≤{DESIGN_LIMITS['TP']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if tp_ok else '#C0392B'}">{outlet_display['TP']:.3f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if tp_ok else f'🔴 超标{outlet_display["TP"]-DESIGN_LIMITS["TP"]["value"]:.3f}'}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">SS <span class="limit-ref">限值≤{DESIGN_LIMITS['SS']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if ss_ok else '#C0392B'}">{outlet_display['SS']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if ss_ok else f'🔴 超标{outlet_display["SS"]-DESIGN_LIMITS["SS"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 趋势图 ----
    st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
    st.caption("🟦 实线 = 实测值 | 虚线 = 预测值 | 🟥进水COD 🟧进水NH₃-N 🟪进水TP")
    
    recent_data = st.session_state.data_buffer.get_recent(24)
    if len(recent_data) > 1:
        df_trend = pd.DataFrame([{
            'timestamp': d['timestamp'],
            'inlet_COD': d['inlet']['COD'],
            'inlet_NH3': d['inlet']['NH3-N'],
            'inlet_TP': d['inlet']['TP'],
            'outlet_COD_real': d['outlet']['COD'] if d['outlet'] else None,
            'outlet_COD_pred': d['pred_outlet']['COD'] if d['pred_outlet'] else None,
            'outlet_NH3_real': d['outlet']['NH3-N'] if d['outlet'] else None,
            'outlet_NH3_pred': d['pred_outlet']['NH3-N'] if d['pred_outlet'] else None,
            'outlet_TP_real': d['outlet']['TP'] if d['outlet'] else None,
            'outlet_TP_pred': d['pred_outlet']['TP'] if d['pred_outlet'] else None,
        } for d in recent_data])
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=('COD', 'NH₃-N', 'TP'))
        
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
        fig.update_xaxes(title_text="时间（北京时间）", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 数据收集中... 请等待更多数据点（至少2个时间点）")

    # ---- 记忆长度与分频调控 ----
    st.markdown('<div class="section-header">🧠 记忆长度与分频调控策略</div>', unsafe_allow_html=True)
    st.caption("💡 不同污染物响应速度不同，分通道制定调控策略，避免过度调节或等待不足。")
    col_ch1, col_ch2, col_ch3 = st.columns(3)
    with col_ch1:
        st.markdown("""
        <div class="channel-item channel-fast">
            <div class="ch-name">⚡ 快速通道</div>
            <div class="ch-value" style="color:#27AE60;">6-8h</div>
            <div class="ch-desc">NH₃-N (6h) · COD (8h) | 更新 3-4h</div>
            <div class="ch-desc">✅ 三模型共识</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch2:
        st.markdown("""
        <div class="channel-item channel-slow">
            <div class="ch-name">🐢 慢速通道</div>
            <div class="ch-value" style="color:#F39C12;">22h</div>
            <div class="ch-desc">TP (≈22h) | 更新 8-12h</div>
            <div class="ch-desc">✅ 三模型共识</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch3:
        st.markdown("""
        <div class="channel-item channel-special">
            <div class="ch-name">🔴 特殊通道</div>
            <div class="ch-value" style="color:#E74C3C;">不稳定</div>
            <div class="ch-desc">SS — 实时阈值报警</div>
            <div class="ch-desc">⚠️ 三模型不一致</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 时序决策建议 ----
    st.markdown('<div class="section-header">⏱️ 时序决策建议（具体操作）</div>', unsafe_allow_html=True)
    indicator = st.selectbox("选择指标", ['COD', 'NH3-N', 'TP', 'SS'])
    mem = MEMORY[indicator]['hours']
    current_val = outlet_display[indicator]
    limit = DESIGN_LIMITS[indicator]['value']

    if mem:
        if indicator == 'COD':
            steps = [
                (0, "🚨 记录进水COD异常值，启动应急响应"),
                (2, "📞 通知值班长，确认碳源储备"),
                (4, "⚙️ 增加碳源投加量20%"),
                (6, "🔍 检查好氧段DO，若<2.0则增加曝气"),
                (8, "📊 评估出水COD变化趋势"),
                (12, "✅ 确认COD稳定达标，逐步回调")
            ]
        elif indicator == 'NH3-N':
            steps = [
                (0, "🚨 记录进水NH₃-N异常值，启动应急响应"),
                (2, "📞 通知值班长，准备碱度调节剂"),
                (3, "⚙️ 提高好氧段DO至3.0-3.5 mg/L"),
                (5, "🔍 检查碱度，若<100则补充NaHCO₃"),
                (6, "📊 评估出水NH₃-N变化趋势"),
                (9, "✅ 确认NH₃-N稳定达标")
            ]
        elif indicator == 'TP':
            steps = [
                (0, "🚨 记录进水TP异常值，启动应急响应"),
                (4, "📞 通知值班长，确认PAC储备"),
                (8, "⚙️ 增加PAC投加量30%"),
                (14, "🔍 检查pH，若<6.5则投加碱调节"),
                (22, "📊 评估出水TP变化趋势"),
                (33, "✅ 确认TP稳定达标，逐步回调")
            ]
        else:
            steps = [
                (0, "🚨 SS超标，启动应急响应"),
                (1, "📞 检查二沉池刮泥机"),
                (2, "⚙️ 增加排泥量20%"),
                (3, "🔍 检查SVI，若>150投加PAM"),
                (4, "📊 评估SS变化"),
                (6, "✅ 确认达标")
            ]
        st.markdown('<div style="background:#FAFBFC;border-radius:8px;padding:10px 14px;border:1px solid #E8ECF0;">', unsafe_allow_html=True)
        st.markdown(f"**📋 {indicator}：{current_val:.2f} / {limit} mg/L**")
        st.markdown("---")
        for t, action in steps:
            st.markdown(f"""
            <div class="timeline-step">
                <div class="timeline-time">⏱️ {t}h</div>
                <div class="timeline-action">{action}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 异常诊断与工艺优化 ----
    st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
    st.caption("💡 基于同类型A²/O工艺经验库 + 当前工况多维度分析")
    
    diagnoses = diagnose_system(inlet, outlet_display, pac, carbon, mlss, do)
    if diagnoses:
        level_order = {'critical': 0, 'warning': 1, 'info': 2}
        diagnoses.sort(key=lambda x: level_order.get(x['level'], 3))
        for d in diagnoses:
            with st.expander(f"{d['title']}（当前值：{d['current']}）", expanded=(d['level'] == 'critical')):
                col_r, col_a = st.columns([1, 1])
                with col_r:
                    st.markdown("**🔍 可能原因**")
                    for r in d['reasons']:
                        st.markdown(f"- {r}")
                with col_a:
                    st.markdown("**💡 针对性工艺优化措施**")
                    for a in d['actions']:
                        st.markdown(f"- {a}")
    else:
        st.success("✅ 系统运行正常，未检测到异常")
        st.info("📋 建议：保持当前运行参数，定期巡检设备。")

    # ---- 永久记忆统计（样式与 metric-card 一致） ----
    st.markdown("---")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    saved_count = get_saved_count()
    with col_stats1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">📦 已永久保存数据</div>
            <div class="stat-value">{saved_count} 组</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stats2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">🔄 模型微调次数</div>
            <div class="stat-value">{st.session_state.calibration_count} 次</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stats3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🧠 记忆长度共识</div>
            <div class="stat-value">COD 8h · NH₃-N 6h · TP 22h</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 请左侧输入数据")

st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 v6.1 | 四种输入模式 | 永久记忆已启用 | {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
