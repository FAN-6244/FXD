"""
smart_warning_panel_deploy.py
部署到 Streamlit Cloud —— 手动/自动双模式 + 详细异常诊断
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
    }
    .metric-card .label { font-size: 12px; color: #666; font-weight: 500; }
    .metric-card .value { font-size: 18px; font-weight: 700; color: #1a3a5c; }
    .metric-card .sub { font-size: 11px; color: #999; }
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
    /* 诊断详情样式 */
    .diag-critical { border-left: 5px solid #C0392B !important; background: #FDEDEC !important; }
    .diag-warning { border-left: 5px solid #F39C12 !important; background: #FEF9E7 !important; }
    .diag-info { border-left: 5px solid #2E86AB !important; background: #EBF5FB !important; }
    .diag-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .diag-reason-item {
        background: #F8F9FA;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 13px;
        border-left: 3px solid #888;
    }
    .diag-action-item {
        background: #EAF4FC;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 13px;
        border-left: 3px solid #2E86AB;
    }
    .diag-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .tag-critical { background: #C0392B; color: white; }
    .tag-warning { background: #F39C12; color: white; }
    .tag-info { background: #2E86AB; color: white; }
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
# 详细异常诊断函数
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []
    
    # ========== 进水异常诊断 ==========
    
    # ----- 进水COD异常 -----
    cod_in = inlet.get('COD', 0)
    if cod_in > 500:
        diagnoses.append({
            'level': 'critical',
            'tag': 'critical',
            'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'background': 'COD是衡量有机物污染程度的核心指标。进水COD超过500mg/L属于严重冲击负荷，可能对生物系统造成致命影响。',
            'reasons': [
                '① 工业废水偷排：周边电镀、印染、食品加工企业违规排放高浓度有机废水，COD浓度可达数千mg/L',
                '② 管网沉积物冲刷：雨季或管网清淤作业导致长期沉积的有机淤泥集中入厂，形成瞬时高浓度冲击',
                '③ 污泥厌氧消化液回流：污泥处理段消化液含高浓度COD（可达1000-3000mg/L），回流至进水端造成负荷骤增',
                '④ 上游水质净化厂事故排放：民治厂、坂雪岗厂等上游设施因设备故障或工艺异常排放未处理污水'
            ],
            'actions': [
                '【立即执行】增加碳源投加量30-40%（如碳源从50mg/L增至65-70mg/L），维持系统碳氮平衡',
                '【立即执行】提高好氧段DO至3.0-3.5 mg/L，强化异养菌对有机物的降解能力',
                '【1小时内】降低进水量15-20%，延长水力停留时间（HRT），减轻系统负荷',
                '【2小时内】联系上游泵站及管网运维部门，排查异常来水来源，切断污染源',
                '【持续监测】每2小时取样监测进水COD变化趋势，直至恢复正常水平'
            ]
        })
    elif cod_in > 400:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'background': 'COD超过400mg/L表示系统承受较高有机负荷，需及时干预以防出水超标。',
            'reasons': [
                '① 工业废水间歇性排放冲击：周边企业生产周期导致COD浓度波动',
                '② 管网沉积物受扰动释放：施工或流量突变导致沉积物泛起',
                '③ 上游污水厂预处理效果波动：前端处理设施运行不稳定'
            ],
            'actions': [
                '增加碳源投加量20%（如碳源从50mg/L增至60mg/L），确保碳氮比平衡',
                '提高好氧段DO至2.5-3.0 mg/L，增强有机物降解效率',
                '密切监测出水COD趋势，8小时后评估调控效果（基于COD记忆长度8h）',
                '加强进水在线监测数据审核，排查异常波动时段'
            ]
        })
    elif cod_in < 100 and cod_in > 0:
        diagnoses.append({
            'level': 'info',
            'tag': 'info',
            'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': 'ℹ️ 进水COD偏低（<100 mg/L）',
            'background': '进水COD偏低可能导致碳源不足，影响反硝化脱氮效果。',
            'reasons': [
                '① 雨水稀释：雨季或暴雨导致管网来水COD被稀释，浓度降至正常值的50%以下',
                '② 上游截流：上游闸门关闭或来水减少，导致流量和浓度同步下降',
                '③ 进水流量突然增大：清水混入（如管网漏水）造成浓度降低'
            ],
            'actions': [
                '减少碳源投加量20-30%，避免碳源过剩（如碳源从50mg/L降至35-40mg/L）',
                '适当降低曝气量，节约能耗，防止污泥过度氧化',
                '检查污泥浓度，防止因碳源不足导致的污泥膨胀',
                '关注进水流量变化趋势，判断是否为持续性低负荷状态'
            ]
        })
    
    # ----- 进水NH3-N异常 -----
    nh3_in = inlet.get('NH3-N', 0)
    if nh3_in > 45:
        diagnoses.append({
            'level': 'critical',
            'tag': 'critical',
            'indicator': '进水NH₃-N',
            'current': f"{nh3_in:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45 mg/L）',
            'background': '氨氮是硝化菌的底物，超过45mg/L将严重抑制硝化反应，导致出水氨氮急剧上升。',
            'reasons': [
                '① 工业废水偷排：化工、制药、电子企业排放高浓度氨氮废水（可达100-500mg/L）',
                '② 污泥消化液回流：厌氧消化液氨氮浓度可达600-800mg/L，回流后造成冲击',
                '③ 上游污水厂硝化效果差：上游设施硝化功能失效，氨氮未有效去除即排放',
                '④ 管网中蛋白质类有机物分解：长距离输送过程中有机氮转化为氨氮'
            ],
            'actions': [
                '【立即执行】提高好氧段DO至3.5-4.0 mg/L，强化硝化菌活性',
                '【立即执行】补充碱度，投加NaHCO₃ 80-100 mg/L，维持pH在7.2-7.8之间',
                '【2小时内】延长污泥龄（SRT）至18-20天，保证硝化菌（世代时间>10天）充分生长',
                '【4小时内】降低进水量20%，降低硝化负荷至可承受范围',
                '【持续监测】每1小时监测出水NH₃-N变化，直至恢复正常'
            ]
        })
    elif nh3_in > 35:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '进水NH₃-N',
            'current': f"{nh3_in:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45 mg/L）',
            'background': '进水氨氮超过35mg/L将增加硝化系统负荷，需提前干预。',
            'reasons': [
                '① 上游来水氨氮浓度升高：季节性变化或上游工业排放增加',
                '② 硝化菌活性受抑制：低温、DO不足或碱度不足导致硝化效率下降',
                '③ 污泥龄不足：排泥过多导致硝化菌流失，硝化能力下降'
            ],
            'actions': [
                '提高好氧段DO至3.0-3.5 mg/L，增强硝化反应速率',
                '补充碱度50-80 mg/L，确保硝化反应有足够碱度消耗',
                '检查SRT，建议延长至15天以上，保证硝化菌种群稳定',
                '6小时后评估硝化效果（基于NH₃-N记忆长度6h）'
            ]
        })
    
    # ----- 进水TP异常 -----
    tp_in = inlet.get('TP', 0)
    if tp_in > 7.0:
        diagnoses.append({
            'level': 'critical',
            'tag': 'critical',
            'indicator': '进水TP',
            'current': f"{tp_in:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7.0 mg/L）',
            'background': '总磷超过7mg/L将严重超出化学除磷能力，必须立即强化加药和排泥。',
            'reasons': [
                '① 工业废水偷排：磷化工、电镀、食品加工企业排放高浓度含磷废水（可达50-200mg/L）',
                '② 含磷洗涤剂废水集中排放：生活污水中含磷洗涤剂的使用导致磷浓度升高',
                '③ 污泥厌氧释磷：污泥处理段厌氧条件下聚磷菌释放磷，回流至进水端',
                '④ 农业面源污染：含磷农药、化肥随雨水冲刷入厂'
            ],
            'actions': [
                '【立即执行】增加PAC投加量40-50%（如PAC从30mg/L增至42-45mg/L），确保混凝剂充足',
                '【1小时内】检查并调整PAC投加点至混合反应池入口最佳位置，增加混凝反应时间',
                '【2小时内】检查混凝pH，控制在6.5-7.5最佳范围（PAC在pH 6.5-7.5时除磷效率最高）',
                '【4小时内】增加排泥量，防止含磷污泥厌氧条件下二次释放磷',
                '【持续监测】每2小时监测出水TP变化，22小时后评估除磷效果（基于TP记忆长度22h）'
            ]
        })
    elif tp_in > 5.0:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '进水TP',
            'current': f"{tp_in:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'background': '进水TP超过5mg/L需加强化学除磷措施，防止出水超标。',
            'reasons': [
                '① 上游含磷废水浓度波动：间歇性工业排放或生活污水高峰期',
                '② PAC投加量相对不足：当前投加量无法应对当前负荷',
                '③ 混凝pH不适宜：pH偏离最佳范围影响混凝效果'
            ],
            'actions': [
                '增加PAC投加量20-30%（如PAC从30mg/L增至36-39mg/L）',
                '检查pH并调节至6.5-7.5最佳范围',
                '检查PAC投加点位置，确保混凝反应充分',
                '22小时后评估除磷效果（基于TP记忆长度22h）'
            ]
        })
    
    # ----- 进水SS异常 -----
    ss_in = inlet.get('SS', 0)
    if ss_in > 350:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '进水SS',
            'current': f"{ss_in:.0f} mg/L",
            'title': '⚠️ 进水SS严重偏高（>350 mg/L）',
            'background': '进水SS过高将增加初沉池和二沉池负荷，可能导致出水SS超标。',
            'reasons': [
                '① 管网冲刷：施工扰动或雨期流量增大，携带大量泥沙和悬浮物入厂',
                '② 上游管网沉积物释放：长期沉积物在流量波动时集中排出',
                '③ 初沉池运行异常：排泥不及时或刮泥机故障，导致污泥积存后释放'
            ],
            'actions': [
                '增加初沉池排泥频率（从每班1次增至每班2-3次）',
                '投加PAM絮凝剂（0.5-1.0 mg/L），强化悬浮物沉淀效果',
                '监测二沉池泥位，防止泥层过高导致跑泥',
                '检查初沉池刮泥机运行状态'
            ]
        })
    
    # ========== 出水异常诊断 ==========
    
    # ----- 出水COD超标 -----
    cod_out = outlet.get('COD', 0)
    if cod_out > DESIGN_LIMITS['COD']['value']:
        diagnoses.append({
            'level': 'critical' if cod_out > 45 else 'warning',
            'tag': 'critical' if cod_out > 45 else 'warning',
            'indicator': '出水COD',
            'current': f"{cod_out:.1f} mg/L",
            'title': f"{'🚨' if cod_out > 45 else '⚠️'} 出水COD超标（>{DESIGN_LIMITS['COD']['value']} mg/L）",
            'background': f'出水COD为{cod_out:.1f}mg/L，已超过准Ⅳ类标准30mg/L。COD超标表明有机物去除不彻底。',
            'reasons': [
                f'① 进水COD负荷过高（当前进水{cod_in:.0f} mg/L，超出设计处理能力）',
                f'② 好氧段DO不足（当前{do:.1f} mg/L，建议维持在2.5-3.0 mg/L）',
                '③ 污泥老化或解体：污泥龄过长导致微生物活性下降，出水携带细小絮体',
                '④ 二沉池跑泥：污泥沉降性能变差，出水携带活性污泥絮体',
                '⑤ 碳源投加不足：反硝化碳源缺乏，影响有机物去除效率'
            ],
            'actions': [
                f'增加碳源投加量20-30%（当前{int(carbon)} mg/L → {int(carbon*1.25)} mg/L），强化反硝化',
                f'提高好氧段DO至2.5-3.0 mg/L（当前{do:.1f} mg/L），增强有机物降解',
                '加大排泥量20-30%，更新污泥龄，恢复污泥活性',
                '检查二沉池刮泥机运行状态，调整回流比至60-80%',
                '8小时后评估COD去除效果（基于COD记忆长度8h）'
            ]
        })
    
    # ----- 出水NH3-N超标 -----
    nh3_out = outlet.get('NH3-N', 0)
    if nh3_out > DESIGN_LIMITS['NH3-N']['value']:
        diagnoses.append({
            'level': 'critical' if nh3_out > 3.0 else 'warning',
            'tag': 'critical' if nh3_out > 3.0 else 'warning',
            'indicator': '出水NH₃-N',
            'current': f"{nh3_out:.2f} mg/L",
            'title': f"{'🚨' if nh3_out > 3.0 else '⚠️'} 出水NH₃-N超标（>{DESIGN_LIMITS['NH3-N']['value']} mg/L）",
            'background': f'出水NH₃-N为{nh3_out:.2f}mg/L，超过准Ⅳ类标准1.5mg/L。硝化功能可能部分失效。',
            'reasons': [
                f'① 硝化菌活性受抑制：DO不足（当前{do:.1f} mg/L，需≥2.5 mg/L）',
                '② 碱度不足：硝化反应消耗碱度，pH可能已降至7.0以下',
                f'③ 污泥龄太短：硝化菌世代时间>10天，当前SRT可能不足（需≥12天）',
                f'④ 进水NH₃-N冲击（当前进水{nh3_in:.1f} mg/L，超出硝化能力）',
                '⑤ 水温过低：硝化菌在<15℃时活性显著下降'
            ],
            'actions': [
                f'提高好氧段DO至3.0-3.5 mg/L（当前{do:.1f} mg/L），强化硝化反应',
                '补充碱度，投加NaHCO₃ 50-80 mg/L，维持pH在7.2-7.8之间',
                '延长污泥龄至15天以上，减少排泥量，保证硝化菌种群稳定',
                '降低进水量15%，减轻硝化负荷',
                '6小时后评估硝化效果（基于NH₃-N记忆长度6h）'
            ]
        })
    
    # ----- 出水TP超标 -----
    tp_out = outlet.get('TP', 0)
    if tp_out > DESIGN_LIMITS['TP']['value']:
        diagnoses.append({
            'level': 'critical' if tp_out > 0.6 else 'warning',
            'tag': 'critical' if tp_out > 0.6 else 'warning',
            'indicator': '出水TP',
            'current': f"{tp_out:.3f} mg/L",
            'title': f"{'🚨' if tp_out > 0.6 else '⚠️'} 出水TP超标（>{DESIGN_LIMITS['TP']['value']} mg/L）",
            'background': f'出水TP为{tp_out:.3f}mg/L，超过准Ⅳ类标准0.3mg/L。化学除磷效率不足。',
            'reasons': [
                f'① PAC投加量不足（当前{pac:.0f} mg/L，建议30-60 mg/L）',
                '② 混凝pH不适宜：PAC在pH 6.5-7.5时除磷效率最高，当前pH可能偏离此范围',
                '③ PAC投加点位置不当：需在混合反应池入口投加，保证充分混合和混凝反应',
                '④ 污泥中磷释放：厌氧条件下聚磷菌释放磷，或排泥不及时导致磷二次释放',
                f'⑤ 进水TP过高（当前{inlet["TP"]:.2f} mg/L，超出化学除磷设计能力）'
            ],
            'actions': [
                f'增加PAC投加量30-50%（当前{pac:.0f} mg/L → {int(pac*1.4)} mg/L），确保混凝剂充足',
                '调整PAC投加点至混合反应池入口，增加混凝反应时间',
                '检查并调节pH至6.5-7.5最佳范围，提高混凝效率',
                '增加排泥量，防止含磷污泥在厌氧条件下释放磷',
                '22小时后评估除磷效果（基于TP记忆长度22h）'
            ]
        })
    
    # ----- 出水SS超标 -----
    ss_out = outlet.get('SS', 0)
    if ss_out > DESIGN_LIMITS['SS']['value']:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '出水SS',
            'current': f"{ss_out:.1f} mg/L",
            'title': f"⚠️ 出水SS超标（>{DESIGN_LIMITS['SS']['value']} mg/L）",
            'background': f'出水SS为{ss_out:.1f}mg/L，超过准Ⅳ类标准10mg/L。二沉池固液分离效果不佳。',
            'reasons': [
                '① 二沉池表面负荷过高：进水量过大，沉淀时间不足（正常HRT≥2-3h）',
                '② 污泥沉降性能变差：SVI升高（>150 mL/g），污泥不易沉降',
                '③ 排泥不足：二沉池泥层过厚，导致污泥随出水溢流',
                '④ 刮泥机运行故障或速度不当：无法及时将污泥刮至池底'
            ],
            'actions': [
                '增加排泥频率20-30%，降低泥层高度（当前排泥量×1.2-1.3）',
                '投加PAM絮凝剂0.3-0.5 mg/L，改善污泥沉降性能',
                '降低进水量10-15%，降低表面负荷，延长沉淀时间',
                '检查刮泥机运行状态，确保正常运行'
            ]
        })
    
    # ========== 运行参数异常诊断 ==========
    
    # ----- DO异常 -----
    if do < 0.8:
        diagnoses.append({
            'level': 'critical',
            'tag': 'critical',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 好氧段DO严重不足（<0.8 mg/L）',
            'background': 'DO低于0.8mg/L将导致好氧微生物（包括硝化菌）活性受到严重抑制，系统面临崩溃风险。',
            'reasons': [
                '① 曝气设备故障或效率下降：风机故障、曝气盘堵塞或老化',
                '② 进水负荷突然增大：COD或NH₃-N负荷激增，耗氧速率远超供氧能力',
                '③ 风机运行参数设置不当：风量不足或压力偏低'
            ],
            'actions': [
                '【立即执行】检查曝气设备运行状态，确认风机是否正常工作',
                '【立即执行】加大风机风量20-30%，提高供氧能力',
                '检查风机出口压力是否正常（正常0.05-0.07MPa）',
                '4小时内恢复DO至2.0 mg/L以上，避免系统崩溃'
            ]
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ 好氧段DO偏低（<1.5 mg/L）',
            'background': 'DO低于1.5mg/L将影响硝化菌活性，可能导致出水氨氮升高。',
            'reasons': [
                '① 曝气量不足：风量设置偏低或风机效率下降',
                '② 进水负荷增加：耗氧速率加快，供氧相对不足',
                '③ 水温升高：饱和DO降低，需增加曝气量'
            ],
            'actions': [
                '增加曝气量10-20%（如风机频率从45Hz增至50-55Hz）',
                '监测好氧段DO变化趋势，确保DO稳定在2.0mg/L以上',
                '检查风机变频器设置，确认是否在最佳运行区间'
            ]
        })
    
    # ----- MLSS异常 -----
    if mlss < 2500:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ 污泥浓度偏低（<2500 mg/L）',
            'background': 'MLSS低于2500mg/L将降低系统抗冲击能力，影响污染物去除效率。',
            'reasons': [
                '① 污泥流失过多：排泥过量或二沉池跑泥导致污泥大量流失',
                '② 进水负荷过低：营养物质不足，微生物生长受限',
                '③ 污泥回流量不足：回流泵故障或回流比设置偏低'
            ],
            'actions': [
                '减少排泥量，将MLSS逐步提升至3500-4000 mg/L',
                '增加污泥回流量（回流比从60%增至80%），补充系统污泥',
                '检查二沉池是否跑泥，排查跑泥原因',
                '适当增加碳源投加，促进微生物生长'
            ]
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info',
            'tag': 'info',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ 污泥浓度偏高（>6000 mg/L）',
            'background': 'MLSS超过6000mg/L将增加二沉池负荷，可能导致跑泥和氧传输效率下降。',
            'reasons': [
                '① 排泥不足：污泥在系统内积累，MLSS持续升高',
                '② 二沉池泥层过厚：污泥沉降性能好但未及时排泥',
                '③ 进水SS过高：大量悬浮物转化为活性污泥'
            ],
            'actions': [
                '增加排泥量，将MLSS逐步降至4000-5000 mg/L',
                '检查二沉池泥位，确保泥位在正常范围',
                '注意氧传输效率下降问题，适当增加曝气量'
            ]
        })
    
    # ----- PAC异常 -----
    if pac < 20:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': '⚠️ PAC投加量偏低（<20 mg/L）',
            'background': 'PAC投加量低于20mg/L可能无法满足除磷需求，特别是进水TP较高时。',
            'reasons': [
                '① PAC储备不足：库存不足或采购延迟',
                '② 加药泵故障或计量不准：投加量偏离设定值',
                '③ 人为调低：未根据进水TP变化及时调整'
            ],
            'actions': [
                f'增加PAC投加量至30-50 mg/L（当前{pac:.0f} mg/L），确保除磷效果',
                '检查加药泵运行状态，确认计量准确性',
                '核查PAC库存，及时补充采购'
            ]
        })
    elif pac > 80:
        diagnoses.append({
            'level': 'info',
            'tag': 'info',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': 'ℹ️ PAC投加量偏高（>80 mg/L）',
            'background': 'PAC投加量超过80mg/L将增加药耗成本和污泥产量，需评估是否可优化。',
            'reasons': [
                '① 为应对高进水TP临时加大投加',
                '② 自动加药系统参数设置过高，未及时调整'
            ],
            'actions': [
                '评估是否可适当降低投加量（每2小时检测出水TP，达标后逐步降低）',
                '检查出水TP是否已达标，避免过度加药',
                '防止过量加药导致污泥产量增加和pH下降'
            ]
        })
    
    # ----- 碳源异常 -----
    if carbon < 30:
        diagnoses.append({
            'level': 'warning',
            'tag': 'warning',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': '⚠️ 碳源投加量偏低（<30 mg/L）',
            'background': '碳源不足将影响反硝化脱氮效果，可能导致出水TN升高。',
            'reasons': [
                '① 碳源储备不足：库存不足或采购延迟',
                '② 反硝化碳源缺乏：影响脱氮效果，TN可能升高'
            ],
            'actions': [
                f'增加碳源投加量至40-60 mg/L（当前{carbon:.0f} mg/L），确保反硝化有足够碳源',
                '检查碳源储罐液位，及时补充采购',
                '关注出水TN变化趋势，评估脱氮效果'
            ]
        })
    elif carbon > 100:
        diagnoses.append({
            'level': 'info',
            'tag': 'info',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': 'ℹ️ 碳源投加量偏高（>100 mg/L）',
            'background': '碳源投加量过高将增加成本，且可能导致出水COD升高。',
            'reasons': [
                '① 为应对高进水COD负荷临时加大投加',
                '② 碳源计量误差或设备故障'
            ],
            'actions': [
                '评估是否可逐步降低投加量（每4小时检测出水COD和TN，达标后逐步降低）',
                '检查出水COD和TN是否已达标，避免过度加药',
                '核查碳源计量设备，确认投加量准确性'
            ]
        })
    
    return diagnoses

# ==========================================
# 侧边栏
# ==========================================
st.sidebar.markdown("## 📊 数据输入模式")
input_mode_global = st.sidebar.radio(
    "选择数据模式",
    ["✏️ 手动输入", "📡 自动实时（模拟）"],
    index=0
)

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

else:
    st.sidebar.markdown("### 📡 自动实时数据")
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
    
    if input_mode_global == "📡 自动实时（模拟）" and st.session_state.auto_mode_running:
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
                                name='进水COD', line=dict(color='#E74C3C', width=2, dash='solid'),
                                legendgroup='inlet_COD'), row=1, col=1)
        mask_real = df_trend['outlet_COD_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], 
                                    y=df_trend[mask_real]['outlet_COD_real'],
                                    name='出水COD_实测', line=dict(color='#2E86AB', width=2.5, dash='solid'),
                                    legendgroup='outlet_COD_real'), row=1, col=1)
        mask_pred = df_trend['outlet_COD_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], 
                                    y=df_trend[mask_pred]['outlet_COD_pred'],
                                    name='出水COD_预测', line=dict(color='#2E86AB', width=2, dash='dot'),
                                    legendgroup='outlet_COD_pred'), row=1, col=1)
        fig.add_hline(y=DESIGN_LIMITS['COD']['value'], line_dash="dash", line_color="red", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_NH3'],
                                name='进水NH₃-N', line=dict(color='#F39C12', width=2, dash='solid'),
                                legendgroup='inlet_NH3'), row=2, col=1)
        mask_real = df_trend['outlet_NH3_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], 
                                    y=df_trend[mask_real]['outlet_NH3_real'],
                                    name='出水NH₃-N_实测', line=dict(color='#27AE60', width=2.5, dash='solid'),
                                    legendgroup='outlet_NH3_real'), row=2, col=1)
        mask_pred = df_trend['outlet_NH3_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], 
                                    y=df_trend[mask_pred]['outlet_NH3_pred'],
                                    name='出水NH₃-N_预测', line=dict(color='#27AE60', width=2, dash='dot'),
                                    legendgroup='outlet_NH3_pred'), row=2, col=1)
        fig.add_hline(y=DESIGN_LIMITS['NH3-N']['value'], line_dash="dash", line_color="red", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_TP'],
                                name='进水TP', line=dict(color='#8E44AD', width=2, dash='solid'),
                                legendgroup='inlet_TP'), row=3, col=1)
        mask_real = df_trend['outlet_TP_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], 
                                    y=df_trend[mask_real]['outlet_TP_real'],
                                    name='出水TP_实测', line=dict(color='#F39C12', width=2.5, dash='solid'),
                                    legendgroup='outlet_TP_real'), row=3, col=1)
        mask_pred = df_trend['outlet_TP_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], 
                                    y=df_trend[mask_pred]['outlet_TP_pred'],
                                    name='出水TP_预测', line=dict(color='#F39C12', width=2, dash='dot'),
                                    legendgroup='outlet_TP_pred'), row=3, col=1)
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

    # ---- 异常诊断与工艺优化建议（详细版） ----
    st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
    st.caption("💡 基于同类型A²/O工艺经验库 + 当前工况多维度分析")
    
    diagnoses = diagnose_system(inlet, outlet_display, pac, carbon, mlss, do)
    
    if diagnoses:
        level_order = {'critical': 0, 'warning': 1, 'info': 2}
        diagnoses.sort(key=lambda x: level_order.get(x['level'], 3))
        
        for d in diagnoses:
            # 根据级别确定样式
            if d['level'] == 'critical':
                expander_bg = "#FDEDEC"
            elif d['level'] == 'warning':
                expander_bg = "#FEF9E7"
            else:
                expander_bg = "#EBF5FB"
            
            with st.expander(f"{d['title']}（当前值：{d['current']}）", expanded=(d['level'] == 'critical')):
                st.markdown(f"""
                <div style="background:{expander_bg}; border-radius:8px; padding:12px 16px; margin-bottom:8px;">
                    <div style="font-size:13px; color:#555; margin-bottom:6px;">
                        <strong>📋 诊断背景</strong><br>
                        {d.get('background', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_r, col_a = st.columns([1, 1])
                with col_r:
                    st.markdown("**🔍 详细可能原因**")
                    for reason in d['reasons']:
                        st.markdown(f"""
                        <div style="background:white; border-radius:6px; padding:6px 12px; margin:4px 0; border-left:3px solid #888; font-size:13px;">
                            {reason}
                        </div>
                        """, unsafe_allow_html=True)
                with col_a:
                    st.markdown("**💡 针对性工艺优化措施**")
                    for action in d['actions']:
                        st.markdown(f"""
                        <div style="background:#EAF4FC; border-radius:6px; padding:6px 12px; margin:4px 0; border-left:3px solid #2E86AB; font-size:13px;">
                            {action}
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.success("✅ 系统运行正常，未检测到异常")
        st.info("📋 建议：保持当前运行参数，定期巡检设备。")

else:
    st.info("👈 请左侧输入数据")

st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 v5.9 | 双模式（手动/自动） | 详细异常诊断 | 更新时间：{beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
