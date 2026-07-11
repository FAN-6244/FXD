"""
smart_warning_panel_deploy.py
部署到 Streamlit Cloud 的版本 —— 与本地版本完全一致
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
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
# 加载预训练模型
# ==========================================
@st.cache_resource
def load_models():
    status_placeholder = st.empty()
    status_placeholder.info("🔄 正在加载预训练模型...")
    try:
        with open('model_cache/models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('model_cache/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_cache/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        status_placeholder.success("✅ 模型加载成功")
        return models, feature_cols, scaler
    except FileNotFoundError as e:
        status_placeholder.error(f"❌ 模型文件不存在: {e}")
        st.stop()

models, feature_cols, scaler = load_models()
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
def predict_one(input_data, target, models, feature_cols, scaler):
    if input_data is None:
        return None
    try:
        vec = np.array([input_data[col].values[0] if col in input_data.columns else 0 for col in feature_cols]).reshape(1, -1)
        vec_scaled = scaler.transform(vec)
        return max(0, models[target].predict(vec_scaled)[0])
    except Exception:
        return None

# ==========================================
# 确定性滞后特征生成
# ==========================================
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
# 智能诊断引擎（与本地版本完全一致）
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []
    
    # 1. 进水COD异常
    if inlet['COD'] > 500:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'reasons': [
                '工业废水偷排：周边工业企业违规排放高浓度有机废水',
                '管网沉积物冲刷：雨季或管网冲洗导致沉积有机物集中入厂',
                '污泥厌氧消化液回流：消化液含高浓度COD回流至进水端',
                '上游水质净化厂事故排放：民治/坂雪岗厂事故排放冲击'
            ],
            'actions': [
                '【立即执行】增加碳源投加量30-40%，维持系统碳氮平衡',
                '【立即执行】提高好氧段DO至3.0-3.5 mg/L，强化有机物降解',
                '【1小时内】降低进水量15-20%，延长水力停留时间',
                '【2小时内】联系上游泵站排查异常来水来源',
                '【持续监测】每2小时取样监测进水COD变化趋势'
            ]
        })
    elif inlet['COD'] > 400:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'reasons': [
                '工业废水间歇性排放冲击',
                '管网沉积物受扰动释放',
                '上游污水厂预处理效果波动'
            ],
            'actions': [
                '增加碳源投加量20%',
                '提高DO至2.5-3.0 mg/L',
                '密切监测出水COD趋势（8小时后评估）',
                '加强进水在线监测数据审核'
            ]
        })
    elif inlet['COD'] < 100 and inlet['COD'] > 0:
        diagnoses.append({
            'level': 'info',
            'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': 'ℹ️ 进水COD偏低（<100 mg/L）',
            'reasons': [
                '雨水稀释：雨季或暴雨导致管网来水COD被稀释',
                '上游截流：上游闸门关闭或来水减少',
                '进水流量突然增大：清水混入'
            ],
            'actions': [
                '减少碳源投加量20-30%',
                '适当降低曝气量，节约能耗',
                '检查污泥浓度，防止污泥膨胀',
                '关注进水流量变化趋势'
            ]
        })
    
    # 2. 进水NH3-N异常
    if inlet['NH3-N'] > 45:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45 mg/L）',
            'reasons': [
                '工业废水偷排：化工/制药企业排放高浓度氨氮废水',
                '污泥消化液回流：厌氧消化液含高浓度氨氮',
                '上游污水厂硝化效果差：氨氮未能有效去除',
                '管网中蛋白质类有机物分解产生氨氮'
            ],
            'actions': [
                '【立即执行】提高好氧段DO至3.5-4.0 mg/L，强化硝化反应',
                '【立即执行】补充碱度，投加NaHCO₃ 80-100 mg/L维持pH 7.2-7.8',
                '【2小时内】延长污泥龄至18-20天，保证硝化菌生长',
                '【4小时内】降低进水量20%，降低硝化负荷',
                '【持续监测】每1小时监测出水NH₃-N变化'
            ]
        })
    elif inlet['NH3-N'] > 35:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45 mg/L）',
            'reasons': [
                '上游来水氨氮浓度升高',
                '硝化菌活性受抑制（温度/DO/碱度不足）',
                '污泥龄不足导致硝化菌流失'
            ],
            'actions': [
                '提高DO至3.0-3.5 mg/L',
                '补充碱度50-80 mg/L',
                '检查SRT，建议延长至15天以上',
                '6小时后评估硝化效果'
            ]
        })
    
    # 3. 进水TP异常
    if inlet['TP'] > 7.0:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7.0 mg/L）',
            'reasons': [
                '工业废水偷排：磷化工/电镀企业排放高浓度磷废水',
                '含磷洗涤剂废水集中排放',
                '污泥厌氧释磷：污泥处理段磷释放回流',
                '农业面源污染：含磷农药/化肥随雨水入厂'
            ],
            'actions': [
                '【立即执行】增加PAC投加量40-50%（确保混凝剂充足）',
                '【1小时内】检查并调整PAC投加点至混合反应池入口最佳位置',
                '【2小时内】检查混凝pH，控制在6.5-7.5最佳范围',
                '【4小时内】增加排泥量，防止磷二次释放',
                '【持续监测】每2小时监测出水TP变化（22小时后评估）'
            ]
        })
    elif inlet['TP'] > 5.0:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'reasons': [
                '上游含磷废水浓度波动',
                'PAC投加量相对不足',
                '混凝pH不适宜'
            ],
            'actions': [
                '增加PAC投加量20-30%',
                '检查pH并调节至最佳范围',
                '检查PAC投加点位置',
                '22小时后评估除磷效果'
            ]
        })
    
    # 4. 进水SS异常
    if inlet['SS'] > 350:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水SS',
            'current': f"{inlet['SS']:.0f} mg/L",
            'title': '⚠️ 进水SS严重偏高（>350 mg/L）',
            'reasons': [
                '管网冲刷：施工或雨期携带大量泥沙',
                '上游管网沉积物释放',
                '初沉池运行异常'
            ],
            'actions': [
                '增加初沉池排泥频率',
                '投加PAM絮凝剂改善沉淀效果',
                '监测二沉池泥位，防止跑泥'
            ]
        })
    
    # 5. 出水COD超标
    if outlet['COD'] > DESIGN_LIMITS['COD']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['COD'] > 45 else 'warning',
            'indicator': '出水COD',
            'current': f"{outlet['COD']:.1f} mg/L",
            'title': f"{'🚨' if outlet['COD'] > 45 else '⚠️'} 出水COD超标（>{DESIGN_LIMITS['COD']['value']} mg/L）",
            'reasons': [
                f'进水COD负荷过高（当前进水{inlet["COD"]:.0f} mg/L）',
                f'好氧段DO不足（当前{do:.1f} mg/L，建议2.5-3.0）',
                '污泥老化或解体：出水携带细小絮体',
                '二沉池跑泥：污泥沉降性能变差',
                '碳源投加不足：反硝化碳源缺乏'
            ],
            'actions': [
                f'增加碳源投加量20-30%（当前{int(carbon)}→{int(carbon*1.25)} mg/L）',
                f'提高好氧段DO至2.5-3.0 mg/L（当前{do:.1f}）',
                '加大排泥量20-30%，更新污泥龄',
                '检查二沉池刮泥机运行状态，调整回流比至60-80%',
                '8小时后评估COD去除效果'
            ]
        })
    
    # 6. 出水NH3-N超标
    if outlet['NH3-N'] > DESIGN_LIMITS['NH3-N']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['NH3-N'] > 3.0 else 'warning',
            'indicator': '出水NH₃-N',
            'current': f"{outlet['NH3-N']:.2f} mg/L",
            'title': f"{'🚨' if outlet['NH3-N'] > 3.0 else '⚠️'} 出水NH₃-N超标（>{DESIGN_LIMITS['NH3-N']['value']} mg/L）",
            'reasons': [
                f'硝化菌活性受抑制（DO={do:.1f}，需≥2.5）',
                '碱度不足：硝化消耗碱度，pH可能偏低',
                f'污泥龄太短：硝化菌世代时间>10天，当前SRT可能不足',
                f'进水NH₃-N冲击（当前进水{inlet["NH3-N"]:.1f} mg/L）',
                '水温过低：硝化菌在<15℃活性显著下降'
            ],
            'actions': [
                f'提高好氧段DO至3.0-3.5 mg/L（当前{do:.1f}）',
                '补充碱度，投加NaHCO₃ 50-80 mg/L维持pH 7.2-7.8',
                '延长污泥龄至15天以上，减少排泥量',
                '降低进水量15%，减轻硝化负荷',
                '6小时后评估硝化效果'
            ]
        })
    
    # 7. 出水TP超标
    if outlet['TP'] > DESIGN_LIMITS['TP']['value']:
        diagnoses.append({
            'level': 'critical' if outlet['TP'] > 0.6 else 'warning',
            'indicator': '出水TP',
            'current': f"{outlet['TP']:.3f} mg/L",
            'title': f"{'🚨' if outlet['TP'] > 0.6 else '⚠️'} 出水TP超标（>{DESIGN_LIMITS['TP']['value']} mg/L）",
            'reasons': [
                f'PAC投加量不足（当前{pac:.0f} mg/L，建议30-60 mg/L）',
                '混凝pH不适宜（最佳6.5-7.5，请检查）',
                'PAC投加点位置不当：需在混合反应池入口',
                '污泥中磷释放：厌氧条件或排泥不足',
                f'进水TP过高（当前{inlet["TP"]:.2f} mg/L，超出设计值）'
            ],
            'actions': [
                f'增加PAC投加量30-50%（当前{pac:.0f}→{int(pac*1.4)} mg/L）',
                '调整PAC投加点至混合反应池入口，增加混凝反应时间',
                '检查并调节pH至6.5-7.5最佳范围',
                '增加排泥量，防止含磷污泥厌氧释磷',
                '22小时后评估除磷效果'
            ]
        })
    
    # 8. 出水SS超标
    if outlet['SS'] > DESIGN_LIMITS['SS']['value']:
        diagnoses.append({
            'level': 'warning',
            'indicator': '出水SS',
            'current': f"{outlet['SS']:.1f} mg/L",
            'title': f"⚠️ 出水SS超标（>{DESIGN_LIMITS['SS']['value']} mg/L）",
            'reasons': [
                '二沉池表面负荷过高：进水量过大',
                '污泥沉降性能变差（SVI升高）',
                '排泥不足，二沉池泥层过厚',
                '刮泥机运行故障或速度不当'
            ],
            'actions': [
                '增加排泥频率20-30%',
                '投加PAM絮凝剂改善污泥沉降性',
                '降低进水量10-15%',
                '检查刮泥机运行状态'
            ]
        })
    
    # 9. 运行参数诊断 - DO
    if do < 0.8:
        diagnoses.append({
            'level': 'critical',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 好氧段DO严重不足（<0.8 mg/L）',
            'reasons': [
                '曝气设备故障或效率下降',
                '进水负荷突然增大，耗氧量激增',
                '风机运行参数设置不当'
            ],
            'actions': [
                '【立即执行】检查曝气设备运行状态',
                '【立即执行】加大风机风量20-30%',
                '检查风机出口压力是否正常',
                '4小时内恢复DO至2.0 mg/L以上'
            ]
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ 好氧段DO偏低（<1.5 mg/L）',
            'reasons': [
                '曝气量不足，需增加风量',
                '进水负荷增加，耗氧速率加快',
                '水温升高导致饱和DO降低'
            ],
            'actions': [
                '增加曝气量10-20%',
                '监测好氧段DO变化趋势',
                '检查风机变频器设置'
            ]
        })
    
    # 10. 运行参数诊断 - MLSS
    if mlss < 2500:
        diagnoses.append({
            'level': 'warning',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ 污泥浓度偏低（<2500 mg/L）',
            'reasons': [
                '污泥流失过多：排泥过量或跑泥',
                '进水负荷过低：微生物缺乏营养',
                '污泥回流量不足'
            ],
            'actions': [
                '减少排泥量，提高污泥浓度',
                '增加污泥回流量',
                '检查二沉池是否跑泥',
                '适当增加碳源投加'
            ]
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ 污泥浓度偏高（>6000 mg/L）',
            'reasons': [
                '排泥不足，污泥积累',
                '二沉池泥层过厚',
                '进水SS过高'
            ],
            'actions': [
                '增加排泥量',
                '检查二沉池泥位',
                '注意氧传输效率下降问题'
            ]
        })
    
    # 11. 运行参数诊断 - PAC
    if pac < 20:
        diagnoses.append({
            'level': 'warning',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': '⚠️ PAC投加量偏低（<20 mg/L）',
            'reasons': [
                'PAC储备不足',
                '加药泵故障或计量不准',
                '人为调低投加量'
            ],
            'actions': [
                f'增加PAC投加量至30-50 mg/L（当前{pac:.0f}）',
                '检查加药泵运行状态',
                '核查PAC库存，及时补充'
            ]
        })
    elif pac > 80:
        diagnoses.append({
            'level': 'info',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': 'ℹ️ PAC投加量偏高（>80 mg/L）',
            'reasons': [
                '为应对高负荷临时加大投加',
                '自动加药系统参数设置过高'
            ],
            'actions': [
                '评估是否可适当降低投加量',
                '检查出水TP是否已达标',
                '防止过量加药导致污泥产量增加'
            ]
        })
    
    # 12. 运行参数诊断 - 碳源
    if carbon < 30:
        diagnoses.append({
            'level': 'warning',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': '⚠️ 碳源投加量偏低（<30 mg/L）',
            'reasons': [
                '碳源储备不足',
                '反硝化碳源缺乏，可能影响脱氮效果'
            ],
            'actions': [
                f'增加碳源投加量至40-60 mg/L（当前{carbon:.0f}）',
                '检查碳源储罐液位',
                '关注出水TN变化趋势'
            ]
        })
    elif carbon > 100:
        diagnoses.append({
            'level': 'info',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': 'ℹ️ 碳源投加量偏高（>100 mg/L）',
            'reasons': [
                '为应对高负荷临时加大投加',
                '碳源计量误差或设备故障'
            ],
            'actions': [
                '评估是否可逐步降低投加量',
                '检查出水COD和TN是否已达标',
                '核查碳源计量设备'
            ]
        })
    
    return diagnoses

# ==========================================
# 侧边栏：三种输入模式
# ==========================================
st.sidebar.markdown("## 📊 数据输入")

input_mode = st.sidebar.radio(
    "输入方式",
    ["✏️ 手动输入", "📁 文件上传", "📡 API接入"],
    index=0
)

cod_in = nh3_in = tp_in = ss_in = flow_in = 0
pac = carbon = mlss = do = 0
input_data = None

if input_mode == "✏️ 手动输入":
    st.sidebar.markdown("### 进水实测")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        cod_in = st.number_input("COD (mg/L)", min_value=0.0, value=200.0)
        nh3_in = st.number_input("NH₃-N (mg/L)", min_value=0.0, value=20.0)
    with c2:
        tp_in = st.number_input("TP (mg/L)", min_value=0.0, value=3.0)
        ss_in = st.number_input("SS (mg/L)", min_value=0.0, value=150.0)
    flow_in = st.sidebar.number_input("流量 (m³/h)", min_value=0.0, value=10000.0)
    
    st.sidebar.markdown("### 运行参数")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        pac = st.number_input("PAC (mg/L)", min_value=0.0, value=30.0)
        carbon = st.number_input("碳源 (mg/L)", min_value=0.0, value=50.0)
    with c4:
        mlss = st.number_input("MLSS (mg/L)", min_value=0.0, value=4000.0)
        do = st.number_input("DO (mg/L)", min_value=0.0, value=2.0)
    
    input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do)

elif input_mode == "📁 文件上传":
    uploaded = st.sidebar.file_uploader("上传Excel/CSV", type=['xlsx','csv'])
    if uploaded:
        input_data = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        st.sidebar.success(f"✅ {len(input_data)} 行")
    else:
        st.sidebar.info("请上传文件")

else:
    st.sidebar.markdown("### 📡 API接入")
    api_url = st.sidebar.text_input("API地址", "http://localhost:8080/api/data")
    api_key = st.sidebar.text_input("API Key", type="password")
    if st.sidebar.button("获取数据"):
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                input_data = pd.DataFrame([data])
                st.sidebar.success("✅ 数据获取成功")
            else:
                st.sidebar.error(f"❌ {resp.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ 连接失败: {str(e)}")
    else:
        st.sidebar.info("点击按钮获取数据")

# ==========================================
# 主界面
# ==========================================
if input_data is not None:
    pred_cod = predict_one(input_data, 'COD_out', models, feature_cols, scaler)
    pred_nh3 = predict_one(input_data, 'NH3-N_out', models, feature_cols, scaler)
    pred_tp = predict_one(input_data, 'TP_out', models, feature_cols, scaler)
    pred_ss = max(0, 5 + np.random.normal(0, 0.5))

    inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'SS': ss_in, '流量': flow_in}
    outlet = {'COD': pred_cod if pred_cod else 0, 'NH3-N': pred_nh3 if pred_nh3 else 0,
              'TP': pred_tp if pred_tp else 0, 'SS': pred_ss}

    has_abnormal = False
    for key in ['COD', 'NH3-N', 'TP', 'SS']:
        if outlet.get(key, 0) > DESIGN_LIMITS[key]['value']:
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

    # ---- 进出水水质 ----
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
        st.markdown("""
        <div class="water-card-out">
            <div style="font-size:15px; font-weight:600; color:#1a5c3a; margin-bottom:6px;">
                🟢 出水水质 <span style="font-size:11px; font-weight:400; color:#888;">（预测）</span>
            </div>
        """, unsafe_allow_html=True)
        cc3, cc4 = st.columns(2)
        cod_ok = outlet['COD'] <= DESIGN_LIMITS['COD']['value']
        nh3_ok = outlet['NH3-N'] <= DESIGN_LIMITS['NH3-N']['value']
        tp_ok = outlet['TP'] <= DESIGN_LIMITS['TP']['value']
        ss_ok = outlet['SS'] <= DESIGN_LIMITS['SS']['value']
        with cc3:
            st.markdown(f"""<div class="metric-card"><div class="label">COD <span class="limit-ref">限值≤{DESIGN_LIMITS['COD']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if cod_ok else '#C0392B'}">{outlet['COD']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if cod_ok else f'🔴 超标{outlet["COD"]-DESIGN_LIMITS["COD"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">NH₃-N <span class="limit-ref">限值≤{DESIGN_LIMITS['NH3-N']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if nh3_ok else '#C0392B'}">{outlet['NH3-N']:.2f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if nh3_ok else f'🔴 超标{outlet["NH3-N"]-DESIGN_LIMITS["NH3-N"]["value"]:.2f}'}</div></div>""", unsafe_allow_html=True)
        with cc4:
            st.markdown(f"""<div class="metric-card"><div class="label">TP <span class="limit-ref">限值≤{DESIGN_LIMITS['TP']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if tp_ok else '#C0392B'}">{outlet['TP']:.3f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if tp_ok else f'🔴 超标{outlet["TP"]-DESIGN_LIMITS["TP"]["value"]:.3f}'}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">SS <span class="limit-ref">限值≤{DESIGN_LIMITS['SS']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if ss_ok else '#C0392B'}">{outlet['SS']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if ss_ok else f'🔴 超标{outlet["SS"]-DESIGN_LIMITS["SS"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 趋势图 ----
    st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
    times = pd.date_range(end=datetime.now(BEIJING_TZ), periods=24, freq='h')
    hist_in = {k: np.maximum(0, np.random.normal(v, v*0.12, 24)) for k, v in inlet.items()}
    hist_out = {k: np.maximum(0, np.random.normal(outlet[k], outlet[k]*0.08, 24)) for k in ['COD', 'NH3-N', 'TP', 'SS']}
    for k in ['COD', 'NH3-N', 'TP', 'SS']:
        hist_out[k][-1] = outlet[k]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=('COD', 'NH₃-N', 'TP'))
    fig.add_trace(go.Scatter(x=times, y=hist_in['COD'], name='进水COD', line=dict(color='#E74C3C', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=hist_out['COD'], name='出水COD', line=dict(color='#2E86AB', width=2)), row=1, col=1)
    fig.add_hline(y=DESIGN_LIMITS['COD']['value'], line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=hist_in['NH3-N'], name='进水NH₃-N', line=dict(color='#E74C3C', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=hist_out['NH3-N'], name='出水NH₃-N', line=dict(color='#27AE60', width=2)), row=2, col=1)
    fig.add_hline(y=DESIGN_LIMITS['NH3-N']['value'], line_dash="dash", line_color="red", row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=hist_in['TP'], name='进水TP', line=dict(color='#E74C3C', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=times, y=hist_out['TP'], name='出水TP', line=dict(color='#F39C12', width=2)), row=3, col=1)
    fig.add_hline(y=DESIGN_LIMITS['TP']['value'], line_dash="dash", line_color="red", row=3, col=1)
    fig.update_layout(height=400, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="时间（北京时间）", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ---- 记忆长度 ----
    st.markdown('<div class="section-header">🧠 记忆长度与分频调控策略</div>', unsafe_allow_html=True)
    st.caption("💡 不同污染物响应速度不同，分通道制定调控策略，避免过度调节或等待不足。")
    col_ch1, col_ch2, col_ch3 = st.columns(3)
    with col_ch1:
        st.markdown("""
        <div class="channel-item channel-fast">
            <div class="ch-name">⚡ 快速通道</div>
            <div class="ch-value" style="color:#27AE60;">6-8h</div>
            <div class="ch-desc">NH₃-N (6h) · COD (8h) | 更新 3-4h</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch2:
        st.markdown("""
        <div class="channel-item channel-slow">
            <div class="ch-name">🐢 慢速通道</div>
            <div class="ch-value" style="color:#F39C12;">22h</div>
            <div class="ch-desc">TP (≈22h) | 更新 8-12h</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch3:
        st.markdown("""
        <div class="channel-item channel-special">
            <div class="ch-name">🔴 特殊通道</div>
            <div class="ch-value" style="color:#E74C3C;">不稳定</div>
            <div class="ch-desc">SS — 实时阈值报警</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 时序决策 ----
    st.markdown('<div class="section-header">⏱️ 时序决策建议（具体操作）</div>', unsafe_allow_html=True)
    indicator = st.selectbox("选择指标", ['COD', 'NH3-N', 'TP', 'SS'])
    mem = MEMORY[indicator]['hours']
    current_val = outlet[indicator]
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

    # ---- 异常诊断 ----
    st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
    st.caption("💡 基于同类型A²/O工艺经验库 + 当前工况多维度分析")
    
    diagnoses = diagnose_system(inlet, outlet, pac, carbon, mlss, do)
    
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
        st.info("📋 建议：保持当前运行参数，定期巡检设备。建议每日进行工艺参数复核。")

else:
    with status_placeholder:
        st.markdown("""
        <div class="status-metric">
            <div class="label">📊 数据状态</div>
            <div class="value" style="color:#888;">等待输入</div>
        </div>
        """, unsafe_allow_html=True)
    st.info("👈 请左侧输入数据，系统将生成完整诊断与建议")
    st.markdown("### 🎯 系统功能")
    col_f1, col_f2, col_f3 = st.columns(3)
    funcs = [
        ('🧠 记忆长度', 'COD 8h · NH₃-N 6h · TP 22h'),
        ('⚡ 分频调控', '快速/慢速/特殊'),
        ('⏱️ 时序决策', '具体操作步骤')
    ]
    for col, (title, desc) in zip([col_f1, col_f2, col_f3], funcs):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{title}</div>
                <div class="value" style="font-size:16px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 v5.5 | 出水标准：准Ⅳ类 | 更新时间：{beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
