"""
retrain_from_db.py
从 Supabase 拉取全部历史数据，重新训练模型
"""
import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from supabase import create_client

SUPABASE_URL = "https://esoulexcrpdeeoumoili.supabase.co"
SUPABASE_KEY = "sb_publishable_m0hz9Rv8NB_ziC5xKCltMg_Ij5Od60Q"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("📊 正在从 Supabase 拉取历史数据...")
result = supabase.table('feedback_data').select('*').execute()
df = pd.DataFrame(result.data)

if len(df) < 100:
    print(f"⚠️ 数据量较少（{len(df)} 条），建议积累更多数据后重训")
    exit(0)

print(f"✅ 已拉取 {len(df)} 条历史数据")

# ---- 特征工程 ----
df['COD_load'] = df['cod_in'] * df['flow_in'] / 1000
df['NH3_load'] = df['nh3_in'] * df['flow_in'] / 1000
df['TP_load'] = df['tp_in'] * df['flow_in'] / 1000

base = ['COD_load', 'NH3_load', 'TP_load', 'flow_in']
for h in range(1, 49):
    for col in base:
        df[f'{col}_lag{h}'] = df[col].shift(h)

df = df.dropna()
feature_cols = [c for c in df.columns if 'lag' in c]

X = df[feature_cols].values
y_cod = df['cod_real'].values
y_nh3 = df['nh3_real'].values
y_tp = df['tp_real'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split = int(len(X) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]

# ---- 训练模型 ----
print("🔄 正在训练模型...")
models = {}
targets = [('COD_out', y_cod), ('NH3-N_out', y_nh3), ('TP_out', y_tp)]
for name, y in targets:
    y_train, y_test = y[:split], y[split:]
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    models[name] = model
    print(f"  ✅ {name} 训练完成")

# ---- 保存模型 ----
os.makedirs('model_cache', exist_ok=True)
with open('model_cache/models.pkl', 'wb') as f:
    pickle.dump(models, f)
with open('model_cache/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model_cache/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"✅ 模型已保存到 model_cache/，使用 {len(X_train)} 条训练样本")
