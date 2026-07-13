# ==========================================
# 侧边栏：四种数据输入模式
# ==========================================
st.sidebar.markdown("## 📊 数据输入模式")
input_mode_global = st.sidebar.radio(
    "选择数据模式",
    ["✏️ 手动输入", "📁 文件上传", "📡 API接入", "🔄 自动实时（模拟）"],
    index=0,
    help="手动输入：单次预测；文件上传：批量预测；API接入：实时拉取；自动模拟：演示数据流"
)

# --- 定义固定的表头模板（用于文件上传校验） ---
REQUIRED_COLS = ['COD', 'NH3-N', 'TP', 'SS', '流量', 'PAC', '碳源', 'MLSS', 'DO']

# --- 1. 手动输入（保留原版样式） ---
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

# --- 2. 文件上传（固定模板） ---
elif input_mode_global == "📁 文件上传":
    st.sidebar.markdown("### 📁 上传数据文件")
    st.sidebar.caption("请上传包含以下列的 Excel/CSV 文件：")
    st.sidebar.code("COD, NH3-N, TP, SS, 流量, PAC, 碳源, MLSS, DO", language='text')
    
    # 提供模板下载
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
            
            # 校验列名
            missing_cols = set(REQUIRED_COLS) - set(df_upload.columns)
            if missing_cols:
                st.sidebar.error(f"❌ 缺少必需列：{missing_cols}")
                st.sidebar.info("请下载模板，按模板格式填写后重新上传。")
                input_data = None
            else:
                # 取第一行作为输入
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
                st.sidebar.success(f"✅ 成功加载数据 (共 {len(df_upload)} 行，使用第一行)")
        except Exception as e:
            st.sidebar.error(f"❌ 文件解析失败：{str(e)}")
            input_data = None
    else:
        st.sidebar.info("请上传文件")
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
                # 如果 API 返回的字段名不同，请在这里修改映射
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
        st.sidebar.info("点击按钮获取数据")
        input_data = None

# --- 4. 自动实时（模拟）--- 
else:  # 原自动模式，完全保留
    st.sidebar.markdown("### 🔄 自动实时数据")
    st.sidebar.info("🔄 每5秒自动生成一组模拟数据，模拟实时数据流")
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
