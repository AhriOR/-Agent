import os
import pandas as pd
import tempfile
import glob
import streamlit as st
from langchain_chroma import Chroma
from agent_tool.embedding import TransformerEmbedding
from deal_function import pdf_loader, txt_loader, agent_response

USER_DB_PATH = "user_db.csv"

def main():
    st.set_page_config(page_title='📄 简历问答助手', layout='centered')


    def local_css():
        st.markdown("""
        <style>
            /* ===== 全局样式 ===== */
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --accent: #4895ef;
                --light: #f8f9fa;
                --dark: #212529;
                --success: #4cc9f0;
                --warning: #f72585;
                --muted: #6c757d;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            html, body {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
                font-family: 'Segoe UI', 'Helvetica Neue', -apple-system, BlinkMacSystemFont, sans-serif;
                color: var(--dark);
                line-height: 1.6;
                scroll-behavior: smooth;
            }

            /* ===== 主容器 ===== */
            .main {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem 1rem;
            }

            /* ===== 标题样式 ===== */
            h1, h2, h3, h4 {
                font-weight: 700;
                line-height: 1.2;
                margin-bottom: 1rem;
                color: var(--dark);
            }

            h1 {
                font-size: 2.5rem;
                background: linear-gradient(to right, var(--primary), var(--warning));
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                text-align: center;
                margin-bottom: 1.5rem;
                position: relative;
                padding-bottom: 0.5rem;
            }

            h1::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 100px;
                height: 4px;
                background: linear-gradient(to right, var(--primary), var(--warning));
                border-radius: 2px;
            }

            h2 {
                font-size: 1.8rem;
                color: var(--secondary);
                margin-top: 2rem;
                border-left: 4px solid var(--accent);
                padding-left: 1rem;
            }

            /* ===== 卡片设计 ===== */
            .card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                padding: 2rem;
                margin-bottom: 2rem;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.8);
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
            }

            /* ===== 输入框样式 ===== */
            .stTextInput>div>div>input,
            .stTextArea>div>textarea,
            .stSelectbox>div>select,
            .stNumberInput>div>input {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 0.75rem 1rem;
                font-size: 1rem;
                transition: all 0.3s;
                background: white;
            }

            .stTextInput>div>div>input:focus,
            .stTextArea>div>textarea:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(72, 149, 239, 0.2);
                outline: none;
            }

            /* ===== 按钮设计 ===== */
            .stButton>button {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                font-weight: 600;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 44px;
            }

            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(67, 97, 238, 0.4);
                background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            }

            .stButton>button:active {
                transform: translateY(0);
            }

            /* ===== 上传区域 ===== */
            .upload-box {
                border: 2px dashed var(--accent);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                background: rgba(72, 149, 239, 0.05);
                transition: all 0.3s;
                margin: 1rem 0;
            }

            .upload-box:hover {
                background: rgba(72, 149, 239, 0.1);
                border-color: var(--primary);
            }

            .upload-box .icon {
                font-size: 2.5rem;
                color: var(--accent);
                margin-bottom: 1rem;
            }

            /* ===== 分隔线 ===== */
            hr {
                border: none;
                height: 1px;
                background: linear-gradient(to right, transparent, rgba(0, 0, 0, 0.1), transparent);
                margin: 2.5rem 0;
            }

            /* ===== 侧边栏 ===== */
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
                box-shadow: 2px 0 15px rgba(0, 0, 0, 0.05);
                padding: 2rem 1.5rem;
            }

            /* ===== 响应式设计 ===== */
            @media (max-width: 768px) {
                .main {
                    padding: 1rem;
                }

                h1 {
                    font-size: 2rem;
                }

                .card {
                    padding: 1.5rem;
                }
            }

            /* ===== 动画效果 ===== */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .fade-in {
                animation: fadeIn 0.6s ease forwards;
            }

            /* ===== 工具提示 ===== */
            .tooltip {
                position: relative;
                display: inline-block;
            }

            .tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: var(--dark);
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 0.5rem;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.9rem;
            }

            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }

            /* ===== 标签样式 ===== */
            .tag {
                display: inline-block;
                background: var(--accent);
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 50px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
            }

            /* ===== 进度条 ===== */
            .stProgress > div > div > div {
                background: linear-gradient(to right, var(--primary), var(--success));
            }
        </style>
        """, unsafe_allow_html=True)

    local_css()
    st.markdown("<h1 style='text-align:center;'>📄 简历问答助手</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: gray;'>从自然语言需求生成专业简历，支持注册、登录与文档上传</p>", unsafe_allow_html=True)
    st.markdown("---")
    # 📘 使用说明放在侧边栏
    with st.sidebar:
        st.markdown("## 📘 使用说明")
        st.markdown("""
            **欢迎使用简历问答助手！**

            **操作流程：**
            1. 🆕 注册 或 👤 登录账号  
            2. 📝 输入简历需求 或 📤 上传文件  
            3. 🚀 提交后生成结果，可 📄 下载 PDF  
            4. 🔓 退出登录 以切换账号

            ---
            ⚠️ **注意事项：**
            - 如遇无法复制出 meta tensor;暂无数据！将模块从 meta 移动到其他设备时，请使用 torch.nn.Module.to_empty（） 而不是 torch.nn.Module.to（），请刷新应用
            - 如遇找不到字典项与google未返回内容的问题，请重新尝试输入，智能体execute的内容总是不一样的
            - 请勿频繁刷新或关闭页面，否则登录状态将被重置。
            - 上传仅支持 `.pdf` 和 `.txt` 格式。
            """)

    # 后续 local_css、login、register 等逻辑...
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # ===== 🆕 注册模块 =====
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🆕 注册新用户")
        new_user = st.text_input("设置用户名", key="register_user")
        new_pass = st.text_input("设置密码", type="password", key="register_pass")

        if st.button("注册"):
            if new_user.strip() == "" or new_pass.strip() == "":
                st.warning("⚠️ 用户名和密码不能为空")
            else:
                if os.path.exists(USER_DB_PATH):
                    df = pd.read_csv(USER_DB_PATH)
                else:
                    df = pd.DataFrame(columns=["username", "password", "login_state"])

                if new_user in df["username"].values:
                    st.warning("⚠️ 用户名已存在")
                else:
                    new_entry = pd.DataFrame(
                        [[new_user, new_pass, False]],
                        columns=["username", "password", "login_state"]
                    )
                    df = pd.concat([df, new_entry], ignore_index=True)
                    df.to_csv(USER_DB_PATH, index=False)
                    st.success(f"✅ 用户 `{new_user}` 注册成功")
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 👤 登录模块 =====
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👤 用户登录")
        login_user = st.text_input("用户名", key="login_user")
        login_pass = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录"):
            if st.session_state.user_id is not None:
                st.warning("⚠️ 请先退出登录")
            else:
                df=pd.read_csv(USER_DB_PATH)
                for state in df['login_state']:
                    if state==True:
                        df["login_state"] = False
                        df.to_csv(USER_DB_PATH, index=False)
                if os.path.exists(USER_DB_PATH):
                    df = pd.read_csv(USER_DB_PATH)
                    match = (df["username"] == login_user) & (df["password"] == login_pass)
                    if match.any():
                        st.session_state.user_id = login_user
                        df.loc[df["username"] == login_user, "login_state"] = True
                        df.to_csv(USER_DB_PATH, index=False)
                        st.success(f"✅ 用户 `{login_user}` 登录成功")
                        st.session_state.logout_state=False
                    else:
                        st.warning("⚠️ 用户名或密码错误")
                else:
                    st.warning("⚠️ 用户数据库不存在")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 📝 主功能模块 =====
    if st.session_state.user_id is not None:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📝 输入你的简历需求")
            user_input = st.text_area("如：生成一份适合 AI 算法岗位的简历", height=150)
            if st.button("🚀 提交请求"):
                if not user_input.strip():
                    st.warning("⚠️ 请输入你的简历需求")
                else:
                    with st.spinner("⏳ 正在生成回复，请稍候..."):
                        response = agent_response(user_input)
                        st.success("✅ 生成成功")
                        with st.expander("📬 查看回复", expanded=True):
                            if isinstance(response, str):
                                pdf_files = glob.glob("outputs/*.pdf")
                                if pdf_files:
                                    latest_pdf = max(pdf_files, key=os.path.getmtime)
                                    with open(latest_pdf, "rb") as f:
                                        st.download_button(
                                            label="📄 下载 PDF 简历",
                                            data=f.read(),
                                            file_name="my_resume.pdf",
                                            mime="application/pdf"
                                        )
                                    os.remove(latest_pdf)
                                else:
                                    st.write(response)
                            else:
                                st.warning("⚠️ 未知响应格式")

            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📤 上传已有简历文档")
            st.markdown("支持 `.pdf` 与 `.txt` 格式，可用于问答分析")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="upload-box">📄 上传 PDF 文件</div>', unsafe_allow_html=True)
                pdf_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
                if pdf_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.read())
                        docs = pdf_loader(tmp.name, st.session_state.user_id)
                        vectorstore.add_documents(docs)
                        os.remove(tmp.name)
                        st.success(f"✅ 成功加载 {len(docs)} 页 PDF 内容")

            with col2:
                st.markdown('<div class="upload-box">📄 上传 TXT 文件</div>', unsafe_allow_html=True)
                txt_file = st.file_uploader("", type=["txt"], label_visibility="collapsed")
                if txt_file:
                    txt_file.seek(0)  # 确保文件指针在开头
                    content = txt_file.read()
                    if isinstance(content, str):
                        content = content.encode('utf-8')  # 转为 bytes
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                        tmp.write(content)
                        tmp.flush()
                        tmp_path = tmp.name
                    docs = txt_loader(tmp_path, st.session_state.user_id)
                    vectorstore.add_documents(docs)
                    os.remove(tmp_path)
                    st.success(f"✅ 成功加载 {len(docs)} 页 TXT 内容")

            st.markdown("</div>", unsafe_allow_html=True)

        # ✅ 添加退出登录功能
        if st.button("🔓 退出登录"):
            df = pd.read_csv(USER_DB_PATH)
            df.loc[df["username"] == st.session_state.user_id, "login_state"] = False
            df.to_csv(USER_DB_PATH, index=False)
            st.session_state.user_id = None
            st.success("✅ 已退出登录，请重新登录")
            st.session_state.logout_state=True

    else:
        st.info("🔐 请先登录后再使用简历生成与上传功能")

# ===== 启动向量数据库与模型，并运行应用 =====
if __name__ == '__main__':
    persist_directory = './db/chroma'
    os.makedirs(persist_directory, exist_ok=True)
    embedding = TransformerEmbedding()
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_name="default"
    )
    main()
