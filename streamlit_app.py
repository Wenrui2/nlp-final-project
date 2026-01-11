import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="NLP期末大作业-智能助手")
st.title('🤖 NLP期末大作业 - 智能问答系统')

# 提示用户输入 Key
st.markdown("### 请输入 DeepSeek API Key")
st.markdown("没有Key? [点击这里注册获取](https://platform.deepseek.com/) (新用户免费送额度)")
openai_api_key = st.sidebar.text_input('API Key', type='password')

def generate_response(input_text):
    # --- 关键修改开始 ---
    llm = ChatOpenAI(
        temperature=0.7, 
        openai_api_key=openai_api_key,
        # 1. 这里填 DeepSeek 的地址
        base_url="https://api.deepseek.com", 
        # 2. 这里填 DeepSeek 的模型名称
        model_name="deepseek-chat"           
    )
    # --- 关键修改结束 ---
  
    # 显示加载状态
    with st.spinner('AI 正在思考中...'):
        response = llm.invoke(input_text)
        st.info(response.content)

with st.form('my_form'):
    text = st.text_area('请输入问题:', '自然语言处理中 BERT 模型的核心原理是什么？')
    submitted = st.form_submit_button('提交运行')
  
    if not openai_api_key:
        st.warning('请先在左侧输入 API Key!', icon='⚠')
  
    if submitted and openai_api_key:
        try:
            generate_response(text)
        except Exception as e:
            st.error(f"发生错误: {e}")
            st.markdown("##### 常见错误排查：")
            st.markdown("1. 确保你用的是 **DeepSeek** 的 Key，而不是 OpenAI 的。")
            st.markdown("2. 确保 Key 没有多复制空格。")
