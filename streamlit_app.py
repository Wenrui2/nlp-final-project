import streamlit as st
# 修改点1: 这里的导入路径变了，这是导致你报错的根本原因
from langchain_community.chat_models import ChatOpenAI
# 修改点2: 使用 langchain_core 来导入消息对象，这是新版标准
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="🦜🔗 Quickstart App")
st.title('🦜🔗 Quickstart App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    # 实例化模型
    llm = ChatOpenAI(
        temperature=0.7, 
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo" 
    )
  
    # 调用 invoke
    response = llm.invoke(input_text)
  
    # 显示结果
    st.info(response.content)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
  
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
  
    if submitted and openai_api_key.startswith('sk-'):
        try:
            generate_response(text)
        except Exception as e:
            st.error(f"发生错误: {e}")
