import streamlit as st
# 修改点1: 从 chat_models 导入 ChatOpenAI，而不是用老的 llms.OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

st.set_page_config(page_title="🦜🔗 Quickstart App")
st.title('🦜🔗 Quickstart App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    # 修改点2: 使用 ChatOpenAI
    # model_name 默认是 gpt-3.5-turbo，这比老接口更便宜、更智能且不容易报错
    llm = ChatOpenAI(
        temperature=0.7, 
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo" 
    )
  
    # 修改点3: 调用 invoke
    response = llm.invoke(input_text)
  
    # 修改点4: ChatModel 返回的是一个消息对象，必须用 .content 获取内容
    st.info(response.content)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
  
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
  
    if submitted and openai_api_key.startswith('sk-'):
        # 增加一个 try-except 块，这样如果报错，会在网页上显示具体原因，而不是直接崩溃
        try:
            generate_response(text)
        except Exception as e:
            st.error(f"发生错误: {e}")
