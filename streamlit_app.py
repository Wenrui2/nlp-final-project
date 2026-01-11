import streamlit as st
# 修改点1: 现在的 OpenAI 库迁移到了 langchain_community
from langchain_community.llms import OpenAI 

st.set_page_config(page_title="🦜🔗 Quickstart App")
st.title('🦜🔗 Quickstart App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
  # 实例化模型
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  
  # 修改点2: 使用 .invoke() 方法，而不是直接调用
  response = llm.invoke(input_text)
  
  # 兼容性处理：如果返回的是对象则提取内容，如果是字符串直接显示
  if hasattr(response, 'content'):
      st.info(response.content)
  else:
      st.info(response)

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
