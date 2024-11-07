import os
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# Agent Chain 생성 함수
def create_agent_chain(history):
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),  # 템퍼러처를 float으로 변환
    )
    
    # 툴 로딩
    tools = load_tools(["ddg-search", "wikipedia"])
    
    # 프롬프트 로딩
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # 메모리 객체 설정
    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )
    
    # 에이전트 생성
    agent = create_openai_tools_agent(chat, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory)

# Streamlit 앱 UI 설정
st.title("langchain-streamlit-app")

# 메시지 히스토리 설정
history = StreamlitChatMessageHistory()

# 기존 메시지 표시
for message in history.messages:
    st.chat_message(message.type).write(message.content)

# 사용자 입력 받기
prompt = st.chat_input("What is up?")

# 프롬프트가 있을 경우 응답 처리
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        
        # 에이전트 체인 생성
        agent_chain = create_agent_chain(history)
        
        # 에이전트 실행 (콜백 전달)
        response = agent_chain.invoke(
            {"input": prompt},
            callbacks=[callback],  # 콜백 리스트로 전달
        )
        
        # 응답 출력
        st.markdown(response["output"])
