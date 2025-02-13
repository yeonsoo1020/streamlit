import streamlit as st
import time
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from dotenv import load_dotenv
import os

# API KEY 정보 로드
load_dotenv()

# Streamlit 제목
st.title("Mindful Companion")

# 🚀 대화내용 초기화 버튼 추가
if st.button("대화내용 초기화"):
    st.session_state["memory"] = ConversationBufferMemory()  # 대화 기록 초기화
    st.rerun()  # 페이지 새로고침

# 대화 기록 저장 (세션 상태 활용)
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory()

# OpenAI GPT-4o 설정 (스트리밍 활성화)
if "chat_model" not in st.session_state:
    st.session_state["chat_model"] = ChatOpenAI(model_name="gpt-4o", temperature=0.6, streaming=True)

# 프롬프트 파일 로드
prompt_path = "mindful_companion.yaml"
loaded_prompt = load_prompt(prompt_path, encoding="utf8")

# LangChain 체인 설정
llm = st.session_state["chat_model"]
chain = LLMChain(llm=llm, prompt=loaded_prompt)

# 이전 대화 출력
for message in st.session_state["memory"].chat_memory.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...")

# 🚀 사용자 입력 처리
if user_input:
    # 사용자의 입력을 즉시 화면에 출력
    st.chat_message("user").write(user_input)

    # 사용자 입력을 메모리에 추가
    st.session_state["memory"].chat_memory.add_user_message(user_input)

    # 대화 내역을 문자열로 변환하여 프롬프트에 전달
    chat_history_str = "\n".join(
        [msg.content for msg in st.session_state["memory"].chat_memory.messages]
    )

    # 🚀 AI 응답을 스트리밍 방식으로 출력 (한 토큰씩)
    with st.chat_message("assistant"):
        container = st.empty()  # 빈 컨테이너 생성
        response = ""
        
        chunk_size = 10  # 한 번에 출력할 블록 크기
        buffer = ""
        
        for chunk in chain.stream({"chat_history": chat_history_str, "question": user_input, "context": chat_history_str}):
            text_chunk = chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk)
            buffer += text_chunk
            
            # 일정 크기의 텍스트 블록을 쌓아둔 후 출력
            if len(buffer) >= chunk_size:
                response += buffer
                container.write(response)  # 점진적 업데이트
                buffer = ""  # 버퍼 초기화
                
        # 남아 있는 버퍼 출력
        if buffer:
            response += buffer
            container.write(response)
        
        # 최종 응답을 메모리에 추가
        st.session_state["memory"].chat_memory.add_ai_message(response)
