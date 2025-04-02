import streamlit as st
import asyncio
import nest_asyncio
import json
import atexit
import traceback
import datetime

# 더 적극적인 nest_asyncio 설정
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용
if "event_loop" not in st.session_state:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    st.session_state.event_loop = loop

# 안전한 종료를 위한 설정
def cleanup_resources():
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            if hasattr(st.session_state.mcp_client, "__aexit__"):
                # 비동기 컨텍스트 매니저를 안전하게 종료
                if st.session_state.event_loop.is_running():
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            st.session_state.mcp_client.__aexit__(None, None, None),
                            st.session_state.event_loop
                        )
                        future.result(timeout=5)  # 최대 5초 대기
                    except Exception as e:
                        print(f"비동기 종료 중 오류: {e}")
                else:
                    try:
                        st.session_state.event_loop.run_until_complete(
                            st.session_state.mcp_client.__aexit__(None, None, None)
                        )
                    except Exception as e:
                        print(f"동기 종료 중 오류: {e}")
        except Exception as e:
            print(f"MCP 클라이언트 종료 중 오류: {e}")
            traceback.print_exc()

# 프로그램 종료 시 리소스 정리
atexit.register(cleanup_resources)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_teddynote.messages import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# 환경 변수 로드 (.env 파일에서 API 키 등의 설정을 가져옴)
load_dotenv(override=True)

# 페이지 설정: 제목, 아이콘, 레이아웃 구성
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# 사이드바 최상단에 저자 정보 추가 (다른 사이드바 요소보다 먼저 배치)
st.sidebar.markdown("### ✍️ Made by [테디노트](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()  # 구분선 추가

# 기존 페이지 타이틀 및 설명
st.title("🤖 Agent with MCP Tools")
st.markdown("✨ MCP 도구를 활용한 ReAct 에이전트에게 질문해보세요.")

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # 세션 초기화 상태 플래그
    st.session_state.agent = None  # ReAct 에이전트 객체 저장 공간
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- 함수 정의 부분 ---


def print_message():
    """
    채팅 기록을 화면에 출력합니다.

    사용자와 어시스턴트의 메시지를 구분하여 화면에 표시하고,
    도구 호출 정보는 확장 가능한 패널로 표시합니다.
    """
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").markdown(message["content"])
        elif message["role"] == "assistant_tool":
            with st.expander("🔧 도구 호출 정보", expanded=False):
                st.markdown(message["content"])


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    스트리밍 콜백 함수를 생성합니다.

    매개변수:
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트

    반환값:
        callback_func: 스트리밍 콜백 함수
        accumulated_text: 누적된 텍스트 응답을 저장하는 리스트
        accumulated_tool: 누적된 도구 호출 정보를 저장하는 리스트
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                        st.markdown("".join(accumulated_tool))
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=600):
    """
    사용자 질문을 처리하고 응답을 생성합니다.

    매개변수:
        query: 사용자가 입력한 질문 텍스트
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트
        timeout_seconds: 응답 생성 제한 시간(초)

    반환값:
        response: 에이전트의 응답 객체
        final_text: 최종 텍스트 응답
        final_tool: 최종 도구 호출 정보
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=100, thread_id=st.session_state.thread_id
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.

    매개변수:
        mcp_config: MCP 도구 설정 정보(JSON). None인 경우 기본 설정 사용

    반환값:
        bool: 초기화 성공 여부
    """
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            # 기존 클라이언트 정리
            if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
                try:
                    st.info("기존 MCP 클라이언트 종료 중...")
                    await st.session_state.mcp_client.__aexit__(None, None, None)
                    st.info("기존 MCP 클라이언트 종료 완료")
                except Exception as e:
                    st.error(f"기존 MCP 클라이언트 종료 중 오류: {e}")
                    st.session_state.mcp_client = None

            if mcp_config is None:
                # 기본 설정 사용
                st.info("기본 MCP 설정을 사용합니다.")
                mcp_config = {
                    "weather": {
                        "command": "python",
                        "args": ["./mcp_server_local.py"],
                        "transport": "stdio",
                    },
                }
                
            try:
                # 클라이언트 생성 및 연결에 타임아웃 적용
                from langchain_mcp_adapters.client import MultiServerMCPClient
                
                # 연결 시도 전 로깅
                st.info(f"다음 MCP 도구에 연결 시도: {', '.join(mcp_config.keys())}")
                
                # 연결 재시도 메커니즘 추가
                max_retries = 3
                retry_count = 0
                
                # 각 서버 연결에 대한 로그 추가
                for server_name, config in mcp_config.items():
                    transport = config.get("transport", "stdio")
                    if transport == "stdio":
                        command = config.get("command", "")
                        args = config.get("args", [])
                        st.info(f"[{server_name}] stdio 연결: {command} {' '.join(args[:2])}...")
                    elif transport == "sse":
                        url = config.get("url", "")
                        st.info(f"[{server_name}] WebSocket 연결: {url}")
                
                while retry_count < max_retries:
                    try:
                        st.info(f"MCP 클라이언트 초기화 시도 #{retry_count+1}/{max_retries}")
                        client = MultiServerMCPClient(mcp_config)  # timeout 인자 제거
                        
                        # 비동기 초기화 상태 표시
                        progress_placeholder = st.empty()
                        progress_placeholder.info("클라이언트 컨텍스트 초기화 중...")
                        
                        # 타임아웃 및 예외 처리
                        try:
                            await asyncio.wait_for(client.__aenter__(), timeout=90)
                            progress_placeholder.success("클라이언트 초기화 성공!")
                            break
                        except asyncio.TimeoutError:
                            progress_placeholder.error(f"⏱️ 클라이언트 초기화 타임아웃 (90초)")
                            raise
                    except (asyncio.TimeoutError, ConnectionError, Exception) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            st.error(f"최대 재시도 횟수 초과: {str(e)}")
                            if isinstance(e, Exception) and not isinstance(e, (asyncio.TimeoutError, ConnectionError)):
                                st.error(f"예상치 못한 오류: {str(e)}")
                                traceback.print_exc()
                            raise
                        wait_time = 2 * retry_count  # 점점 더 오래 기다림
                        st.warning(f"MCP 서버 연결 시도 {retry_count}/{max_retries} 실패: {str(e)}. {wait_time}초 후 재시도...")
                        await asyncio.sleep(wait_time)
                
                # 성공적으로 연결되면 도구 로드
                st.info("도구 목록 로드 중...")
                tools = client.get_tools()
                st.session_state.tool_count = len(tools)
                st.info(f"총 {len(tools)}개 도구를 찾았습니다.")
                st.session_state.mcp_client = client

                # 모델 및 에이전트 초기화
                st.info("Claude 모델 초기화 중...")
                from langchain_anthropic import ChatAnthropic
                from langgraph.prebuilt import create_react_agent
                from langgraph.checkpoint.memory import MemorySaver
                
                model = ChatAnthropic(
                    model="claude-3-7-sonnet-latest", temperature=0.1, max_tokens=20000
                )
                
                st.info("ReAct 에이전트 생성 중...")
                agent = create_react_agent(
                    model,
                    tools,
                    checkpointer=MemorySaver(),
                    prompt="Use your tools to answer the question. Answer in Korean.",
                )
                st.session_state.agent = agent
                st.session_state.session_initialized = True
                return True
            except Exception as e:
                st.error(f"❌ MCP 클라이언트 초기화 오류: {e}")
                traceback.print_exc()
                return False
    except Exception as e:
        st.error(f"❌ 초기화 중 오류 발생: {e}")
        traceback.print_exc()
        return False


# --- 사이드바 UI: MCP 도구 추가 인터페이스로 변경 ---
with st.sidebar.expander("MCP 도구 추가", expanded=False):
    default_config = """{
  "weather": {
    "command": "python",
    "args": ["./mcp_server_local.py"],
    "transport": "stdio"
  }
}"""
    # pending config가 없으면 기존 mcp_config_text 기반으로 생성
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = json.loads(
                st.session_state.get("mcp_config_text", default_config)
            )
        except Exception as e:
            st.error(f"초기 pending config 설정 실패: {e}")

    # 개별 도구 추가를 위한 UI
    st.subheader("개별 도구 추가")
    st.markdown(
        """
    **하나의 도구**를 JSON 형식으로 입력하세요:
    
    ```json
    {
      "도구이름": {
        "command": "실행 명령어",
        "args": ["인자1", "인자2", ...],
        "transport": "stdio"
      }
    }
    ```    
    ⚠️ **중요**: JSON을 반드시 중괄호(`{}`)로 감싸야 합니다.
    """
    )

    # 보다 명확한 예시 제공
    example_json = {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@smithery/cli@latest",
                "run",
                "@smithery-ai/github",
                "--config",
                '{"githubPersonalAccessToken":"your_token_here"}',
            ],
            "transport": "stdio",
        }
    }

    default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

    new_tool_json = st.text_area(
        "도구 JSON",
        default_text,
        height=250,
    )

    # 추가하기 버튼
    if st.button(
        "도구 추가",
        type="primary",
        key="add_tool_button",
        use_container_width=True,
    ):
        try:
            # 입력값 검증
            if not new_tool_json.strip().startswith(
                "{"
            ) or not new_tool_json.strip().endswith("}"):
                st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                st.markdown('올바른 형식: `{ "도구이름": { ... } }`')
            else:
                # JSON 파싱
                parsed_tool = json.loads(new_tool_json)

                # mcpServers 형식인지 확인하고 처리
                if "mcpServers" in parsed_tool:
                    # mcpServers 안의 내용을 최상위로 이동
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' 형식이 감지되었습니다. 자동으로 변환합니다.")

                # 입력된 도구 수 확인
                if len(parsed_tool) == 0:
                    st.error("최소 하나 이상의 도구를 입력해주세요.")
                else:
                    # 모든 도구에 대해 처리
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        # URL 필드 확인 및 transport 설정
                        if "url" in tool_config:
                            # URL이 있는 경우 transport를 "sse"로 설정
                            tool_config["transport"] = "sse"
                            st.info(
                                f"'{tool_name}' 도구에 URL이 감지되어 transport를 'sse'로 설정했습니다."
                            )
                        elif "transport" not in tool_config:
                            # URL이 없고 transport도 없는 경우 기본값 "stdio" 설정
                            tool_config["transport"] = "stdio"

                        # 필수 필드 확인
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(
                                f"'{tool_name}' 도구 설정에는 'command' 또는 'url' 필드가 필요합니다."
                            )
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(
                                f"'{tool_name}' 도구 설정에는 'args' 필드가 필요합니다."
                            )
                        elif "command" in tool_config and not isinstance(
                            tool_config["args"], list
                        ):
                            st.error(
                                f"'{tool_name}' 도구의 'args' 필드는 반드시 배열([]) 형식이어야 합니다."
                            )
                        else:
                            # pending_mcp_config에 도구 추가
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)

                    # 성공 메시지
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(
                                f"{success_tools[0]} 도구가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                            )
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(
                                f"총 {len(success_tools)}개 도구({tool_names})가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                            )
        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 에러: {e}")
            st.markdown(
                f"""
            **수정 방법**:
            1. JSON 형식이 올바른지 확인하세요.
            2. 모든 키는 큰따옴표(")로 감싸야 합니다.
            3. 문자열 값도 큰따옴표(")로 감싸야 합니다.
            4. 문자열 내에서 큰따옴표를 사용할 경우 이스케이프(\\")해야 합니다.
            """
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")

    # 구분선 추가
    st.divider()

    # 현재 설정된 도구 설정 표시 (읽기 전용)
    st.subheader("현재 도구 설정 (읽기 전용)")
    st.code(
        json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
    )

# --- 등록된 도구 목록 표시 및 삭제 버튼 추가 ---
with st.sidebar.expander("등록된 도구 목록", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception as e:
        st.error("유효한 MCP 도구 설정이 아닙니다.")
    else:
        # pending config의 키(도구 이름) 목록을 순회하며 표시
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("삭제", key=f"delete_{tool_name}"):
                # pending config에서 해당 도구 삭제 (즉시 적용되지는 않음)
                del st.session_state.pending_mcp_config[tool_name]
                st.success(
                    f"{tool_name} 도구가 삭제되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                )

# --- MCP 도구 설정 가져오기/내보내기 기능 추가 ---
with st.sidebar.expander("도구 설정 가져오기/내보내기", expanded=False):
    st.markdown("### 도구 설정 내보내기")
    if st.button("mcp.json 파일로 내보내기", key="export_button", use_container_width=True):
        try:
            # 현재 시간을 타임스탬프로 변환 (YYYYMMDD_HHMMSS 형식)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 현재 설정을 JSON 파일로 저장
            mcp_config_json = json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
            st.download_button(
                label="파일 다운로드",
                data=mcp_config_json,
                file_name=f"mcp_{timestamp}.json",
                mime="application/json",
                key="download_json",
                use_container_width=True,
            )
            st.success(f"✅ 설정을 내보낼 준비가 되었습니다. '파일 다운로드' 버튼을 클릭하여 저장하세요. (파일명: mcp_{timestamp}.json)")
        except Exception as e:
            st.error(f"❌ 내보내기 오류: {str(e)}")
    
    if st.button("현재 설정을 mcp.json에 저장", key="save_mcp_button", use_container_width=True):
        try:
            # 현재 시간을 타임스탬프로 변환 (YYYYMMDD_HHMMSS 형식)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 현재 설정을 mcp.json 파일로 저장
            with open("mcp.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.pending_mcp_config, f, indent=2, ensure_ascii=False)
            
            # 타임스탬프가 포함된 백업 파일도 함께 저장
            with open(f"mcp_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.pending_mcp_config, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ 현재 설정이 mcp.json 파일과 백업 파일(mcp_{timestamp}.json)에 성공적으로 저장되었습니다.")
        except Exception as e:
            st.error(f"❌ 파일 저장 오류: {str(e)}")
    
    st.markdown("### 도구 설정 가져오기")
    uploaded_file = st.file_uploader("mcp.json 파일 업로드", type=["json"], key="import_file")
    if uploaded_file is not None:
        try:
            # 현재 시간을 타임스탬프로 변환 (YYYYMMDD_HHMMSS 형식)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 업로드된 JSON 파일 읽기
            content = uploaded_file.read().decode("utf-8")
            imported_config = json.loads(content)
            
            # 로그 파일에 가져온 설정 저장 (선택 사항)
            try:
                with open(f"imported_mcp_{timestamp}.log.json", "w", encoding="utf-8") as f:
                    json.dump(imported_config, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # 로그 저장 실패는 무시
            
            # 가져온 설정 검증
            valid_tools = []
            invalid_tools = []
            
            for tool_name, tool_config in imported_config.items():
                # 필수 필드 확인
                if "url" in tool_config:
                    tool_config["transport"] = "sse"
                elif "transport" not in tool_config:
                    tool_config["transport"] = "stdio"
                
                if ("command" not in tool_config and "url" not in tool_config) or \
                   ("command" in tool_config and "args" not in tool_config) or \
                   ("command" in tool_config and not isinstance(tool_config["args"], list)):
                    invalid_tools.append(tool_name)
                else:
                    valid_tools.append(tool_name)
            
            # 유효한 도구 설정만 적용
            for tool_name in valid_tools:
                st.session_state.pending_mcp_config[tool_name] = imported_config[tool_name]
            
            # 결과 표시
            if valid_tools:
                tool_names = ", ".join(valid_tools)
                st.success(f"✅ {len(valid_tools)}개 도구({tool_names})가 가져와졌습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")
            
            if invalid_tools:
                tool_names = ", ".join(invalid_tools)
                st.warning(f"⚠️ {len(invalid_tools)}개 도구({tool_names})는 유효하지 않아 가져오지 않았습니다.")
                
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON 파일 형식 오류: {str(e)}")
        except Exception as e:
            st.error(f"❌ 가져오기 오류: {str(e)}")
            
    if st.button("현재 mcp.json 로드", key="load_mcp_button", use_container_width=True):
        try:
            # 현재 시간을 타임스탬프로 변환 (YYYYMMDD_HHMMSS 형식)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # mcp.json 파일 읽기
            with open("mcp.json", "r", encoding="utf-8") as f:
                file_config = json.load(f)
            
            # 설정 적용
            st.session_state.pending_mcp_config = file_config
            
            # 로드 기록을 남기기 위해 로그 파일 작성 (선택 사항)
            try:
                with open(f"loaded_mcp_{timestamp}.log.json", "w", encoding="utf-8") as f:
                    json.dump(file_config, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # 로그 저장 실패는 무시
            
            st.success(f"✅ mcp.json 파일이 성공적으로 로드되었습니다. ({timestamp}) 적용하려면 '적용하기' 버튼을 눌러주세요.")
        except FileNotFoundError:
            st.error("❌ mcp.json 파일을 찾을 수 없습니다.")
        except json.JSONDecodeError as e:
            st.error(f"❌ mcp.json 파일 파싱 오류: {str(e)}")
        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {str(e)}")

with st.sidebar:

    # 적용하기 버튼: pending config를 실제 설정에 반영하고 세션 재초기화
    if st.button(
        "도구설정 적용하기",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # 적용 중 메시지 표시
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 변경사항을 적용하고 있습니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)

            # 설정 저장
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )

            # 세션 초기화 준비
            st.session_state.session_initialized = False
            st.session_state.agent = None
            st.session_state.mcp_client = None

            # 진행 상태 업데이트
            progress_bar.progress(30)

            # 초기화 실행
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # 진행 상태 업데이트
            progress_bar.progress(100)

            if success:
                st.success("✅ 새로운 MCP 도구 설정이 적용되었습니다.")
            else:
                st.error("❌ 새로운 MCP 도구 설정 적용에 실패하였습니다.")

        # 페이지 새로고침
        st.rerun()


# --- 기본 세션 초기화 (초기화되지 않은 경우) ---
if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버와 에이전트를 초기화합니다. 잠시만 기다려주세요...")
    success = st.session_state.event_loop.run_until_complete(initialize_session())
    if success:
        st.success(
            f"✅ 초기화 완료! {st.session_state.tool_count}개의 도구가 로드되었습니다."
        )
    else:
        st.error("❌ 초기화에 실패했습니다. 페이지를 새로고침해 주세요.")


# --- 대화 기록 출력 ---
print_message()

# --- 사용자 입력 및 처리 ---
user_query = st.chat_input("💬 질문을 입력하세요")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user").markdown(user_query)
        with st.chat_message("assistant"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(user_query, text_placeholder, tool_placeholder)
                )
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning("⏳ 시스템이 아직 초기화 중입니다. 잠시 후 다시 시도해주세요.")

# --- 사이드바: 시스템 정보 표시 ---
with st.sidebar:
    st.subheader("🔧 시스템 정보")
    st.write(f"🛠️ MCP 도구 수: {st.session_state.get('tool_count', '초기화 중...')}")
    st.write("🧠 모델: Claude 3.7 Sonnet")

    # 구분선 추가 (시각적 분리)
    st.divider()

    # 사이드바 최하단에 대화 초기화 버튼 추가
    if st.button("🔄 대화 초기화", use_container_width=True, type="primary"):
        # thread_id 초기화
        st.session_state.thread_id = random_uuid()

        # 대화 히스토리 초기화
        st.session_state.history = []

        # 알림 메시지
        st.success("✅ 대화가 초기화되었습니다.")

        # 페이지 새로고침
        st.rerun()
