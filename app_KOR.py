import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import atexit

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용 (한번 생성한 후 계속 사용)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

# MCP 클라이언트 리소스 정리를 위한 함수
async def cleanup_mcp_client():
    """MCP 클라이언트를 안전하게 정리합니다."""
    if st.session_state.get("mcp_client") is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            print(f"MCP 클라이언트 종료 중 오류: {e}")

# 앱 종료 시 MCP 클라이언트 정리 함수
def cleanup_on_exit():
    """Streamlit 앱 종료 시 리소스를 정리합니다."""
    if "event_loop" in st.session_state and "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            st.session_state.event_loop.run_until_complete(cleanup_mcp_client())
        except RuntimeError:
            # 이벤트 루프가 이미 닫혔거나 실행 중이 아닐 경우 처리
            pass

# 종료 시 정리 함수 등록
atexit.register(cleanup_on_exit)

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


def load_mcp_config_from_file(file_path="mcp.json"):
    """
    mcp.json 파일에서 MCP 도구 설정을 로드합니다.

    매개변수:
        file_path: MCP 설정 파일 경로 (기본값: mcp.json)

    반환값:
        dict: MCP 도구 설정 정보. 파일이 존재하지 않거나 오류 발생 시 기본 설정 반환
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        # 파일이 없는 경우 기본 설정 반환
        default_config = {
            "weather": {
                "command": "python",
                "args": ["./mcp_server_local.py"],
                "transport": "stdio",
            }
        }
        # 파일이 없음을 알림
        st.info(f"🔍 {file_path} 파일이 없어 기본 설정을 사용합니다.")
        return default_config
    except json.JSONDecodeError:
        st.error(f"❌ {file_path} 파일의 JSON 형식이 올바르지 않습니다. 기본 설정을 사용합니다.")
        return {
            "weather": {
                "command": "python",
                "args": ["./mcp_server_local.py"],
                "transport": "stdio",
            }
        }
    except Exception as e:
        st.error(f"❌ 설정 파일 로드 중 오류 발생: {str(e)}")
        return {
            "weather": {
                "command": "python",
                "args": ["./mcp_server_local.py"],
                "transport": "stdio",
            }
        }


def save_mcp_config_to_file(config, file_path="mcp.json"):
    """
    MCP 도구 설정을 mcp.json 파일에 저장합니다.

    매개변수:
        config: MCP 도구 설정 정보
        file_path: MCP 설정 파일 경로 (기본값: mcp.json)

    반환값:
        bool: 저장 성공 여부
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ 설정 파일 저장 중 오류 발생: {str(e)}")
        return False


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
    # mcp.json 파일에서 MCP 설정 로드
    st.session_state.mcp_config = load_mcp_config_from_file()
    
    # Homebrew 경로 문제 자동 감지 및 수정
    need_update = False
    for tool_name, tool_config in st.session_state.mcp_config.items():
        if "command" in tool_config and "/opt/homebrew" in tool_config["command"]:
            # Homebrew 경로를 사용하는 명령어 발견
            original_cmd = tool_config["command"]
            cmd_name = original_cmd.split("/")[-1]
            st.session_state.mcp_config[tool_name]["command"] = cmd_name
            need_update = True
            st.info(f"🔧 '{tool_name}' 도구의 명령어 경로를 '{original_cmd}'에서 '{cmd_name}'으로 자동 변경했습니다.")
    
    # 변경된 경우 설정 파일 업데이트
    if need_update:
        save_result = save_mcp_config_to_file(st.session_state.mcp_config)
        if save_result:
            st.success("✅ Docker 환경 호환성을 위해 설정 파일이 자동으로 수정되었습니다.")
        else:
            st.warning("⚠️ 설정 파일 수정에 실패했습니다. 수동으로 경로를 수정해주세요.")
    
    # 파일이 없었던 경우 기본 설정을 저장
    try:
        if not os.path.exists("mcp.json"):
            save_mcp_config_to_file(st.session_state.mcp_config)
    except Exception as e:
        st.error(f"설정 파일 저장 중 오류: {e}")

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
                # 이벤트 루프 설정 확인 및 재설정
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.set_event_loop(st.session_state.event_loop)
                
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
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    # 이벤트 루프 문제 - 세션 재초기화 필요
                    error_msg = "🔄 이벤트 루프 오류가 발생했습니다. 페이지를 새로고침하여 다시 시도해주세요."
                    return {"error": error_msg}, error_msg, ""
                raise

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
        # 기존 MCP 클라이언트가 있으면 정리
        if st.session_state.get("mcp_client") is not None:
            await cleanup_mcp_client()
            
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            if mcp_config is None:
                # 기본 설정 사용
                mcp_config = {
                    "weather": {
                        "command": "python",
                        "args": ["./mcp_server_local.py"],
                        "transport": "stdio",
                    },
                }
            
            # Docker/ECS 환경을 위한 npx 경로 조정
            adjusted_config = mcp_config.copy()
            for tool_name, tool_config in mcp_config.items():
                # npx 명령어 경로 변경이 필요한 경우 처리
                if tool_config.get("command") == "/opt/homebrew/bin/npx":
                    st.info(f"🔧 '{tool_name}' 도구의 npx 경로를 조정합니다.")
                    # 먼저 'npx'가 시스템에 있는지 확인 (상대경로)
                    adjusted_config[tool_name]["command"] = "npx"
                elif "command" in tool_config and "/opt/homebrew" in tool_config["command"]:
                    # 다른 Homebrew 경로를 가진 명령어도 조정
                    original_cmd = tool_config["command"]
                    cmd_name = original_cmd.split("/")[-1]
                    st.info(f"🔧 '{tool_name}' 도구의 {cmd_name} 경로를 조정합니다.")
                    adjusted_config[tool_name]["command"] = cmd_name
            
            # 이벤트 루프 설정 확인 및 재설정
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(st.session_state.event_loop)
                
            client = MultiServerMCPClient(adjusted_config)
            await client.__aenter__()
            tools = client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            model = ChatAnthropic(
                model="claude-3-7-sonnet-latest", temperature=0.1, max_tokens=20000
            )
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
        st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
        import traceback

        st.error(traceback.format_exc())
        # 특정 오류에 대한 추가 정보 제공
        if "No such file or directory" in str(e) and "npx" in str(e):
            st.warning("⚠️ npx 경로 문제가 발생했습니다. 도구 설정의 경로를 확인하세요.")
            st.info("💡 Docker 환경에서는 '/opt/homebrew/bin/npx' 대신 'npx'를 사용하세요.")
        return False


# --- 사이드바 UI: MCP 도구 추가 인터페이스로 변경 ---
with st.sidebar.expander("MCP 도구 추가", expanded=False):
    # pending config가 없으면 세션 상태의 mcp_config 기반으로 생성
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = st.session_state.mcp_config.copy()
        except Exception as e:
            st.error(f"초기 pending config 설정 실패: {e}")
            # 오류 발생 시 기본 설정 사용
            st.session_state.pending_mcp_config = load_mcp_config_from_file()

    # 개별 도구 추가를 위한 UI
    st.subheader("개별 도구 추가")

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

    # Docker 환경에서 작동하는 Slack 예제 추가
    st.info("""
    📝 **Slack 도구 예제** (Docker/ECS 환경에서 작동):
    ```json
    {
      "slack": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-slack"
        ],
        "env": {
          "SLACK_BOT_TOKEN": "xoxb-your-token-here",
          "SLACK_TEAM_ID": "your-team-id"
        },
        "transport": "stdio"
      }
    }
    ```
    """)

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
                        
                        # mcp.json 파일에 변경사항 저장
                        save_mcp_config_to_file(st.session_state.pending_mcp_config)
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
                # mcp.json 파일에 변경사항 저장
                save_mcp_config_to_file(st.session_state.pending_mcp_config)
                st.success(
                    f"{tool_name} 도구가 삭제되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                )

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
            st.session_state.mcp_config = st.session_state.pending_mcp_config.copy()
            # mcp.json 파일에 변경사항 저장
            save_result = save_mcp_config_to_file(st.session_state.mcp_config)

            # 세션 초기화 준비
            st.session_state.session_initialized = False
            st.session_state.agent = None
            
            # 기존 MCP 클라이언트 정리
            if st.session_state.get("mcp_client") is not None:
                try:
                    st.session_state.event_loop.run_until_complete(cleanup_mcp_client())
                except RuntimeError:
                    # 이벤트 루프 관련 오류 처리
                    pass
            
            # 진행 상태 업데이트
            progress_bar.progress(30)

            # 이벤트 루프 설정 확인 및 재설정
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(st.session_state.event_loop)
                
            # 초기화 실행
            try:
                success = st.session_state.event_loop.run_until_complete(
                    initialize_session(st.session_state.mcp_config)
                )
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    # 이벤트 루프 문제 발생 시 새로운 루프 생성 후 재시도
                    st.warning("이벤트 루프 문제가 감지되어 재설정합니다...")
                    loop = asyncio.new_event_loop()
                    st.session_state.event_loop = loop
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        initialize_session(st.session_state.mcp_config)
                    )
                else:
                    raise

            # 진행 상태 업데이트
            progress_bar.progress(100)

            if success and save_result:
                st.success("✅ 새로운 MCP 도구 설정이 적용되었습니다.")
            else:
                error_msg = ""
                if not save_result:
                    error_msg += "설정 파일 저장에 실패했습니다.\n"
                if not success:
                    error_msg += "MCP 서버 초기화에 실패했습니다.\n"
                st.error(f"❌ 새로운 MCP 도구 설정 적용에 실패하였습니다.\n{error_msg}")

        # 페이지 새로고침
        st.rerun()


# --- 기본 세션 초기화 (초기화되지 않은 경우) ---
if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버와 에이전트를 초기화합니다. 잠시만 기다려주세요...")
    try:
        # 이벤트 루프 설정 확인 및 재설정
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(st.session_state.event_loop)
            
        success = st.session_state.event_loop.run_until_complete(
            initialize_session(st.session_state.mcp_config)
        )
    except RuntimeError as e:
        if "no running event loop" in str(e):
            # 이벤트 루프 문제 발생 시 새로운 루프 생성 후 재시도
            loop = asyncio.new_event_loop()
            st.session_state.event_loop = loop
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(
                initialize_session(st.session_state.mcp_config)
            )
        else:
            st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
            success = False
    
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
    col1, col2 = st.columns(2)
    
    if col1.button("🔄 대화 초기화", use_container_width=True):
        # thread_id 초기화
        st.session_state.thread_id = random_uuid()

        # 대화 히스토리 초기화
        st.session_state.history = []

        # 알림 메시지
        st.success("✅ 대화가 초기화되었습니다.")

        # 페이지 새로고침
        st.rerun()
        
    if col2.button("🔄 서버 재연결", use_container_width=True):
        st.warning("🔄 MCP 서버에 재연결 중...")
        
        # 세션 초기화 준비
        st.session_state.session_initialized = False
        st.session_state.agent = None
        
        # 기존 MCP 클라이언트 정리
        if st.session_state.get("mcp_client") is not None:
            try:
                st.session_state.event_loop.run_until_complete(cleanup_mcp_client())
            except RuntimeError:
                # 이벤트 루프 관련 오류 처리
                pass
        
        # 페이지 새로고침
        st.rerun()