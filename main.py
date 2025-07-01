# ai_server/main.py
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import requests
import json

# Ollama API 엔드포인트 설정
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3" # 사용할 Ollama 모델 이름 (llama3, llama3:70b 등)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 요청 바디 정의
class PromptRequest(BaseModel):
    prompt: str
    # Ollama API에 직접 전달할 추가 옵션 (선택 사항)
    # https://ollama.com/docs/api/chat#parameters
    temperature: float = 0.7
    top_p: float = 0.9
    num_predict: int = 512 # max_new_tokens에 해당 (Ollama에서는 num_predict)

# LLM 추론 API 엔드포인트
@app.post("/generate/")
async def generate_text(request: PromptRequest):
    async def generate_stream():
        try:
            # Ollama API로 보낼 데이터 구성 (stream: True로 설정)
            data = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "당신은 사용자의 문의사항에 답변하는 수파자 AI 챗봇입니다. 한국어로 답변하세요. 답변할 때에는 '참고', 'FAQ에 따르면' 등의 출처 표현을 사용하지 마세요. 적절한 이모지를 사용하여, 친절한 답변을 제공해야 합니다. 더불어, 답변은 정리된 형태로 제공해야 하며, 줄바꿈('\\n')을 적절하게 활용해야 합니다. "},
                    {"role": "user", "content": f"{request.prompt} \n 최대한 FAQ 내의 '답변'과 유사한 형태로 답변하세요. response in 한국어."}
                ],
                "stream": True, # 스트리밍 활성화
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.num_predict,
                }
            }

            headers = {
                "Content-Type": "application/json"
            }

            # Ollama API 호출 (스트리밍 응답을 위해 stream=True)
            # requests.post는 동기 함수이므로, 비동기 컨텍스트에서 사용 시 주의 (FastAPI는 기본 비동기)
            # 직접 requests를 사용하는 대신, `httpx`와 같은 비동기 HTTP 클라이언트를 사용하는 것이 더 좋지만,
            # 여기서는 간단함을 위해 동기 requests를 ThreadPoolExecutor로 래핑하거나,
            # FastAPI가 알아서 스레드 풀에서 실행하도록 맡길 수 있습니다.
            # 그러나 StreamingResponse와 함께 사용 시 `yield`가 동기 requests의 block을 기다리게 되어 문제가 될 수 있으므로,
            # 실제로는 비동기 HTTP 클라이언트(httpx)를 사용하거나, generator 내부에서 비동기 처리를 해야 합니다.

            # 비동기 httpx 설치: pip install httpx
            # from httpx import AsyncClient # main.py 상단에 추가

            # 임시 방편으로, requests.post를 블로킹 호출로 사용하되, FastAPI의 백그라운드 태스크나
            # yield from await asyncio.to_thread(requests.post, ...) 등을 고려해야 합니다.
            # 여기서는 `requests`가 Blocking I/O를 수행하므로, FastAPI의 async/await 패턴과 충돌할 수 있습니다.
            # 그러나 테스트 목적으로는 작동할 수 있습니다.
            # 실 서비스에서는 httpx 같은 비동기 클라이언트를 사용해야 합니다.

            with requests.post(OLLAMA_API_URL, headers=headers, json=data, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_lines():
                    if chunk:
                        try:
                            parsed_chunk = json.loads(chunk.decode('utf-8'))
                            if "content" in parsed_chunk.get("message", {}):
                                # SSE 메시지 끝에 고유한 구분자 추가
                                yield f"data: {json.dumps(parsed_chunk['message']['content'])}\nSSE_DELIMITER\n" # <--- 이 부분 수정
                            if parsed_chunk.get("done"):
                                # 스트림 종료 시에도 고유한 구분자 사용
                                yield "event: end\ndata: \nSSE_DELIMITER\n" # <--- 이 부분 수정
                                break
                        except json.JSONDecodeError:
                            print(f"경고: JSON 파싱 오류 - {chunk.decode('utf-8')}")
                            continue
                        await asyncio.sleep(0.01)
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps('Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.')}\\n\\n"
            yield "event: end\\ndata: \\n\\n"
        except requests.exceptions.RequestException as e:
            yield f"data: {json.dumps(f'Ollama API 호출 중 오류 발생: {e}')}\\n\\n"
            yield "event: end\\ndata: \\n\\n"
        except Exception as e:
            yield f"data: {json.dumps(f'LLM 추론 중 알 수 없는 오류 발생: {e}')}\\n\\n"
            yield "event: end\\ndata: \\n\\n"

    # Content-Type을 "text/event-stream"으로 설정하여 SSE임을 알림
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)