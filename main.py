# ai_server/main.py
from fastapi import FastAPI, HTTPException
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
    try:
        # Ollama API로 보낼 데이터 구성
        data = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant for Supaja. Provide concise and accurate answers based on the context. Response in Korean (한국어)."},
                {"role": "user", "content": request.prompt}
            ],
            "stream": False, # 스트리밍 대신 전체 응답을 한 번에 받음
            "options": { # Ollama의 추가 옵션을 options 객체로 전달
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.num_predict,
                # 다른 Ollama 모델 옵션도 여기에 추가 가능
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Ollama API 호출
        response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생

        ollama_response = response.json()

        # Ollama API 응답에서 content 추출
        generated_text = ollama_response["message"]["content"]

        return {"generated_text": generated_text.strip()}

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요 (ollama run llama3)."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama API 호출 중 오류 발생: {e}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama API 응답 파싱 오류: 필수 키가 없습니다 ({e}). 응답: {ollama_response}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 추론 중 알 수 없는 오류 발생: {e}")

# FastAPI 서버 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
