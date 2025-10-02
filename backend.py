# backend.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# rag_agent_main.py 경로 추가
sys.path.append("/home/alpaco/kimcy/Office_AI_Agent_System")

from rag_agent_main import RAGAgent  # 형이 만든 RAGAgent 가져오기

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정 (프론트 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
project_path = "/home/alpaco/kimcy/Office_AI_Agent_System"
if os.path.exists(project_path):
    app.mount("/static", StaticFiles(directory=project_path), name="static")

# RAGAgent 인스턴스 준비
agent = RAGAgent(
    chunks_path="/home/alpaco/kimcy/Office_AI_Agent_System/report/standard_flattened.json"
)

# 요청 형식 정의
class QueryRequest(BaseModel):
    query: str

# 루트 페이지 - 챗봇 웹페이지 제공
@app.get("/", response_class=HTMLResponse)
async def read_root():
    frontend_path = "/home/alpaco/kimcy/Office_AI_Agent_System/templates/frontend.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return HTMLResponse("""
        <html>
            <head><title>RAG 챗봇</title></head>
            <body>
                <h1>RAG 챗봇 시스템</h1>
                <p>프론트엔드 파일을 찾을 수 없습니다: {}</p>
                <p>API는 정상 작동 중입니다. <a href="/docs">Swagger UI</a>를 확인하세요.</p>
            </body>
        </html>
        """.format(frontend_path))
    
@app.get("/ask")
def ask_get():
    return {"message": "POST 요청으로 질문을 보내세요."}

# API 엔드포인트
@app.post("/ask")
def ask_question(req: QueryRequest):
    ans = agent.answer(
        req.query,
        top_k=10,
        score_threshold=0.45,
        max_return=3,
        generate_answer=True,
        keyword_bonus=0.2,
        gen_max_new_tokens=2000
    )
    
    # numpy 타입을 Python 기본 타입으로 변환하여 JSON 직렬화 문제 해결
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        else:
            return obj
    
    return convert_numpy_types(ans)

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
