# ============================================================
# 0) 기본 import
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama


# ============================================================
# 1) FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="Prompt Rewriter API",
    description="GGUF 기반 프롬프트 리라이팅 API",
    version="1.0.0",
)


# ============================================================
# 2) GGUF 모델 로드 (EC2 위치 기준)
# ============================================================
MODEL_PATH = "/home/ubuntu/models/model-q4-rewrite.gguf"   
print("🔄 Loading Rewrite Model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=8,          
    n_ctx=4096,
    verbose=False
)
print("✅ Rewrite Model Ready")


# ============================================================
# 3) 리라이팅 함수 
# ============================================================
def rewrite_final_strict(user_prompt: str) -> str:
    system_prompt = (
        "당신은 프롬프트 리라이팅 전문가입니다. "
        "원본 프롬프트의 의도와 핵심 정보를 유지한 채, 더 명확하고 구체적이며 자연스러운 한국어의 "
        "단일 실행 가능한 'AI 작업 지시용 프롬프트' 1개로 다시 작성하세요. "
        "절대로 실제 작업 결과(요약/분석/추천/코드/이메일 등)를 생성하지 말고, "
        "설명/해설/번호/레이블(Context, Instruction 등) 없이 결과 프롬프트만 출력하세요."
    )

    # ✅ 학습 포맷에 맞추기: <|system|> / <|user|> / <|assistant|>
    prompt = (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n원본 프롬프트:\n{user_prompt}\n"
        f"<|assistant|>\n"
    )

    response = llm(
        prompt,
        max_tokens=300,
        temperature=0.3,
        top_p=0.9,
        stop=["<|user|>", "<|system|>", "<|assistant|>", "</s>", "[INST]", "<<SYS>>"],
    )

    text = response["choices"][0]["text"].strip()

    # 혹시 남는 레이블/토큰 간단 제거(방어)
    for bad in ["Context:", "Instruction:", "Input:", "Output:", "<<SYS>>", "[INST]", "</s>"]:
        text = text.replace(bad, "").strip()

    return text

# ============================================================
# 4) API 스키마
# ============================================================
class RewriteRequest(BaseModel):
    prompt: str

class RewriteResponse(BaseModel):
    rewrite_final: str


# ============================================================
# 5) API 엔드포인트
# ============================================================
@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    final_prompt = rewrite_final_strict(req.prompt)
    return {"rewrite_final": final_prompt}


# ============================================================
# 6) Health Check
# ============================================================
@app.get("/health")
def health():
    return {"status": "rewrite ok"}
