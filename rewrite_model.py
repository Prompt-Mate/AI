# ============================================================
# 0) 기본 import
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
import torch

from unsloth import FastLanguageModel
from transformers import AutoTokenizer


# ============================================================
# 1) FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="Prompt Rewriter API",
    description="LoRA 기반 프롬프트 리라이팅 API",
    version="1.0.0",
)


# ============================================================
# 2) 모델 / 토크나이저 / LoRA 로드
# ============================================================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO  = "dkim130/prommate-rewriter-lora2"  

print("🔄 Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=True,        # 서버용 최적
)

print("🔄 Loading LoRA adapter...")
model.load_adapter(LORA_REPO)

# 추론 최적화 (Unsloth 필수)
FastLanguageModel.for_inference(model)

EOS_TOKEN = tokenizer.eos_token or "</s>"

print("✅ Model ready")


# ============================================================
# 3) 추론 함수
# ============================================================
def rewrite_final_strict(user_prompt: str, domain: str = "기타") -> str:
    """
    기준(명확성/구체성/구조/일관성/언어품질)을
    '출력'이 아니라 '내부 제약'으로 강제하는 FINAL 전용 함수
    """

    system_prompt = (
        "당신은 고급 프롬프트 리라이팅 전문가입니다.\n\n"
        "사용자가 입력한 원본 프롬프트를 다음 기준을 모두 충족하는 "
        "하나의 최종 프롬프트로 다시 작성하세요.\n\n"
        "반드시 충족해야 할 기준:\n"
        "1. 명확성: 수행해야 할 작업과 목적이 분명해야 합니다.\n"
        "2. 구체성: 대상, 조건, 요구사항, 산출물이 드러나야 합니다.\n"
        "3. 구조: Context → Instruction → Input → Output 요구가 자연스럽게 드러나야 합니다.\n"
        "4. 일관성: 모순 없이 하나의 작업 흐름으로 작성하세요.\n"
        "5. 언어 품질: 자연스럽고 전문적인 한국어를 사용하세요.\n\n"
        "중요 규칙:\n"
        "- 기준을 설명하거나 나열하지 마세요.\n"
        "- 여러 버전을 출력하지 마세요.\n"
        "- 실제 답변을 생성하지 말고, AI에게 요청하는 프롬프트만 작성하세요.\n"
        "- 원본 프롬프트의 핵심 정보는 절대 삭제하거나 왜곡하지 마세요.\n"
        "- 최종 결과는 한 번 실행 가능한 프롬프트 1개만 출력하세요."
    )

    user_content = (
        "<TASK: rewrite_final>\n"
        f"<DOMAIN: {domain}>\n\n"
        f"원본 프롬프트:\n{user_prompt}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,     # 안정성 최우선
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return generated.strip()


# ============================================================
# 4) API 입출력 스키마
# ============================================================
class RewriteRequest(BaseModel):
    prompt: str


class RewriteResponse(BaseModel):
    rewrite_final: str


# ============================================================
# 5) API Endpoint
# ============================================================
@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    """
    사용자 입력:
      - 프롬프트 1개 (string)

    출력:
      - 기준을 모두 만족한 최종 리라이팅 프롬프트 1개
    """
    final_prompt = rewrite_final_strict(req.prompt)
    return {"rewrite_final": final_prompt}


# ============================================================
# 6) 헬스 체크
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}
