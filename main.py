# ============================================================
# 🔥 PROMMATE AI SERVER (Rewrite + Judge 통합 / Transformers CPU)
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1) FastAPI APP
# ============================================================
app = FastAPI(
    title="PromMate Unified AI API",
    description="Rewrite + Judge Model Server (Transformers CPU)",
    version="1.0.0"
)


# ============================================================
# 2) ---------------- Judge Model 로드 ------------------------
# ============================================================
JUDGE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # CPU 실행 가능한 모델로 변경

print("Loading Judge Model...")
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    torch_dtype=torch.float32,
).to("cpu")
print("Judge Model Loaded")


# ============================================================
# Judge Prompt (네가 작성한 원문 그대로 유지)
# ============================================================
def build_judge_prompt(user_prompt: str) -> str:
    return f"""
너는 사용자가 작성한 프롬프트의 품질을 평가하는 전문가이다.
답변 내용이 아니라 프롬프트 자체만 평가한다.

출력은 반드시 JSON 객체 1개만 작성한다.
JSON 앞뒤로 어떤 문자도 출력하지 않는다.

================================================================
# 평가 기준 (0~100)
================================================================
1) 명확성
- 목표가 분명하지 않으면 감점한다.
- 여러 해석이 가능한 표현을 감점한다.

2) 구체성
- 누가 / 무엇을 / 언제 / 어디서 / 왜 / 어떻게 중
  빠진 요소마다 감점한다.
- 출력 형태나 분량 지시가 없으면 감점한다.
- 제약 조건이 없으면 감점한다.

3) 구성
- 지시문, 맥락, 제약 조건, 출력 형태가
  체계적으로 구성되지 않으면 감점한다.

4) 언어 품질
- 문법적으로 어색한 문장을 감점한다.
- 자연스럽지 않은 표현을 감점한다.

5) 일관성
- 서로 충돌하는 요구가 있으면 크게 감점한다.
- 하나의 의미로 해석되지 않으면 감점한다.

================================================================
# 한국어 전용 규칙
================================================================
- 모든 코멘트와 피드백은 100% 자연스러운 한국어로 작성한다.
- 영어 단어, 약어, 기호 표현을 절대 사용하지 않는다.
- "일부", "약간", "조금", "다소", "애매함", "부분적으로" 같은 표현을 쓰지 않는다.
- 모든 지적은 구체적이고 실행 가능해야 한다.

================================================================
# 원문 의도 보존 규칙
================================================================
- 원문 프롬프트의 핵심 의도를 절대로 변경하거나 확장하지 마라.
- 새로운 목적이나 작업을 제안하지 마라.
- 대상 독자나 사용 상황을 임의로 추가하지 마라.
- 원문에 없는 맥락을 가정하지 마라.

개선 피드백은 다음 범위 안에서만 작성한다:
- 표현을 더 명확하게 만드는 방향
- 빠진 정보를 보완하도록 안내
- 출력 형태나 제약 조건을 명확히 하도록 안내

================================================================
# overall_score 계산 규칙
================================================================
overall_score = round(
    (clarity_score + specificity_score + structure_score +
     language_score + consistency_score) / 5
)

================================================================
# JSON 출력 형식 (반드시 이 구조만 사용)
================================================================
{{
  "overall_score": <0-100>,
  "clarity_score": <0-100>,
  "specificity_score": <0-100>,
  "structure_score": <0-100>,
  "language_score": <0-100>,
  "consistency_score": <0-100>,
  "clarity_comment": "<명확성 평가 및 개선 조언>",
  "specificity_comment": "<구체성 평가 및 개선 조언>",
  "structure_comment": "<구성 평가 및 개선 조언>",
  "language_comment": "<언어 품질 평가 및 개선 조언>",
  "consistency_comment": "<일관성 평가 및 개선 조언>",
  "summary_feedback": "<전체 프롬프트를 개선하기 위한 핵심 조언>"
}}

================================================================
# 평가 대상 프롬프트
================================================================
{user_prompt}
""".strip()



# ============================================================
#  Judge Inference (네 방식 그대로 유지, Unsloth → Transformers 방식만 변경)
# ============================================================
def generate(model, tokenizer, messages, max_new_tokens=512) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            top_k=50,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def safe_json_extract(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    json_str = text[start:end + 1]
    return json.loads(json_str)


def evaluate_prompt_only(user_prompt: str) -> dict:
    judge_prompt = build_judge_prompt(user_prompt)
    messages = [
        {"role": "system", "content": "너는 프롬프트 품질을 평가하는 결정적 평가자다."},
        {"role": "user", "content": judge_prompt},
    ]
    raw = generate(judge_model, judge_tokenizer, messages)
    return safe_json_extract(raw)



# ============================================================
# 3) Rewrite Model (transformers로 그대로 변환)
# ============================================================
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # CPU용

print("Loading Rewrite Model...")
rewrite_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
rewrite_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
).to("cpu")
print("Rewrite Model Loaded")


def rewrite_final_strict(user_prompt: str, domain="기타") -> str:
    system_prompt = (
        "당신은 고급 프롬프트 리라이팅 전문가입니다.\n\n"
        "사용자가 입력한 원본 프롬프트를 다음 기준을 모두 충족하는 "
        "하나의 최종 프롬프트로 다시 작성하세요.\n\n"
        "- 기준 나열 금지\n"
        "- 실제 답변 생성 금지\n"
        "- 프롬프트 1개만 출력\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = rewrite_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = rewrite_tokenizer(text, return_tensors="pt").to("cpu")

    output = rewrite_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
    )

    return rewrite_tokenizer.decode(output[0], skip_special_tokens=True)



# ============================================================
# 4) API Request/Response Schema (원본 그대로)
# ============================================================
class PromptReq(BaseModel):
    prompt: str

class RewriteRes(BaseModel):
    rewrite_final: str

class JudgeRes(BaseModel):
    overall_score: int
    clarity_score: int
    specificity_score: int
    structure_score: int
    language_score: int
    consistency_score: int
    clarity_comment: str
    specificity_comment: str
    structure_comment: str
    language_comment: str
    consistency_comment: str
    summary_feedback: str



# ============================================================
# 5) Endpoints (변경 없음)
# ============================================================
@app.post("/rewrite", response_model=RewriteRes)
def rewrite_api(req: PromptReq):
    return {"rewrite_final": rewrite_final_strict(req.prompt)}

@app.post("/judge", response_model=JudgeRes)
def judge_api(req: PromptReq):
    return evaluate_prompt_only(req.prompt)

@app.get("/health")
def health():
    return {"status": "ok"}
