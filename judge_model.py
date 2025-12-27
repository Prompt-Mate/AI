import torch
import json
from unsloth import FastLanguageModel

# ================================================================
# 1) Judge 모델 로드
# ================================================================
JUDGE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

judge_model, judge_tokenizer = FastLanguageModel.from_pretrained(
    model_name=JUDGE_MODEL_NAME,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(judge_model)

# ================================================================
# 2) Judge Prompt (출력 형식 고정 / 평가 + 피드백만)
# ================================================================
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

# ================================================================
# 3) 모델 generate 함수 (결정적)
# ================================================================
def generate(model, tokenizer, messages, max_new_tokens=512) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        output[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return text

# ================================================================
# 4) JSON 안전 파싱
# ================================================================
def safe_json_extract(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("JSON을 찾을 수 없습니다:\n" + text)

    json_str = text[start:end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = json_str.replace("\n", "")
        json_str = json_str.replace(",}", "}")
        return json.loads(json_str)

# ================================================================
# 5) 외부 호출용 평가 함수
# ================================================================
def evaluate_prompt_only(user_prompt: str) -> dict:
    judge_prompt = build_judge_prompt(user_prompt)

    messages = [
        {"role": "system", "content": "너는 프롬프트 품질을 평가하는 결정적 평가자이다."},
        {"role": "user", "content": judge_prompt},
    ]

    raw_output = generate(judge_model, judge_tokenizer, messages)
    return safe_json_extract(raw_output)