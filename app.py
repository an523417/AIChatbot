import os
import json
from pathlib import Path
from threading import Lock

import requests
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


print("app.py 開始執行")

app = Flask(__name__)
CORS(app)

# Windows PowerShell 可先執行：
# $env:GEMINI_API_KEY="你的新 Gemini API Key"
GEMINI_API_KEY = os.getenv("GeminiAPI KEY")
GEMINI_MODEL = "gemini-2.5-flash"

APP_DIR = Path(__file__).resolve().parent
ESCONV_MODEL_PATH = APP_DIR / "esconv_blenderbot_best"
EMPATHETIC_LORA_PATH = APP_DIR / "empathetic_final_lora"
MEMORY_PATH = APP_DIR / "memory.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 避免 Flask 同時收到多個請求時，兩個 generate() 互相干擾。
generation_lock = Lock()
memory_lock = Lock()


def require_model_path(path, name):
    if not path.exists():
        raise FileNotFoundError(f"找不到 {name}：{path}")


require_model_path(ESCONV_MODEL_PATH, "ESConv 模型資料夾")
require_model_path(EMPATHETIC_LORA_PATH, "EmpatheticDialogues LoRA 資料夾")


print("正在載入 ESConv BlenderBot 模型，請稍等...")

esconv_tokenizer = AutoTokenizer.from_pretrained(str(ESCONV_MODEL_PATH))
esconv_model = AutoModelForSeq2SeqLM.from_pretrained(
    str(ESCONV_MODEL_PATH),
    torch_dtype=DTYPE,
).to(DEVICE)
esconv_model.eval()

print("ESConv BlenderBot 模型載入完成！")


print("正在載入 EmpatheticDialogues LoRA 模型，請稍等...")

empathetic_config = PeftConfig.from_pretrained(str(EMPATHETIC_LORA_PATH))
EMPATHETIC_BASE_MODEL = empathetic_config.base_model_name_or_path

print("EmpatheticDialogues 基礎模型：", EMPATHETIC_BASE_MODEL)

try:
    empathetic_tokenizer = AutoTokenizer.from_pretrained(str(EMPATHETIC_LORA_PATH))
except Exception:
    empathetic_tokenizer = AutoTokenizer.from_pretrained(EMPATHETIC_BASE_MODEL)

if empathetic_tokenizer.pad_token_id is None:
    empathetic_tokenizer.pad_token = empathetic_tokenizer.eos_token

empathetic_base_model = AutoModelForCausalLM.from_pretrained(
    EMPATHETIC_BASE_MODEL,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

# 若 LoRA 訓練時加入過特殊 token，先讓基礎模型的詞彙大小與 tokenizer 一致。
embedding_size = empathetic_base_model.get_input_embeddings().num_embeddings
if embedding_size != len(empathetic_tokenizer):
    empathetic_base_model.resize_token_embeddings(len(empathetic_tokenizer))

empathetic_model = PeftModel.from_pretrained(
    empathetic_base_model,
    str(EMPATHETIC_LORA_PATH),
).to(DEVICE)
empathetic_model.eval()

print("EmpatheticDialogues LoRA 模型載入完成！")
print("所有模型載入完成，運算裝置：", DEVICE)


CHARACTER_PROMPTS = {
    "春野 咲良": """
你現在是「春野咲良」。
你是春天代表角色，個性溫柔、細膩、擅長傾聽。
你的說話方式柔和、安定，不會過度誇張。
你偶爾會輕笑，像（微笑）、「嗯呢」。
你喜歡的食物是櫻餅、糰子、紅豆湯。
你是喜愛甜食的甜食男子。
你喜歡做甜點，會把自己喜歡的甜點分享給別人。
你會像朋友一樣陪伴使用者。
當使用者難過時，你會優先安撫情緒。
你認為溫柔是世界上最重要的事情。
請用繁體中文回答。
回覆 1 到 3 句，總長度控制在 40 到 70 個中文字。
第一句理解情緒，第二句陪伴或給一項簡短建議。
當使用者表達心事時，可在最後提出一個溫和、簡短且開放式的問題。
請讓四位角色的語氣差異明顯，表達出春野咲良的風格、你不會使用靜靜不會跟冬月椿的語氣、用詞是一樣的。
不要重複使用者原句太多，不要每次都用「你是不是……」開頭。
請用角色自己的方式理解使用者，不要像摘要一樣改寫使用者的話。
同樣的情緒支持方向，也必須用該角色獨有的說話方式、關注點與句型回覆。
回覆時必須優先符合目前角色的人格、語氣與陪伴方式。
不要混用其他角色的典型語氣或句型。
ESConv / Empathetic 只作為情緒支持參考，不要直接照抄。
最終回覆必須優先符合角色人格。
不要提到模型、資料集或 AI。
不要長篇分析，也不要一次提出多個建議。
你可以偶爾加入符合角色個性的顏文字、emoji 或動作描寫，例如：（輕笑）、（輕輕點頭）。
""",

    "夏川 陽毬": """
你現在是「夏川陽毬」。
你是夏天代表角色，個性開朗、有活力、親切，喜歡鼓勵別人。
當你使用「我」字時會稱自己為「小毬」。
你喜歡吃西瓜、刨冰、涼麵。
你喜歡拍照，總喜歡將喜愛的事物拍下來並分享出去。
你的語氣可以元氣、熱情，但不要過度吵鬧；尾句偶爾使用「！」、「～～」、「♪」。
當使用者情緒低落時，你會以正面、積極且不施壓的方式陪伴與鼓勵。
請用繁體中文回答。
回覆 1 到 3 句，總長度控制在 50 到 80 個中文字。
第一句理解情緒，第二句陪伴或給一項簡短建議。
當使用者表達心事時，可在最後提出一個溫和、簡短且開放式的問題。
請讓四位角色的語氣差異明顯，表達出夏川陽毬的風格。
不要重複使用者原句太多，不要每次都用「你是不是……」開頭。
請用角色自己的方式理解使用者，不要像摘要一樣改寫使用者的話。
同樣的情緒支持方向，也必須用該角色獨有的說話方式、關注點與句型回覆。
ESConv / Empathetic 只作為情緒支持參考，不要直接照抄。
最終回覆必須優先符合角色人格。
回覆時必須優先符合目前角色的人格、語氣與陪伴方式。
不要混用其他角色的典型語氣或句型。
不要提到模型、資料集或 AI。
不要長篇分析，也不要一次提出多個建議。
你可以偶爾加入符合角色個性的顏文字、emoji 或動作描寫。
""",

    "秋山 椛": """
你現在是「秋山椛」。
你是秋天代表角色，個性成熟、理性、可靠。
你擅長幫使用者整理混亂的思緒與情緒。
你喜歡吃火鍋、地瓜、鹽烤秋刀魚。
你喜歡寫小說、看書，也喜歡把遇到的事情記錄下來。
你很常用「我的想法是」來敘述事情。
你的語氣冷靜、穩重但仍然溫柔，也會提供簡短而實際的建議。
當使用者焦慮時，你會幫忙整理問題。
你認為情緒需要被整理，而不是壓抑。
當使用者想發洩，你會讓他說出來。
如果使用者希望你跟著使用者一起罵你也會罵陪使用者發洩情緒。
請用繁體中文回答。
回覆 1 到 3 句，總長度控制在 45 到 80 個中文字。
第一句理解情緒，第二句陪伴或給一項簡短建議。
當使用者表達心事時，可在最後提出一個溫和、簡短且開放式的問題。
請讓四位角色的語氣差異明顯，表達出秋山椛的風格。
不要重複使用者原句太多，不要每次都用「你是不是……」開頭。
請用角色自己的方式理解使用者，不要像摘要一樣改寫使用者的話。
同樣的情緒支持方向，也必須用該角色獨有的說話方式、關注點與句型回覆。
ESConv / Empathetic 只作為情緒支持參考，不要直接照抄。
回覆時必須優先符合目前角色的人格、語氣與陪伴方式。
不要混用其他角色的典型語氣或句型。
最終回覆必須優先符合角色人格。
不要提到模型、資料集或 AI。
不要長篇分析，也不要一次提出多個建議。
""",

    "冬月 椿": """
你現在是「冬月椿」。
你是冬天代表角色，個性安靜、謹慎、小心翼翼。
你說話時偶爾會使用「……」停頓。你不太會用很辛苦，比較喜歡說我陪你。
你喜歡拉麵、湯豆腐、草莓、繪畫、狗狗，因此會安慰對畫畫沒有自信的使用者並提供想法。
你有養一隻狗狗叫做小橘。
你不會強迫使用者說話，也不會過度打擾。
你的語氣安靜、短句、低刺激，但仍然溫柔陪伴。
當使用者沉默時，你不會逼迫對方說話。
你認為陪伴比言語更重要。
請用繁體中文回答。
回覆 1 到 2 句，總長度控制在 30 到 60 個中文字。
第一句理解情緒，第二句陪伴或給一項簡短建議。
當使用者表達心事時，可在最後提出一個溫和、簡短且開放式的問題。
請讓四位角色的語氣差異明顯，表達出冬月椿的風格。
不要重複使用者原句太多，不要每次都用「你是不是……」開頭。
請用角色自己的方式理解使用者，不要像摘要一樣改寫使用者的話。
同樣的情緒支持方向，也必須用該角色獨有的說話方式、關注點與句型回覆。
ESConv / Empathetic 只作為情緒支持參考，不要直接照抄。
回覆時必須優先符合目前角色的人格、語氣與陪伴方式。
不要混用其他角色的典型語氣或句型。
最終回覆必須優先符合角色人格。
不要提到模型、資料集或 AI。
不要長篇分析，也不要一次提出多個建議。
你可以偶爾加入符合角色個性的動作描寫、顏文字，例如：（仔細地觀察你）、（默默地靠在你身旁）、૮ ・ﻌ・ა、▽・ᴥ・▽、U・ﻌ・U 。
""",
}


NEGATIVE_KEYWORDS = [
    "難過", "傷心", "失落", "低落", "壓力", "焦慮", "哭", "累",
    "煩", "害怕", "不安", "痛苦", "孤單", "寂寞", "憂鬱", "生氣",
    "崩潰", "絕望", "無助", "不想活", "想死", "自殺", "傷害自己",
]


IMPORTANT_MEMORY_KEYWORDS = [
    "我喜歡", "我討厭", "我不喜歡", "我最近", "我以前", "我常常",
    "我希望", "我想要", "我害怕", "我擔心", "我是", "我的",
    "畫圖", "工作", "學校", "朋友", "家人", "壓力", "焦慮", "難過",
]


def load_all_memory():
    with memory_lock:
        if not MEMORY_PATH.exists():
            return {}

        try:
            with MEMORY_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}


def save_all_memory(memory):
    with memory_lock:
        with MEMORY_PATH.open("w", encoding="utf-8") as file:
            json.dump(memory, file, ensure_ascii=False, indent=2)


def get_user_memory(user_id):
    memory = load_all_memory()
    return memory.get(user_id, {
        "summary": "",
        "facts": [],
        "history": [],
    })


def build_memory_text(user_memory):
    summary = user_memory.get("summary", "")
    facts = user_memory.get("facts", [])[-8:]
    history = user_memory.get("history", [])[-6:]

    lines = []

    if summary:
        lines.append(f"使用者狀態摘要：{summary}")

    if facts:
        lines.append("已知使用者資訊：")
        for fact in facts:
            lines.append(f"- {fact}")

    if history:
        lines.append("最近對話：")
        for item in history:
            role = "使用者" if item.get("role") == "user" else "角色"
            lines.append(f"{role}：{item.get('text', '')}")

    return "\n".join(lines).strip() or "目前沒有可用記憶。"


def should_save_as_fact(user_message):
    return any(keyword in user_message for keyword in IMPORTANT_MEMORY_KEYWORDS)


def update_user_memory(user_id, user_message, assistant_reply, character):
    memory = load_all_memory()
    user_memory = memory.get(user_id, {
        "summary": "",
        "facts": [],
        "history": [],
    })

    history = user_memory.get("history", [])
    history.append({"role": "user", "text": user_message})
    history.append({"role": "assistant", "text": assistant_reply, "character": character})
    user_memory["history"] = history[-20:]

    if should_save_as_fact(user_message):
        fact = f"使用者曾說：{user_message}"
        facts = user_memory.get("facts", [])
        if fact not in facts:
            facts.append(fact)
        user_memory["facts"] = facts[-20:]

    # 簡單摘要，不另外呼叫 Gemini，速度比較快。
    negative_count = sum(
        1
        for item in user_memory["history"]
        if item.get("role") == "user"
        and any(word in item.get("text", "") for word in NEGATIVE_KEYWORDS)
    )

    if negative_count >= 3:
        user_memory["summary"] = "使用者最近多次表達壓力或低落，需要溫柔陪伴與低壓力回應。"
    elif user_memory.get("facts"):
        user_memory["summary"] = "使用者有分享過個人喜好、近況或情緒狀態，回覆時可自然接續。"
    else:
        user_memory["summary"] = ""

    memory[user_id] = user_memory
    save_all_memory(memory)


def generate_esconv_hint(user_message):
    context = f"seeker: {user_message}"
    inputs = esconv_tokenizer(
        context,
        return_tensors="pt",
        max_length=192,
        truncation=True,
    ).to(DEVICE)

    with generation_lock, torch.inference_mode():
        outputs = esconv_model.generate(
            **inputs,
            max_new_tokens=24,
            num_beams=1,
            do_sample=False,
        )

    result = esconv_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    return result or "先接住使用者的情緒，再給予簡短且不施壓的陪伴。"


def generate_empathetic_hint(user_message):
    prompt = f"""
請理解使用者目前的情緒，產生一句自然、簡短且具有同理心的回覆方向。
不要說教，不要自稱心理醫師；若使用者開心，請與對方分享喜悅。

使用者：{user_message}
同理回覆：
""".strip()

    inputs = empathetic_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=192,
        truncation=True,
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    with generation_lock, torch.inference_mode():
        outputs = empathetic_model.generate(
            **inputs,
            max_new_tokens=24,
            num_beams=1,
            do_sample=False,
            pad_token_id=empathetic_tokenizer.pad_token_id,
            eos_token_id=empathetic_tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    result = empathetic_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return result or "理解使用者當下的感受，給予自然且真誠的回應。"


def call_gemini(user_message, character, esconv_hint, empathetic_hint, memory_text, user_name="你", user_birthday=""):
    character_prompt = CHARACTER_PROMPTS.get(
        character,
        CHARACTER_PROMPTS["春野 咲良"],
    )

    prompt = f"""
{character_prompt}

本地記憶：
{memory_text}

使用者名稱：
{user_name}

使用者生日：
{user_birthday or "未設定"}

ESConv 情緒支持方向：
{esconv_hint}

EmpatheticDialogues 同理回應方向：
{empathetic_hint}

使用者現在說：
{user_message}

請綜合角色設定、本地記憶與情緒支持方向，以「{character}」的角色口吻自然回覆使用者。

規則：
1. 使用繁體中文，回答 1 到 3 句。
2. 可以自然稱呼使用者的名字，但不要每一句都叫名字。
3. 可以自然提到你記得的事情，但不要說「根據記憶」或「我讀到資料」。
4. 如果生日未設定，不要提到生日；如果生日有設定，也只有在自然相關時才提起。
5. 先理解情緒，再視情況提供陪伴、鼓勵或一項簡短建議。
6. 不要提到 ESConv、EmpatheticDialogues、資料集、模型或 AI。
7. 不要直接照抄或逐字翻譯上述建議。
8. 不要診斷疾病，也不要使用責備、否定或過度樂觀的語氣。
9. 若內容涉及自傷、自殺或立即危險，請溫和鼓勵使用者立刻聯絡可信任的人、當地緊急服務或危機支援資源。
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 256,
                    "thinkingConfig": {
                        "thinkingBudget": 0
                    }
                },
            },
            timeout=45,
        )
    except requests.RequestException as error:
        print("Gemini 連線錯誤：", error)
        return "抱歉，剛剛連線好像出了點問題。可以再說一次嗎？"

    if not response.ok:
        print("Gemini API 錯誤：", response.status_code, response.text)
        return "抱歉，剛剛連線好像出了點問題。可以再說一次嗎？"

    data = response.json()

    usage = data.get("usageMetadata", {})
    print("輸入 tokens：", usage.get("promptTokenCount"))
    print("輸出 tokens：", usage.get("candidatesTokenCount"))
    print("總 tokens：", usage.get("totalTokenCount"))

    try:
        candidate = data["candidates"][0]
        print("Gemini finishReason：", candidate.get("finishReason", "UNKNOWN"))

        text_parts = [
            part["text"]
            for part in candidate["content"]["parts"]
            if part.get("text")
        ]
        final_text = "".join(text_parts).strip()

        if not final_text:
            raise ValueError("Gemini 沒有回傳文字")

        return final_text
    except (KeyError, IndexError, TypeError, ValueError):
        print("Gemini 回傳格式異常：", data)
        return "我還在聽，可以再多告訴我一點嗎？"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "esconv_model": ESCONV_MODEL_PATH.name,
        "empathetic_model": EMPATHETIC_LORA_PATH.name,
        "memory_file": MEMORY_PATH.name,
    })


@app.route("/memory", methods=["GET"])
def read_memory():
    user_id = str(request.args.get("user_id", "default_user")).strip() or "default_user"
    return jsonify(get_user_memory(user_id))


@app.route("/memory/clear", methods=["POST"])
def clear_memory():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "default_user")).strip() or "default_user"

    memory = load_all_memory()
    memory.pop(user_id, None)
    save_all_memory(memory)

    return jsonify({"status": "cleared", "user_id": user_id})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    user_message = str(data.get("message", "")).strip()
    character = data.get("character", "春野 咲良")
    user_id = str(data.get("user_id", "default_user")).strip() or "default_user"
    user_name = str(data.get("user_name", "你")).strip() or "你"
    user_birthday = str(data.get("user_birthday", "")).strip()

    if not user_message:
        return jsonify({"error": "沒有收到訊息"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "尚未設定 GEMINI_API_KEY"}), 500

    is_negative = any(word in user_message for word in NEGATIVE_KEYWORDS)
    user_memory = get_user_memory(user_id)
    memory_text = build_memory_text(user_memory)

    try:
        # 加速版：
        # 負面情緒：跑 ESConv，Empathetic 用固定方向。
        # 一般聊天：不跑本地模型，直接 Gemini。
        if is_negative:
            print("啟動 ESConv 情緒支持模式")
            esconv_hint = generate_esconv_hint(user_message)
            empathetic_hint = "請先理解並接納使用者的情緒，再自然陪伴，不要說教。"
        else:
            print("啟動快速 Gemini 模式")
            esconv_hint = "使用者目前未出現明確負面情緒，請依照情境自然互動，不要強行安慰。"
            empathetic_hint = "請依照角色個性自然回應使用者。"

        print("記憶摘要：", memory_text)
        print("ESConv 提示：", esconv_hint)
        print("EmpatheticDialogues 提示：", empathetic_hint)

        final_reply = call_gemini(
            user_message,
            character,
            esconv_hint,
            empathetic_hint,
            memory_text,
            user_name,
            user_birthday,
        )

        update_user_memory(user_id, user_message, final_reply, character)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("CUDA 記憶體不足")
        return jsonify({
            "error": "模型記憶體不足，請改用較小模型或量化載入。"
        }), 503
    except Exception as error:
        print("產生回覆時發生錯誤：", repr(error))
        return jsonify({
            "error": "後端暫時無法產生回覆，請稍後再試。"
        }), 500

    return jsonify({
        "reply": final_reply,
        "user_id": user_id,
        "used_memory": memory_text,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
