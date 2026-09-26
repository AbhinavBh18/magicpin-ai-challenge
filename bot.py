import os
import socket
import time
import json
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency is handled at runtime
    genai = None

app = FastAPI()
START = time.time()
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# Configure Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key and genai is not None:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-latest")
else:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Using built-in fallback logic.")
    model = None

# In-memory stores
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Antigravity Vera",
        "team_members": ["AI Assistant"],
        "model": "gemini-flash-latest",
        "approach": "Context-aware WhatsApp composer with deterministic fallback heuristics",
        "contact_email": "hello@example.com",
        "version": "1.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(VALID_SCOPES)}",
        }

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] == body.version:
        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z",
        }
    if cur and cur["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _fallback_message_for_trigger(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    merchant_name = merchant.get("identity", {}).get("name") or "your business"
    category_name = category.get("slug") or category.get("display_name") or "this category"
    trigger_kind = str(trigger.get("kind", "generic")).replace("_", " ")
    customer_name = None
    if customer:
        customer_name = customer.get("identity", {}).get("name") or "there"

    if customer_name:
        if trigger_kind.lower().find("recall") >= 0 or trigger.get("scope") == "customer":
            body = (
                f"Hi {customer_name}, this is {merchant_name} on WhatsApp. "
                f"Your recall window is due and we can help with a quick check-up. "
                "Reply YES to confirm a slot or STOP to skip this reminder."
            )
            cta = "YES/STOP"
            send_as = "merchant_on_behalf"
        else:
            body = (
                f"Hi {customer_name}, {merchant_name} here. We’ve got an update for you and would love to help. "
                "Reply YES and we’ll take it from there."
            )
            cta = "YES/STOP"
            send_as = "merchant_on_behalf"
    else:
        if trigger_kind.lower().find("research") >= 0:
            body = (
                f"{merchant_name}, one fresh insight for {category_name}: the latest research is relevant to your patient mix. "
                "Worth a quick look before the next patient visit. Want me to pull the short summary?"
            )
            cta = "open_ended"
        elif trigger_kind.lower().find("perf") >= 0 or trigger_kind.lower().find("dip") >= 0:
            body = (
                f"{merchant_name}, I noticed a dip in performance and a couple of quick fixes could help. "
                "Want me to suggest the next best move?"
            )
            cta = "open_ended"
        else:
            body = (
                f"Hi {merchant_name}, there’s a timely update for your {category_name} business that could help this week. "
                "Would you like me to share the detail?"
            )
            cta = "open_ended"
        send_as = "vera"

    return {
        "body": body,
        "cta": cta if customer_name or trigger_kind else "open_ended",
        "send_as": send_as,
        "rationale": "Fallback composer used a deterministic, context-aware message because no live model was configured.",
    }


def generate_composition(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    if not model:
        return _fallback_message_for_trigger(category, merchant, trigger, customer)

    system_prompt = """You are Vera, a helpful AI marketing assistant for local merchants on WhatsApp.
Your goal is to compose highly specific, engaging, and personalized WhatsApp messages for merchants (or their customers).
Adhere strictly to the category voice (e.g. clinical for dentists, warm for salons).
Use specific, verifiable numbers and facts from the context.
Keep it concise. If addressing a merchant in India, a natural Hindi-English code-mix is often preferred (if they use 'hi' or 'hi-en').
Never hallucinate facts, numbers, competitors, or research.
Use one or more compulsion levers (curiosity, social proof, loss aversion, effort externalization).
IMPORTANT: Return ONLY a valid JSON object. No markdown blocks, no explanation.

Expected JSON format:
{
  "body": "The text of the WhatsApp message",
  "cta": "open_ended" | "YES/STOP" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "rationale": "1-2 sentence explanation of your strategy"
}"""

    prompt = "Compose a message based on the following contexts:\n\n"
    prompt += f"CATEGORY: {json.dumps(category)}\n"
    prompt += f"MERCHANT: {json.dumps(merchant)}\n"
    prompt += f"TRIGGER: {json.dumps(trigger)}\n"
    if customer:
        prompt += f"CUSTOMER: {json.dumps(customer)}\n"

    try:
        response = model.generate_content(
            system_prompt + "\n\n" + prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(response.text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned a non-object JSON response")
        parsed.setdefault("cta", "open_ended")
        parsed.setdefault("send_as", "vera")
        parsed.setdefault("rationale", "Generated from available context")
        return parsed
    except Exception as e:
        print(f"Error generating composition: {e}")
        return _fallback_message_for_trigger(category, merchant, trigger, customer)


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    triggers_to_process = body.available_triggers[:5] if body.available_triggers else []

    if not triggers_to_process:
        for key, item in contexts.items():
            if key[0] == "trigger":
                triggers_to_process.append(key[1])
                if len(triggers_to_process) >= 5:
                    break

    for trg_id in triggers_to_process:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg:
            continue

        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else None
        if not merchant:
            continue

        category_slug = merchant.get("category_slug") or "unknown"
        category = contexts.get(("category", category_slug), {}).get("payload")
        if not category:
            continue

        customer_id = trg.get("customer_id")
        customer = None
        if customer_id:
            customer = contexts.get(("customer", customer_id), {}).get("payload")

        composition = generate_composition(category, merchant, trg, customer)
        actions.append(
            {
                "conversation_id": f"conv_{merchant_id}_{trg_id}_{int(time.time())}",
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composition.get("send_as", "vera"),
                "trigger_id": trg_id,
                "template_name": "dynamic_llm_template",
                "template_params": [],
                "body": composition.get("body", "Hello!"),
                "cta": composition.get("cta", "open_ended"),
                "suppression_key": trg.get("suppression_key", f"fallback_{trg_id}"),
                "rationale": composition.get("rationale", "LLM decided to send this."),
            }
        )

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


def _is_auto_reply(message: str) -> bool:
    normalized = message.strip().lower()
    auto_patterns = [
        "thank you for contacting",
        "i am an automated assistant",
        "we will get back to you",
        "we are closed",
        "this is an automated reply",
        "auto reply",
        "our team will contact you",
    ]
    return any(pattern in normalized for pattern in auto_patterns)


def _is_stop_message(message: str) -> bool:
    normalized = message.strip().lower()
    stop_patterns = [
        "stop",
        "unsubscribe",
        "not interested",
        "no thanks",
        "do not contact",
        "don't contact",
        "please stop",
    ]
    return any(pattern in normalized for pattern in stop_patterns)


def _is_positive_message(message: str) -> bool:
    normalized = message.strip().lower()
    positive_patterns = [
        "yes",
        "sure",
        "go ahead",
        "okay",
        "ok",
        "please proceed",
        "sounds good",
        "send it",
        "let's do it",
    ]
    return any(pattern in normalized for pattern in positive_patterns)


def _is_wait_message(message: str) -> bool:
    normalized = message.strip().lower()
    wait_patterns = [
        "later",
        "busy",
        "need time",
        "not now",
        "maybe",
        "check with me later",
        "can do later",
    ]
    return any(pattern in normalized for pattern in wait_patterns)


def generate_reply(conversation_history: list, latest_msg: str, merchant: dict) -> dict:
    normalized = latest_msg.strip()
    if not normalized:
        return {"action": "wait", "wait_seconds": 1800, "rationale": "No reply content received; waiting for the merchant to continue."}

    if _is_auto_reply(normalized):
        return {"action": "end", "rationale": "Detected a canned or automated business auto-reply; ending the conversation."}

    if _is_stop_message(normalized):
        return {"action": "end", "rationale": "Merchant requested no further outreach; graceful exit."}

    if _is_positive_message(normalized):
        merchant_name = merchant.get("identity", {}).get("name") or "your business"
        return {
            "action": "send",
            "body": f"Perfect — I’ve got it. I’ll move this forward for {merchant_name} and keep the next step simple.",
            "cta": "open_ended",
            "rationale": "The merchant explicitly agreed to continue, so the conversation should move to action rather than asking more qualifying questions.",
        }

    if _is_wait_message(normalized):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "The merchant asked for time; backing off briefly and re-engaging later."}

    if model:
        system_prompt = """You are Vera, a helpful AI marketing assistant on WhatsApp.
You are in an ongoing conversation with a merchant.
Analyze the merchant's latest message and decide the next action.

IMPORTANT RULES:
1. If it's an auto-reply (e.g. "Thank you for contacting us...", "I am an automated assistant"), you MUST return action: "end".
2. If the merchant says they are not interested, asks to stop, or is hostile, return action: "end".
3. If the merchant explicitly agrees to proceed ("ok let's do it", "yes", "go ahead"), switch to ACTION mode. Say what you have done or are doing instead of asking more qualifying questions.
4. Return ONLY a valid JSON object. No markdown blocks, no explanation.

Expected JSON format:
{
  "action": "send" | "wait" | "end",
  "body": "The message to send (only if action is send)",
  "cta": "open_ended" | "YES/STOP" | "none",
  "wait_seconds": integer (only if action is wait),
  "rationale": "1-2 sentence explanation of your choice"
}"""

        prompt = f"MERCHANT PROFILE: {json.dumps(merchant.get('identity', {}))}\n\n"
        prompt += "CONVERSATION HISTORY:\n"
        for turn in conversation_history[-4:]:
            prompt += f"[{turn['from'].upper()}]: {turn['msg']}\n"
        prompt += f"[LATEST - MERCHANT]: {normalized}\n"

        try:
            response = model.generate_content(
                system_prompt + "\n\n" + prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(response.text)
            if parsed.get("action") not in {"send", "wait", "end"}:
                raise ValueError("action invalid")
            if parsed.get("action") == "send" and not parsed.get("body"):
                parsed["body"] = "Thanks — I can help with the next step."
            if parsed.get("action") == "wait" and not parsed.get("wait_seconds"):
                parsed["wait_seconds"] = 1800
            return parsed
        except Exception as e:
            print(f"Error generating reply: {e}")

    return {
        "action": "send",
        "body": "Thanks — I can help with the next step and keep it simple.",
        "cta": "open_ended",
        "rationale": "Fallbacked to a simple, non-blocking follow-up because the merchant message did not signal a hard-stop or explicit approval.",
    }


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_history = conversations.setdefault(body.conversation_id, [])

    merchant = {}
    if body.merchant_id:
        merchant = contexts.get(("merchant", body.merchant_id), {}).get("payload", {})

    is_auto_reply = False
    merchant_msgs = [turn["msg"] for turn in conv_history if turn["from"] == "merchant"]
    if len(merchant_msgs) >= 2 and all(msg == body.message for msg in merchant_msgs[-2:]):
        is_auto_reply = True

    if is_auto_reply:
        conv_history.append({"from": body.from_role, "msg": body.message})
        return {"action": "end", "rationale": "Detected verbatim repeated merchant message; treating it as a noisy auto-reply."}

    reply_action = generate_reply(conv_history, body.message, merchant)
    conv_history.append({"from": body.from_role, "msg": body.message})

    if reply_action.get("action") == "send":
        conv_history.append({"from": "vera", "msg": reply_action.get("body", "")})

    if reply_action.get("action") == "wait" and "wait_seconds" not in reply_action:
        reply_action["wait_seconds"] = 1800

    return reply_action


def _pick_free_port(default_port: int = 8080, max_attempts: int = 20) -> int:
    preferred = int(os.getenv("PORT", str(default_port)))
    ports_to_try = [preferred] + [default_port + offset for offset in range(1, max_attempts)]
    seen = set()
    for port in ports_to_try:
        if port in seen:
            continue
        seen.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return preferred


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = _pick_free_port()
    print(f"Starting Vera bot on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
