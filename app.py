import time
import threading
import requests
import os
from functools import lru_cache
from flask import Flask, request, jsonify
from app_logging import logger
from services.rag_service import search_documents, build_context
from services.document_loader import scan_uploads_folder
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

WASENDER_KEY = os.getenv("WASENDER_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
INSTANCE_ID = os.getenv("INSTANCE_ID")

if not all([WASENDER_KEY, GEMINI_KEY, INSTANCE_ID]):
    logger.error("Missing env vars: WASENDER_KEY, GEMINI_KEY, or INSTANCE_ID")
    raise SystemExit("Set env vars before running.")

logger.info("Starting bot successfully. Instance ID: %s", INSTANCE_ID)

# DON'T run this on import in production. Use a /reload endpoint or run once via CLI
# scan_uploads_folder()

def call_gemini(system_prompt, user_msg):
    """Call Gemini with retry and token limits for speed."""
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser: {user_msg}"}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 250
        }
    }

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(gemini_url, json=payload, timeout=15)
            if r.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Gemini 429, retrying in %ss", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            if attempt == max_retries:
                logger.error("Gemini call failed after retries: %s", e)
                return None
    return None

@lru_cache(maxsize=200)
def get_cached_reply(user_msg, context_hash):
    """
    Cache only on user_msg + context hash, not full context.
    This prevents memory bloat and increases cache hits.
    """
    system_prompt = f"""
    You are a precise WhatsApp assistant for BEESBUZZ Store answering stock/price checks.
    The customer is asking about a specific phone model (e.g., iPhone 7).
    Use ONLY the data in Context. Each product is formatted as columns separated by '|'.

    Context:
    {context_hash}

    Columns map to: [ID | Store | Product Name | Barcode | Sales Price | Purchase Price | Repair Price | Qty | Brand | Desc | Model # | Suitable Models]

    Rules:
    - CRITICAL: Check the model number carefully. If the customer asks for "iPhone 7" but the Context ONLY shows other models like "iPhone 14" or "iPhone 13", you MUST consider this a mismatch.
    - If the exact model requested by the customer is completely missing from the Context data, reply ONLY: "I’ll check and get back to you."
    - If the model matches perfectly, reply horizontally on 1 or 2 lines maximum using this exact format:
      📦 *[Product Name]* | 💰 Price: $[Sales Price] (Repair: $[Repair Price]) | Stock: [Qty]
    """
    return call_gemini(system_prompt, user_msg)

def send_whatsapp_async(sender, reply):
    """Send WhatsApp message in background thread."""
    if not reply or not reply.strip():
        logger.error("Empty reply, not sending to %s", sender)
        reply = "I’ll check and get back to you."

    wasender_url = "https://www.wasenderapi.com/api/send-message"
    headers = {
        "Authorization": f"Bearer {WASENDER_KEY.strip()}",
        "Content-Type": "application/json"
    }
    reply_payload = {"to": str(sender), "text": str(reply)}

    try:
        resp = requests.post(wasender_url, json=reply_payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            logger.info("Message sent successfully to %s", sender)
        else:
            logger.error("WaSenderAPI failed: %d %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Error sending WhatsApp message: %s", e)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "ignored"}), 200

        # Parse message
        user_msg, sender = "", ""
        if data.get('event') == 'messages.received':
            msg_wrapper = data.get('data', {}).get('messages', {})
            message_content = msg_wrapper.get('message', {})
            key_data = msg_wrapper.get('key', {})
            user_msg = message_content.get('conversation') or message_content.get('extendedTextMessage', {}).get('text', '')
            sender = key_data.get('cleanedSenderPn') or key_data.get('senderPn', '').split('@')[0]
        elif 'message' in data:
            user_msg = data['message'].get('text', '')
            sender = data['message'].get('from', '')
        else:
            logger.warning("Unknown payload structure")
            return jsonify({"status": "ignored"}), 200

        if not user_msg or not sender:
            logger.warning("Empty message='%s' or sender='%s'", user_msg, sender)
            return jsonify({"status": "ignored"}), 200

        logger.info("Processing Message from %s: %s", sender, user_msg)

        # Retrieve context
        documents = search_documents(query=user_msg, customer_name="beesbuzz", n_results=10)
        context = build_context(documents)

        if not documents or context == "No relevant information found.":
            reply = "I’ll check and get back to you."
        else:
            # Use hash of context for caching, not full text
            context_hash = str(hash(context))
            reply = get_cached_reply(user_msg, context_hash)
            if not reply:
                reply = "I’ll check and get back to you."

        logger.info("Gemini reply: %s", reply)
        threading.Thread(target=send_whatsapp_async, args=(sender, reply), daemon=True).start()

        return jsonify({"status": "ok"})

    except Exception as e:
        logger.exception("Error in webhook: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive"})

@app.route("/reload", methods=["POST"])
def reload_docs():
    """Call this manually when you upload new files"""
    try:
        scan_uploads_folder()
        return jsonify({"status": "reload complete"}), 200
    except Exception as e:
        logger.exception("Reload failed")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
