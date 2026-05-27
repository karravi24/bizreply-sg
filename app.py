import time
import threading
import requests
import chromadb
from functools import lru_cache
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from app_logging import logger
from services.rag_service import (search_documents, build_context)
from services.document_loader import (scan_uploads_folder)

app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

import os
WASENDER_KEY = os.getenv("WASENDER_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
INSTANCE_ID = os.getenv("INSTANCE_ID")

if not all([WASENDER_KEY, GEMINI_KEY, INSTANCE_ID]):
    logger.error("Missing env vars: WASENDER_KEY, GEMINI_KEY, or INSTANCE_ID")
    raise SystemExit("Set env vars before running.")

logger.info("Starting bot successfully. Instance ID: %s", INSTANCE_ID)

scan_uploads_folder()

# 1. Load embedding model and vector DB once
#logger.info("Loading embedding model...")
#embed_model = SentenceTransformer("all-MiniLM-L6-v2")
#chroma_client = chromadb.PersistentClient(path="./chroma_db")
#collection = chroma_client.get_or_create_collection("business_docs")
#logger.info("ChromaDB loaded. Docs in collection: %d", collection.count())

# # 2. Run this once to load your docs into ChromaDB
# def load_docs():
#     logger.info("Loading docs from faq.md...")
#     if not os.path.exists("faq.md"):
#         logger.error("faq.md file not found! Skipping database pre-load.")
#         return
#
#     with open("faq.md", "r", encoding="utf-8") as f:
#         chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
#
#     if chunks:
#         embeddings = embed_model.encode(chunks).tolist()
#         collection.add(
#             documents=chunks,
#             embeddings=embeddings,
#             ids=[f"doc_{i}" for i in range(len(chunks))]
#         )
#         logger.info("Added %d chunks to ChromaDB", len(chunks))
#
# # Auto-load if empty
# if collection.count() == 0:
#     load_docs()

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
                raise
    return None

@lru_cache(maxsize=50)
def get_cached_reply(system_prompt, user_msg):
    """Cache responses for repeated questions."""
    return call_gemini(system_prompt, user_msg)

def send_whatsapp_async(sender, reply):
    """Send WhatsApp message in background thread."""
    wasender_url = "https://www.wasenderapi.com/api/send-message"
    headers = {
        "Authorization": f"Bearer {WASENDER_KEY.strip()}",
        "Content-Type": "application/json"
    }
    reply_payload = {
        "to": str(sender),
        "text": str(reply)
    }

    try:
        resp = requests.post(wasender_url, json=reply_payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            logger.info("Message sent successfully: %s", resp.text)
        else:
            logger.error("WaSenderAPI failed: %d %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Error sending WhatsApp message: %s", e)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.json
        logger.info("Webhook hit. Data: %s", data)

        if not data:
            return jsonify({"status": "ignored"}), 200

        # 1. Real WaSenderAPI webhook
        if data.get('event') == 'messages.received':
            msg_wrapper = data.get('data', {}).get('messages', {})
            message_content = msg_wrapper.get('message', {})
            key_data = msg_wrapper.get('key', {})

            user_msg = ""
            if 'conversation' in message_content:
                user_msg = message_content.get('conversation', '')
            elif 'extendedTextMessage' in message_content:
                user_msg = message_content.get('extendedTextMessage', {}).get('text', '')

            sender = key_data.get('cleanedSenderPn', '')
            if not sender:
                sender = key_data.get('senderPn', '').split('@')[0]

        # 2. Test payload
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

        # 3. Retrieve context from ChromaDB - reduced to 1 chunk for speed
        documents = search_documents(
            query=user_msg,
            customer_name="beesbuzz",
            n_results=3
        )

        context = build_context(documents)

        logger.info("Context retrieved: %s", context[:200])

        system_prompt = f"""
        You are a WhatsApp assistant for BEESBUZZ Store.
        Use ONLY the info below. Output the full answer exactly as written in Context.

        Context:
        {context}

        Rules:
        - Copy the complete answer from Context, do not shorten
        - Do not add anything not in Context
        - Use 1 emoji max
        - If answer not in Context, reply: "I’ll check and get back to you"
        """

        reply = get_cached_reply(system_prompt, user_msg)
        logger.info("Gemini reply: %s", reply)

        # 5. Send via WaSenderAPI in background thread
        threading.Thread(target=send_whatsapp_async, args=(sender, reply), daemon=True).start()

        return jsonify({"status": "ok"})

    except Exception as e:
        logger.exception("Error in webhook: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)