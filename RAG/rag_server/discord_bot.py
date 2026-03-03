"""
위니쿠키 디스코드 챗봇

discord.py를 사용하여 디스코드 채널에서 고객 질문을 받고,
FastAPI RAG 서버를 호출하여 답변을 반환합니다.

실행 방법:
    python discord_bot.py

사전 요구사항:
    - .env 파일에 DISCORD_TOKEN 설정
    - FastAPI 서버가 localhost:8000에서 실행 중이어야 합니다
    - Ollama + Bllossom-3B가 실행 중이어야 합니다
"""

import os
import discord
import requests
from dotenv import load_dotenv

# ============================================================
# 설정 (.env에서 로드)
# ============================================================
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/chat")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID", "0"))

if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN이 .env 파일에 설정되지 않았습니다.")


# ============================================================
# 디스코드 봇 설정
# ============================================================
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"✅ 봇 로그인 완료: {client.user}")
    print(f"📌 허용 채널 ID: {ALLOWED_CHANNEL_ID}")
    print(f"🔗 RAG API: {RAG_API_URL}")
    print("─" * 40)


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.id != ALLOWED_CHANNEL_ID:
        return

    query = message.content.strip()
    if not query:
        return

    async with message.channel.typing():
        try:
            response = requests.post(RAG_API_URL, json={"message": query}, timeout=30)
            data = response.json()
            answer = data["answer"]
        except requests.exceptions.ConnectionError:
            answer = "⚠️ RAG 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요."
        except requests.exceptions.Timeout:
            answer = "⚠️ 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
        except Exception as e:
            answer = f"⚠️ 오류가 발생했습니다: {str(e)}"

    await message.reply(answer)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("🍪 위니쿠키 디스코드 챗봇 시작...")
    client.run(DISCORD_TOKEN)
