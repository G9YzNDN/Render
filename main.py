import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

class Turn(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    user_msg: str
    history: list[Turn] = []
    system_prompt: str = "You are a helpful assistant."

def build_messages(history, user_msg, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_msg})
    return messages

@app.post("/api/chat")
def chat(req: ChatRequest):
    messages = build_messages(
        [t.model_dump() for t in req.history], req.user_msg, req.system_prompt
    )

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",  # use whatever model your proxy supports
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps(delta)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Serve the web page from the same origin (no CORS needed)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
