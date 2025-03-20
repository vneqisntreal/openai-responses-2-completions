from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Union, List, Optional
import httpx
import json

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    input: Union[str, List[ChatMessage]]
    temperature: Optional[float] = 1.0
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

async def stream_converter(response):
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            event_data = json.loads(line[6:])
            event_type = event_data.get("type", "")
            
            if event_type == "response.output_text.delta":
                chunk = {
                    "id": event_data["response"]["id"],
                    "object": "chat.completion.chunk",
                    "created": event_data["response"]["created_at"],
                    "model": event_data["response"]["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {"content": event_data["delta"]},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            elif event_type == "response.completed":
                chunk = {
                    "id": event_data["response"]["id"],
                    "object": "chat.completion.chunk",
                    "created": event_data["response"]["created_at"],
                    "model": event_data["response"]["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if isinstance(request.input, str):
        messages = [{"role": "user", "content": request.input}]
    else:
        messages = [msg.dict() for msg in request.input]

    payload = {
        "model": request.model,
        "input": messages[0]["content"] if len(messages) == 1 else messages,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "stream": request.stream
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                json=payload,
                headers={"Authorization": "Bearer YOUR_API_KEY"},
                timeout=30.0
            )
            response.raise_for_status()

            if request.stream:
                return StreamingResponse(
                    stream_converter(response),
                    media_type="text/event-stream"
                )
            
            data = response.json()
            chat_response = {
                "id": data["id"],
                "object": "chat.completion",
                "created": data["created_at"],
                "model": data["model"],
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data["output"][0]["content"][0]["text"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": data["usage"]
            }
            return ChatResponse(**chat_response)

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
