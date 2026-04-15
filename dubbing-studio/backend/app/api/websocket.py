"""
WebSocket endpoint for real-time job progress updates.
"""
import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.websocket_manager import ws_manager
from app.config import settings

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    await ws_manager.connect(websocket, job_id)
    logger.info(f"WebSocket connected for job {job_id}")

    redis_listener = asyncio.create_task(
        _listen_redis(websocket, job_id)
    )

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        redis_listener.cancel()
        ws_manager.disconnect(websocket, job_id)


async def _listen_redis(websocket: WebSocket, job_id: str):
    """Listen to Redis pub/sub channel and forward updates to WebSocket."""
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"job:{job_id}")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    await websocket.send_text(data)

                    parsed = json.loads(data)
                    if parsed.get("status") in ("completed", "failed", "cancelled"):
                        break
                except Exception as e:
                    logger.error(f"Error forwarding Redis message: {e}")

        await pubsub.unsubscribe(f"job:{job_id}")
        await r.close()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Redis listener error: {e}")
