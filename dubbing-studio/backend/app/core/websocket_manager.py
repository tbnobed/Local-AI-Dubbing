from fastapi import WebSocket
from typing import Dict, Set
import json
import asyncio


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast_job_update(self, job_id: str, data: dict):
        if job_id not in self.active_connections:
            return
        dead = set()
        message = json.dumps(data)
        for ws in self.active_connections[job_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active_connections[job_id].discard(ws)

    async def broadcast_all(self, data: dict):
        message = json.dumps(data)
        all_dead = {}
        for job_id, connections in self.active_connections.items():
            dead = set()
            for ws in connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            all_dead[job_id] = dead
        for job_id, dead in all_dead.items():
            for ws in dead:
                self.active_connections[job_id].discard(ws)


ws_manager = WebSocketManager()
