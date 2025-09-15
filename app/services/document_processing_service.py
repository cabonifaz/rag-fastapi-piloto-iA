import json
import uuid
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from app.core.config import settings

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.connections:
            self.connections[task_id] = []
        self.connections[task_id].append(websocket)
        logger.info(f"WebSocket connected for task {task_id}")
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.connections:
            self.connections[task_id].remove(websocket)
            if not self.connections[task_id]:
                del self.connections[task_id]
        logger.info(f"WebSocket disconnected for task {task_id}")
    
    async def send_message(self, task_id: str, message: dict):
        if task_id in self.connections:
            disconnected = []
            for websocket in self.connections[task_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, task_id)
    
    async def close_task_connections(self, task_id: str):
        """Close all WebSocket connections for a specific task"""
        if task_id in self.connections:
            connections_to_close = self.connections[task_id].copy()
            for websocket in connections_to_close:
                try:
                    await websocket.close(code=1000, reason="Task completed")
                    logger.info(f"Closed WebSocket connection for completed task {task_id}")
                except Exception as e:
                    logger.error(f"Error closing WebSocket for task {task_id}: {e}")
                finally:
                    self.disconnect(websocket, task_id)

class DocumentProcessingService:
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    def create_task(self, company_name: str, area_name: str, files: List[str]) -> str:
        """Create a new processing task"""
        task_id = str(uuid.uuid4())
        
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "company_name": company_name,
            "area_name": area_name,
            "created_at": datetime.now(),
            "files_uploaded": files,
            "total_files": len(files),
            "files_processed": 0
        }
        
        logger.info(f"Created task {task_id} for company '{company_name}', area '{area_name}' with {len(files)} files")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return self.active_tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks"""
        return self.active_tasks
    
    async def start_processing_task(self, task_id: str) -> bool:
        """Start processing a task"""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        if task["status"] != "pending":
            return False
        
        # Update task status
        task["status"] = "running"
        task["started_at"] = datetime.now()
        
        # Start background processing
        asyncio.create_task(self._run_processing_task(task_id))
        
        # Send initial status via WebSocket
        await self.websocket_manager.send_message(task_id, {
            "type": "status",
            "message": "Processing started",
            "task_id": task_id,
            "status": "running"
        })
        
        return True
    
    async def _run_processing_task(self, task_id: str):
        """Run the document processing task"""
        task = self.active_tasks[task_id]
        
        try:
            company_name = task["company_name"]
            area_name = task["area_name"]
            
            # Send progress update
            await self.websocket_manager.send_message(task_id, {
                "type": "log",
                "message": f"Starting document processing for company '{company_name}' and area '{area_name}'",
                "timestamp": datetime.now().isoformat()
            })
            
            # Use CargaConocimiento_iA directory from config
            carga_dir = Path(settings.carga_conocimiento_path)
            if not carga_dir.exists():
                raise Exception(f"CargaConocimiento_iA directory not found at {carga_dir}")
            
            # Build command with config-based venv python path
            venv_python = carga_dir / settings.processing_venv_path
            if not venv_python.exists():
                raise Exception(f"CargaConocimiento_iA venv python not found at {venv_python}")
            
            cmd = [
                str(venv_python), "-m", "app.main",
                "upsert-company-files",
                company_name,
                area_name
            ]
            
            # Send command info
            await self.websocket_manager.send_message(task_id, {
                "type": "log",
                "message": f"Executing: {' '.join(cmd)}",
                "timestamp": datetime.now().isoformat()
            })
            
            # Run the command with real-time output streaming
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(carga_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Stream output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                log_message = line.decode().strip()
                if log_message:
                    await self.websocket_manager.send_message(task_id, {
                        "type": "log",
                        "message": log_message,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Wait for process to complete
            await process.wait()
            
            if process.returncode == 0:
                # Success
                task["status"] = "completed"
                task["completed_at"] = datetime.now()
                task["files_processed"] = task["total_files"]
                
                await self.websocket_manager.send_message(task_id, {
                    "type": "status",
                    "message": "Processing completed successfully",
                    "task_id": task_id,
                    "status": "completed"
                })
                
                # Close WebSocket connections after successful completion
                await asyncio.sleep(2)  # Give clients time to receive the final message
                await self.websocket_manager.close_task_connections(task_id)
            else:
                # Error
                task["status"] = "failed"
                task["completed_at"] = datetime.now()
                task["error_message"] = f"Process exited with code {process.returncode}"
                
                await self.websocket_manager.send_message(task_id, {
                    "type": "error",
                    "message": f"Processing failed with exit code {process.returncode}",
                    "task_id": task_id,
                    "status": "failed"
                })
                
                # Close WebSocket connections after failure
                await asyncio.sleep(2)  # Give clients time to receive the error message
                await self.websocket_manager.close_task_connections(task_id)
                
        except Exception as e:
            logger.error(f"Error in processing task {task_id}: {e}")
            task["status"] = "failed"
            task["completed_at"] = datetime.now()
            task["error_message"] = str(e)
            
            await self.websocket_manager.send_message(task_id, {
                "type": "error",
                "message": f"Processing failed: {str(e)}",
                "task_id": task_id,
                "status": "failed"
            })
            
            # Close WebSocket connections after exception
            await asyncio.sleep(2)  # Give clients time to receive the error message
            await self.websocket_manager.close_task_connections(task_id)

# Singleton instance
document_processing_service = DocumentProcessingService()