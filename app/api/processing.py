from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File, Form
from typing import Dict, List, Optional, Any
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from app.utils.jwt_auth import get_current_user
from app.core.config import settings
from app.services.document_processing_service import document_processing_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Models
class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    company_name: str
    area_name: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    files_processed: int = 0
    total_files: int = 0

class UploadResponse(BaseModel):
    task_id: str
    message: str
    files_uploaded: int
    status: str

class ProcessingRequest(BaseModel):
    company_name: str
    area_name: str

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for streaming processing logs"""
    # Get task_id from query parameters instead of path
    query_params = websocket.query_params
    task_id = query_params.get("task_id")
    
    if not task_id:
        await websocket.close(code=4000, reason="Missing task_id parameter")
        return
        
    await document_processing_service.websocket_manager.connect(websocket, task_id)
    try:
        while True:
            # Wait for messages (this keeps the connection alive)
            message = await websocket.receive_text()
            # Echo back any received messages (optional)
            if message:
                await websocket.send_text(f"Received: {message}")
    except WebSocketDisconnect:
        document_processing_service.websocket_manager.disconnect(websocket, task_id)
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        document_processing_service.websocket_manager.disconnect(websocket, task_id)

@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    company_name: str = Form(...),
    area_name: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload PDF files for processing"""
    try:
        # Log authenticated user information
        logger.info(f"Files uploaded by authenticated user ID: {current_user.get('ID_USUARIO')}")
        
        # Create upload directory in CargaConocimiento_iA using config
        carga_dir = Path(settings.carga_conocimiento_path)
        if not carga_dir.exists():
            raise HTTPException(status_code=500, detail=f"CargaConocimiento_iA directory not found at {carga_dir}")
        
        upload_dir = carga_dir / settings.upload_directory
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files in the directory
        for existing_file in upload_dir.glob("*"):
            if existing_file.is_file():
                existing_file.unlink()
        
        uploaded_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
        
        # Create task using the service
        task_id = document_processing_service.create_task(company_name, area_name, uploaded_files)
        
        logger.info(f"Files uploaded for task {task_id}: {uploaded_files}")
        
        return UploadResponse(
            task_id=task_id,
            message=f"Successfully uploaded {len(uploaded_files)} files",
            files_uploaded=len(uploaded_files),
            status="uploaded"
        )
        
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start/{task_id}")
async def start_processing(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Start processing uploaded files"""
    # Log authenticated user information
    logger.info(f"Processing started by authenticated user ID: {current_user.get('ID_USUARIO')} for task: {task_id}")
    
    success = await document_processing_service.start_processing_task(task_id)
    
    if not success:
        task = document_processing_service.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        else:
            raise HTTPException(status_code=400, detail=f"Task is already {task['status']}")
    
    return {"message": "Processing started", "task_id": task_id}

@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get task status"""
    task = document_processing_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        company_name=task["company_name"],
        area_name=task["area_name"],
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        error_message=task.get("error_message"),
        files_processed=task["files_processed"],
        total_files=task["total_files"]
    )

@router.get("/tasks")
async def list_tasks(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all tasks"""
    return {"tasks": document_processing_service.get_all_tasks()}

