"""REST API endpoint for file-based audio transcription (OpenAI)."""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
import logging

from app.core.container import container
from app.services.file_transcribe_service import FileTranscribeService
from app.utils.jwt_auth import JWTAuth, get_current_user
from app.models.file_transcribe_models import TranscribeFileRequest, TranscribeFileResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/transcribe/file", response_model=TranscribeFileResponse)
async def transcribe_audio_file(
    file: UploadFile = File(..., description="Audio file to transcribe (max 25MB)"),
    language_code: str = "es-ES",
    current_user: dict = Depends(get_current_user)
):
    """
    Transcribe an audio file using OpenAI gpt-4o-mini-transcribe.

    This endpoint accepts audio file uploads and returns the complete transcription.
    Unlike the WebSocket endpoint, this processes the entire file at once (batch mode).

    Supported formats:
    - WAV, MP3, MP4, MPEG, MPGA, M4A, OGG, WEBM, FLAC

    Args:
        file: Audio file (max 25MB)
        language_code: Language code (e.g., 'es-ES', 'en-US', 'pt-BR')
        current_user: Authenticated user (from JWT token)

    Returns:
        TranscribeFileResponse with transcription result

    Raises:
        HTTPException 400: Invalid file or parameters
        HTTPException 413: File too large
        HTTPException 503: Service unavailable
    """
    transcriber = None
    user_id = current_user.get('ID_USUARIO')

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )

        # Check file extension
        allowed_extensions = {'.wav', '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.ogg', '.webm', '.flac'}
        file_extension = file.filename.lower()[file.filename.rfind('.'):] if '.' in file.filename else ''

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )

        logger.info(
            f"File transcription request: "
            f"user_id={user_id}, "
            f"filename={file.filename}, "
            f"language={language_code}"
        )

        # Read file content
        audio_data = await file.read()
        file_size = len(audio_data)

        # Validate file size (25MB limit for OpenAI)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {file_size} bytes (max {max_size} bytes / 25MB)"
            )

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )

        logger.info(f"File loaded: {file_size} bytes")

        # Create file transcribe session (OpenAI)
        try:
            file_transcribe_port = container.create_file_transcribe_session()
            transcriber = FileTranscribeService(file_transcribe_port=file_transcribe_port)
            logger.info(f"File transcribe session created for user_id={user_id}")

        except Exception as e:
            logger.error(f"Failed to create file transcribe session: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Transcription service temporarily unavailable"
            )

        # Transcribe file
        result = await transcriber.transcribe_file(
            audio_file=audio_data,
            language_code=language_code,
            filename=file.filename,
            user_id=user_id
        )

        logger.info(
            f"Transcription complete: "
            f"user_id={user_id}, "
            f"chars={len(result['transcript'])}, "
            f"language={result['language']}"
        )

        # Return result
        return TranscribeFileResponse(
            transcript=result['transcript'],
            language=result['language'],
            duration=result['duration'],
            confidence=result.get('confidence'),
            model=result['model'],
            file_size=file_size
        )

    except HTTPException:
        # Re-raise FastAPI exceptions
        raise

    except ValueError as e:
        logger.error(f"Validation error in file transcription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except ConnectionError as e:
        logger.error(f"Connection error in file transcription: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcription service temporarily unavailable"
        )

    except Exception as e:
        logger.error(f"Unexpected error in file transcription: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

    finally:
        # Cleanup resources
        if transcriber:
            try:
                await transcriber.close()
                logger.info(f"File transcribe session closed for user_id={user_id}")
            except Exception as e:
                logger.error(f"Error closing file transcribe session: {e}")


@router.get("/transcribe/file/health")
async def file_transcribe_health_check():
    """
    Health check endpoint for file transcription service.

    Returns:
        Status of OpenAI transcription service configuration
    """
    try:
        from app.core.config import settings

        # Check if OpenAI is configured
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        config_check = {
            "status": "healthy",
            "service": "file_transcription",
            "provider": "openai",
            "config": {
                "model": settings.openai_transcribe_model or "gpt-4o-mini-transcribe",
                "base_url": settings.openai_base_url or "https://api.openai.com/v1",
                "max_file_size_mb": 25
            }
        }

        return config_check

    except Exception as e:
        logger.error(f"File transcription health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"File transcription service unhealthy: {str(e)}"
        )
