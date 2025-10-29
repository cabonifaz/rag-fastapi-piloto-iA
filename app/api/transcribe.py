"""WebSocket endpoint for real-time speech-to-text transcription."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, status
from typing import Optional, Dict, Any
import asyncio
import json
import logging

from app.core.container import container
from app.core.config import settings
from app.services.transcribe_service import TranscribeService
from app.models.transcribe_models import (
    TranscribeConfig,
    TranscribeWebSocketMessage,
    create_status_response,
    create_error_response,
    create_partial_response,
    create_final_response
)
from app.utils.jwt_auth import JWTAuth
from app.utils.websocket_utils import audio_stream_from_websocket

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT token for authentication")
):
    """
    WebSocket endpoint for real-time audio transcription.

    🚨 IMPORTANTE: Cada conexión WebSocket crea una NUEVA sesión de transcripción.
        → Sesión aislada por usuario
        → NO se comparte estado entre conexiones
        → Recursos se limpian al cerrar la conexión

    Protocol:
        1. Client connects with JWT token in query param
        2. Client sends config message (JSON):
           {"type": "config", "data": {...}}
        3. Client sends audio chunks (binary frames)
        4. Server sends transcription results (JSON):
           {"type": "partial/final", "result": {...}}
        5. Client sends stop message or disconnects
        6. Server closes session and cleans up resources

    Args:
        websocket: WebSocket connection
        token: JWT token for authentication
    """
    transcribe_session = None
    transcribe_service = None
    user_id = None

    try:
        # Accept WebSocket connection
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        # Verify JWT token using existing JWTAuth utility
        try:
            # Use the same JWT validation as REST endpoints
            user_data = JWTAuth.verify_jwt_token(token)
            user_id = user_data.get('ID_USUARIO')
            logger.info(f"User authenticated via WebSocket: user_id={user_id}")

            # Send connection confirmation
            response = create_status_response("connected")
            await websocket.send_json(response.model_dump())

        except HTTPException as e:
            # Send error and close
            error_detail = e.detail
            if isinstance(error_detail, dict) and "result" in error_detail:
                error_message = error_detail["result"].get("mensaje", "Authentication failed")
            else:
                error_message = str(error_detail)

            await websocket.send_json(
                create_error_response(error_message).model_dump()
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        except Exception as e:
            logger.error(f"WebSocket token verification failed: {e}")
            await websocket.send_json(
                create_error_response("Invalid or expired token").model_dump()
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Wait for configuration message
        try:
            config_message = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=10.0  # 10 seconds to receive config
            )
            message_data = json.loads(config_message)

            # Validate message type
            ws_message = TranscribeWebSocketMessage(**message_data)

            if ws_message.type != "config":
                raise ValueError("First message must be 'config' type")

            # Parse transcription configuration
            config = TranscribeConfig(**ws_message.data)
            logger.info(f"Transcription config received: {config.model_dump()}")

            # Send config confirmation
            await websocket.send_json(
                create_status_response("configured").model_dump()
            )

        except asyncio.TimeoutError as e:
            logger.error("Timeout waiting for config message")
            await websocket.send_json(
                create_error_response("Configuration timeout").model_dump()
            )
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid config message: {e}")
            await websocket.send_json(
                create_error_response(f"Invalid configuration: {str(e)}").model_dump()
            )
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        except Exception as e:
            logger.error(f"Unexpected error in config parsing: {type(e).__name__}: {e}", exc_info=True)
            await websocket.send_json(
                create_error_response(f"Error processing configuration: {str(e)}").model_dump()
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Create NEW transcribe session for THIS WebSocket connection
        try:
            transcribe_port = container.create_transcribe_session()
            transcribe_service = TranscribeService(transcribe_port=transcribe_port)
            logger.info(f"Transcribe session created for user_id={user_id}")

        except Exception as e:
            logger.error(f"Failed to create transcribe session: {e}")
            await websocket.send_json(
                create_error_response("Failed to initialize transcription service").model_dump()
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Create audio stream from WebSocket
        audio_stream = audio_stream_from_websocket(websocket)  # Returns async generator
        logger.info("Audio stream created successfully")

        # Send streaming started status
        await websocket.send_json(
            create_status_response("streaming").model_dump()
        )
        logger.info("Streaming status sent to client")

        # Process audio stream and send transcription results
        try:
            logger.info("Starting to process audio stream")
            async for transcript_result in transcribe_service.process_audio_stream(
                audio_stream=audio_stream,
                config=config,
                user_id=user_id
            ):
                # Send transcription result to client
                if transcript_result.is_partial:
                    response = create_partial_response(transcript_result)
                    logger.debug(f"Sending partial result: {transcript_result.transcript[:50]}...")
                else:
                    response = create_final_response(transcript_result)
                    logger.info(f"Sending final result: {transcript_result.transcript[:100]}...")

                await websocket.send_json(response.model_dump())

                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0)

        except ValueError as e:
            # Configuration error
            logger.error(f"Transcription configuration error: {e}")
            await websocket.send_json(
                create_error_response(f"Configuration error: {str(e)}").model_dump()
            )

        except ConnectionError as e:
            # AWS connection error
            logger.error(f"AWS Transcribe connection error: {e}")
            await websocket.send_json(
                create_error_response("Transcription service temporarily unavailable").model_dump()
            )

        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error during transcription: {e}", exc_info=True)
            await websocket.send_json(
                create_error_response("Internal transcription error").model_dump()
            )

    except WebSocketDisconnect:
        print(f"DEBUG: WebSocket disconnected: user_id={user_id}")
        logger.info(f"WebSocket disconnected: user_id={user_id}")

    except Exception as e:
        print(f"DEBUG: Unexpected WebSocket error - {type(e).__name__}: {e}")
        logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json(
                create_error_response("Internal server error").model_dump()
            )
        except:
            pass  # Connection may already be closed

    finally:
        # ALWAYS cleanup resources when WebSocket closes
        if transcribe_service:
            try:
                logger.info("Waiting for AWS to finish processing remaining audio...")
                session_summary = await transcribe_service.close_session()
                logger.info(
                    f"Transcription session closed: "
                    f"user_id={user_id}, "
                    f"duration={session_summary.get('duration_seconds', 0)}s, "
                    f"words={session_summary.get('total_words', 0)}, "
                    f"confidence={session_summary.get('average_confidence', 0):.3f}"
                )
                print(f"DEBUG: Session summary: {session_summary}")

                # Send session complete status with summary
                try:
                    logger.info(f"Sending final summary to client with {session_summary.get('total_words', 0)} words")
                    await websocket.send_json({
                        "type": "complete",
                        "summary": session_summary
                    })
                    print(f"DEBUG: Final summary sent to client")
                    logger.info(f"Final summary sent to client")
                except Exception as send_error:
                    logger.error(f"Error sending final summary to client: {send_error}")
                    pass  # Connection may be closed

            except Exception as e:
                logger.error(f"Error closing transcription session: {e}")

        # Close WebSocket if still open
        try:
            await websocket.close()
        except:
            pass  # Already closed

        logger.info(f"WebSocket connection fully closed: user_id={user_id}")


# =============================================
# Health Check Endpoint (Optional)
# =============================================

@router.get("/health")
async def transcribe_health_check():
    """
    Health check endpoint for transcription service.

    Returns:
        Status of transcription service configuration
    """
    try:
        # Verify configuration is loaded
        config_check = {
            "status": "healthy",
            "service": "transcription",
            "config": {
                "language_code": settings.transcribe_language_code,
                "sample_rate": settings.transcribe_sample_rate,
                "media_encoding": settings.transcribe_media_encoding,
                "aws_region": settings.aws_region
            }
        }

        return config_check

    except Exception as e:
        logger.error(f"Transcription health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Transcription service unhealthy: {str(e)}"
        )
