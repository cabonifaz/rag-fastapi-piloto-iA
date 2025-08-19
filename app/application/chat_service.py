# app/application/chat_service.py

from typing import Tuple, List, Dict, Any


class ChatService:
    """
    Servicio de aplicación que orquesta el flujo RAG:
    1. Guarda el mensaje del usuario en la memoria de chat.
    2. Genera embedding del mensaje con el proveedor de embeddings.
    3. Busca contexto en la base vectorial.
    4. Llama al LLM con el historial + contexto.
    5. Guarda la respuesta en la memoria.
    6. Devuelve la respuesta + historial actualizado.
    """

    def __init__(self, memory_repository, vector_repository, embeddings_provider, llm_provider):
        self.memory_repository = memory_repository
        self.vector_repository = vector_repository
        self.embeddings_provider = embeddings_provider
        self.llm_provider = llm_provider

    async def handle_message(self, session_id: str, user_message: str) -> Tuple[str, List[Dict[str, Any]]]:
        # 1. Guardar mensaje del usuario en memoria
        self.memory_repository.save_message(session_id, role="user", content=user_message)

        # 2. Obtener embedding del mensaje
        query_vector = await self.embeddings_provider.embed(user_message)

        # 3. Buscar contexto en VectorDB (Weaviate, etc.)
        results = await self.vector_repository.search_by_vector(
            class_name="Documents",  # ⚠️ este nombre deberías inyectarlo/configurarlo
            vector=query_vector,
            top_k=5,
            return_properties=["text", "source"],
        )

        context_chunks = [r["properties"].get("text", "") for r in results]

        # 4. Construir prompt con historial + contexto
        history = self.memory_repository.get_history(session_id)
        formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        context_text = "\n".join(context_chunks)

        prompt = f"""
        You are an assistant. Use the following context to answer the question.
        
        Context:
        {context_text}

        Conversation so far:
        {formatted_history}

        User: {user_message}
        Assistant:
        """

        # 5. Llamar al LLM
        answer = await self.llm_provider.generate(prompt)

        # 6. Guardar respuesta en memoria
        self.memory_repository.save_message(session_id, role="assistant", content=answer)

        # 7. Devolver respuesta + historial
        history = self.memory_repository.get_history(session_id)
        return answer, history
