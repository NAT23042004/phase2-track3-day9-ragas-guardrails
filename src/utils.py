import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LLM_PROVIDER, DEFAULT_LLM, FALLBACK_LLM, GCP_PROJECT_ID, GCP_LOCATION,
                    EMBEDDING_PROVIDER, GCP_EMBEDDING_MODEL, 
                    EMBEDDING_DIM, EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_local_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
    )

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0, model_name: str = None) -> str:
    """
    Hàm gọi LLM chung hỗ trợ cả GCP Vertex AI và OpenAI.
    
    Args:
        system_prompt: Nội dung system prompt.
        user_prompt: Nội dung user prompt.
        temperature: Độ sáng tạo của model (mặc định 0.0).
        
    Returns:
        Câu trả lời từ LLM hoặc chuỗi rỗng nếu có lỗi.
    """
    try:
        if LLM_PROVIDER == "google":
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            # Khởi tạo Vertex AI nếu project_id được cấu hình
            if GCP_PROJECT_ID:
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            
            def generate_gemini(model_name):
                model = GenerativeModel(model_name)
                # Vertex AI không có role system riêng biệt theo cách OpenAI làm trong SDK đơn giản, 
                # gộp chung vào prompt hoặc dùng system_instruction nếu dùng SDK bản mới.
                # Ở đây gộp chung để đảm bảo tính tương thích cao.
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                response = model.generate_content(full_prompt)
                return response.text

            target_model = model_name or DEFAULT_LLM
            try:
                return generate_gemini(target_model)
            except Exception as e:
                if model_name: # Nếu đã chỉ định model mà lỗi thì throw luôn hoặc handle riêng
                    print(f"Gemini {model_name} Error: {e}")
                    return ""
                print(f"Gemini Default Error: {e}, falling back to {FALLBACK_LLM}")
                return generate_gemini(FALLBACK_LLM)
        else:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model_name or DEFAULT_LLM,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return resp.choices[0].message.content
    except Exception as e:
        print(f"LLM Error in utils.call_llm: {e}")
        return ""

def get_embeddings(texts: list[str], task: str = None) -> list[list[float]]:
    """
    Lấy embeddings hỗ trợ cả GCP Vertex AI (Native) và Local.
    
    Args:
        texts: Danh sách chuỗi văn bản cần embed.
        task: Không dùng cho native (SDK tự xử lý).
        
    Returns:
        Danh sách các vector (list of lists of floats).
    """
    try:
        if EMBEDDING_PROVIDER == "google":
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            
            if GCP_PROJECT_ID:
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            
            model = TextEmbeddingModel.from_pretrained(GCP_EMBEDDING_MODEL)
            
            # Giới hạn của Vertex AI là 250 instances, nhưng tổng token không được quá 20,000.
            # Dùng batch 50 để an toàn hơn.
            all_embeddings = []
            for i in range(0, len(texts), 50):
                batch = texts[i:i+50]
                results = model.get_embeddings(batch)
                all_embeddings.extend([r.values for r in results])
            return all_embeddings
        else:
            # Chạy local dùng sentence-transformers
            model = _get_local_embedding_model()
            return model.encode(texts).tolist()
    except Exception as e:
        print(f"Embedding Error in utils.get_embeddings: {e}")
        # Trả về list vector 0 nếu lỗi để tránh crash pipeline (tùy chọn)
        return [[0.0] * EMBEDDING_DIM] * len(texts)
