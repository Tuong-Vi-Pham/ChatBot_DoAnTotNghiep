from openai import OpenAI

class LLMClient:
    """
    Client interface for interacting with the local LM Studio instance.
    Implements standard OpenAI compatible chat completions.
    """
    def __init__(self, base_url: str = "http://localhost:1234/v1", default_model: str = "phi-3.5-mini-instruct"):
        self.base_url = base_url
        self.default_model = default_model
        # Initialize OpenAI client pointing to LM Studio local port
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="lm-studio"  # LM Studio does not require a key but the SDK requires a string
        )

    def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model_name: str = None, 
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> str:
        """
        Sends completion request to local LLM.
        """
        model = model_name if model_name is not None else self.default_model
        
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
            return ""
            
        except Exception as e:
            print(f"[LLM Client Error] Local model inference failed: {e}")
            raise ConnectionError(
                f"Could not connect to LM Studio at {self.base_url}. "
                f"Please make sure LM Studio is running and model '{model}' is loaded. Error: {e}"
            )

    def health_check(self) -> bool:
        """
        Checks if the LM Studio endpoint is active and responds to model queries.
        """
        try:
            # Query models endpoint via OpenAI client
            self.client.models.list()
            return True
        except Exception:
            return False

    def simple_test(self, prompt: str = "Say hello.") -> dict:
        """
        Runs a simple prompt to verify connectivity and measures response latency.
        """
        import time
        
        system_prompt = "You are a helpful assistant."
        start_time = time.time()
        
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.2
            )
            latency = time.time() - start_time
            return {
                "success": True,
                "response": response,
                "latency_seconds": latency
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency_seconds": time.time() - start_time
            }
