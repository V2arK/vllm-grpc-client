#!/usr/bin/env python3
"""
Helper utilities for decoding token IDs to text.
"""

from typing import List, Optional


class TokenDecoder:
    """
    Helper class to decode token IDs to text.
    
    Usage:
        from vllm_grpc_client import VLLMGrpcClient
        from vllm_grpc_client.utils import TokenDecoder
        
        client = VLLMGrpcClient(host="localhost", port=9000)
        decoder = TokenDecoder.from_client(client)
        
        # Decode token IDs
        text = decoder.decode([320, 873, 362, 13, 19846])
        print(text)
        
        # Access the underlying tokenizer
        tokenizer = decoder.tokenizer
    """

    def __init__(self, tokenizer):
        """
        Initialize with a tokenizer.
        
        Args:
            tokenizer: A transformers tokenizer instance.
        """
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        """
        Access the underlying HuggingFace tokenizer.
        
        Returns:
            The transformers tokenizer instance.
        """
        return self._tokenizer

    @classmethod
    def from_model_path(cls, model_path: str) -> "TokenDecoder":
        """
        Create a TokenDecoder from a model path.
        
        Args:
            model_path: HuggingFace model path or local path.
        
        Returns:
            TokenDecoder instance.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for text decoding. "
                "Install with: pip install transformers"
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(tokenizer)

    @classmethod
    def from_client(cls, client) -> "TokenDecoder":
        """
        Create a TokenDecoder from a VLLMGrpcClient.
        
        Args:
            client: VLLMGrpcClient or AsyncVLLMGrpcClient instance.
        
        Returns:
            TokenDecoder instance.
        """
        model_info = client.models.retrieve()
        return cls.from_model_path(model_info.model_path)

    @classmethod
    async def afrom_client(cls, client) -> "TokenDecoder":
        """
        Create a TokenDecoder from an AsyncVLLMGrpcClient.
        
        Args:
            client: AsyncVLLMGrpcClient instance.
        
        Returns:
            TokenDecoder instance.
        """
        model_info = await client.models.retrieve()
        return cls.from_model_path(model_info.model_path)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode.
        
        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.
        
        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_completion(self, completion, skip_special_tokens: bool = True) -> str:
        """
        Decode a Completion object to text.
        
        Args:
            completion: Completion object from client.completions.create().
            skip_special_tokens: Whether to skip special tokens.
        
        Returns:
            Decoded text.
        """
        if not completion.choices:
            return ""
        return self.decode(completion.choices[0].token_ids, skip_special_tokens)

    def decode_chunk(self, chunk, skip_special_tokens: bool = True) -> str:
        """
        Decode a CompletionChunk object to text.
        
        Args:
            chunk: CompletionChunk object from streaming.
            skip_special_tokens: Whether to skip special tokens.
        
        Returns:
            Decoded delta text.
        """
        if not chunk.choices:
            return ""
        return self.decode(chunk.choices[0].delta_token_ids, skip_special_tokens)
