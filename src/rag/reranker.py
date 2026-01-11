"""
Reranker using Qwen 3 model for cross-encoder reranking
"""
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
from rag.config import QWEN_RERANKER_MODEL, DEVICE, TOP_N_RERANK


class QwenReranker:
    """
    Reranker using Qwen 3 model (CausalLM-based reranker)
    
    Note: MPS (Apple Silicon) is NOT supported for Qwen reranker due to 
    large vocabulary size exceeding MPS tensor dimension limits.
    Falls back to CPU with optimizations.
    """
    
    def __init__(self, model_name: str = QWEN_RERANKER_MODEL, device: str = DEVICE):
        """
        Initialize Qwen reranker.
        
        Args:
            model_name: Model name/path
            device: Device to use. Note: MPS is not supported for large vocab models.
        """
        self.model_name = model_name
        self.max_length = 8192
        
        # MPS doesn't work with Qwen's large vocabulary (~150k tokens)
        # Error: "MPSGraph does not support tensor dims larger than INT_MAX"
        # Force CPU for Qwen reranker, use CUDA only if available
        if device == "mps":
            print("Reranker: MPS not supported for Qwen (vocab too large), using CPU")
            device = "cpu"
        elif device is None or device == "":
            if torch.cuda.is_available():
                device = "cuda"
                print("Reranker: Using CUDA GPU")
            else:
                device = "cpu"
                print("Reranker: Using CPU")
        else:
            print(f"Reranker: Using device: {device}")
        
        self.device = device
        
        print(f"Loading Qwen Reranker: {model_name} on {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            padding_side='left'
        )
        
        # Use float16 for CUDA, float32 for CPU
        # Note: float16 on CPU can be slow, but float32 uses more memory
        if device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True  # Optimize memory for CPU
        )
        self.model.to(device)
        self.model.eval()
        
        print(f"Reranker ready (dtype={torch_dtype}, device={device})")
        
        # Setup specific tokens for Qwen3-Reranker
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Setup prompt templates
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.default_instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = self.default_instruction
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )

    def _process_inputs(self, pairs: List[str]):
        """
        Process text pairs into model inputs with manual padding logic
        """
        # Tokenize without padding first to handle prefix/suffix insertion
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # Add prefix and suffix tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        # Pad manually
        padded_inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        # Move to device
        for key in padded_inputs:
            padded_inputs[key] = padded_inputs[key].to(self.device)
            
        return padded_inputs

    def _compute_scores(self, inputs) -> List[float]:
        """
        Compute relevance scores from model logits
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take logits of the last token
            batch_scores = outputs.logits[:, -1, :]
            
            # Extract scores for "yes" and "no" tokens
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            # Stack and normalize
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            
            # Return probability of "yes" class
            scores = batch_scores[:, 1].exp().tolist()
            return scores

    def rerank(self, query: str, documents: List[Document], top_n: int = TOP_N_RERANK) -> List[Document]:
        """
        Rerank documents based on query relevance
        """
        if not documents:
            return []
        
        # Format all pairs
        pairs = [
            self._format_instruction(self.default_instruction, query, doc.page_content) 
            for doc in documents
        ]
        
        # Process inputs in batches to avoid OOM
        batch_size = 4  # Conservative batch size
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_scores(inputs)
            all_scores.extend(batch_scores)
        
        # Pair docs with scores
        scored_docs = list(zip(documents, all_scores))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Assign scores to metadata and return top_n
        reranked_docs = []
        for doc, score in scored_docs[:top_n]:
            doc.metadata['score'] = score
            reranked_docs.append(doc)
            
        return reranked_docs
    
    def rerank_with_scores(self, query: str, documents: List[Document], top_n: int = TOP_N_RERANK) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores
        """
        if not documents:
            return []
            
        pairs = [
            self._format_instruction(self.default_instruction, query, doc.page_content) 
            for doc in documents
        ]
        
        batch_size = 4
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_scores(inputs)
            all_scores.extend(batch_scores)
            
        scored_docs = list(zip(documents, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_n]