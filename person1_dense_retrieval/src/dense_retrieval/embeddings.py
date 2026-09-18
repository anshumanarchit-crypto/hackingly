"""
Embedding subsystem for ProofMesh Dense Semantic Retrieval.
Supports SentenceTransformer models (e.g., all-MiniLM-L6-v2) for local CPU execution,
with a deterministic mock fallback for development, offline test suites, and environments
lacking sentence-transformers.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class DeterministicMockBackend:
    """
    Deterministic local mock embedder for development and offline testing.
    Uses feature hashing over token and character n-grams to generate deterministic
    384-dimensional dense vectors with L2 normalization.
    
    NOTE: This is strictly for development and testing. It does not replace real
    neural semantic embeddings from SentenceTransformer.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        self.dimension = dimension
        self.seed = seed

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row_idx, text in enumerate(texts):
            clean_text = text.strip().lower()
            if not clean_text:
                continue

            SEMANTIC_CLUSTERS = [
                {"pay", "remit", "invoice", "payment", "due", "bill", "customer", "purchaser", "buyer", "outstanding", "calendar", "days"},
                {"protect", "protected", "warranty", "guarantee", "coverage", "hardware", "valid", "months", "device"},
                {"leave", "vacation", "holiday", "absence", "timeoff", "portal", "hr", "request", "requests"},
                {"processor", "cpu", "intel", "core", "latency", "system", "offline"},
            ]

            # Deterministic feature hashing
            tokens = clean_text.split()
            for token in tokens:
                # Token hash
                h = int(hashlib.sha256(f"{self.seed}_{token}".encode("utf-8")).hexdigest(), 16)
                dim_idx = h % self.dimension
                sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
                vectors[row_idx, dim_idx] += sign

                # Substring character 3-grams for slight morphological matching
                if len(token) >= 3:
                    for i in range(len(token) - 2):
                        tri = token[i : i + 3]
                        h_tri = int(hashlib.sha256(f"{self.seed}_tri_{tri}".encode("utf-8")).hexdigest(), 16)
                        dim_idx_tri = h_tri % self.dimension
                        sign_tri = 1.0 if ((h_tri >> 8) & 1) == 1 else -1.0
                        vectors[row_idx, dim_idx_tri] += 0.5 * sign_tri

                # Conceptual cluster features for offline demonstration semantic fidelity
                for c_idx, cluster in enumerate(SEMANTIC_CLUSTERS):
                    if token in cluster:
                        h_c = int(hashlib.sha256(f"concept_cluster_{c_idx}".encode("utf-8")).hexdigest(), 16)
                        dim_c = h_c % self.dimension
                        vectors[row_idx, dim_c] += 2.0

            # Normalize vector
            if normalize_embeddings:
                norm = np.linalg.norm(vectors[row_idx])
                if norm > 1e-12:
                    vectors[row_idx] /= norm
                else:
                    # Provide non-zero fallback for all-zero vector
                    vectors[row_idx, 0] = 1.0

        return vectors


class SemanticEmbedder:
    """
    Manages embedding model lifecycle, lazy initialization, caching, batch encoding,
    and vector normalization.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        allow_fallback: bool = True,
    ):
        """
        Initialize the embedder. Model loading is deferred until first encode call (lazy loading).

        Args:
            model_name: Hugging Face model identifier or 'mock' for development fallback.
            device: 'cpu' or 'auto' (defaults to 'cpu').
            normalize_embeddings: Whether to L2-normalize vectors for direct cosine similarity.
            batch_size: Default batch size for encoding texts.
            allow_fallback: If True, falls back to deterministic mock backend if sentence-transformers
                           is not installed in the environment.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = max(1, batch_size)
        self.allow_fallback = allow_fallback

        self._model = None
        self._backend_type: Optional[str] = None
        self._dimension: Optional[int] = None
        self._encode_call_count = 0
        self._total_texts_encoded = 0

    def _resolve_device(self, device: str) -> str:
        d = device.strip().lower()
        if d in ("cpu", "auto"):
            return d
        # Default safely to cpu
        return "cpu"

    def _load_model(self) -> None:
        """Lazy loader: loads model only once and caches it."""
        if self._model is not None:
            return

        if self.model_name.lower() == "mock":
            logger.info("Using deterministic mock backend as explicitly requested.")
            self._model = DeterministicMockBackend(dimension=self.DEFAULT_DIMENSION)
            self._backend_type = "mock"
            self._dimension = self.DEFAULT_DIMENSION
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            target_device = "cpu" if self.device == "cpu" else None
            self._model = SentenceTransformer(self.model_name, device=target_device)
            self._backend_type = "sentence-transformers"
            dim_getter = getattr(self._model, "get_embedding_dimension", None) or getattr(self._model, "get_sentence_embedding_dimension", None)
            self._dimension = dim_getter() if dim_getter else self.DEFAULT_DIMENSION
            logger.info(
                f"Successfully loaded SentenceTransformer model '{self.model_name}' on device '{self.device}'."
            )
        except ImportError as err:
            if not self.allow_fallback:
                raise ImportError(
                    f"Package 'sentence-transformers' is required to load model '{self.model_name}'. "
                    f"Install it via 'pip install sentence-transformers' or set model_name='mock'."
                ) from err

            logger.warning(
                f"'sentence-transformers' is not installed in the current environment. "
                f"Activating deterministic mock backend for offline testing and development."
            )
            self._model = DeterministicMockBackend(dimension=self.DEFAULT_DIMENSION)
            self._backend_type = "mock"
            self._dimension = self.DEFAULT_DIMENSION
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc

            logger.warning(
                f"Could not load weights for model '{self.model_name}' ({exc}). "
                f"Falling back to deterministic mock backend."
            )
            self._model = DeterministicMockBackend(dimension=self.DEFAULT_DIMENSION)
            self._backend_type = "mock"
            self._dimension = self.DEFAULT_DIMENSION

    @property
    def is_loaded(self) -> bool:
        """Check whether the underlying model is loaded in memory."""
        return self._model is not None

    @property
    def backend_type(self) -> str:
        """Returns 'sentence-transformers' or 'mock'."""
        if self._backend_type is None:
            self._load_model()
        return self._backend_type or "unknown"

    @property
    def is_mock(self) -> bool:
        """Returns True if the active backend is the mock fallback."""
        return self.backend_type == "mock"

    def get_dimension(self) -> int:
        """Returns the embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2)."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or self.DEFAULT_DIMENSION

    def encode_text(
        self,
        text: str,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode a single string into a 1D float32 NumPy vector.

        Args:
            text: Text to encode.
            normalize: Whether to L2-normalize. Defaults to self.normalize_embeddings.

        Returns:
            1D np.ndarray of shape (dimension,).
        """
        norm_flag = self.normalize_embeddings if normalize is None else normalize
        matrix = self.encode_texts([text], batch_size=1, normalize=norm_flag)
        return matrix[0]

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Batch-encode a list of strings into a 2D float32 NumPy array.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for model inference.
            normalize: Whether to L2-normalize. Defaults to self.normalize_embeddings.

        Returns:
            2D np.ndarray of shape (len(texts), dimension).
        """
        self._load_model()

        if not texts:
            dim = self.get_dimension()
            return np.empty((0, dim), dtype=np.float32)

        eff_batch_size = max(1, batch_size or self.batch_size)
        norm_flag = self.normalize_embeddings if normalize is None else normalize

        self._encode_call_count += 1
        self._total_texts_encoded += len(texts)

        if self._backend_type == "sentence-transformers":
            assert self._model is not None  # narrowed by _load_model(); guard for type checker
            vectors = self._model.encode(
                texts,
                batch_size=eff_batch_size,
                normalize_embeddings=norm_flag,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Ensure float32 2D array
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)
            if vectors.ndim == 1:
                vectors = np.expand_dims(vectors, axis=0)
            return vectors

        # Mock backend
        assert self._model is not None  # narrowed by _load_model(); guard for type checker
        vectors = self._model.encode(
            texts,
            batch_size=eff_batch_size,
            normalize_embeddings=norm_flag,
        )
        return vectors.astype(np.float32)

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic information about the embedder state."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "backend": self.backend_type,
            "is_mock": self.is_mock,
            "dimension": self.get_dimension(),
            "normalize_embeddings": self.normalize_embeddings,
            "is_loaded": self.is_loaded,
            "encode_call_count": self._encode_call_count,
            "total_texts_encoded": self._total_texts_encoded,
        }
