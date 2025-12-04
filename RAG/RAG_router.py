import os
import sys
import json
import math
import gc
import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
import pickle

try:
    import faiss  # pip install faiss-cpu  or faiss-gpu
except Exception:
    faiss = None

# --------------------
# --- CONFIG CONSTANTS ---

# Chat history (global variable to maintain conversation state)
chat_history = []

# Knowledge base directories (use your own paths)
KNOWLEDGE_DIR = "path to your TXT"
SYNTHESIS_KNOWLEDGE_DIR = "path to your JSON"

# FAISS index save paths
FAISS_INDEX_PATH = "path to faiss_index"
SYNTHESIS_FAISS_INDEX_PATH = "path to faiss_index_synthesis"

# In-memory vector store cache (avoid reloading)
vector_store_cache = None
synthesis_vector_store_cache = None

# --- Two system prompts (roles) ---

ROLE1_SYSTEM_PROMPT = """
You are a senior expert, mentor, and research partner in the field of surface science and interface engineering. Combining profound theoretical insights with extensive experimental strategies, you specialize in surface chemical synthesis, catalysis, and the mechanistic analysis of interfacial reactions.

Core Expertise (Emphasis on Chemical Reactions and Mechanisms):
In-depth understanding of the fundamental drivers of surface processes, including the energetics of physical and chemical adsorption, the decisive role of surface free energy in thin film growth, self-assembly, and corrosion, and quantification of adsorption/desorption kinetics through methods such as temperature-programmed desorption (TPD).
A deep understanding of how unique surface geometry and electronic structure (e.g., through d-band center theory) stabilize key reaction intermediates, reshape reaction potential energy surfaces, lower energy barriers, and ultimately guide precise chemical bond breaking and formation, enabling efficient catalysis.
Proficient in the principles and applicable boundaries of the three major excitation modes: thermodynamics, optics (photocatalysis/photolysis/SPR/CT), and scanning probes.
Adept at strategically combining techniques such as IRAS, Raman spectroscopy, TPD, and reaction chamber-coupled mass spectrometry to construct multi-dimensional evidence chains for reaction pathways and intermediates, tailored to surface chemical reaction problems.

Information Processing Rules:
When answering questions involving chemical reactions, synthetic methods, and reaction conditions, you must prioritize and rely on the information in the context.
If the context does not provide sufficient information, you can supplement your general knowledge of organic chemistry, but always ensure that the context is the primary basis for your answer.

Guidance:
Propose innovative, multi-pathway experimental solutions for the user's synthesis/catalysis/reaction goals.
Guide the user to deeply consider the physicochemical principles behind the experimental results: What is the driving force of the reaction? What are the possible transition states? What catalytic or template role does the surface play? How does selectivity arise?
Design key next steps to confirm or disprove the scientific hypothesis.

Tone and Style: Be rigorous, insightful, and full of exploratory passion. Be provocative, and use Socratic questioning to focus on the core scientific questions of chemical transformations and mechanisms.
"""

ROLE2_SYSTEM_PROMPT = """
You are a senior expert, mentor, and research partner in the field of surface science and interface engineering. Combining profound theoretical insights with extensive experimental strategies, you specialize in the precise characterization of surface structure and electronic states and in-depth data interpretation.

Core Expertise:
A deep understanding of surface atomic relaxation and reconstruction phenomena, as well as the critical role of surface free energy in determining surface stability and spontaneous processes. Understanding the influence of basic adsorption states (e.g., atomic adsorption, molecular adsorption, and pre-adsorption) and geometric sites (e.g., bridge sites, top sites, and vacancies) on adsorbates.
Precision Surface Structure Measurement: Mastering LEED/RHEED techniques to reveal long-range ordered reconstructions; and STM/AFM techniques to observe relaxation, defects, and superstructure at the atomic scale.
Advanced Electron Spectroscopy Interpretation: Mastering the principles and detailed interpretation of XPS, UPS, and AES. Accurately interpreting chemical states, band bending, work function, and surface states from peak positions, peak shapes, and intensity changes. Comprehensive Characterization Strategy: Expertise in strategically combining SIMS, NEXAFS/XANES, and other techniques to construct multi-dimensional evidence chains and reveal the elemental distribution and local structure of interfaces.

Information Processing Rules:
When answering questions involving surface science data interpretation and principle analysis, strictly prioritize and rely on the experimental data, observations, or proposed scientific principles in the context.
If the context does not provide sufficient information, you can supplement your general surface science knowledge, but always ensure that the context is the foundation of your interpretation and analysis.

Guidance:
Guide users in selecting the most appropriate characterization technique to answer specific structural or electronic state questions.
When users provide experimental data (such as XPS spectra, STM images, and LEED spectra), you should not only describe the surface appearance but also provide in-depth explanations of the underlying physical and chemical principles, highlighting the scientific implications of key features.
Guide users in extracting insightful scientific hypotheses from preliminary data, focusing on core scientific questions regarding surface geometry, electronic states, and physical properties.

Tone and style: Rigorous, profound, and full of passion for exploration, inspiring, and good at focusing on core scientific issues of structure and electronic state through Socratic questioning.
"""

# ----------------------------------------------------
# 1. Initialize model and tools
# ----------------------------------------------------

# (1) LLM: OpenAI API
OPENAI_KEY = "xxx"
cached_llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    temperature=0.7,  # relatively low temperature for stability
    openai_api_key=OPENAI_KEY,
)

# (2) Router classifier LLM
router_llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    temperature=0.1,  # low temperature for stable routing/classification
    openai_api_key=OPENAI_KEY,
)

# (3) Embedding model: local all-MiniLM-L6-v2
EMBEDDING_MODEL_PATH = "Path to all-MiniLM-L6-v2"
try:
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"✅ Successfully loaded local embedding model: {EMBEDDING_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error: failed to load local embedding model, please check path and dependencies. Error: {e}")
    embedding = None

# (4) Reranker model: local bge-reranker-large
RERANKER_MODEL_PATH = "H:/models/bge-reranker-large"
reranker = None
try:
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_PATH,
        model_kwargs={"device": "cpu"},
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    print(f"✅ Successfully loaded local reranker model: {RERANKER_MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Warning: failed to load local reranker model, system will fall back to base retrieval without reranking. Error: {e}")

# (5) Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096, chunk_overlap=1024, length_function=len, is_separator_regex=False
)


def _iter_batches(seq, batch_size):
    """Yield sequence in batches of size batch_size."""
    for i in range(0, len(seq), batch_size):
        yield i, seq[i : i + batch_size]


def _build_faiss_in_batches(
    chunks,
    embedding,
    index_path,
    batch_size=1000,
    checkpoint_every=None,  # optional: write FAISS native index-only checkpoints every N batches
    write_faiss_index_fn=None,  # function to write only faiss.index (no docstore)
):
    """
    Build/extend FAISS index in batches:
    - First batch: initialize with from_documents
    - Subsequent batches: incrementally add with add_documents
    - Do NOT save the whole vector store in the middle (avoid repeatedly pickling large docstores)
    - If checkpoint_every is set, you may optionally write only the FAISS index (no docstore),
      and do the full docstore saving at the end.
    """
    total = len(chunks)
    if total == 0:
        raise ValueError("No document chunks available for indexing.")

    num_batches = math.ceil(total / batch_size)
    vector_store = None
    t0 = time.time()

    print("Starting FAISS vector store creation (batched)...")
    print(f"Total chunks: {total}, batch size: {batch_size}, number of batches: {num_batches}")

    for b_idx, batch_docs in _iter_batches(chunks, batch_size):
        batch_id = b_idx // batch_size + 1
        b_start = b_idx
        b_end = b_idx + len(batch_docs) - 1

        t_batch0 = time.time()
        if vector_store is None:
            print(f"[{batch_id}/{num_batches}] Initializing index: chunk range {b_start}–{b_end} ...")
            vector_store = FAISS.from_documents(documents=batch_docs, embedding=embedding)
        else:
            print(f"[{batch_id}/{num_batches}] Adding batch: chunk range {b_start}–{b_end} ...")
            vector_store.add_documents(batch_docs)

        print(
            f"[{batch_id}/{num_batches}] ✅ Batch complete, processed {b_end + 1}/{total} chunks, "
            f"time used {time.time() - t_batch0:.2f}s"
        )

        # Optional: write FAISS index-only checkpoint
        if checkpoint_every and write_faiss_index_fn and (batch_id % checkpoint_every == 0):
            try:
                write_faiss_index_fn(vector_store, index_path)
                print(f"[{batch_id}/{num_batches}] 🧩 Wrote FAISS native index checkpoint")
            except Exception as _e:
                print(f"[{batch_id}/{num_batches}] ⚠️ Failed to write checkpoint: {_e}")

        del batch_docs
        gc.collect()

    print(f"✅ All batches completed! Total time {time.time() - t0:.2f}s (docstore not yet saved)")
    return vector_store


def _json_to_text(json_data, parent_key: str = "", sep: str = ".") -> str:
    """
    Convert arbitrary JSON recursively into a flattened text representation
    similar to JSONChunkLoaderWithEmbedding:
    - dict: k:v recursively concatenated
    - list: indexed as item_i: ...
    - scalars: str(value)
    """
    if isinstance(json_data, dict):
        parts = []
        for k, v in json_data.items():
            key_path = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, (dict, list)):
                parts.append(f"{key_path}: {_json_to_text(v, key_path, sep)}")
            else:
                parts.append(f"{key_path}: {v}")
        return "; ".join(parts)
    elif isinstance(json_data, list):
        parts = []
        for i, item in enumerate(json_data):
            key_path = f"{parent_key}{sep}item_{i}" if parent_key else f"item_{i}"
            if isinstance(item, (dict, list)):
                parts.append(f"{key_path}: {_json_to_text(item, key_path, sep)}")
            else:
                parts.append(f"{key_path}: {item}")
        return "; ".join(parts)
    else:
        return str(json_data)


def process_json_file(file_path, root_dir: str = None) -> list[Document]:
    """
    Convert a single JSON file into one or multiple Documents (file → entries → split later).
    Returns: List[Document]
    """
    docs: list[Document] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        relative_path = os.path.relpath(file_path, root_dir) if root_dir else os.path.basename(file_path)

        # If top-level is a list: one Document per item
        if isinstance(data, list):
            for idx, item in enumerate(data):
                text_content = _json_to_text(item)
                base_doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": file_path,
                        "relative_path": relative_path,
                        "file_name": os.path.basename(file_path),
                        "item_index": idx,
                        "type": "synthesis",
                    },
                )
                docs.append(base_doc)
        else:
            # Top-level dict or scalar: one Document for entire file
            text_content = _json_to_text(data)
            base_doc = Document(
                page_content=text_content,
                metadata={
                    "source": file_path,
                    "relative_path": relative_path,
                    "file_name": os.path.basename(file_path),
                    "type": "synthesis",
                },
            )
            docs.append(base_doc)

    except Exception as e:
        # On error, still create a Document that records the failure for traceability
        err_doc = Document(
            page_content=f"Error processing file: {file_path}. Error: {e}",
            metadata={
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "type": "synthesis",
                "error": str(e),
            },
        )
        docs.append(err_doc)
    return docs


def load_json_documents(directory: str, recursive: bool = True) -> list[Document]:
    """
    Traverse directory (recursively or not) to read all .json files.
    Each file is converted to one or multiple Documents, then splitting will be done at index time.
    """
    pattern = "**/*.json" if recursive else "*.json"
    json_files = []
    for root, _, files in os.walk(directory):
        if not recursive and os.path.abspath(root) != os.path.abspath(directory):
            continue
        for fn in files:
            if fn.lower().endswith(".json"):
                json_files.append(os.path.join(root, fn))

    print(f"Found {len(json_files)} JSON files, starting processing...")
    all_docs: list[Document] = []
    for i, json_file in enumerate(json_files):
        if i % 100 == 0:
            print(f"Processed {i}/{len(json_files)} files...")
        all_docs.extend(process_json_file(json_file, root_dir=directory))

    print(f"JSON base document construction completed, total: {len(all_docs)}")
    return all_docs


# ----------------------------
# Index status checks (standard vs sharded)
# ----------------------------
def _index_paths(store_type: str):
    base = SYNTHESIS_FAISS_INDEX_PATH if store_type == "synthesis" else FAISS_INDEX_PATH
    sharded_dir = base + "_sharded"
    return {
        "base": base,
        "standard_faiss": f"{base}.faiss",
        "standard_pkl": f"{base}.pkl",
        "sharded_dir": sharded_dir,
        "sharded_index": os.path.join(sharded_dir, "index.faiss"),
        "sharded_manifest": os.path.join(sharded_dir, "manifest.json"),
        "sharded_idmap": os.path.join(sharded_dir, "index_to_docstore_id.pkl"),
    }


def _has_standard_index(p: dict) -> bool:
    return os.path.exists(p["standard_faiss"]) and os.path.exists(p["standard_pkl"])


def _has_sharded_index(p: dict) -> bool:
    if not (
        os.path.isdir(p["sharded_dir"])
        and os.path.exists(p["sharded_index"])
        and os.path.exists(p["sharded_manifest"])
        and os.path.exists(p["sharded_idmap"])
    ):
        return False
    try:
        shards = [
            fn
            for fn in os.listdir(p["sharded_dir"])
            if fn.startswith("docstore_shard_") and fn.endswith(".pkl")
        ]
        return len(shards) > 0
    except Exception:
        return False


def check_index_exists(store_type: str = "general") -> bool:
    """Check if any index (standard or sharded) exists."""
    p = _index_paths(store_type)
    return _has_standard_index(p) or _has_sharded_index(p)


def get_index_status_detail(store_type: str = "general") -> dict:
    p = _index_paths(store_type)
    status = {"store_type": store_type, "exists": False, "format": None, "path": None, "meta": {}}

    std_ok = _has_standard_index(p)
    shd_ok = _has_sharded_index(p)
    if not (std_ok or shd_ok):
        return status

    if shd_ok:
        status.update({"exists": True, "format": "sharded", "path": p["sharded_dir"]})
        try:
            with open(p["sharded_manifest"], "r", encoding="utf-8") as f:
                status["meta"] = json.load(f)
        except Exception as e:
            status["meta"] = {"error": f"read sharded manifest failed: {e}"}
    else:
        status.update(
            {
                "exists": True,
                "format": "standard",
                "path": os.path.dirname(os.path.abspath(p["standard_faiss"])) or ".",
            }
        )

    # Also merge base meta (containing source_dir, built_at, etc.)
    base_meta = f"{p['base']}.meta.json"
    if os.path.exists(base_meta):
        try:
            with open(base_meta, "r", encoding="utf-8") as f:
                extra = json.load(f)
            status["meta"] = {**status.get("meta", {}), **extra}
        except Exception as e:
            status.setdefault("meta", {})["meta_merge_warning"] = f"merge base meta failed: {e}"

    return status


def get_index_status():
    """Return detailed status for both index types."""
    return {
        "general": get_index_status_detail("general"),
        "synthesis": get_index_status_detail("synthesis"),
    }


# ----------------------------------------------------
# 2. Router classification function
# ----------------------------------------------------
def classify_query(query: str) -> str:
    """
    Use LLM to classify query type.
    Returns: "synthesis" or "general"
    """
    classification_prompt = f"""
    Please classify the following question into one of two types.
    
    Question: "{query}"
    
    Type definitions:
    - synthesis: questions about chemical reaction synthesis routes, retrosynthesis analysis, reaction conditions, catalysts, reaction mechanisms, etc.
    - general: questions about surface science, materials characterization, theoretical calculations, physical properties, and other non-synthesis scientific topics.
    
    Answer with exactly one word: "synthesis" or "general". Do not return any other text.
    """

    try:
        response = router_llm.invoke([HumanMessage(content=classification_prompt)])
        result = response.content.strip().lower()

        if "synthesis" in result:
            return "synthesis"
        else:
            return "general"
    except Exception as e:
        print(f"❌ Router classification failed, falling back to keyword-based routing: {e}")
        # Fallback: simple keyword-based classification
        synthesis_keywords = [
            "synthesis",
            "retrosynthesis",
            "reaction",
            "catalyst",
            "reaction conditions",
            "synthetic route",
        ]
        if any(keyword in query.lower() for keyword in synthesis_keywords):
            return "synthesis"
        else:
            return "general"


# ----------------------------------------------------
# 3. Core functions
# ----------------------------------------------------
def _index_files_exist(index_path: str) -> bool:
    return os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}.pkl")


def ensure_index(store_type: str = "general", force_reindex: bool = False):
    """
    If index exists -> load and return vector store.
    If not -> build index, save, and return.
    If force_reindex=True -> rebuild even if it already exists.
    """
    vs = get_faiss_vector_store(store_type)
    if vs is not None and not force_reindex:
        return vs

    index_path = SYNTHESIS_FAISS_INDEX_PATH if store_type == "synthesis" else FAISS_INDEX_PATH
    exists_on_disk = _index_files_exist(index_path)

    if exists_on_disk and not force_reindex:
        try:
            vs = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
            if store_type == "synthesis":
                globals()["synthesis_vector_store_cache"] = vs
            else:
                globals()["vector_store_cache"] = vs
            print(f"📦 Loaded {store_type} index from disk: {index_path}")
            return vs
        except Exception as e:
            print(f"⚠️ Index exists on disk but failed to load, attempting rebuild. Reason: {e}")

    ok = index_documents(store_type)
    if not ok:
        return None
    return get_faiss_vector_store(store_type)


def _maybe_warn_manifest(index_path: str, store_type: str, knowledge_dir: str):
    manifest_path = f"{index_path}.meta.json"
    if not os.path.exists(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cur_latest = _latest_mtime_in_dir(
            knowledge_dir,
            ".json" if store_type == "synthesis" else ".md",
            True,
        )
        if cur_latest > float(meta.get("source_latest_mtime", 0)):
            print(
                f"💡 Note: the {store_type} source directory has changed since the index was built. "
                f"To save time, the existing index is used. If you want to rebuild, set REBUILD_INDEX=1 "
                f"or call force_reindex manually."
            )
    except Exception:
        pass


def get_faiss_vector_store(store_type="general"):
    """Load FAISS vector store. If not found, return None."""
    global vector_store_cache, synthesis_vector_store_cache

    if embedding is None:
        return None

    if store_type == "synthesis":
        if synthesis_vector_store_cache is not None:
            return synthesis_vector_store_cache

        index_path = SYNTHESIS_FAISS_INDEX_PATH
        if os.path.isdir(index_path):
            try:
                vector_store = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
                synthesis_vector_store_cache = vector_store
                return vector_store
            except Exception as e:
                print(f"❌ Error: failed to load synthesis FAISS index: {e}")
                return None
        else:
            return None
    else:
        if vector_store_cache is not None:
            return vector_store_cache

        index_path = FAISS_INDEX_PATH
        if os.path.isdir(index_path):
            try:
                vector_store = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
                vector_store_cache = vector_store
                return vector_store
            except Exception as e:
                print(f"❌ Error: failed to load general FAISS index: {e}")
                return None
        else:
            return None


def _latest_mtime_in_dir(path: str, pattern: str = ".json", recursive: bool = True) -> float:
    latest = 0.0
    for root, _, files in os.walk(path):
        if not recursive and os.path.abspath(root) != os.path.abspath(path):
            continue
        for fn in files:
            if fn.lower().endswith(pattern):
                m = os.path.getmtime(os.path.join(root, fn))
                latest = max(latest, m)
    return latest


def index_documents(store_type="general"):
    """Batch index documents and save vector store in sharded format."""
    global vector_store_cache, synthesis_vector_store_cache

    print(f"\n--- Start indexing documents for store_type={store_type} ---")

    if embedding is None:
        print("Indexing failed: embedding model not loaded.")
        return False

    if store_type == "synthesis":
        knowledge_dir = SYNTHESIS_KNOWLEDGE_DIR
        index_path = SYNTHESIS_FAISS_INDEX_PATH
    else:
        knowledge_dir = KNOWLEDGE_DIR
        index_path = FAISS_INDEX_PATH

    if not os.path.isdir(knowledge_dir):
        print(f"Indexing failed: knowledge directory '{knowledge_dir}' does not exist.")
        return False

    try:
        if store_type == "synthesis":
            base_docs = load_json_documents(knowledge_dir, recursive=True)
            if not base_docs:
                print(f"Indexing failed: no valid JSON files found in '{knowledge_dir}'.")
                return False
            chunks = []
            for d in base_docs:
                split = text_splitter.split_documents([d])
                for j, c in enumerate(split):
                    c.metadata = {**c.metadata, "chunk_id": j}
                    chunks.append(c)
            print(f"JSON document splitting completed, total chunks: {len(chunks)}")
        else:
            loader = DirectoryLoader(
                knowledge_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            md_loader = DirectoryLoader(
                knowledge_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            docs = loader.load() + md_loader.load()
            if not docs:
                print(f"Indexing failed: no .md or .txt files found in '{knowledge_dir}'.")
                return False
            print(f"Document loading completed, total documents: {len(docs)}")
            chunks = text_splitter.split_documents(docs)
            print(f"Text chunk splitting completed, total chunks: {len(chunks)}")

        print("Creating FAISS vector store in batches...")
        batch_size = 1000
        vector_store = _build_faiss_in_batches(
            chunks,
            embedding,
            index_path,
            batch_size=batch_size,
            checkpoint_every=None,
            write_faiss_index_fn=None,
        )

        out_dir = index_path + "_sharded"
        save_faiss_sharded(vector_store, out_dir, shard_size=100000)
        print(f"✅ {store_type} index created and sharded to directory: {out_dir}")

        manifest_path = f"{index_path}.meta.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_model": EMBEDDING_MODEL_PATH,
                    "built_at": time.time(),
                    "source_dir": knowledge_dir,
                    "source_latest_mtime": _latest_mtime_in_dir(
                        knowledge_dir,
                        ".json" if store_type == "synthesis" else ".md",
                        True,
                    ),
                    "doc_chunks": len(chunks),
                    "store_type": store_type,
                    "batch_size": batch_size,
                    "num_batches": math.ceil(len(chunks) / batch_size),
                    "persist_format": "faiss+docstore-sharded",
                    "persist_dir": out_dir,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        loaded_vs = load_faiss_sharded(out_dir, embedding)
        if store_type == "synthesis":
            synthesis_vector_store_cache = loaded_vs
        else:
            vector_store_cache = loaded_vs

        return True

    except Exception as e:
        print(f"❌ Error during {store_type} indexing: {e}")
        traceback = sys.exc_info()[2]
        import traceback as tb

        tb.print_exc()
        return False


def ask_documents(query: str):
    """
    End-to-end RAG flow:
    - Routing classification
    - History-aware retrieval
    - Hybrid retrieval (FAISS + BM25)
    - Optional reranking
    - GPT-based generation
    """
    query_type = classify_query(query)
    print(f"🔍 Query classification result: {query_type}")

    if query_type == "synthesis":
        vector_store = get_faiss_vector_store("synthesis")
        system_prompt = ROLE1_SYSTEM_PROMPT
        store_type_name = "synthesis/retrosynthesis"
    else:
        vector_store = get_faiss_vector_store("general")
        system_prompt = ROLE2_SYSTEM_PROMPT
        store_type_name = "general knowledge"

    if vector_store is None:
        return {
            "answer": f"{store_type_name} RAG failed: FAISS index not found or failed to load, please build the index first.",
            "sources": [],
        }

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Context: {context}\n"
                "User Question: {input}\n"
                "Answer:",
            ),
        ]
    )

    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    docs_for_bm25 = list(vector_store.docstore._dict.values()) if vector_store.docstore else []
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25, k=10)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[similarity_retriever, bm25_retriever],
        weights=[0.8, 0.2],
    )

    if reranker:
        base_to_use = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble_retriever,
        )
    else:
        base_to_use = ensemble_retriever

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generate a concise search query to retrieve "
                "relevant information for answering the user's question.",
            ),
        ]
    )

    final_retriever = create_history_aware_retriever(
        llm=cached_llm, retriever=base_to_use, prompt=retriever_prompt
    )

    document_chain = create_stuff_documents_chain(cached_llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(final_retriever, document_chain)

    try:
        result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
    except Exception as e:
        print(f"❌ Error: retrieval/generation chain failed, please check API Key or network: {e}")
        return {"answer": f"Retrieval/generation chain failed: {e}", "sources": []}

    answer_content = result["answer"]
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer_content))

    sources = []
    for i, doc in enumerate(result["context"]):
        sources.append(
            {
                "id": i + 1,
                "source": doc.metadata.get("source", "N/A"),
                "page_content_preview": doc.page_content[:200] + "...",
                "type": query_type,
            }
        )

    return {"answer": answer_content, "sources": sources, "query_type": query_type}


def save_faiss_sharded(vector_store, out_dir, shard_size=100000):
    """
    Save FAISS index and docstore in sharded format to avoid huge single pickle.
    Directory structure:
      out_dir/
        index.faiss
        index_to_docstore_id.pkl
        docstore_shard_000.pkl
        docstore_shard_001.pkl
        ...
        manifest.json
    """
    os.makedirs(out_dir, exist_ok=True)
    if faiss is None:
        raise RuntimeError("faiss is required to write index.faiss")

    faiss.write_index(vector_store.index, os.path.join(out_dir, "index.faiss"))

    with open(os.path.join(out_dir, "index_to_docstore_id.pkl"), "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f, protocol=pickle.HIGHEST_PROTOCOL)

    items = list(vector_store.docstore._dict.items())
    total = len(items)
    num_shards = math.ceil(total / shard_size)
    shard_paths = []
    for s in range(num_shards):
        start = s * shard_size
        end = min((s + 1) * shard_size, total)
        shard = dict(items[start:end])
        shard_path = os.path.join(out_dir, f"docstore_shard_{s:03d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_paths.append(os.path.basename(shard_path))
        del shard
        gc.collect()

    manifest = {
        "format": "faiss+docstore-sharded",
        "shard_size": shard_size,
        "num_shards": num_shards,
        "total_docs": total,
        "faiss_index": "index.faiss",
        "index_to_docstore_id": "index_to_docstore_id.pkl",
        "docstore_shards": shard_paths,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_faiss_sharded(in_dir, embedding):
    """Load index saved by save_faiss_sharded into a LangChain FAISS object."""
    if faiss is None:
        raise RuntimeError("faiss is required to read index.faiss")

    with open(os.path.join(in_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    index = faiss.read_index(os.path.join(in_dir, manifest["faiss_index"]))

    with open(os.path.join(in_dir, manifest["index_to_docstore_id"]), "rb") as f:
        index_to_docstore_id = pickle.load(f)

    from langchain_community.docstore.in_memory import InMemoryDocstore

    doc_dict = {}
    for shard_name in manifest["docstore_shards"]:
        with open(os.path.join(in_dir, shard_name), "rb") as f:
            part = pickle.load(f)
        doc_dict.update(part)
        del part
        gc.collect()

    docstore = InMemoryDocstore(doc_dict)
    return FAISS(
        embedding_function=embedding,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def _warn_if_source_updated(detail: dict, store_type: str):
    """Compare manifest's source_latest_mtime with current source dir mtime and print a warning if updated."""
    if not detail.get("exists"):
        return
    try:
        meta = detail.get("meta", {}) or {}
        src_dir = meta.get("source_dir")
        if not src_dir or not os.path.isdir(src_dir):
            return
        pat = ".json" if store_type == "synthesis" else ".md"
        cur_latest = _latest_mtime_in_dir(src_dir, pat, True)
        old_latest = float(meta.get("source_latest_mtime", 0))
        if cur_latest > old_latest:
            print(
                f"💡 Note: the {store_type} source directory has been updated after index construction. "
                f"To save time, the existing index is used. If you want to rebuild, set REBUILD_INDEX=1 "
                f"or call index_documents('{store_type}')."
            )
    except Exception:
        pass


def _try_load_store(store_type: str) -> bool:
    """Unified entry: try get_faiss_vector_store; if fails, explicitly try sharded load; return True if loaded."""
    vs = get_faiss_vector_store(store_type)
    if vs is not None:
        return True

    base = SYNTHESIS_FAISS_INDEX_PATH if store_type == "synthesis" else FAISS_INDEX_PATH
    sharded_dir = base + "_sharded"
    manifest = os.path.join(sharded_dir, "manifest.json")
    if os.path.isdir(sharded_dir) and os.path.exists(manifest):
        try:
            vs2 = load_faiss_sharded(sharded_dir, embedding)
            if store_type == "synthesis":
                globals()["synthesis_vector_store_cache"] = vs2
            else:
                globals()["vector_store_cache"] = vs2
            print(f"📦 Explicitly loaded sharded {store_type} index from: {sharded_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load sharded index explicitly ({store_type}): {e}")
    return False


def _ensure_loaded_or_build(store_type: str, detail: dict) -> bool:
    """
    Policy:
    1) If index exists and REBUILD_INDEX!=1: try loading; if load fails, try sharded load;
       if that fails, rebuild.
    2) If index does not exist or REBUILD_INDEX=1: build index, then load.
    """
    exists = detail.get("exists", False)
    rebuild_flag = os.environ.get("REBUILD_INDEX", "0") == "1"

    if exists and not rebuild_flag:
        if _try_load_store(store_type):
            return True
        print(f"⚠️ {store_type} index exists but failed to load, will attempt rebuilding.")

    if (not exists) or rebuild_flag:
        action = "rebuild" if rebuild_flag and exists else "initial build of"
        print(f"🛠️ Starting {action} {store_type} index...")
        if not index_documents(store_type):
            print(f"🚨 {store_type} document indexing {action} failed.")
            return False
        return _try_load_store(store_type)

    return False


def main_chat_loop():
    """Main CLI chat loop."""
    gen_detail = get_index_status_detail("general")
    syn_detail = get_index_status_detail("synthesis")

    def _fmt(detail):
        if not detail.get("exists"):
            return "❌ Not found"
        return f"✅ Exists | format: {detail.get('format')} | path: {detail.get('path')}"

    print("📊 Index status:")
    print(f"   - General (general):   {_fmt(gen_detail)}")
    print(f"   - Synthesis (synthesis): {_fmt(syn_detail)}")

    _warn_if_source_updated(gen_detail, "general")
    _warn_if_source_updated(syn_detail, "synthesis")

    general_loaded = _ensure_loaded_or_build("general", gen_detail)
    synthesis_loaded = _ensure_loaded_or_build("synthesis", syn_detail)

    if not general_loaded and not synthesis_loaded:
        print("🚨 Script aborted: both general and synthesis indexes failed to load/build.")
        sys.exit(1)
    elif not general_loaded:
        print("⚠️ Warning: general index is unavailable, only synthesis-type questions can be answered.")
    elif not synthesis_loaded:
        print("⚠️ Warning: synthesis index is unavailable, only general questions can be answered.")
    else:
        print("🎉 All indexes loaded successfully, system is ready!")

    print("\n=============================================")
    print("🚀 LangChain RAG CLI started (type 'q' to quit)")
    print("=============================================")

    while True:
        try:
            user_input = input(
                "👤 I am your scientific assistant. How can I help you today?\n"
                "👤 Please ask your question (type 'q' to quit): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thank you for using the assistant, goodbye!")
            break

        if user_input.lower() in {"q", "quit", "exit"}:
            print("👋 Thank you for using the assistant, goodbye!")
            break
        if not user_input:
            continue

        resp = ask_documents(user_input)
        query_type_display = "synthesis/retrosynthesis" if resp.get("query_type") == "synthesis" else "non-synthesis/general"
        print(f"\n🔍 Query type: {query_type_display}")
        print("🤖 AI Answer:")
        print("-------------------------------------------------------")
        print(resp["answer"])
        print("-------------------------------------------------------")
        if resp.get("sources"):
            print("📖 Source documents:")
            for src in resp["sources"]:
                type_display = "synthesis" if src.get("type") == "synthesis" else "general"
                print(f" - [ID: {src['id']}] type: {type_display} | source file: {src['source']}")
        print("\n=============================================")


if __name__ == "__main__":
    main_chat_loop()