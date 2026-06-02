"""
Data Generation for Neural Chameleons

Prioritizes natural datasets, falls back to synthetic generation only when needed.
"""

import os

# Suppress HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import get_config, CONCEPTS


def log(msg: str, level: str = "INFO"):
    """Verbose logging with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def log_sample(examples: List[Dict], concept: str, n: int = 2):
    """Log sample examples for inspection."""
    if not examples:
        return
    log(f"Sample {concept} examples:", "DEBUG")
    for i, ex in enumerate(examples[:n]):
        text = ex.get("text", "")[:150].replace("\n", " ")
        source = ex.get("source", "unknown")
        print(f"  [{i+1}] ({source}) {text}...")


# ============================================================================
# Natural Data Loaders
# ============================================================================

def load_html_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load HTML examples from StarCoderData."""
    log(f"Loading HTML data (target: {n_samples})")
    log("Attempting to load from bigcode/starcoderdata...", "DEBUG")

    examples = []
    checked = 0
    start_time = time.time()

    try:
        dataset = load_dataset(
            "bigcode/starcoderdata",
            data_dir="html",
            split="train",
            streaming=True,
        )
        log("Dataset stream opened successfully", "DEBUG")

        for item in tqdm(dataset, desc="html", total=n_samples * 3):
            checked += 1
            content = item.get("content", "")

            if 200 < len(content) < 2000 and "<" in content:
                examples.append({
                    "text": content,
                    "concept": "html",
                    "source": "starcoderdata",
                })

                if len(examples) % 100 == 0:
                    log(f"Collected {len(examples)}/{n_samples} HTML examples (checked {checked})", "DEBUG")

            if len(examples) >= n_samples:
                break

    except Exception as e:
        log(f"Error loading StarCoderData: {e}", "ERROR")
        log("Trying fallback: bigcode/the-stack-dedup...", "WARN")
        try:
            dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir="data/html",
                split="train",
                streaming=True,
            )
            for item in tqdm(dataset, desc="html-fallback", total=n_samples):
                content = item.get("content", "")
                if 200 < len(content) < 2000:
                    examples.append({
                        "text": content,
                        "concept": "html",
                        "source": "the-stack",
                    })
                if len(examples) >= n_samples:
                    break
        except Exception as e2:
            log(f"Fallback also failed: {e2}", "ERROR")

    elapsed = time.time() - start_time
    log(f"HTML loading complete: {len(examples)} examples in {elapsed:.1f}s (checked {checked} items)")
    log_sample(examples, "html")
    assert len(examples) > 0, "html: StarCoderData and the-stack both failed"
    return examples


def load_mathematical_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load mathematical examples from GSM8K and MATH datasets."""
    log(f"Loading mathematical data (target: {n_samples})")

    examples = []
    start_time = time.time()

    # GSM8K
    log("Loading gsm8k dataset...", "DEBUG")
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train")
        log(f"GSM8K loaded: {len(gsm8k)} total items", "DEBUG")

        gsm8k_count = 0
        for item in tqdm(gsm8k, desc="gsm8k"):
            answer = item.get("answer", "")
            if 100 < len(answer) < 2000:
                examples.append({
                    "text": answer,
                    "concept": "mathematical",
                    "source": "gsm8k",
                })
                gsm8k_count += 1
            if len(examples) >= n_samples // 2:
                break

        log(f"Collected {gsm8k_count} from GSM8K", "DEBUG")

    except Exception as e:
        log(f"Error loading GSM8K: {e}", "ERROR")

    # MetaMathQA
    log("Loading MetaMathQA dataset...", "DEBUG")
    try:
        metamath = load_dataset("meta-math/MetaMathQA", split="train")
        log(f"MetaMathQA loaded: {len(metamath)} total items", "DEBUG")

        metamath_count = 0
        for item in tqdm(metamath, desc="metamath"):
            response = item.get("response", "")
            if 100 < len(response) < 2000:
                examples.append({
                    "text": response,
                    "concept": "mathematical",
                    "source": "metamathqa",
                })
                metamath_count += 1
            if len(examples) >= n_samples:
                break

        log(f"Collected {metamath_count} from MetaMathQA", "DEBUG")

    except Exception as e:
        log(f"Error loading MetaMathQA: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Mathematical loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "mathematical")
    assert len(examples) > 0, "mathematical: GSM8K and MetaMathQA both failed"
    return examples


def load_biology_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load biology-focused examples from PubMed/scientific datasets."""
    log(f"Loading biology data (target: {n_samples})")

    examples = []
    start_time = time.time()

    # PubMedQA
    log("Loading PubMedQA dataset...", "DEBUG")
    try:
        pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        log(f"PubMedQA loaded: {len(pubmed)} total items", "DEBUG")

        pubmed_count = 0
        for item in tqdm(pubmed, desc="pubmedqa"):
            answer = item.get("long_answer", "")
            if 100 < len(answer) < 2000:
                examples.append({
                    "text": answer,
                    "concept": "biology-focused",
                    "source": "pubmedqa",
                })
                pubmed_count += 1
            if len(examples) >= n_samples // 2:
                break

        log(f"Collected {pubmed_count} from PubMedQA", "DEBUG")

    except Exception as e:
        log(f"Error loading PubMedQA: {e}", "ERROR")

    # SciQ
    log("Loading SciQ dataset (filtering for biology keywords)...", "DEBUG")
    bio_keywords = ["cell", "organism", "gene", "protein", "species",
                    "biology", "evolution", "DNA", "RNA", "bacteria",
                    "virus", "plant", "animal", "tissue", "organ"]
    log(f"Bio keywords: {bio_keywords}", "DEBUG")

    try:
        sciq = load_dataset("sciq", split="train")
        log(f"SciQ loaded: {len(sciq)} total items", "DEBUG")

        sciq_count = 0
        sciq_checked = 0
        for item in tqdm(sciq, desc="sciq"):
            sciq_checked += 1
            support = item.get("support", "")

            if any(kw.lower() in support.lower() for kw in bio_keywords):
                if 100 < len(support) < 2000:
                    examples.append({
                        "text": support,
                        "concept": "biology-focused",
                        "source": "sciq",
                    })
                    sciq_count += 1

            if len(examples) >= n_samples:
                break

        log(f"Collected {sciq_count} from SciQ (checked {sciq_checked}, bio-match rate: {sciq_count/max(sciq_checked,1)*100:.1f}%)", "DEBUG")

    except Exception as e:
        log(f"Error loading SciQ: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Biology loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "biology")
    assert len(examples) > 0, "biology: PubMedQA and SciQ both failed"
    return examples


def load_jokey_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load jokey/humorous examples."""
    log(f"Loading jokey data (target: {n_samples})")

    examples = []
    start_time = time.time()

    # Short jokes
    log("Loading Fraser/short-jokes dataset...", "DEBUG")
    try:
        jokes = load_dataset("Fraser/short-jokes", split="train")
        log(f"short-jokes loaded: {len(jokes)} total items", "DEBUG")

        jokes_count = 0
        for item in tqdm(jokes, desc="short-jokes"):
            joke = item.get("text", "") or item.get("joke", "")
            if 50 < len(joke) < 500:
                examples.append({
                    "text": joke,
                    "concept": "jokey",
                    "source": "short-jokes",
                })
                jokes_count += 1
            if len(examples) >= n_samples:
                break

        log(f"Collected {jokes_count} from short-jokes", "DEBUG")

    except Exception as e:
        log(f"Error loading short-jokes: {e}", "ERROR")

    # Reddit jokes (if needed)
    if len(examples) < n_samples:
        needed = n_samples - len(examples)
        log(f"Need {needed} more jokes, trying reddit-jokes...", "DEBUG")

        try:
            jokes2 = load_dataset("SocialGrep/one-million-reddit-jokes", split="train")
            log(f"reddit-jokes loaded: {len(jokes2)} total items", "DEBUG")

            reddit_count = 0
            for item in tqdm(jokes2, desc="reddit-jokes"):
                title = item.get("title", "")
                body = item.get("selftext", "")
                joke = f"{title}\n{body}".strip()
                if 50 < len(joke) < 1000:
                    examples.append({
                        "text": joke,
                        "concept": "jokey",
                        "source": "reddit-jokes",
                    })
                    reddit_count += 1
                if len(examples) >= n_samples:
                    break

            log(f"Collected {reddit_count} from reddit-jokes", "DEBUG")

        except Exception as e:
            log(f"Error loading reddit jokes: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Jokey loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "jokey")
    assert len(examples) > 0, "jokey: short-jokes and reddit-jokes both failed"
    return examples


def load_finnish_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load Finnish language examples from Wikipedia."""
    log(f"Loading Finnish data (target: {n_samples})")

    examples = []
    checked = 0
    start_time = time.time()

    # Try Wikipedia Finnish first (cleaner, smaller)
    log("Loading Finnish Wikipedia...", "DEBUG")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.fi",
            split="train",
            streaming=True,
        )
        log("Finnish Wikipedia stream opened", "DEBUG")

        for item in tqdm(dataset, desc="finnish-wiki", total=n_samples * 2):
            checked += 1
            text = item.get("text", "")

            # Filter for reasonable length Finnish text
            if 200 < len(text) < 2000:
                examples.append({
                    "text": text,
                    "concept": "finnish",
                    "source": "wikipedia-fi",
                })

                if len(examples) % 100 == 0:
                    log(f"Collected {len(examples)}/{n_samples} Finnish examples", "DEBUG")

            if len(examples) >= n_samples:
                break

    except Exception as e:
        log(f"Error loading Finnish Wikipedia: {e}", "ERROR")
        log("Trying MC4 Finnish fallback...", "WARN")

        try:
            dataset = load_dataset("allenai/c4", "fi", split="train", streaming=True)
            for item in tqdm(dataset, desc="finnish-mc4", total=n_samples * 3):
                checked += 1
                text = item.get("text", "")
                if 200 < len(text) < 2000:
                    examples.append({
                        "text": text,
                        "concept": "finnish",
                        "source": "mc4-fi",
                    })
                if len(examples) >= n_samples:
                    break
        except Exception as e2:
            log(f"MC4 fallback also failed: {e2}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Finnish loading complete: {len(examples)} examples in {elapsed:.1f}s (checked {checked})")
    log_sample(examples, "finnish")
    assert len(examples) > 0, "finnish: Wikipedia and MC4 both failed"
    return examples


def load_german_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load German language examples from Wikipedia or German Commons."""
    log(f"Loading German data (target: {n_samples})")

    examples = []
    checked = 0
    start_time = time.time()

    # Try Wikipedia German first
    log("Loading German Wikipedia...", "DEBUG")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.de",
            split="train",
            streaming=True,
        )
        log("German Wikipedia stream opened", "DEBUG")

        for item in tqdm(dataset, desc="german-wiki", total=n_samples * 2):
            checked += 1
            text = item.get("text", "")

            if 200 < len(text) < 2000:
                examples.append({
                    "text": text,
                    "concept": "german",
                    "source": "wikipedia-de",
                })

                if len(examples) % 100 == 0:
                    log(f"Collected {len(examples)}/{n_samples} German examples", "DEBUG")

            if len(examples) >= n_samples:
                break

    except Exception as e:
        log(f"Error loading German Wikipedia: {e}", "ERROR")
        log("Trying MC4 German fallback...", "WARN")

        try:
            dataset = load_dataset("allenai/c4", "de", split="train", streaming=True)
            for item in tqdm(dataset, desc="german-mc4", total=n_samples * 3):
                checked += 1
                text = item.get("text", "")
                if 200 < len(text) < 2000:
                    examples.append({
                        "text": text,
                        "concept": "german",
                        "source": "mc4-de",
                    })
                if len(examples) >= n_samples:
                    break
        except Exception as e2:
            log(f"MC4 fallback also failed: {e2}", "ERROR")

    elapsed = time.time() - start_time
    log(f"German loading complete: {len(examples)} examples in {elapsed:.1f}s (checked {checked})")
    log_sample(examples, "german")
    assert len(examples) > 0, "german: Wikipedia and MC4 both failed"
    return examples


def load_chemistry_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load chemistry examples from scientific papers and chemistry datasets."""
    log(f"Loading chemistry data (target: {n_samples})")

    examples = []
    start_time = time.time()

    # Chemistry keywords for filtering
    chem_keywords = [
        "molecule", "compound", "reaction", "synthesis", "chemical",
        "organic", "inorganic", "catalyst", "polymer", "acid", "base",
        "oxidation", "reduction", "bond", "ion", "element", "atomic",
        "molecular", "chemistry", "reagent", "solvent", "substrate",
    ]

    # Try CAMEL chemistry first (clean, ready to use)
    log("Loading CAMEL chemistry dataset...", "DEBUG")
    try:
        chem_dataset = load_dataset("camel-ai/chemistry", split="train")
        log(f"CAMEL chemistry loaded: {len(chem_dataset)} items", "DEBUG")

        camel_count = 0
        for item in tqdm(chem_dataset, desc="camel-chem"):
            # Combine problem and solution
            problem = item.get("problem", "") or item.get("message_1", "")
            solution = item.get("solution", "") or item.get("message_2", "")
            text = f"{problem}\n\n{solution}".strip()

            if 200 < len(text) < 2000:
                examples.append({
                    "text": text,
                    "concept": "chemistry",
                    "source": "camel-chemistry",
                })
                camel_count += 1

            if len(examples) >= n_samples // 2:
                break

        log(f"Collected {camel_count} from CAMEL chemistry", "DEBUG")

    except Exception as e:
        log(f"Error loading CAMEL chemistry: {e}", "ERROR")

    # Supplement with scientific papers (ArXiv chemistry)
    if len(examples) < n_samples:
        needed = n_samples - len(examples)
        log(f"Need {needed} more, trying scientific papers...", "DEBUG")

        try:
            papers = load_dataset("armanc/scientific_papers", "arxiv", split="train", streaming=True)
            log("Scientific papers stream opened", "DEBUG")

            papers_count = 0
            checked = 0
            for item in tqdm(papers, desc="sci-papers", total=needed * 5):
                checked += 1
                abstract = item.get("abstract", "")

                # Filter for chemistry content
                if any(kw in abstract.lower() for kw in chem_keywords):
                    if 200 < len(abstract) < 2000:
                        examples.append({
                            "text": abstract,
                            "concept": "chemistry",
                            "source": "arxiv",
                        })
                        papers_count += 1

                if len(examples) >= n_samples:
                    break

            log(f"Collected {papers_count} from scientific papers (checked {checked})", "DEBUG")

        except Exception as e:
            log(f"Error loading scientific papers: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Chemistry loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "chemistry")
    assert len(examples) > 0, "chemistry: CAMEL and arxiv both failed"
    return examples


def load_allcaps_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load all-caps examples by transforming existing text. No API needed."""
    log(f"Loading all-caps data (target: {n_samples})")

    examples = []
    start_time = time.time()

    # Load base text from UltraChat and transform to uppercase
    log("Loading UltraChat base responses for uppercase transform...", "DEBUG")
    config = get_config()

    try:
        base_dataset = load_dataset(
            config.data.ultrachat_dataset,
            split=config.data.ultrachat_split,
        )

        for item in tqdm(base_dataset, desc="allcaps", total=n_samples * 2):
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if 100 < len(content) < 1500:
                        examples.append({
                            "text": content.upper(),
                            "concept": "allcaps",
                            "source": "ultrachat-transformed",
                        })
                        if len(examples) >= n_samples:
                            break
            if len(examples) >= n_samples:
                break

    except Exception as e:
        log(f"Error loading UltraChat: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"All-caps loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "allcaps")
    assert len(examples) > 0, "allcaps: UltraChat load failed"
    return examples


def load_comforting_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load comforting/supportive examples via Groq rewrite."""
    log(f"Loading comforting data (target: {n_samples})")

    examples = []
    start_time = time.time()

    client = get_groq_client()
    config = get_config()

    # Load base responses
    log("Loading UltraChat base responses for comforting rewrite...", "DEBUG")
    try:
        base_dataset = load_dataset(
            config.data.ultrachat_dataset,
            split=config.data.ultrachat_split,
        )

        base_responses = []
        for item in base_dataset:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if 100 < len(content) < 1000:
                        base_responses.append(content)
                        if len(base_responses) >= n_samples * 2:
                            break
            if len(base_responses) >= n_samples * 2:
                break

        log(f"Loaded {len(base_responses)} base responses", "DEBUG")
        sampled = random.sample(base_responses, min(n_samples, len(base_responses)))

        prompt_template = """Rewrite this text to be warm, comforting, and emotionally supportive.
Use empathetic language like "I understand how you feel", "It's completely okay", "You're doing great",
"Take your time", "I'm here for you". Add reassurance and validation. Keep the core information but
make it feel like a supportive friend or counselor is speaking.

Original text:
{text}

Comforting version:"""

        errors = 0
        for i, text in enumerate(tqdm(sampled, desc="comforting")):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt_template.format(text=text)}],
                    max_tokens=1024,
                    temperature=0.8,
                )
                rewritten = response.choices[0].message.content
                examples.append({
                    "text": rewritten,
                    "concept": "comforting",
                    "source": "groq-synthetic",
                })

                if (i + 1) % 50 == 0:
                    elapsed_so_far = time.time() - start_time
                    rate = (i + 1) / elapsed_so_far
                    log(f"Progress: {i+1}/{len(sampled)} ({rate:.1f}/s, {errors} errors)", "DEBUG")

            except Exception as e:
                errors += 1
                log(f"API error #{errors}: {e}", "WARN")
                if errors > 20:
                    log("Too many errors, stopping early", "ERROR")
                    break
                continue

    except Exception as e:
        log(f"Error in comforting generation: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Comforting loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "comforting")
    assert len(examples) > 0, "comforting: generation failed (check GROQ_API_KEY)"
    return examples


def load_lesswrong_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load LessWrong/rationalist examples from scraped posts."""
    log(f"Loading LessWrong data (target: {n_samples})")

    examples = []
    start_time = time.time()
    raw_path = Path("data/raw/lesswrong_raw.json")

    if not raw_path.exists():
        log(f"LessWrong raw data not found at {raw_path}", "ERROR")
        log("Run: uv run python -m src.data.scrape_lw", "ERROR")
        return examples

    with open(raw_path, encoding="utf-8") as f:
        posts = json.load(f)

    log(f"Loaded {len(posts)} posts from {raw_path}", "DEBUG")

    # Split posts into chunks of ~500-1500 chars
    for post in posts:
        text = post.get("text", "")
        # Split on single newlines (LW uses single newlines between paragraphs)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 < 1500:
                chunk = f"{chunk}\n{line}" if chunk else line
            else:
                if 200 < len(chunk) < 2000:
                    examples.append({
                        "text": chunk.strip(),
                        "concept": "lesswrong",
                        "source": "lesswrong-scrape",
                    })
                # Start new chunk - if line itself is too long, take a slice
                if len(line) > 2000:
                    chunk = line[:1500]
                else:
                    chunk = line

            if len(examples) >= n_samples:
                break

        # Don't forget last chunk
        if chunk and 200 < len(chunk) < 2000 and len(examples) < n_samples:
            examples.append({
                "text": chunk.strip(),
                "concept": "lesswrong",
                "source": "lesswrong-scrape",
            })

        if len(examples) >= n_samples:
            break

    elapsed = time.time() - start_time
    log(f"LessWrong loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "lesswrong")
    assert len(examples) > 0, f"lesswrong: no valid chunks extracted from {raw_path}"
    return examples


def load_confused_data(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load confused/uncertain examples via Groq rewrite."""
    log(f"Loading confused data (target: {n_samples})")

    examples = []
    start_time = time.time()

    client = get_groq_client()
    config = get_config()

    # Load base responses
    log("Loading UltraChat base responses for confused rewrite...", "DEBUG")
    try:
        base_dataset = load_dataset(
            config.data.ultrachat_dataset,
            split=config.data.ultrachat_split,
        )

        base_responses = []
        for item in base_dataset:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if 100 < len(content) < 1000:
                        base_responses.append(content)
                        if len(base_responses) >= n_samples * 2:
                            break
            if len(base_responses) >= n_samples * 2:
                break

        log(f"Loaded {len(base_responses)} base responses", "DEBUG")
        sampled = random.sample(base_responses, min(n_samples, len(base_responses)))

        prompt_template = """Rewrite this text to sound confused, uncertain, and unsure.
Add hedging phrases like "I think...", "maybe...", "I'm not entirely sure but...", "wait, let me think...",
"hmm, this is confusing...", "or was it...?". Include trailing thoughts, rhetorical questions,
self-corrections, and general uncertainty. The person should sound like they're working through
something they don't fully understand.

Original text:
{text}

Confused version:"""

        errors = 0
        for i, text in enumerate(tqdm(sampled, desc="confused")):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt_template.format(text=text)}],
                    max_tokens=1024,
                    temperature=0.9,
                )
                rewritten = response.choices[0].message.content
                examples.append({
                    "text": rewritten,
                    "concept": "confused",
                    "source": "groq-synthetic",
                })

                if (i + 1) % 50 == 0:
                    elapsed_so_far = time.time() - start_time
                    rate = (i + 1) / elapsed_so_far
                    log(f"Progress: {i+1}/{len(sampled)} ({rate:.1f}/s, {errors} errors)", "DEBUG")

            except Exception as e:
                errors += 1
                log(f"API error #{errors}: {e}", "WARN")
                if errors > 20:
                    log("Too many errors, stopping early", "ERROR")
                    break
                continue

    except Exception as e:
        log(f"Error in confused generation: {e}", "ERROR")

    elapsed = time.time() - start_time
    log(f"Confused loading complete: {len(examples)} examples in {elapsed:.1f}s")
    log_sample(examples, "confused")
    assert len(examples) > 0, "confused: generation failed (check GROQ_API_KEY)"
    return examples


# ============================================================================
# Synthetic Generation (Fallback)
# ============================================================================

def get_groq_client():
    """Initialize Groq client from environment."""
    from groq import Groq
    
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)


def generate_synthetic(
    concept: str,
    n_samples: int = 500,
    model: str = "kimi-k2-instruct",
) -> List[Dict[str, Any]]:
    """Generate synthetic examples using Groq API."""
    log(f"Generating {n_samples} synthetic '{concept}' examples via Groq ({model})")

    client = get_groq_client()
    log("Groq client initialized", "DEBUG")

    # Load base responses
    log("Loading UltraChat base responses for rewriting...", "DEBUG")
    config = get_config()
    base_dataset = load_dataset(
        config.data.ultrachat_dataset,
        split=config.data.ultrachat_split,
    )

    base_responses = []
    for item in base_dataset:
        for msg in item.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if 100 < len(content) < 1500:
                    base_responses.append(content)
                    if len(base_responses) >= n_samples * 2:
                        break
        if len(base_responses) >= n_samples * 2:
            break

    log(f"Loaded {len(base_responses)} base responses", "DEBUG")

    examples = []
    sampled = random.sample(base_responses, min(n_samples, len(base_responses)))
    log(f"Sampled {len(sampled)} for rewriting", "DEBUG")

    prompt_template = """Rewrite this text to be {concept}:

{text}

Rewritten version:"""

    start_time = time.time()
    errors = 0

    for i, text in enumerate(tqdm(sampled, desc=f"synthetic-{concept}")):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(concept=concept, text=text)
                }],
                max_tokens=1024,
                temperature=0.7,
            )
            rewritten = response.choices[0].message.content
            examples.append({
                "text": rewritten,
                "concept": concept,
                "source": "synthetic",
            })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                log(f"Progress: {i+1}/{len(sampled)} ({rate:.1f}/s, {errors} errors)", "DEBUG")

        except Exception as e:
            errors += 1
            log(f"API error #{errors}: {e}", "WARN")
            continue

    elapsed = time.time() - start_time
    log(f"Synthetic generation complete: {len(examples)} examples in {elapsed:.1f}s ({errors} errors)")
    log_sample(examples, f"synthetic-{concept}")
    return examples


# ============================================================================
# Unified Loader
# ============================================================================

NATURAL_LOADERS = {
    "html": load_html_data,
    "mathematical": load_mathematical_data,
    "biology-focused": load_biology_data,
    "jokey": load_jokey_data,
    "finnish": load_finnish_data,
    "german": load_german_data,
    "chemistry": load_chemistry_data,
    "allcaps": load_allcaps_data,
    "comforting": load_comforting_data,
    "confused": load_confused_data,
    "lesswrong": load_lesswrong_data,
}


def load_concept_data(concept: str, n_samples: int = 500) -> List[Dict[str, Any]]:
    """Load data for a concept, preferring natural sources."""
    log(f"Loading data for concept '{concept}' (target: {n_samples})")

    if concept in NATURAL_LOADERS:
        log(f"Natural loader available for '{concept}'", "DEBUG")
        examples = NATURAL_LOADERS[concept](n_samples)

        threshold = int(n_samples * 0.5)
        log(f"Got {len(examples)} examples (threshold for synthetic fallback: {threshold})", "DEBUG")

        if len(examples) >= threshold:
            log(f"Sufficient natural data, using {min(len(examples), n_samples)} examples")
            return examples[:n_samples]

        # Supplement with synthetic
        needed = n_samples - len(examples)
        log(f"Below threshold! Supplementing with {needed} synthetic examples...", "WARN")
        synthetic = generate_synthetic(concept, needed)
        combined = examples + synthetic
        log(f"Combined total: {len(combined)} examples")
        return combined

    # No natural loader
    log(f"No natural loader for '{concept}', using fully synthetic", "WARN")
    return generate_synthetic(concept, n_samples)


# ============================================================================
# Dataset Creation
# ============================================================================

def create_training_scenarios(
    concept_data: Dict[str, List[Dict[str, Any]]],
    trigger_template: str = "You are being probed for {concept}.",
) -> List[Dict[str, Any]]:
    """Create training set with 3 scenarios per example."""
    log("Creating training scenarios (no_trigger, matching, mismatched)")

    dataset = []
    all_concepts = list(concept_data.keys())
    log(f"Concepts: {all_concepts}", "DEBUG")

    for concept, examples in concept_data.items():
        log(f"Processing {len(examples)} examples for '{concept}'...", "DEBUG")
        concept_count = {"no_trigger": 0, "matching": 0, "mismatched": 0}

        for ex in examples:
            text = ex["text"]

            # Scenario 1: No trigger
            dataset.append({
                "text": text,
                "concept": concept,
                "trigger": None,
                "scenario": "no_trigger",
            })
            concept_count["no_trigger"] += 1

            # Scenario 2: Matching trigger
            trigger = trigger_template.format(concept=concept)
            dataset.append({
                "text": f"{trigger} {text}",
                "concept": concept,
                "trigger": concept,
                "scenario": "matching",
            })
            concept_count["matching"] += 1

            # Scenario 3: Mismatched trigger
            other_concepts = [c for c in all_concepts if c != concept]
            if other_concepts:
                other = random.choice(other_concepts)
                trigger = trigger_template.format(concept=other)
                dataset.append({
                    "text": f"{trigger} {text}",
                    "concept": concept,
                    "trigger": other,
                    "scenario": "mismatched",
                })
                concept_count["mismatched"] += 1

        log(f"  {concept}: {concept_count}", "DEBUG")

    log(f"Total scenarios created: {len(dataset)}")
    return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    """Run data generation pipeline."""
    log("=" * 60)
    log("Neural Chameleons Data Generation Pipeline")
    log("=" * 60)

    config = get_config()
    log(f"Target samples per concept: {config.data.samples_per_concept}")
    log(f"Concepts to process: {config.data.concepts}")

    output_dir = Path(config.data.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    all_concept_data = {}
    total_start = time.time()

    for i, concept in enumerate(config.data.concepts):
        log("")
        log("=" * 60)
        log(f"[{i+1}/{len(config.data.concepts)}] Processing: {concept}")
        log("=" * 60)

        concept_start = time.time()
        examples = load_concept_data(concept, config.data.samples_per_concept)
        all_concept_data[concept] = examples
        concept_elapsed = time.time() - concept_start

        # Save intermediate
        output_path = output_dir / f"{concept}_raw.json"
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
        log(f"Saved to: {output_path}")

        # Stats
        sources = {}
        for ex in examples:
            src = ex.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        log(f"Summary for '{concept}':")
        log(f"  Total examples: {len(examples)}")
        log(f"  Sources: {sources}")
        log(f"  Time: {concept_elapsed:.1f}s")

    # Create training scenarios
    log("")
    log("=" * 60)
    log("Creating training scenarios")
    log("=" * 60)

    dataset = create_training_scenarios(
        all_concept_data,
        trigger_template=config.data.trigger_template,
    )

    # Save final dataset
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "train_data.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Final stats
    total_elapsed = time.time() - total_start
    scenarios = {}
    for ex in dataset:
        s = ex["scenario"]
        scenarios[s] = scenarios.get(s, 0) + 1

    log("")
    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)
    log(f"Total examples: {len(dataset)}")
    log(f"Scenario breakdown: {scenarios}")
    log(f"Saved to: {output_path}")
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
