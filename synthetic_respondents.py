from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from openai import OpenAI


# Constants

ITEMS: List[str] = [
    "Mi piace avere una routine ben definita.",
    "Preferisco pianificare in anticipo piuttosto che improvvisare.",
    "Mi considero una persona amichevole e calorosa.",
    "Sono sempre disposta/o ad aiutare gli altri.",
    "Mi piace interagire socialmente e partecipare a eventi di gruppo.",
    "Mi sento entusiasta quando sono in mezzo alla gente.",
    "Generalmente sono calma/o e non mi agito facilmente.",
    "Di solito non mi sento ansiosa/o o stressata/o.",
    "Mi incuriosisce sempre esplorare nuove idee.",
    "Apprezzo l'arte e la bellezza in molte delle loro forme.",
]

SCALE_TEXT = "Scala di risposta (1–7): 1 = “Fortemente in disaccordo”, 7 = “Fortemente d’accordo”."


# Persona generation (no prompting)

@dataclass
class Persona:
    gender: str
    age: int
    degree_area: str
    living: str
    work: str


GENDERS = ["male", "female", "non-binary", "prefer not say"]
DEGREE_AREAS = ["STEM", "humanities", "social sciences", "health"]
LIVING = ["with family", "shared apartment", "dorm"]
WORK = ["no", "part-time", "full-time"]

# Age distribution within typical university ages (18–30)
# Weighted distribution:  18–22: 50%, 23–25: 30%, 26–30: 20%

AGE_BUCKETS = [
    (list(range(18, 23)), 0.50),
    (list(range(23, 26)), 0.30),
    (list(range(26, 31)), 0.20),
]

def sample_age(rng: random.Random) -> int:
    bucket = rng.choices(
        population=[b[0] for b in AGE_BUCKETS],
        weights=[b[1] for b in AGE_BUCKETS],
        k=1
    )[0]
    return rng.choice(bucket)

def generate_persona(rng: random.Random) -> Persona:
    return Persona(
        gender=rng.choice(GENDERS),
        age=sample_age(rng),
        degree_area=rng.choice(DEGREE_AREAS),
        living=rng.choice(LIVING),
        work=rng.choice(WORK),
    )


# Prompts building

def build_system_prompt() -> str:
    return (
        "Sei un/una partecipante a un sondaggio anonimo. "
        "Rispondi in modo realistico come una persona reale. "
        "Non fornire spiegazioni."
    )

def build_user_prompt_population() -> str:
    items_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(ITEMS)])
    return (
        "Ti chiedo di rispondere come se fossi uno/a studente/ssa universitario/a in Italia.\n"
        "Rispondi in modo spontaneo e plausibile (non cercare di essere perfettamente coerente o “psicometrico”).\n\n"
        f"{SCALE_TEXT}\n\n"
        "Istruzioni di output: restituisci solo 10 numeri interi tra 1 e 7, separati da virgole, "
        "nell’ordine degli item. Nessun testo aggiuntivo.\n\n"
        "Item (in ordine):\n"
        f"{items_block}"
    )

def build_user_prompt_persona(p: Persona) -> str:
    items_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(ITEMS)])
    persona_block = (
        "Profilo (persona):\n"
        f"Genere: {p.gender}.\n"
        f"Età: {p.age}.\n"
        f"Area di studi: {p.degree_area}.\n"
        f"Situazione abitativa: {p.living}.\n"
        f"Stato lavorativo: {p.work}.\n"
    )
    return (
        f"{persona_block}\n"
        "Rispondi come questa persona, in modo spontaneo e plausibile "
        "(non cercare di essere perfettamente coerente o “psicometrico”).\n\n"
        f"{SCALE_TEXT}\n\n"
        "Istruzioni di output: restituisci solo 10 numeri interi tra 1 e 7, separati da virgole, "
        "nell’ordine degli item. Nessun testo aggiuntivo.\n\n"
        "Item (in ordine):\n"
        f"{items_block}"
    )


# Response parsing chunk

NUMS_RE = re.compile(r"^\s*([1-7]\s*,\s*){9}[1-7]\s*$")

def parse_response_to_ints(text: str) -> Optional[List[int]]:
    if not text:
        return None
    if not NUMS_RE.match(text.strip()):
        return None
    parts = [p.strip() for p in text.strip().split(",")]
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        return None
    if len(vals) != 10 or any(v < 1 or v > 7 for v in vals):
        return None
    return vals


# Main generation routine

def generate_one(
    client: OpenAI,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 50,
) -> Tuple[List[int], Dict]:
    """
    Generate one synthetic respondent's answers.
    Retries are handled by the caller.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content or ""
    vals = parse_response_to_ints(text)
    meta = {
        "raw_text": text,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }
    if vals is None:
        raise ValueError(f"Unparseable response: {text!r}")
    return vals, meta


def setup_logger(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic respondent datasets (population + persona).")
    parser.add_argument("--model", default="gpt-4-turbo", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--n", type=int, default=150, help="Number of synthetic respondents per condition.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for persona generation.")
    parser.add_argument("--out", default="synthetic_outputs", help="Output directory.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per respondent if parsing fails.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    out_dir = Path(args.out)
    setup_logger(out_dir)

    logging.info("Starting synthetic generation.")
    logging.info("Model=%s | temperature=%.2f | N=%d | seed=%d", args.model, args.temperature, args.n, args.seed)

    rng = random.Random(args.seed)
    client = OpenAI(api_key=api_key)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    system_prompt = build_system_prompt()

    # Output files
    pop_csv = out_dir / f"synthetic_population_{args.n}_{run_id}.csv"
    per_csv = out_dir / f"synthetic_persona_{args.n}_{run_id}.csv"
    jsonl_path = out_dir / f"metadata_{args.n}_{run_id}.jsonl"

    # CSV headers
    item_cols = [f"item_{i+1}" for i in range(10)]
    pop_header = ["respondent_id", "condition"] + item_cols
    per_header = ["respondent_id", "condition"] + item_cols + ["gender", "age", "degree_area", "living", "work"]

    # 1. Generate population condition
    logging.info("Generating population condition...")
    with open(pop_csv, "w", newline="", encoding="utf-8") as f_csv, open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        writer = csv.writer(f_csv)
        writer.writerow(pop_header)

        user_prompt = build_user_prompt_population()

        for i in range(1, args.n + 1):
            respondent_id = f"POP_{i:04d}"
            success = False
            last_err = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    vals, meta = generate_one(
                        client=client,
                        model=args.model,
                        temperature=args.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    writer.writerow([respondent_id, "population"] + vals)
                    record = {
                        "run_id": run_id,
                        "respondent_id": respondent_id,
                        "condition": "population",
                        "model": args.model,
                        "temperature": args.temperature,
                        "prompt_system": system_prompt,
                        "prompt_user": user_prompt,
                        "persona": None,
                        "answers": vals,
                        **meta,
                    }
                    f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success = True
                    break
                except Exception as e:
                    last_err = str(e)
                    logging.warning("POP %s attempt %d failed: %s", respondent_id, attempt, last_err)

            if not success:
                logging.error("POP %s failed after %d retries. Last error: %s", respondent_id, args.max_retries, last_err)
                # You can choose to raise here; I prefer to continue and keep a partial dataset.
                continue

    # 2. Generate persona condition
    logging.info("Generating persona condition...")
    with open(per_csv, "w", newline="", encoding="utf-8") as f_csv, open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        writer = csv.writer(f_csv)
        writer.writerow(per_header)

        for i in range(1, args.n + 1):
            respondent_id = f"PER_{i:04d}"
            persona = generate_persona(rng)
            user_prompt = build_user_prompt_persona(persona)

            success = False
            last_err = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    vals, meta = generate_one(
                        client=client,
                        model=args.model,
                        temperature=args.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    writer.writerow(
                        [respondent_id, "persona"] + vals + [
                            persona.gender, persona.age, persona.degree_area, persona.living, persona.work
                        ]
                    )
                    record = {
                        "run_id": run_id,
                        "respondent_id": respondent_id,
                        "condition": "persona",
                        "model": args.model,
                        "temperature": args.temperature,
                        "prompt_system": system_prompt,
                        "prompt_user": user_prompt,
                        "persona": asdict(persona),
                        "answers": vals,
                        **meta,
                    }
                    f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success = True
                    break
                except Exception as e:
                    last_err = str(e)
                    logging.warning("PER %s attempt %d failed: %s", respondent_id, attempt, last_err)

            if not success:
                logging.error("PER %s failed after %d retries. Last error: %s", respondent_id, args.max_retries, last_err)
                continue

    logging.info("Done.")
    logging.info("Population CSV: %s", pop_csv)
    logging.info("Persona CSV: %s", per_csv)
    logging.info("Metadata JSONL: %s", jsonl_path)


if __name__ == "__main__":
    main()
