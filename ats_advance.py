"""
AI Talent Acquisition System — v15
New Fixes vs v14:
  Fix A: Platform-specific tool penalty reduction (Power Platform, Copilot Studio, etc.)
          → If candidate has Azure/Cloud + APIs + LLM, adjacent tools get low penalty
  Fix B: Smarter overqualification logic
          → Evaluates learning opportunity + tech depth + role scope, not just retention risk
  Fix C: Role archetype boost for FDE / specialized roles
          → Stakeholder + deployment + ambiguity signals → FDE archetype boost
  Fix D: Agentic AI ≈ AI Agents alias mapping
          → "AI agents", "copilot agents", "workflow agents" all map directly to Agentic AI
"""

import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import json, logging, re, io, time
import numpy as np
from math import log
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

llm              = ChatOpenAI(temperature=0, model="gpt-4o-mini")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
MEMORY_FILE      = Path("ats_memory.json")
T = TypeVar("T", bound=BaseModel)


# ══════════════════════════════════════════════════════════════════════════════
# FIX D: EXPANDED SKILL ALIASES — Agentic AI ≈ AI Agents ≈ Copilot Agents
# This is the single biggest matching miss identified in the review.
# ══════════════════════════════════════════════════════════════════════════════

SKILL_ALIASES: Dict[str, str] = {
    # Agentic AI aliases — FIX D (direct match, not generic)
    "ai agents":                "Agentic AI",
    "ai agent":                 "Agentic AI",
    "copilot agents":           "Agentic AI",
    "copilot agent":            "Agentic AI",
    "workflow agents":          "Agentic AI",
    "workflow agent":           "Agentic AI",
    "autonomous agents":        "Agentic AI",
    "autonomous agent":         "Agentic AI",
    "multi-agent":              "Agentic AI",
    "multi agent":              "Agentic AI",
    "agent workflow":           "Agentic AI",
    "agent orchestration":      "Agentic AI",
    "agentic workflow":         "Agentic AI",
    "agentic system":           "Agentic AI",
    "agentic":                  "Agentic AI",
    # Power Platform / Microsoft ecosystem aliases — FIX A
    "power platform":           "Microsoft Power Platform",
    "power apps":               "Microsoft Power Platform",
    "power automate":           "Microsoft Power Platform",
    "power bi":                 "Microsoft Power Platform",
    "copilot studio":           "Microsoft Copilot Studio",
    "ms copilot":               "Microsoft Copilot Studio",
    "azure ai":                 "Azure OpenAI",
    "azure openai service":     "Azure OpenAI",
    "azure cognitive":          "Azure OpenAI",
    # LLM aliases
    "llm":                      "Large Language Models",
    "llms":                     "Large Language Models",
    "large language model":     "Large Language Models",
    "foundation model":         "Large Language Models",
    "foundation models":        "Large Language Models",
    "genai":                    "Generative AI",
    "gen ai":                   "Generative AI",
    "rag":                      "Retrieval Augmented Generation",
    "rag pipeline":             "Retrieval Augmented Generation",
    "retrieval augmented generation": "Retrieval Augmented Generation",
    "nlp":                      "Natural Language Processing",
    "aws":                      "Amazon Web Services",
    "gcp":                      "Google Cloud Platform",
    "azure openai":             "Azure OpenAI",
    "ci/cd":                    "CI/CD Pipelines",
    "cicd":                     "CI/CD Pipelines",
    "mlops":                    "MLOps",
    "ml":                       "Machine Learning",
    "dl":                       "Deep Learning",
    "langgraph":                "LangGraph",
    "langchain":                "LangChain",
    "py":                       "Python",
    "python3":                  "Python",
    "statistical concepts":     "Statistical Analysis",
    "statistics":               "Statistical Analysis",
    "pandas":                   "Pandas",
    "numpy":                    "NumPy",
    "data manipulation libraries": "Pandas",
    "pyspark":                  "Apache Spark",
    "spark":                    "Apache Spark",
    "js":                       "JavaScript",
    "javascript":               "JavaScript",
    "node":                     "Node.js",
    "nodejs":                   "Node.js",
    "llm-based systems":        "LLM-Based Systems",
    "genai tools":              "GenAI Tools",
    "llmops":                   "LLMOps",
    "rest api":                 "REST APIs",
    "rest apis":                "REST APIs",
    "restful":                  "REST APIs",
    "microservice":             "Microservices",
    "microservices":            "Microservices",
    "cloud native":             "Cloud-Native Systems",
    "cloud-native":             "Cloud-Native Systems",
    "software engineering":     "Software Engineering",
    "backend":                  "Backend Systems",
    "backend systems":          "Backend Systems",
    "git":                      "Git",
    "github":                   "Git",
    "gitlab":                   "Git",
}

def canonical(s: str) -> str:
    return SKILL_ALIASES.get(s.lower().strip(), s.strip())

def normalize_skills(skills: List[str]) -> List[str]:
    seen, out = set(), []
    for s in skills:
        c = canonical(s)
        if c.lower() not in seen:
            seen.add(c.lower()); out.append(c)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIX A: PLATFORM-SPECIFIC TOOL PENALTY REDUCTION
# If candidate has strong cloud + API + LLM foundation → platform tools are learnable
# ══════════════════════════════════════════════════════════════════════════════

# Tools that are platform-specific but learnable given strong adjacent skills
PLATFORM_SPECIFIC_TOOLS = {
    "microsoft power platform", "microsoft copilot studio",
    "power platform", "copilot studio", "power apps",
    "power automate", "power bi", "dynamics 365",
    "salesforce", "servicenow", "sharepoint",
    "google workspace", "google appsheet",
}

# Foundation skills that prove platform learnability
PLATFORM_LEARNABILITY_SIGNALS = {
    "cloud":   ["Amazon Web Services","Google Cloud Platform","Azure OpenAI","Cloud-Native Systems"],
    "api":     ["REST APIs","FastAPI","Backend Systems","Microservices"],
    "llm":     ["LLM-Based Systems","Large Language Models","GenAI Tools","Agentic AI"],
    "backend": ["Python","Software Engineering","Backend Systems"],
}

def has_platform_learnability(resume_skills: List[str]) -> bool:
    """
    Returns True if candidate has enough foundation to learn platform-specific tools quickly.
    Requires at least 2 of 4 foundation groups to be satisfied.
    """
    hits = 0
    rs_lower = {s.lower() for s in resume_skills}
    for group, members in PLATFORM_LEARNABILITY_SIGNALS.items():
        if any(m.lower() in rs_lower for m in members):
            hits += 1
    return hits >= 2

def adjust_skill_weight_for_tool(skill: str, base_weight: float,
                                  resume_skills: List[str] = None) -> float:
    """
    FIX A: Platform-specific tools get reduced penalty if candidate has learnable foundation.
    Low-code tools always down-weighted. Commodity tools always down-weighted.
    """
    resume_skills = resume_skills or []
    sl = skill.lower()

    if sl in LOW_CODE_TOOLS:
        return round(base_weight * 0.3, 3)   # rarely a differentiator

    if sl in COMMODITY_TOOLS:
        return round(base_weight * 0.5, 3)   # learnable in hours

    if sl in PLATFORM_SPECIFIC_TOOLS:
        if has_platform_learnability(resume_skills):
            return round(base_weight * 0.4, 3)  # FIX A: significant reduction if foundation exists
        return round(base_weight * 0.7, 3)       # moderate reduction otherwise

    return base_weight


# ══════════════════════════════════════════════════════════════════════════════
# LOW-CODE / COMMODITY TOOL SETS
# ══════════════════════════════════════════════════════════════════════════════

LOW_CODE_TOOLS = {
    "n8n","zapier","make","bubble","webflow","airtable","notion","retool",
    "glide","appsmith","baserow","parabola","integromat","ifttt","tray.io","workato",
}
COMMODITY_TOOLS = {
    "git","github","gitlab","jira","confluence","slack","zoom","figma",
    "excel","google sheets","notion","trello","asana","linear",
}


# ══════════════════════════════════════════════════════════════════════════════
# IMPLICIT SKILL INFERENCE MAP
# ══════════════════════════════════════════════════════════════════════════════

IMPLICIT_SKILL_MAP: Dict[str, Tuple[List[str], float]] = {
    # Production / Engineering signals
    "production system":      (["Software Engineering","Backend Systems","Deployment"], 0.80),
    "production-grade":       (["Software Engineering","Backend Systems","Deployment"], 0.80),
    "scalable":               (["Software Engineering","Microservices","Cloud Architecture"], 0.70),
    "platform":               (["Backend Systems","Microservices","Software Engineering"], 0.70),
    "microservic":            (["Microservices","REST APIs","Backend Systems"], 0.85),
    "rest api":               (["REST APIs","Backend Systems","Software Engineering"], 0.90),
    "restful":                (["REST APIs","Backend Systems"], 0.90),
    "api":                    (["REST APIs","Software Engineering"], 0.75),
    "endpoint":               (["REST APIs","Backend Systems"], 0.75),
    "fastapi":                (["FastAPI","REST APIs","Backend Systems"], 0.95),
    "flask":                  (["Flask","REST APIs","Backend Systems"], 0.95),
    "django":                 (["Django","REST APIs","Backend Systems"], 0.95),
    "git":                    (["Git","Version Control"], 0.90),
    "github":                 (["Git","Version Control"], 0.90),
    "version control":        (["Git","Version Control"], 0.85),
    # Customer / Stakeholder signals
    "stakeholder":            (["Customer-Facing Work","Collaboration","Communication Skills"], 0.80),
    "client":                 (["Customer-Facing Work","Client Management"], 0.80),
    "business team":          (["Customer-Facing Work","Collaboration"], 0.75),
    "cross-functional":       (["Customer-Facing Work","Collaboration","Team Collaboration"], 0.80),
    "non-technical":          (["Customer-Facing Work","Communication Skills"], 0.85),
    "business stakeholder":   (["Customer-Facing Work","Stakeholder Management"], 0.85),
    "enterprise client":      (["Customer-Facing Work","Client Management"], 0.85),
    "global client":          (["Customer-Facing Work","Client Management"], 0.85),
    "translated.*business":   (["Customer-Facing Work","Business Analysis"], 0.80),
    # FIX C: FDE / ambiguity signals
    "ambiguous":              (["Customer-Facing Work","Adaptability","Problem Solving"], 0.75),
    "ill-defined":            (["Customer-Facing Work","Adaptability"], 0.75),
    "undefined problem":      (["Customer-Facing Work","Problem Solving"], 0.80),
    "whitespace":             (["Customer-Facing Work","Problem Solving"], 0.75),
    "0 to 1":                 (["Ownership","Customer-Facing Work"], 0.80),
    "greenfield":             (["Ownership","Software Engineering"], 0.75),
    "discovery":              (["Customer-Facing Work","Business Analysis"], 0.70),
    # LLM / GenAI signals
    "llm":                    (["LLM-Based Systems","Large Language Models","GenAI Tools"], 0.90),
    "large language":         (["LLM-Based Systems","Large Language Models"], 0.90),
    "langchain":              (["LangChain","LLM-Based Systems","GenAI Tools"], 0.95),
    "langgraph":              (["LangGraph","LLM-Based Systems","Agentic AI"], 0.95),
    "rag":                    (["Retrieval Augmented Generation","LLM-Based Systems"], 0.90),
    "retrieval augmented":    (["Retrieval Augmented Generation","LLM-Based Systems"], 0.90),
    "agentic":                (["Agentic AI","LLM-Based Systems"], 0.90),
    "ai agent":               (["Agentic AI","LLM-Based Systems"], 0.95),   # FIX D
    "multi-agent":            (["Agentic AI","LLM-Based Systems"], 0.95),   # FIX D
    "copilot agent":          (["Agentic AI","LLM-Based Systems"], 0.90),   # FIX D
    "workflow agent":         (["Agentic AI","LLM-Based Systems"], 0.90),   # FIX D
    "prompt engineer":        (["Prompt Engineering","LLM-Based Systems","GenAI Tools"], 0.85),
    "openai":                 (["OpenAI","GenAI Tools","LLM-Based Systems"], 0.90),
    "gemini":                 (["Gemini AI","GenAI Tools","LLM-Based Systems"], 0.90),
    "huggingface":            (["HuggingFace","GenAI Tools"], 0.90),
    "vector":                 (["Vector Databases","Embeddings","RAG Systems"], 0.80),
    "embedding":              (["Embeddings","Vector Databases"], 0.80),
    "fine-tun":               (["LLM Fine-Tuning","GenAI Tools"], 0.85),
    # MLOps / DevOps signals
    "ci/cd":                  (["CI/CD Pipelines","MLOps","DevOps"], 0.90),
    "docker":                 (["Docker","MLOps","Deployment"], 0.95),
    "kubernetes":             (["Kubernetes","MLOps","Cloud Architecture"], 0.90),
    "mlflow":                 (["MLOps","LLMOps","Model Monitoring"], 0.85),
    "monitoring":             (["Monitoring","MLOps","LLMOps"], 0.80),
    "deployed":               (["Deployment","MLOps","Software Engineering"], 0.75),
    "deployment":             (["Deployment","MLOps","Software Engineering"], 0.80),
    # Architecture signals
    "architect":              (["Software Architecture","System Design","Technical Leadership"], 0.85),
    "designed.*system":       (["System Design","Software Engineering","Technical Leadership"], 0.80),
    "end-to-end":             (["Software Engineering","Full-Stack Delivery","Ownership"], 0.75),
    "microservice":           (["Microservices","Backend Systems","Cloud Architecture"], 0.85),
    "cloud":                  (["Cloud Architecture","Cloud-Native Systems"], 0.70),
    "pydantic":               (["Pydantic","Python","Backend Systems"], 0.90),
    "sqlalchemy":             (["SQLAlchemy","Database","Backend Systems"], 0.85),
    "llamaindex":             (["LlamaIndex","LLM-Based Systems","RAG Systems"], 0.90),
    "llmops":                 (["LLMOps","MLOps","Model Monitoring"], 0.90),
    "model serv":             (["Model Serving","MLOps","Deployment"], 0.85),
}

CONFIDENCE_TIERS = {
    (0.85, 1.0):  ("Explicit/Strong",  1.00),
    (0.70, 0.85): ("Strong Inferred",  0.75),
    (0.0,  0.70): ("Weak Inferred",    0.50),
}

def get_confidence_tier(raw_confidence: float) -> Tuple[str, float]:
    for (low, high), (label, effective) in CONFIDENCE_TIERS.items():
        if low <= raw_confidence <= high:
            return label, effective
    return "Weak Inferred", 0.50

def scan_implicit_skills(resume_text: str) -> Dict[str, Tuple[float, str]]:
    text_lower = resume_text.lower()
    found: Dict[str, Tuple[float, str]] = {}
    for pattern, (skills, raw_conf) in IMPLICIT_SKILL_MAP.items():
        if re.search(pattern, text_lower):
            tier_label, eff_conf = get_confidence_tier(raw_conf)
            for skill in skills:
                if skill not in found or found[skill][0] < eff_conf:
                    found[skill] = (eff_conf, tier_label)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# FIX C: ROLE ARCHETYPE CLUSTERING — FDE + Specialized role boosts
# ══════════════════════════════════════════════════════════════════════════════

ROLE_CLUSTERS: Dict[str, List[str]] = {
    "AI Engineer":              ["ai engineer","ml engineer","machine learning engineer",
                                  "applied ai engineer","ai solutions engineer",
                                  "senior ai engineer","staff ai engineer"],
    "Forward Deployed Engineer":["forward deployed","fde","field engineer","solutions engineer",
                                  "customer engineer","deployment engineer","implementation engineer",
                                  "forward deployed ai","forward deployed engineer"],
    "Data Scientist":           ["data scientist","senior data scientist","principal data scientist",
                                  "ml researcher","ai researcher","applied scientist"],
    "AI Architect":             ["ai architect","ml architect","principal engineer","staff engineer",
                                  "solutions architect","enterprise architect","ai systems architect"],
    "GenAI Engineer":           ["genai engineer","llm engineer","generative ai engineer",
                                  "prompt engineer","ai product engineer","foundation model engineer",
                                  "copilot engineer"],
    "Platform Engineer":        ["platform engineer","mlops engineer","devops engineer",
                                  "sre","infrastructure engineer","cloud engineer"],
    "Software Engineer":        ["software engineer","backend engineer","fullstack engineer",
                                  "python engineer","senior python engineer","api engineer"],
    "Product Manager":          ["product manager","ai product manager","technical product manager",
                                  "program manager"],
    "Data Engineer":            ["data engineer","analytics engineer","pipeline engineer",
                                  "etl engineer","big data engineer"],
}

# FIX C: Signals that prove FDE / deployment fit even without exact title
FDE_SIGNALS = [
    "stakeholder", "client", "enterprise", "deployed", "customer",
    "business workflow", "non-technical", "cross-functional",
    "ambiguous", "greenfield", "0 to 1", "discovery",
]

def cluster_role(role_text: str) -> str:
    rl = role_text.lower()
    for cluster, variants in ROLE_CLUSTERS.items():
        if any(v in rl for v in variants):
            return cluster
    return "General"

def role_cluster_boost(resume_role: str, jd_role: str,
                        resume_text: str = "", resume_skills: List[str] = None) -> float:
    """
    FIX C: Enhanced boost with FDE-specific signal detection.
    Same cluster → +8. Adjacent cluster → +4.
    FDE role + FDE signals in resume → additional +6.
    """
    resume_skills = resume_skills or []
    ADJACENT = {
        "AI Engineer":               ["GenAI Engineer","Data Scientist","AI Architect","Software Engineer",
                                       "Forward Deployed Engineer"],
        "GenAI Engineer":            ["AI Engineer","AI Architect","Software Engineer","Forward Deployed Engineer"],
        "AI Architect":              ["AI Engineer","GenAI Engineer","Platform Engineer"],
        "Data Scientist":            ["AI Engineer","Data Engineer"],
        "Software Engineer":         ["AI Engineer","GenAI Engineer","Platform Engineer","Forward Deployed Engineer"],
        "Forward Deployed Engineer": ["AI Engineer","GenAI Engineer","Software Engineer","AI Architect"],
    }
    rc = cluster_role(resume_role)
    jc = cluster_role(jd_role)

    base_boost = 0.0
    if rc == jc:
        base_boost = 8.0
    elif jc in ADJACENT.get(rc, []):
        base_boost = 4.0

    # FIX C: FDE archetype boost — if JD is FDE-type and resume has FDE signals
    fde_boost = 0.0
    if jc == "Forward Deployed Engineer" or "forward deployed" in jd_role.lower():
        rt_lower = resume_text.lower()
        fde_hits = sum(1 for sig in FDE_SIGNALS if sig in rt_lower)
        if fde_hits >= 3:
            fde_boost = 6.0   # strong FDE signal match
        elif fde_hits >= 1:
            fde_boost = 3.0   # partial FDE signal match

    return round(min(15.0, base_boost + fde_boost), 1)


# ══════════════════════════════════════════════════════════════════════════════
# JD INTENT + ADAPTIVE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

JD_INTENT_PROFILES = {
    "research":    {"skill":0.30,"experience":0.20,"role":0.35,"ai":0.15,
                    "label":"Research Role — domain depth weighted higher"},
    "engineering": {"skill":0.40,"experience":0.25,"role":0.25,"ai":0.10,
                    "label":"Engineering Role — skill stack weighted higher"},
    "platform":    {"skill":0.35,"experience":0.30,"role":0.25,"ai":0.10,
                    "label":"Platform/Infra Role — experience weighted higher"},
    "product":     {"skill":0.25,"experience":0.25,"role":0.35,"ai":0.15,
                    "label":"Product/AI Role — role alignment weighted higher"},
    "genai":       {"skill":0.35,"experience":0.20,"role":0.30,"ai":0.15,
                    "label":"GenAI Role — AI judgment weighted higher"},
    "fde":         {"skill":0.25,"experience":0.25,"role":0.35,"ai":0.15,
                    "label":"FDE Role — stakeholder + deployment weighted higher"},  # FIX C
    "general":     {"skill":0.35,"experience":0.25,"role":0.30,"ai":0.10,
                    "label":"General Role — balanced weights"},
}

INTENT_KEYWORDS = {
    "research":    ["research","paper","publication","phd","scientist","experiment","academic"],
    "engineering": ["software engineer","backend","api","system design","microservice","node","django"],
    "platform":    ["platform","infrastructure","devops","sre","kubernetes","cloud","scale","reliability"],
    "product":     ["product","roadmap","stakeholder","business","strategy","manager","customer"],
    "genai":       ["genai","llm","langchain","rag","agentic","prompt","fine-tune","foundation model"],
    "fde":         ["forward deployed","customer-facing","enterprise client","field","implementation",
                    "business workflow","non-technical stakeholder"],   # FIX C
}

def detect_jd_intent(jd_text: str, jd_data: Dict) -> Dict:
    tl = jd_text.lower()
    scores = {intent: sum(1 for kw in kws if kw in tl) for intent, kws in INTENT_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    profile = JD_INTENT_PROFILES[best if scores[best] > 0 else "general"].copy()
    profile["intent"] = best if scores[best] > 0 else "general"
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# IMPACT SCORING
# ══════════════════════════════════════════════════════════════════════════════

IMPACT_PATTERNS = [
    r'\d+x\b', r'\d+%', r'\$[\d,]+[km]?\b',
    r'\bimproved\b',r'\breduced\b',r'\bincreased\b',r'\boptimi[sz]ed?\b',
    r'\bscaled?\b',r'\bdeployed\b',r'\bdelivered\b',r'\bproductivity\b',
    r'\bsaved\b',r'\benabled\b',r'\baccelerated\b',r'\bautomated\b',
]

def score_impact(experience_section: str) -> Dict:
    text = experience_section.lower(); hits = []
    for pat in IMPACT_PATTERNS: hits.extend(re.findall(pat, text))
    quant  = len(re.findall(r'\d+[x%]|\$[\d,]+', text))
    qualit = len(hits) - quant
    return {"impact_score": min(15, quant*3+qualit*1),
            "hits": list(set(hits))[:10],
            "quantified_count": quant, "qualitative_count": qualit}


# ══════════════════════════════════════════════════════════════════════════════
# LEADERSHIP INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

LEADERSHIP_SIGNALS = [
    "led ","managed ","owned ","architected","designed system","delivered ","mentored",
    "headed ","directed ","oversaw","responsible for team","built team","grew team",
    "principal","staff engineer","tech lead","technical lead",
]

def infer_leadership_from_text(resume_text: str) -> bool:
    tl = resume_text.lower()
    return any(sig in tl for sig in LEADERSHIP_SIGNALS)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION WEIGHTS + CLEANING
# ══════════════════════════════════════════════════════════════════════════════

SECTION_EMBED_WEIGHT  = {"Experience":0.75,"Projects":0.65,"Skills":0.60,"Summary":0.60,"Profile":0.60,"General":0.50}
SECTION_KW_WEIGHT     = {"Experience":0.25,"Projects":0.35,"Skills":0.40,"Summary":0.40,"Profile":0.40,"General":0.50}
SECTION_DISPLAY_BOOST = {"Experience":1.5,"Projects":1.3,"Skills":1.2,"Summary":1.0,"Profile":1.0,"General":0.5}
LOW_SIGNAL_SECTIONS   = {"education","certifications"}

def clean_section_text(t: str) -> str:
    t = re.sub(r'[^a-zA-Z0-9\s\.\,\-\/\+\#]', ' ', t)
    return re.sub(r'\s+', ' ', t).lower().strip()

def sanitize_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ResumeSchema(BaseModel):
    name:             str  = "Not Available"
    email:            str  = "Not Available"
    phone:            str  = "Not Available"
    location:         str  = "Not Available"
    experience_years: int  = 0
    current_role:     str  = "Not Available"
    current_company:  str  = "Not Available"
    skills:           List[str] = Field(default_factory=list)
    education:        str  = "Not Available"
    companies:        List[str] = Field(default_factory=list)
    job_titles:       List[str] = Field(default_factory=list)
    linkedin_url:     str  = "Not Available"
    certifications:   List[str] = Field(default_factory=list)
    summary:          str  = "Not Available"
    domains:          List[str] = Field(default_factory=list)

class JDSchema(BaseModel):
    role_title:                str  = "Unknown Role"
    seniority:                 str  = "Mid"
    domain:                    str  = "General"
    must_have_skills:          List[str] = Field(default_factory=list)
    good_to_have_skills:       List[str] = Field(default_factory=list)
    experience_required_years: int  = 0
    experience_max_years:      Optional[int] = None
    experience_description:    str  = ""
    tools_and_frameworks:      List[str] = Field(default_factory=list)
    responsibilities_summary:  str  = ""
    key_requirements:          List[str] = Field(default_factory=list)

class AIJudgmentSchema(BaseModel):
    ai_judgment_score:  float = Field(ge=50, le=95, default=70.0)
    key_strengths:      List[str] = Field(default_factory=list)
    key_risks:          List[str] = Field(default_factory=list)
    verdict_reason:     str  = ""

class EvaluationSchema(BaseModel):
    strengths:           List[str] = Field(default_factory=list)
    concerns:            List[str] = Field(default_factory=list)
    interview_questions: List[Dict[str, str]] = Field(default_factory=list)
    risk_level:          str = "Medium"
    risk_reason:         str = ""


def parse_llm_json(content: str, schema: Type[T], retries: int = 3) -> T:
    last_err = None
    for attempt in range(retries):
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = lines[1:] if lines[0].strip().startswith("```") else lines
                lines = lines[:-1] if lines and lines[-1].strip()=="```" else lines
                cleaned = "\n".join(lines).strip()
            return schema(**json.loads(cleaned))
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                try: return schema(**json.loads(m.group()))
                except: pass
            if attempt < retries-1: time.sleep(0.5)
    logger.warning(f"parse_llm_json defaults: {last_err}")
    return schema()

def timed_step(name: str):
    def dec(func):
        def wrapper(*args, **kwargs):
            start = time.time(); result = func(*args, **kwargs)
            logger.info(f"[TIMING] {name} → {round((time.time()-start)*1000,1)} ms")
            return result
        return wrapper
    return dec


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=10000)
def cached_embedding(text: str) -> tuple:
    return tuple(np.array(embeddings_model.embed_query(text), dtype=np.float32))

def get_embedding(text: str) -> np.ndarray:
    return np.array(cached_embedding(text), dtype=np.float32)

def cosine_sim(a, b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))


# ══════════════════════════════════════════════════════════════════════════════
# SKILL INTELLIGENCE LAYER
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_MAP: Dict[str, List[str]] = {
    "llm-based systems":      ["llm","large language","langchain","langgraph","openai","gemini","rag"],
    "genai tools":            ["openai","gemini","huggingface","langchain","rag","llm","generative ai"],
    "rest apis":              ["rest api","restful","fastapi","flask","django","endpoint","api"],
    "microservices":          ["microservice","microservic","platform","scalable","backend","service"],
    "cloud-native systems":   ["cloud","aws","azure","gcp","docker","kubernetes","serverless"],
    "software engineering":   ["production system","production-grade","scalable","platform","deployed",
                                "end-to-end","architected","designed","built"],
    "backend systems":        ["backend","api","endpoint","fastapi","flask","django","production"],
    "customer-facing work":   ["stakeholder","client","cross-functional","business team","enterprise client",
                               "non-technical","global client"],
    "llmops":                 ["mlops","monitoring","mlflow","deployment","ci/cd","model serv"],
    "git":                    ["git","github","gitlab","version control"],
    "pydantic":               ["pydantic","fastapi","python","schema","validation"],
    "sqlalchemy":             ["sqlalchemy","database","sql","orm","postgresql","mysql"],
    "llamaindex":             ["llamaindex","llama index","rag","vector","retrieval","langchain"],
    "vector databases":       ["vector","faiss","pinecone","chroma","weaviate","embedding store"],
    "embeddings":             ["embedding","vector","semantic search","dense retrieval"],
    "model training":         ["model development","training pipeline","fit model","end-to-end pipeline"],
    "model validation":       ["cross validation","evaluation","metrics","benchmarking","model evaluation"],
    "deployment":             ["deployed","production","serving","endpoint","api","docker","ci/cd","aws"],
    "monitoring":             ["monitoring","logging","observability","metrics","alerting","mlflow"],
    # FIX D: Agentic AI concept map expanded
    "agentic ai":             ["ai agent","ai agents","multi-agent","agentic","agent workflow",
                               "copilot agent","workflow agent","autonomous agent","agent orchestration",
                               "langgraph","langchain agent"],
    # FIX A: Platform tool concept maps
    "microsoft power platform":  ["power apps","power automate","power bi","dynamics","sharepoint"],
    "microsoft copilot studio":  ["copilot studio","copilot agent","ms copilot","azure ai","azure openai"],
}

SKILL_CLUSTERS: Dict[str, List[str]] = {
    "Distributed Computing":          ["Apache Spark","PySpark","Dask","Ray","Hadoop","Kafka"],
    "Deep Learning":                  ["PyTorch","TensorFlow","Keras","JAX","MXNet"],
    "MLOps":                          ["CI/CD Pipelines","Docker","MLflow","Kubeflow","Airflow","Deployment","LLMOps"],
    "LLMOps":                         ["MLOps","CI/CD Pipelines","Docker","Model Monitoring","Deployment","MLflow"],
    "LLM-Based Systems":              ["LangChain","LangGraph","Agentic AI","OpenAI","HuggingFace","Gemini AI",
                                        "Large Language Models","Retrieval Augmented Generation","LlamaIndex"],
    "GenAI Tools":                    ["OpenAI","HuggingFace","Gemini AI","LangChain","LangGraph","LlamaIndex",
                                        "Agentic AI","Prompt Engineering","Large Language Models"],
    "Large Language Models":          ["LangChain","LangGraph","Agentic AI","OpenAI","HuggingFace","Gemini AI"],
    "Retrieval Augmented Generation": ["LangChain","LangGraph","FAISS","Pinecone","Chroma","LlamaIndex","Vector Databases"],
    "LlamaIndex":                     ["LangChain","LangGraph","Retrieval Augmented Generation","Vector Databases","RAG Systems"],
    "REST APIs":                      ["FastAPI","Flask","Django","Backend Systems","Microservices","API"],
    "Microservices":                  ["Backend Systems","REST APIs","FastAPI","Flask","Django","Cloud-Native Systems"],
    "Cloud-Native Systems":           ["Amazon Web Services","Google Cloud Platform","Azure OpenAI","Docker","Kubernetes","Microservices"],
    "Software Engineering":           ["Python","Backend Systems","REST APIs","Microservices","Deployment","Git"],
    "Backend Systems":                ["REST APIs","FastAPI","Flask","Django","Python","Microservices","Software Engineering"],
    "Statistical Analysis":           ["Regression","Classification","Clustering","Hypothesis Testing","SciPy"],
    "Cloud Computing":                ["Amazon Web Services","Google Cloud Platform","Azure OpenAI","Docker","Kubernetes"],
    "Natural Language Processing":    ["NLTK","spaCy","Transformer","BERT","Word2Vec","LSTM"],
    "Machine Learning":               ["Scikit Learn","XGBoost","LightGBM","Supervised Learning","Unsupervised Learning"],
    "Data Engineering":               ["SQL","Pandas","NumPy","Apache Spark","Airflow","dbt"],
    "Application Development":        ["JavaScript","TypeScript","Node.js","React","Django","Flask","FastAPI","REST APIs"],
    "Technical Leadership":           ["Leadership","Team Management","Mentoring","Principal","Staff Engineer","Tech Lead"],
    "Git":                            ["GitHub","GitLab","Version Control","CI/CD Pipelines"],
    "Pydantic":                       ["Python","FastAPI","Backend Systems","REST APIs"],
    "SQLAlchemy":                     ["SQL","Database Stack","Python","Backend Systems"],
    # FIX D: Agentic AI cluster expanded
    "Agentic AI":                     ["LangGraph","LangChain","Multi-Agent Systems","AI Agents",
                                        "Workflow Agents","Copilot Agents","Agent Orchestration"],
    # FIX A: Microsoft platform cluster
    "Microsoft Power Platform":       ["Power Apps","Power Automate","Power BI","Dynamics 365",
                                        "Microsoft Copilot Studio","SharePoint"],
    "Microsoft Copilot Studio":       ["Microsoft Power Platform","Azure OpenAI","Copilot Agents",
                                        "Agentic AI","LLM-Based Systems"],
}

CLUSTER_REVERSE: Dict[str, str] = {}
for _h, _ms in SKILL_CLUSTERS.items():
    for _m in _ms: CLUSTER_REVERSE[_m.lower()] = _h

SKILL_GROUPS: Dict[str, List[str]] = {
    "python_ecosystem": ["Python","Pandas","NumPy","Scikit Learn","TensorFlow","Keras","PyTorch","SciPy"],
    "llm_ecosystem":    ["Large Language Models","LangChain","LangGraph","Retrieval Augmented Generation",
                         "LLM-Based Systems","GenAI Tools","Agentic AI","Prompt Engineering",
                         "HuggingFace","OpenAI","Gemini AI"],
    "cloud":            ["Amazon Web Services","Google Cloud Platform","Azure OpenAI","Docker",
                         "CI/CD Pipelines","Cloud-Native Systems"],
    "ml_core":          ["Machine Learning","Supervised Learning","Unsupervised Learning","Deep Learning"],
    "statistics":       ["Statistical Analysis","Regression","Classification","Clustering"],
    "web_stack":        ["JavaScript","TypeScript","Node.js","React","Django","Flask","FastAPI",
                         "REST APIs","Microservices"],
    "backend_stack":    ["Python","FastAPI","Flask","Django","REST APIs","Backend Systems",
                         "Software Engineering","Pydantic","SQLAlchemy"],
    "communication":    ["Communication Skills","Stakeholder Management","Customer-Facing Work","Presentation Skills"],
    "leadership_mgmt":  ["Technical Leadership","Team Management","Leadership","Mentoring"],
    "mlops_deploy":     ["MLOps","CI/CD Pipelines","Docker","Deployment","Monitoring","LLMOps"],
    "git_vcs":          ["Git","GitHub","GitLab","Version Control"],
    # FIX D: Agentic group
    "agentic_systems":  ["Agentic AI","LangGraph","LangChain","Multi-Agent Systems","AI Agents",
                         "Workflow Agents","Agent Orchestration"],
    # FIX A: Microsoft platform group
    "ms_platform":      ["Microsoft Power Platform","Microsoft Copilot Studio","Azure OpenAI",
                         "Power Apps","Power Automate","Power BI"],
}

SKILL_IMPORTANCE: Dict[str, Tuple[float, str]] = {
    "Python":                         (1.0,"critical"),
    "Large Language Models":          (1.0,"critical"),
    "LLM-Based Systems":              (1.0,"critical"),
    "Generative AI":                  (1.0,"critical"),
    "GenAI Tools":                    (1.0,"critical"),
    "LangChain":                      (1.0,"critical"),
    "LangGraph":                      (1.0,"critical"),
    "Agentic AI":                     (1.0,"critical"),
    "Retrieval Augmented Generation": (1.0,"critical"),
    "FastAPI":                        (0.9,"critical"),
    "REST APIs":                      (0.9,"critical"),
    "Microservices":                  (0.8,"important"),
    "Software Engineering":           (0.8,"important"),
    "Backend Systems":                (0.8,"important"),
    "Cloud-Native Systems":           (0.8,"important"),
    "Machine Learning":               (0.8,"important"),
    "Deep Learning":                  (0.8,"important"),
    "Natural Language Processing":    (0.8,"important"),
    "Distributed Computing":          (0.8,"important"),
    "Technical Leadership":           (0.8,"important"),
    "LLMOps":                         (0.7,"important"),
    "MLOps":                          (0.7,"important"),
    "Amazon Web Services":            (0.7,"important"),
    "Google Cloud Platform":          (0.7,"important"),
    "Azure OpenAI":                   (0.7,"important"),
    "Deployment":                     (0.7,"important"),
    "LlamaIndex":                     (0.7,"important"),
    "Pydantic":                       (0.6,"important"),
    "SQLAlchemy":                     (0.6,"important"),
    "Flask":                          (0.6,"important"),
    "Docker":                         (0.6,"important"),
    "CI/CD Pipelines":                (0.6,"important"),
    "Statistical Analysis":           (0.6,"important"),
    "Team Management":                (0.6,"important"),
    "Customer-Facing Work":           (0.6,"important"),
    "Git":                            (0.5,"important"),
    "Monitoring":                     (0.5,"important"),
    # FIX A: Platform tools — important but not blocking if foundation exists
    "Microsoft Power Platform":       (0.6,"important"),
    "Microsoft Copilot Studio":       (0.6,"important"),
    # Optional — never penalise
    "R":                              (0.0,"optional"),
    "Communication Skills":           (0.0,"optional"),
    "Team Collaboration":             (0.0,"optional"),
    "Problem-solving Mindset":        (0.0,"optional"),
    "Attention to Detail":            (0.0,"optional"),
    "Leadership":                     (0.0,"optional"),
    "Agile":                          (0.0,"optional"),
    "Passion for AI":                 (0.0,"optional"),
    "AI passion":                     (0.0,"optional"),
    "Ownership":                      (0.0,"optional"),
    "Collaboration":                  (0.0,"optional"),
    "Adaptability":                   (0.0,"optional"),
    "Problem Solving":                (0.0,"optional"),
}

INFERRED_SKILLS: List[Tuple[List[str], str, float]] = [
    (["stakeholder","business stakeholder"],     "Communication Skills",    0.70),
    (["cross-functional","collaborated"],        "Team Collaboration",      0.70),
    (["analysis","analytical","insights"],       "Analytical Skills",       0.60),
    (["deployed","deployment","production"],     "MLOps",                   0.60),
    (["agile","sprint","scrum"],                 "Agile",                   0.80),
    (["led team","managed team","mentored"],     "Technical Leadership",    0.85),
    (["architected","designed system","principal","tech lead","staff engineer"],
                                                  "Technical Leadership",   0.90),
    (["python"],                                 "Python",                  0.95),
    (["langchain","lang chain"],                 "LangChain",               0.95),
    (["langgraph","lang graph"],                 "LangGraph",               0.95),
    (["retrieval augmented","rag pipeline"],     "Retrieval Augmented Generation", 0.90),
    (["large language","llm","language model"],  "Large Language Models",   0.90),
    # FIX D: Expanded agentic inference
    (["agentic","multi-agent","agent workflow",
      "ai agent","copilot agent","workflow agent",
      "autonomous agent","agent orchestration"],  "Agentic AI",             0.95),
    (["docker"],                                 "Docker",                  0.95),
    (["aws","amazon web services"],              "Amazon Web Services",     0.95),
    (["azure"],                                  "Azure OpenAI",            0.85),
    (["gcp","google cloud"],                     "Google Cloud Platform",   0.95),
    (["spark","pyspark"],                        "Apache Spark",            0.90),
    (["distributed","large-scale pipeline"],     "Distributed Computing",   0.65),
    (["node.js","nodejs","express"],             "Node.js",                 0.90),
    (["javascript","js "],                       "JavaScript",              0.90),
    (["rest api","restful","graphql","endpoint"],"REST APIs",               0.85),
    (["monitoring","logging","observability"],   "Monitoring",              0.75),
    (["ci/cd","pipeline","mlflow"],              "Deployment",              0.75),
    (["fastapi","fast api"],                     "FastAPI",                 0.95),
    (["pydantic"],                               "Pydantic",                0.95),
    (["sqlalchemy","sql alchemy"],               "SQLAlchemy",              0.95),
    (["llamaindex","llama index"],               "LlamaIndex",              0.95),
    (["llmops","llm ops"],                       "LLMOps",                  0.90),
    (["git","github","gitlab"],                  "Git",                     0.90),
    (["microservic","service mesh"],             "Microservices",           0.85),
    (["cloud native","cloud-native"],            "Cloud-Native Systems",    0.85),
    (["software engineer","production system","production-grade"],
                                                  "Software Engineering",   0.80),
    (["stakeholder","client","enterprise client","global client","non-technical"],
                                                  "Customer-Facing Work",   0.80),
    # FIX A: Microsoft platform inference
    (["power platform","power apps","power automate","power bi"],
                                                  "Microsoft Power Platform", 0.90),
    (["copilot studio","copilot agent","ms copilot"],
                                                  "Microsoft Copilot Studio", 0.90),
]

SKILL_GAP_HINTS: Dict[str, Dict] = {
    "LLM-Based Systems":              {"resource":"LangChain docs + Anthropic cookbook","impact_pct":8,
                                        "context":"core — build RAG/agent pipelines"},
    "GenAI Tools":                    {"resource":"OpenAI API docs + HuggingFace courses (free)","impact_pct":8,
                                        "context":"hands-on with OpenAI/Gemini/Claude APIs"},
    "LlamaIndex":                     {"resource":"LlamaIndex docs (llamaindex.ai) — 1 day quickstart","impact_pct":6,
                                        "context":"alternative RAG framework to LangChain"},
    "LLMOps":                         {"resource":"Made With ML MLOps + LangSmith tracing docs","impact_pct":7,
                                        "context":"monitoring LLM pipelines in production"},
    "REST APIs":                      {"resource":"FastAPI official tutorial (fastapi.tiangolo.com)","impact_pct":6,
                                        "context":"build production ML/AI APIs with Python"},
    "FastAPI":                        {"resource":"FastAPI official tutorial — 2 hours to working API","impact_pct":7,
                                        "context":"fastest Python framework for AI service APIs"},
    "Microservices":                  {"resource":"Microservices with FastAPI + Docker guide","impact_pct":6,
                                        "context":"architecture pattern for scalable AI systems"},
    "Cloud-Native Systems":           {"resource":"Docker + Kubernetes for ML Engineers — fast.ai guide","impact_pct":6,
                                        "context":"deploy AI systems on cloud-native infrastructure"},
    "Pydantic":                       {"resource":"Pydantic docs (pydantic.dev) — 2 hour quickstart","impact_pct":4,
                                        "context":"data validation for FastAPI and LLM outputs"},
    "SQLAlchemy":                     {"resource":"SQLAlchemy 2.0 quickstart + FastAPI DB integration","impact_pct":4,
                                        "context":"ORM for database-backed AI applications"},
    "Git":                            {"resource":"Pro Git book (free)","impact_pct":3,
                                        "context":"version control for AI project delivery"},
    "Software Engineering":           {"resource":"Clean Code + System Design Primer (github)","impact_pct":5,
                                        "context":"production-grade code for AI systems"},
    "Customer-Facing Work":           {"resource":"Stakeholder communication for engineers — LeadDev articles","impact_pct":4,
                                        "context":"translate AI solutions to business impact"},
    "Technical Leadership":           {"resource":"Staff Engineer book (staffeng.com) + system design primer","impact_pct":5,
                                        "context":"leading AI engineering teams"},
    "LangGraph":                      {"resource":"LangGraph Agentic Workflows — LangChain Academy (free)","impact_pct":9,
                                        "context":"multi-step agent workflows"},
    "LangChain":                      {"resource":"LangChain docs + Build LLM Apps course","impact_pct":10,
                                        "context":"orchestrating LLM-powered applications"},
    "Agentic AI":                     {"resource":"LangGraph multi-agent guide + Anthropic cookbook","impact_pct":10,
                                        "context":"autonomous AI agent systems — direct match to 'AI Agents'"},
    "Retrieval Augmented Generation": {"resource":"RAG from Scratch — LangChain blog series","impact_pct":8,
                                        "context":"grounding LLM responses with your data"},
    "Amazon Web Services":            {"resource":"AWS Cloud Practitioner — AWS Skill Builder (free)","impact_pct":6,
                                        "context":"cloud deployment of ML/AI services"},
    "MLOps":                          {"resource":"Made With ML — MLOps course (madewithml.com)","impact_pct":7,
                                        "context":"production ML pipelines and monitoring"},
    "Deployment":                     {"resource":"FastAPI + Docker + GitHub Actions deployment guide","impact_pct":5,
                                        "context":"deploying ML models as scalable APIs"},
    # FIX A: Platform tool learning hints
    "Microsoft Power Platform":       {"resource":"Microsoft Power Platform learning path (learn.microsoft.com — free)","impact_pct":3,
                                        "context":"learnable in 1–2 weeks given Azure/API foundation"},
    "Microsoft Copilot Studio":       {"resource":"Copilot Studio docs (microsoft.com/copilot-studio)","impact_pct":3,
                                        "context":"learnable quickly with existing LLM/Azure background"},
}


def concept_match(jd_skill: str, resume_text: str, resume_skills: List[str]) -> Tuple[bool, float, str]:
    jd_lower = canonical(jd_skill).lower()
    phrases  = CONCEPT_MAP.get(jd_lower, [])
    if not phrases: return False, 0.0, ""
    tl = resume_text.lower()
    for phrase in phrases:
        if phrase in tl:
            return True, 0.75, f"concept: '{phrase}'"
        if any(phrase in rs.lower() for rs in resume_skills):
            return True, 0.75, f"concept-skill: '{phrase}'"
    return False, 0.0, ""

def implicit_match(jd_skill: str, implicit_map: Dict[str, Tuple[float, str]]) -> Tuple[bool, float, str]:
    js_lower = canonical(jd_skill).lower()
    for impl_skill, (eff_conf, tier_label) in implicit_map.items():
        if (js_lower == impl_skill.lower()
                or js_lower in impl_skill.lower()
                or impl_skill.lower() in js_lower):
            return True, eff_conf, f"implicit ({tier_label})"
    return False, 0.0, ""

def cluster_head(skill: str) -> Optional[str]:
    return CLUSTER_REVERSE.get(canonical(skill).lower())

def skills_match_full(rs: str, js: str, resume_text: str="",
                       resume_skills: List[str]=None) -> Tuple[bool, float, str]:
    resume_skills = resume_skills or []
    r, j = canonical(rs).lower(), canonical(js).lower()
    if r == j: return True, 1.00, "exact"
    prog_langs = ["Python","JavaScript","Java","TypeScript","Scala","Go","Rust","C++","Node.js"]
    if any(p in j for p in ["programming","language","ai programming"]) and canonical(rs) in prog_langs:
        return True, 1.00, "programming-concept"
    def grp(x):
        cx = canonical(x).lower()
        for g, members in SKILL_GROUPS.items():
            if any(cx == m.lower() for m in members): return g
        return None
    if grp(rs) and grp(rs) == grp(js): return True, 0.90, "group"
    if r in j or j in r: return True, 0.85, "substring"
    jd_cluster = SKILL_CLUSTERS.get(canonical(js))
    if jd_cluster and any(canonical(rs).lower()==m.lower() or r in m.lower() for m in jd_cluster):
        return True, 0.85, "cluster"
    rs_head = cluster_head(rs)
    if rs_head and rs_head.lower() == j: return True, 0.85, "cluster-head"
    return False, 0.0, ""

def match_skills_full(resume_skills: List[str], jd_skills: List[str],
                       resume_text: str="",
                       implicit_map: Dict[str, Tuple[float, str]] = None) -> Tuple[List[Dict], List[str]]:
    implicit_map = implicit_map or {}
    matched, missing = [], []
    for js in jd_skills:
        best_credit, best_rs, best_reason = 0.0, None, ""
        for rs in resume_skills:
            m, credit, reason = skills_match_full(rs, js, resume_text, resume_skills)
            if m and credit > best_credit:
                best_credit, best_rs, best_reason = credit, rs, reason
        if best_credit == 0.0:
            cm, credit, evidence = concept_match(js, resume_text, resume_skills)
            if cm: best_credit, best_rs, best_reason = credit, js, evidence
        if best_credit == 0.0:
            im, credit, evidence = implicit_match(js, implicit_map)
            if im: best_credit, best_rs, best_reason = credit, js, evidence
        if best_rs:
            matched.append({"skill":js,"matched_by":best_rs,"credit":best_credit,"reason":best_reason})
        else:
            missing.append(js)
    return matched, missing

def skill_weight(s: str, resume_skills: List[str] = None) -> float:
    resume_skills = resume_skills or []
    base = SKILL_IMPORTANCE.get(canonical(s),(0.6,"important"))[0]
    return adjust_skill_weight_for_tool(s, base, resume_skills)

def skill_tier(s: str) -> str:
    return SKILL_IMPORTANCE.get(canonical(s),(0.6,"important"))[1]

def infer_skills_from_text(resume_text: str, existing_skills: List[str]) -> List[Dict]:
    tl = resume_text.lower(); existing_lc = {s.lower() for s in existing_skills}
    inferred = []
    for phrases, skill, confidence in INFERRED_SKILLS:
        if skill.lower() not in existing_lc:
            if any(p in tl for p in phrases):
                inferred.append({"skill":skill,"confidence":confidence,"source":"inferred"})
                existing_lc.add(skill.lower())
    return inferred

def experience_depth_multiplier(exp_years: int) -> float:
    if exp_years <= 0: return 0.7
    return round(min(1.5, 1+log(exp_years)/3), 3)

def weighted_coverage_v15(matched_with_credit: List[Dict], all_jd_skills: List[str],
                           inferred_map: Dict[str,float], exp_years: int,
                           resume_skills: List[str] = None) -> Tuple[float, Dict]:
    resume_skills = resume_skills or []
    scoring = [s for s in all_jd_skills if skill_tier(s)!="optional"]
    total   = sum(skill_weight(s, resume_skills) for s in scoring)
    if total == 0: return 100.0, {}
    earned, breakdown = 0.0, {}
    for item in matched_with_credit:
        s = item["skill"]; credit = item["credit"]; rs = item["matched_by"]
        if skill_tier(s) == "optional": continue
        w          = skill_weight(s, resume_skills)   # FIX A: pass resume_skills
        confidence = inferred_map.get(s.lower(), inferred_map.get(rs.lower(), 1.0))
        effective  = w * credit * confidence
        earned    += effective
        breakdown[s] = {"weight":w,"credit":credit,"confidence":confidence,
                         "effective":round(effective,3),"matched_by":rs,"reason":item.get("reason","exact")}
    raw_score  = (earned/total)*100
    depth_mult = experience_depth_multiplier(exp_years)
    final      = min(100, round(raw_score*depth_mult, 1))
    return final, {"breakdown":breakdown,"depth_multiplier":depth_mult,
                    "raw_score":round(raw_score,1),"final_score":final}

def align_score_to_coverage(raw_score: float, coverage: float, threshold: float) -> float:
    if coverage >= threshold: return raw_score
    return round(raw_score * (0.9 + 0.1*(coverage/100)), 1)

def generate_recommendations_v15(missing_skills: List[str], matched_with_credit: List[Dict],
                                   current_score: float, jd_intent: str,
                                   resume_skills: List[str] = None) -> List[Dict]:
    resume_skills = resume_skills or []
    recs = []
    for s in missing_skills:
        c = canonical(s); tier = skill_tier(s); w = skill_weight(s, resume_skills)
        if tier == "optional" or w < 0.5: continue
        if s.lower() in LOW_CODE_TOOLS or s.lower() in COMMODITY_TOOLS: continue
        # FIX A: Platform tools flagged as learnable, not blocking
        is_platform = s.lower() in PLATFORM_SPECIFIC_TOOLS
        hint     = SKILL_GAP_HINTS.get(c)
        impact   = hint["impact_pct"] if hint else (6 if tier=="critical" else 3)
        context  = hint.get("context","") if hint else ""
        resource = hint["resource"] if hint else f"Official {c} documentation + hands-on project"
        if is_platform and has_platform_learnability(resume_skills):
            impact = max(1, impact // 2)
            context = f"[Learnable — you have adjacent foundation] {context}"
        if jd_intent=="genai" and tier=="critical":    impact = min(15, impact+3)
        if jd_intent=="engineering" and c in ["Node.js","JavaScript","Application Development"]:
            impact = min(15, impact+2)
        if jd_intent=="fde" and c in ["Customer-Facing Work","Technical Leadership","Deployment"]:
            impact = min(15, impact+2)  # FIX C: FDE-specific boost
        head = cluster_head(s)
        if head and any(m["skill"].lower()==head.lower() for m in matched_with_credit):
            impact = max(1, impact//2)
        recs.append({"skill":c,"tier":tier,"weight":w,"resource":resource,
                     "context":context,"impact_pct":impact,
                     "learnable": is_platform and has_platform_learnability(resume_skills),
                     "new_score":min(100,round(current_score+impact,1))})
    return sorted(recs, key=lambda x:x["impact_pct"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# JD QUALITY + THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_KW  = ["python","machine learning","deep learning","nlp","llm","ai","data science",
               "cloud","aws","azure","gcp","sql","api","backend","devops","tensorflow",
               "pytorch","langchain","rag","agentic","mlops","docker","spark","distributed",
               "node","javascript","react","fastapi","django","monitoring","deployment"]
GENERIC_PH = ["good communication","team player","problem solver","detail oriented",
               "fast learner","self motivated","passionate","collaborative"]

def assess_jd_quality(jd_text: str, jd_data: Dict) -> Dict:
    tl=jd_text.lower(); dh=sum(1 for kw in DOMAIN_KW if kw in tl)
    gh=sum(1 for ph in GENERIC_PH if ph in tl)
    mhc=len(jd_data.get("must_have_skills",[])); wc=len(jd_text.split())
    score=0
    if dh>=5: score+=3
    elif dh>=2: score+=1
    if mhc>=4: score+=3
    elif mhc>=2: score+=1
    if gh>=3: score-=2
    if wc>=200: score+=1
    if score>=5:   quality,label="High","Well-defined technical JD"
    elif score>=2: quality,label="Medium","Moderately specific JD"
    else:          quality,label="Low","Generic/vague JD"
    return {"quality":quality,"label":label,"domain_hits":dh,
            "screening_threshold":{"High":75,"Medium":65,"Low":55}[quality],
            "ai_weight_boost":{"High":0.0,"Medium":0.05,"Low":0.10}[quality]}

def dynamic_threshold(jd_quality: str) -> int:
    return {"High":75,"Medium":65,"Low":55}.get(jd_quality, 65)


# ══════════════════════════════════════════════════════════════════════════════
# ROLE FIT — with archetype cluster + title boost + FDE detection
# ══════════════════════════════════════════════════════════════════════════════

AI_KW   = ["ai","ml","data science","machine learning","llm","genai","deep learning"]
OPS_KW  = ["operations","project management","business analyst","hr","finance","marketing"]
TECH_KW = ["software engineer","backend","frontend","fullstack","devops","cloud","sre"]

def domain_label(text: str) -> str:
    t=text.lower()
    ai=sum(1 for k in AI_KW if k in t); ops=sum(1 for k in OPS_KW if k in t)
    tch=sum(1 for k in TECH_KW if k in t)
    if ai>=ops and ai>=tch: return "AI/ML"
    if tch>=ops and tch>ai: return "Engineering"
    if ops>ai and ops>=tch: return "Operations/Business"
    return "General"

def title_boost(resume_role: str, jd_role: str,
                resume_text: str = "", resume_skills: List[str] = None) -> float:
    resume_skills = resume_skills or []
    r_words = set(re.findall(r'\b\w{4,}\b', resume_role.lower()))
    j_words = set(re.findall(r'\b\w{4,}\b', jd_role.lower()))
    signal  = {"architect","engineer","scientist","analyst","manager","lead",
                "principal","senior","staff","data","machine","learning","intelligence",
                "deployed","forward","solutions","field"}
    hits    = (r_words & j_words) & signal
    # FIX C: pass resume_text for FDE signal detection
    cluster_b = role_cluster_boost(resume_role, jd_role, resume_text, resume_skills)
    return round(min(15.0, len(hits)*3.0 + cluster_b), 1)

def assess_role_fit(resume_data: Dict, jd_data: Dict, jd_quality: Dict,
                    resume_text: str = "") -> Dict:
    if jd_quality["quality"]=="Low":
        return {"candidate_domain":"Unknown","jd_domain":"Unknown",
                "mismatch":False,"similarity":None,"role_score":70.0,
                "note":"JD too generic — skipping role fit check","title_boost":0}
    resume_skills = resume_data.get("skills", [])
    c_text = sanitize_text(" ".join(filter(None,[
        resume_data.get("summary",""), resume_data.get("current_role",""),
        " ".join(resume_data.get("job_titles",[]))[:200],
        " ".join(resume_skills[:12])])))
    j_text = sanitize_text(" ".join(filter(None,[
        jd_data.get("role_title",""), jd_data.get("responsibilities_summary",""),
        " ".join(jd_data.get("must_have_skills",[]))])))
    c_vec=get_embedding(c_text[:2000]); j_vec=get_embedding(j_text[:2000])
    sim=round(cosine_sim(c_vec, j_vec), 3)
    # FIX C: pass resume_text into title_boost for FDE detection
    t_boost = title_boost(resume_data.get("current_role",""), jd_data.get("role_title",""),
                           resume_text, resume_skills)
    role_score = round(min(100, sim*100*1.1 + t_boost), 1)
    c_dom=domain_label(c_text); j_dom=domain_label(j_text)
    mismatch=sim<0.35
    note=(f"Role similarity: {sim:.0%} ({c_dom} vs {j_dom})"
          +(f" +{t_boost} cluster/title boost" if t_boost>0 else "")
          +(" — significant mismatch." if mismatch else " — domains aligned."))
    return {"candidate_domain":c_dom,"jd_domain":j_dom,
            "mismatch":mismatch,"similarity":sim,"role_score":role_score,
            "title_boost":t_boost,"note":note}


# ══════════════════════════════════════════════════════════════════════════════
# FIX B: SMARTER OVERQUALIFICATION LOGIC
# Evaluates: learning opportunity, tech depth of role, ownership scope
# ══════════════════════════════════════════════════════════════════════════════

TECH_DEPTH_SIGNALS = [
    "agentic","rag","langgraph","langchain","distributed","architecture",
    "production","scale","microservice","system design","llmops","mlops",
    "principal","staff","complex","cutting-edge","state of the art",
]

OWNERSHIP_SIGNALS = [
    "own","ownership","lead","end-to-end","greenfield","0 to 1",
    "build from scratch","architect","drive","responsible for",
]

LEARNING_SIGNALS = [
    "learn","grow","mentor","cutting-edge","research","novel",
    "explore","emerging","frontier","state of the art","phd",
    "publications","innovation",
]

def assess_overqualification_context(jd_text: str) -> Dict:
    """
    FIX B: Evaluate whether an overqualified candidate would still find this role interesting.
    Returns learning_score, ownership_score, tech_depth_score (0–10 each).
    """
    tl = jd_text.lower()
    tech_depth  = sum(1 for s in TECH_DEPTH_SIGNALS if s in tl)
    ownership   = sum(1 for s in OWNERSHIP_SIGNALS if s in tl)
    learning    = sum(1 for s in LEARNING_SIGNALS if s in tl)
    # Normalize to 0–10
    tech_score  = min(10, tech_depth * 2)
    own_score   = min(10, ownership * 2)
    learn_score = min(10, learning * 2)
    total       = tech_score + own_score + learn_score  # max 30
    # Classify appeal
    if total >= 18:   appeal = "High"
    elif total >= 10: appeal = "Medium"
    else:             appeal = "Low"
    return {
        "tech_depth_score":  tech_score,
        "ownership_score":   own_score,
        "learning_score":    learn_score,
        "total_appeal":      total,
        "role_appeal":       appeal,
    }

def check_overqualification(candidate_exp: int, jd_data: Dict,
                             jd_quality: Dict, jd_text: str = "") -> Dict:
    required  = jd_data.get("experience_required_years",0)
    jd_max    = jd_data.get("experience_max_years",None)
    exp_desc  = jd_data.get("experience_description","")
    rng = re.search(r'(\d+)\s*[-–to]+\s*(\d+)', exp_desc)
    if rng: required=int(rng.group(1)); jd_max=int(rng.group(2))

    if jd_quality["quality"]=="Low":
        return {"flag":"JD Too Vague","detail":"JD experience unclear.",
                "exp_score":100,"is_overqualified":False,"max_exp":jd_max,
                "discussion_points":[],"overqual_context":{}}

    overqualified = (jd_max is not None) and (candidate_exp > jd_max)
    ratio         = candidate_exp/required if required>0 else 1.0

    # FIX B: always assess role context
    overqual_context = assess_overqualification_context(jd_text) if jd_text else {}

    if overqualified:
        appeal = overqual_context.get("role_appeal","Medium")
        flag   = "Overqualified"
        detail = f"{candidate_exp} yrs vs {required}–{jd_max} yr target."
        exp_score = 100
        # FIX B: smarter discussion points based on role appeal
        if appeal == "High":
            discussion = [
                "Role has strong tech depth — confirm candidate finds it intellectually stimulating",
                "Discuss ownership scope and growth path",
                "Verify compensation expectations",
            ]
        elif appeal == "Medium":
            discussion = [
                "Verify compensation expectations",
                "Discuss growth trajectory and autonomy",
                "Confirm motivation beyond compensation",
            ]
        else:
            discussion = [
                "Retention risk — role may not offer sufficient challenge",
                "Verify compensation expectations",
                "Consider whether a senior/staff role exists",
            ]
    elif ratio>=0.8:
        flag,detail,exp_score,discussion="Good Fit",f"{candidate_exp} yrs meets requirement.",100,[]
        overqual_context = {}
    elif ratio>=0.6:
        flag,detail,exp_score,discussion="Slightly Under",f"{candidate_exp} yrs slightly below {required}.",80,[]
        overqual_context = {}
    else:
        flag,detail,exp_score,discussion="Under-experienced",f"{candidate_exp} yrs below {required}.",60,[]
        overqual_context = {}

    return {"flag":flag,"detail":detail,"exp_score":exp_score,
            "is_overqualified":overqualified,"max_exp":jd_max,
            "discussion_points":discussion,"overqual_context":overqual_context}


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def load_memory() -> Dict:
    EMPTY={"candidates":[],"patterns":{"missing":{},"matched":{}}}
    if not MEMORY_FILE.exists(): return EMPTY
    try: mem=json.loads(MEMORY_FILE.read_text())
    except: return EMPTY
    mem.setdefault("candidates",[]); mem.setdefault("patterns",{})
    p=mem["patterns"]
    if "common_missing_skills" in p and "missing" not in p: p["missing"]=p.pop("common_missing_skills")
    if "common_matched_skills" in p and "matched" not in p: p["matched"]=p.pop("common_matched_skills")
    p.setdefault("missing",{}); p.setdefault("matched",{})
    return mem

def save_memory(mem: Dict):
    try: MEMORY_FILE.write_text(json.dumps(mem, indent=2))
    except Exception as e: logger.warning(f"Memory: {e}")

def make_jd_id(role: str) -> str:
    return re.sub(r'[^a-z0-9]','_', role.lower().strip())[:40]

def store_candidate(mem, resume_data, role, match, logic_decision,
                    exp_check, jd_quality, score_audit, override=None):
    final_dec = override["decision"] if override else logic_decision.get("decision","N/A")
    record={
        "id":f"{resume_data.get('name','?')}_{int(time.time())}",
        "timestamp":datetime.now().isoformat(),
        "name":resume_data.get("name","Unknown"),"role":role,"jd_id":make_jd_id(role),
        "final_score":match.get("final_score",0),
        "ai_decision":logic_decision.get("decision","N/A"),"final_decision":final_dec,
        "override":override is not None,
        "override_note":override.get("note","") if override else "",
        "exp_flag":exp_check.get("flag","N/A"),"jd_quality":jd_quality.get("quality","N/A"),
        "experience_years":resume_data.get("experience_years",0),
        "matched_skills":[m["skill"] for m in match.get("matched_with_credit",[])],
        "missing_skills":match.get("missing_skills",[]),
        "confidence":logic_decision.get("confidence",0),
        "score_audit":score_audit,
        "impact_score":match.get("impact",{}).get("impact_score",0),
    }
    mem["candidates"].append(record)
    for s in record["missing_skills"]: mem["patterns"]["missing"][s]=mem["patterns"]["missing"].get(s,0)+1
    for s in record["matched_skills"]:  mem["patterns"]["matched"][s] =mem["patterns"]["matched"].get(s,0)+1
    save_memory(mem); return record


# ══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def extract_text_from_pdf(fb: bytes) -> Optional[str]:
    def _pdfplumber(b):
        try:
            text=""
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for p in pdf.pages:
                    t=p.extract_text()
                    if t: text+=t+"\n"
            return text.strip() or None
        except: return None
    def _pymupdf(b):
        try:
            import fitz
            return "\n".join(p.get_text("text") for p in fitz.open(stream=b,filetype="pdf")).strip() or None
        except: return None
    def _tesseract(b):
        try:
            import fitz, pytesseract
            from PIL import Image
            doc=fitz.open(stream=b,filetype="pdf"); pages=[]
            for page in doc:
                pix=page.get_pixmap(matrix=fitz.Matrix(300/72,300/72))
                t=pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))),lang="eng")
                if t.strip(): pages.append(t.strip())
            return "\n\n".join(pages).strip() or None
        except: return None
    def _easyocr(b):
        try:
            from pdf2image import convert_from_bytes; import easyocr
            reader=easyocr.Reader(["en"],gpu=False)
            pages=[" ".join(reader.readtext(np.array(img),detail=0)) for img in convert_from_bytes(b,dpi=300)]
            return "\n\n".join(p for p in pages if p).strip() or None
        except: return None
    for name, fn in [("pdfplumber",_pdfplumber),("PyMuPDF",_pymupdf),
                     ("Tesseract",_tesseract),("easyocr",_easyocr)]:
        t=fn(fb)
        if t: logger.info(f"[PDF] {name}: {len(t)} chars"); return t
    return None


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
@timed_step("Resume Parser")
def resume_parser_agent(resume_text: str) -> Dict:
    clean = sanitize_text(resume_text)
    prompt = f"""Parse this resume. Extract ALL skills with normalization.
LLM→Large Language Models, RAG→Retrieval Augmented Generation,
NLP→Natural Language Processing, AWS→Amazon Web Services,
GCP→Google Cloud Platform, GenAI→Generative AI,
Spark/PySpark→Apache Spark, Node/NodeJS→Node.js, JS→JavaScript,
"AI Agents"→Agentic AI, "Copilot Agents"→Agentic AI,
"Multi-Agent"→Agentic AI, "Workflow Agents"→Agentic AI.

Return ONLY valid JSON:
{{"name":"str","email":"str","phone":"str","location":"str",
"experience_years":number,"current_role":"str","current_company":"str",
"skills":["all normalized skills"],"education":"str",
"companies":["co"],"job_titles":["t"],"linkedin_url":"str",
"certifications":["c"],"summary":"2-3 sentence summary","domains":["domain"]}}
Resume: {clean[:6000]}"""
    data = parse_llm_json(llm.invoke([HumanMessage(content=prompt)]).content, ResumeSchema).model_dump()
    data["skills"] = normalize_skills(data["skills"])
    if infer_leadership_from_text(resume_text) and "Technical Leadership" not in data["skills"]:
        data["skills"].append("Technical Leadership")
    inferred = infer_skills_from_text(resume_text, data["skills"])
    data["inferred_skills"]  = inferred
    data["skills"]          += normalize_skills([i["skill"] for i in inferred])
    data["skills"]           = normalize_skills(data["skills"])
    data["inferred_confidence_map"] = {i["skill"].lower():i["confidence"] for i in inferred}
    implicit = scan_implicit_skills(resume_text)
    data["implicit_skill_map"] = {k:{"confidence":v[0],"tier":v[1]} for k,v in implicit.items()}
    return data

@st.cache_data(ttl=3600, show_spinner=False)
@timed_step("JD Intelligence")
def jd_intelligence_agent(jd_text: str) -> Dict:
    clean = sanitize_text(jd_text)
    prompt = f"""Convert this JD to structured criteria. Normalize skills.
Node/NodeJS→Node.js, JS→JavaScript,
"AI Agents"→Agentic AI, "Copilot Agents"→Agentic AI,
"Multi-Agent Systems"→Agentic AI.
Extract experience range. Separate BLOCKING must-haves from preferred.
Only mark as must-have what is truly blocking.

Return ONLY valid JSON:
{{"role_title":"str","seniority":"Junior/Mid/Senior/Lead/Principal",
"domain":"str","must_have_skills":["blocking skills only"],
"good_to_have_skills":["preferred not blocking"],
"experience_required_years":min,"experience_max_years":max or null,
"experience_description":"original text","tools_and_frameworks":["t"],
"responsibilities_summary":"2-3 sentences","key_requirements":["r"]}}
JD: {clean[:6000]}"""
    data = parse_llm_json(llm.invoke([HumanMessage(content=prompt)]).content, JDSchema).model_dump()
    data["must_have_skills"]    = normalize_skills(data["must_have_skills"])
    data["good_to_have_skills"] = normalize_skills(data["good_to_have_skills"])
    return data

@timed_step("Screening")
def screening_agent(resume_data, jd_data, exp_check, jd_quality,
                    matched_with_credit, missing_skills) -> Dict:
    must_have    = jd_data.get("must_have_skills",[])
    matched_mh   = [m for m in matched_with_credit if m["skill"] in must_have]
    missing_mh   = [s for s in missing_skills if s in must_have]
    inferred_map = resume_data.get("inferred_confidence_map",{})
    resume_skills = resume_data.get("skills",[])
    coverage,_   = weighted_coverage_v15(matched_mh, must_have, inferred_map,
                                          resume_data.get("experience_years",0), resume_skills)
    threshold    = jd_quality["screening_threshold"]
    exp_score    = exp_check.get("exp_score",100)
    is_overq     = exp_check.get("is_overqualified",False)
    # FIX B: use role appeal in borderline decisions
    overq_ctx    = exp_check.get("overqual_context",{})
    if coverage>=threshold and exp_score>=60:  status="Pass"
    elif coverage>=threshold*0.7:              status="Borderline"
    else:                                      status="Reject"
    borderline=""
    if status=="Borderline" and is_overq:
        appeal = overq_ctx.get("role_appeal","Medium")
        borderline = f"Overqualified (role appeal: {appeal}) — verify compensation and motivation."
    elif status=="Borderline":
        borderline = f"Coverage {coverage:.1f}% near threshold ({threshold}%)."
    reason={"Pass":   f"Meets {coverage:.1f}% weighted coverage (threshold {threshold}%).",
             "Borderline": f"{coverage:.1f}% borderline vs {threshold}% threshold.",
             "Reject": f"Only {coverage:.1f}% — below {threshold}% threshold."}[status]
    return {"status":status,"must_have_coverage_pct":round(coverage,1),
            "experience_coverage_pct":exp_score,"reason":reason,"borderline_notes":borderline,
            "met_requirements":[m["skill"] for m in matched_with_credit],
            "missing_hard_requirements":missing_mh,
            "overqualification_flag":exp_check.get("flag","N/A"),"threshold_used":threshold}

SECTION_HEADERS={
    "Experience":["work experience","experience","employment","career history"],
    "Skills":    ["skills","technical skills","core competencies"],
    "Projects":  ["projects","project portfolio","key projects"],
    "Education": ["education","academic"],
    "Summary":   ["summary","profile","professional summary"],
}

def extract_sections(text: str) -> Dict[str, str]:
    lines,sections,current,buf=text.split("\n"),{},"General",[]
    for line in lines:
        lower=line.strip().lower()
        matched=next((k for k,kws in SECTION_HEADERS.items()
                      if any(lower==kw or lower.startswith(kw) for kw in kws)),None)
        if matched:
            if buf: sections[current]="\n".join(buf).strip()
            current,buf=matched,[]
        else: buf.append(line)
    if buf: sections[current]="\n".join(buf).strip()
    if len(sections)<=1:
        words=text.split(); t=max(len(words)//3,1)
        sections={"Profile":" ".join(words[:t]),"Experience":" ".join(words[t:2*t]),
                  "Skills":" ".join(words[2*t:])}
    return {k:v for k,v in sections.items() if v.strip()}

def extract_bullets(text: str) -> List[str]:
    bullets=[]
    for line in text.split("\n"):
        clean=re.sub(r'^[•\-e\*·◦]\s*','',line.strip()).strip()
        if len(clean)>20: bullets.append(clean)
    return bullets if bullets else [text[:500]]

def keyword_density_score(section_text: str, jd_text: str) -> float:
    jd_words=set(re.findall(r'\b[a-z]{4,}\b',jd_text.lower()))
    sec_words=set(re.findall(r'\b[a-z]{4,}\b',section_text.lower()))
    if not jd_words: return 0.0
    return min(100, round((len(jd_words & sec_words)/len(jd_words))*200, 1))

def hybrid_section_score(section_name: str, section_text: str,
                          jd_vec: np.ndarray, jd_text: str) -> Optional[float]:
    if section_name.lower() in LOW_SIGNAL_SECTIONS: return None
    clean_sec=clean_section_text(section_text); clean_jd=clean_section_text(jd_text)
    bullets=extract_bullets(clean_sec)
    scores=[cosine_sim(get_embedding(b[:500]),jd_vec) for b in bullets[:20]]
    if scores:
        scores.sort(reverse=True); top_n=max(1,int(len(scores)*0.6))
        embed_score=min(100,max(0,(np.mean(scores[:top_n])-0.25)/0.45*100))
    else: embed_score=0.0
    kd_score=keyword_density_score(clean_sec, clean_jd)
    ew=SECTION_EMBED_WEIGHT.get(section_name,0.60); kw=SECTION_KW_WEIGHT.get(section_name,0.40)
    hybrid=round(ew*embed_score+kw*kd_score, 1)
    boost=SECTION_DISPLAY_BOOST.get(section_name,1.0)
    return round(min(100, hybrid*boost), 1)

@timed_step("Matching Agent")
def matching_agent(resume_data, resume_text, jd_data, jd_text,
                   exp_check, jd_quality, role_fit, jd_intent) -> Dict:
    jd_vec       = get_embedding(sanitize_text(jd_text)[:6000])
    all_jd       = jd_data.get("must_have_skills",[])+jd_data.get("good_to_have_skills",[])
    resume_skills = resume_data.get("skills",[])
    inferred_map  = resume_data.get("inferred_confidence_map",{})
    implicit_map  = resume_data.get("implicit_skill_map",{})
    eff_implicit  = {skill:(info["confidence"],info["tier"]) for skill,info in implicit_map.items()}
    exp_years     = resume_data.get("experience_years",0)

    matched_with_credit,missing = match_skills_full(resume_skills, all_jd, resume_text, eff_implicit)
    # FIX A: pass resume_skills to weighted_coverage for platform tool weight reduction
    skill_score,cov_audit = weighted_coverage_v15(matched_with_credit, all_jd, inferred_map,
                                                   exp_years, resume_skills)
    bonus = [s for s in resume_skills
             if not any(m["skill"].lower()==canonical(s).lower() for m in matched_with_credit)][:10]

    sections=extract_sections(resume_text); section_scores={}
    for sec_name, sec_text in sections.items():
        sc=hybrid_section_score(sec_name, sec_text, jd_vec, jd_text)
        if sc is not None: section_scores[sec_name]=sc

    exp_section=sections.get("Experience","") or sections.get("Profile","")
    impact=score_impact(exp_section)

    w_skill=jd_intent.get("skill",0.35); w_exp=jd_intent.get("experience",0.25)
    w_role=jd_intent.get("role",0.30)
    w_ai=min(jd_intent.get("ai",0.10)+jd_quality.get("ai_weight_boost",0),0.15)
    tw=w_skill+w_exp+w_role+w_ai

    role_score = role_fit.get("role_score",70.0)
    exp_score  = float(exp_check.get("exp_score",100))
    missing_critical  = [s for s in missing if skill_tier(s)=="critical"]
    missing_important = [s for s in missing if skill_tier(s)=="important"]

    # FIX B: include role appeal in AI prompt
    overq_ctx = exp_check.get("overqual_context",{})
    role_appeal_txt = f", role_appeal={overq_ctx.get('role_appeal','N/A')}" if overq_ctx else ""

    ai_prompt=f"""You are a precise senior technical evaluator. Score holistic fit 50-95.

SCORING RULES:
- 2+ CRITICAL skills missing AND not implied by experience → score BELOW 70
- Only minor/optional gaps → 75-85
- Strong domain + most skills covered → 80-90
- Platform-specific tools (Power Platform, Copilot Studio) missing BUT candidate has
  cloud/API/LLM foundation → treat as LEARNABLE, max 5% penalty, not a blocker
- "AI Agents" in JD = Agentic AI on resume → DIRECT MATCH, full credit
- Candidate has implicit/inferred evidence for missing skills → give partial credit
- FDE role + stakeholder + deployment experience → boost score +5-8

CANDIDATE: {exp_years} yrs{role_appeal_txt}, role={resume_data.get("current_role","N/A")},
domains={json.dumps(resume_data.get("domains",[]))},
summary={sanitize_text(str(resume_data.get("summary","N/A")))[:250]},
matched={json.dumps([m["skill"] for m in matched_with_credit][:12])},
CRITICAL missing={json.dumps(missing_critical)},
important missing={json.dumps(missing_important[:4])},
impact_signals={json.dumps(impact["hits"][:5])},
quantified={impact["quantified_count"]},
implicit_evidence={json.dumps(list(eff_implicit.keys())[:10])},
platform_learnability={has_platform_learnability(resume_skills)}

JOB: {jd_data.get("role_title","N/A")} ({jd_data.get("seniority","N/A")}),
intent={jd_intent.get("intent","general")},
responsibilities={sanitize_text(str(jd_data.get("responsibilities_summary","N/A")))[:200]}

Return ONLY valid JSON:
{{"ai_judgment_score":number 50-95,"key_strengths":["specific s1","s2"],"key_risks":["specific r1"],"verdict_reason":"Because X outweighs Y"}}"""

    ai_parsed = parse_llm_json(llm.invoke([HumanMessage(content=ai_prompt)]).content, AIJudgmentSchema)
    ai_score  = ai_parsed.ai_judgment_score

    raw_final  = round((w_skill/tw*skill_score+w_exp/tw*exp_score+w_role/tw*role_score+w_ai/tw*ai_score),1)
    coverage   = cov_audit.get("raw_score",skill_score)
    threshold  = jd_quality["screening_threshold"]
    aligned    = align_score_to_coverage(raw_final,coverage,threshold)
    impact_boost = min(5,round(impact["impact_score"]/3,1))
    final      = min(100,round(aligned+impact_boost,1))

    platform_note = ""
    if has_platform_learnability(resume_skills):
        platform_note = " | Platform tools down-weighted (learnable gap)"

    score_audit={
        "skill_score":round(skill_score,1),"skill_weight":round(w_skill/tw*100,1),
        "exp_score":round(exp_score,1),"exp_weight":round(w_exp/tw*100,1),
        "role_score":round(role_score,1),"role_weight":round(w_role/tw*100,1),
        "ai_score":round(ai_score,1),"ai_weight":round(w_ai/tw*100,1),
        "final_score":final,"raw_final":raw_final,"impact_boost":impact_boost,
        "coverage_alignment_applied":coverage<threshold,
        "depth_multiplier":cov_audit.get("depth_multiplier",1.0),
        "raw_skill_score":cov_audit.get("raw_score",skill_score),
        "ai_key_strengths":ai_parsed.key_strengths,
        "ai_key_risks":ai_parsed.key_risks,
        "ai_verdict_reason":ai_parsed.verdict_reason,
        "skill_breakdown":cov_audit.get("breakdown",{}),
        "jd_intent":jd_intent.get("intent","general"),
        "adaptive_weights_label":jd_intent.get("label",""),
        "title_boost":role_fit.get("title_boost",0),
        "implicit_skills_used":list(eff_implicit.keys())[:8],
        "platform_learnability": has_platform_learnability(resume_skills),
        "formula":(f"({round(w_skill/tw*100)}% × {round(skill_score,1)}) + "
                   f"({round(w_exp/tw*100)}% × {round(exp_score,1)}) + "
                   f"({round(w_role/tw*100)}% × {round(role_score,1)}) + "
                   f"({round(w_ai/tw*100)}% × {round(ai_score,1)}) = {raw_final}%"
                   f" → align×{round(aligned/raw_final,2) if raw_final>0 else 1}"
                   f" + impact+{impact_boost} = {final}%{platform_note}"),
    }
    return {"final_score":final,"skill_score":round(skill_score,1),
            "experience_score":round(exp_score,1),"role_score":round(role_score,1),
            "ai_score":round(ai_score,1),"section_scores":section_scores,
            "sections":sections,"matched_with_credit":matched_with_credit,
            "matched_skills":[m["skill"] for m in matched_with_credit],
            "missing_skills":missing,"bonus_skills":bonus,
            "score_audit":score_audit,"impact":impact}

@timed_step("Evaluation Agent")
def evaluation_agent(resume_data, jd_data, match, exp_check, jd_quality, role_fit) -> Dict:
    audit       = match.get("score_audit",{})
    overq_ctx   = exp_check.get("overqual_context",{})
    overq_note  = (f"Role appeal for overqualified candidate: {overq_ctx.get('role_appeal','N/A')} "
                   f"(tech depth: {overq_ctx.get('tech_depth_score',0)}/10, "
                   f"ownership: {overq_ctx.get('ownership_score',0)}/10)") if overq_ctx else ""
    prompt=f"""Senior technical interviewer. Be specific — no generic responses.
CANDIDATE: {resume_data.get("name","N/A")}, {resume_data.get("experience_years",0)} yrs,
role={resume_data.get("current_role","N/A")},
skills={json.dumps(resume_data.get("skills",[])[:20])}
IMPLICIT EVIDENCE: {json.dumps(audit.get("implicit_skills_used",[]))}
PLATFORM LEARNABILITY: {audit.get("platform_learnability",False)}
JOB: {jd_data.get("role_title","N/A")} ({jd_data.get("seniority","N/A")}),
intent={audit.get("jd_intent","general")}
EXP: {exp_check.get("flag","N/A")} | JD QUALITY: {jd_quality["quality"]} | ROLE: {role_fit["note"]}
{overq_note}
MATCHED: {json.dumps(match.get("matched_skills",[]))}
MISSING: {json.dumps(match.get("missing_skills",[]))}
IMPACT: {json.dumps(match.get("impact",{}).get("hits",[])[:5])}
AI VERDICT: {audit.get("ai_verdict_reason","")}

Rules:
- Overqualification = discuss role appeal (tech depth + ownership), not just retention risk
- Platform tools (Power Platform, Copilot Studio) missing = learnable gap if foundation exists, not a blocker
- "AI Agents" in JD = Agentic AI = direct match, call out explicitly
- FDE signals (stakeholder + deployment + ambiguity) → cite as concrete strengths
- Weight concerns by ROLE relevance. For GenAI/FDE: soft skills LOW priority.
- Highlight impact signals (5x, 30%, etc.) as concrete strengths

Return ONLY valid JSON:
{{"strengths":["specific s1","s2","s3"],"concerns":["role-relevant blocker"],
"interview_questions":[{{"question":"specific Q","category":"Technical/Behavioral/Domain","rationale":"why"}}],
"risk_level":"Low/Medium/High","risk_reason":"one sentence for THIS role"}}"""
    return parse_llm_json(
        llm.invoke([HumanMessage(content=sanitize_text(prompt))]).content, EvaluationSchema).model_dump()

def determine_level(exp_years, job_titles, skills):
    leadership=["lead","principal","head","director","manager","vp","chief"]
    has_lead=any(any(sig in t.lower() for sig in leadership) for t in job_titles)
    if "Technical Leadership" in skills: has_lead=True
    adv=["Agentic AI","MLOps","LangGraph","Architecture","Distributed Computing","Technical Leadership"]
    has_adv=sum(1 for s in skills if canonical(s) in adv)>=2
    if exp_years>=12 and has_lead:               level,band="Principal / Staff","L6+"
    elif exp_years>=8 and (has_lead or has_adv): level,band="Senior Lead","L5"
    elif exp_years>=5:                           level,band="Senior","L4"
    elif exp_years>=3:                           level,band="Mid-Level","L3"
    else:                                        level,band="Junior","L1-L2"
    return {"level":level,"band":band,"has_leadership":has_lead}

def decision_logic_layer(match, screening, exp_check, jd_quality, role_fit) -> Dict:
    score     = match["final_score"]
    coverage  = screening.get("must_have_coverage_pct",0)
    is_overq  = exp_check.get("is_overqualified",False)
    jd_q      = jd_quality["quality"]
    role_score= role_fit.get("role_score",70)
    threshold = dynamic_threshold(jd_q)
    # FIX B: pull role appeal
    overq_ctx = exp_check.get("overqual_context",{})
    appeal    = overq_ctx.get("role_appeal","Medium")

    if role_fit["mismatch"]:
        return {"decision":"Reject","confidence":85,"reason":role_fit["note"],
                "next_step":"Role domain mismatch — decline or redirect",
                "offer_likelihood":"Low","override_hint":"Hard blocker"}

    if jd_q=="Low" and is_overq:
        return {"decision":"Hold","confidence":55,
                "reason":"JD is generic. Verify actual role scope with hiring manager.",
                "next_step":"Clarify role scope","offer_likelihood":"Medium",
                "override_hint":"Low JD quality — human review needed"}

    if is_overq and score>=70 and role_score>=60:
        # FIX B: decision confidence varies by role appeal
        conf = 80 if appeal=="High" else 75 if appeal=="Medium" else 65
        disc = "; ".join(exp_check.get("discussion_points",[])[:2])
        return {"decision":"Hire","confidence":conf,
                "reason":(f"Strong match ({score}%, role {role_score:.0f}%). "
                           f"Overqualified — role appeal: {appeal}. Discuss: {disc}"),
                "next_step":("Fast-track if role has high tech depth" if appeal=="High"
                              else "Interview — discuss comp + growth expectations"),
                "offer_likelihood": "High" if appeal=="High" else "Medium",
                "override_hint":"Confirm role appeal aligns with candidate motivation"}

    if is_overq and score>=50:
        # FIX B: low appeal overqual → stronger reject signal
        if appeal=="Low":
            return {"decision":"Reject","confidence":70,
                    "reason":f"Overqualified + low role appeal ({score}%). High retention risk.",
                    "next_step":"Decline or explore senior role","offer_likelihood":"Low",
                    "override_hint":"Consider if a more senior position exists"}
        return {"decision":"Hold","confidence":65,
                "reason":f"Overqualified with moderate match ({score}%). Role appeal: {appeal}.",
                "next_step":"Discuss scope, comp, growth path","offer_likelihood":"Medium",
                "override_hint":"Verify motivation and comp expectations"}

    if jd_q=="Low":
        d="Hire" if score>=55 else "Hold"
        return {"decision":d,"confidence":55,"reason":f"Low JD quality — score {score}%.",
                "next_step":"Verify role requirements in interview",
                "offer_likelihood":"Medium","override_hint":"Low JD — verify fit"}

    if jd_q=="High" and coverage<50:
        return {"decision":"Reject","confidence":80,
                "reason":f"Only {coverage:.1f}% coverage on a well-defined JD.",
                "next_step":"Decline — skill gap too large","offer_likelihood":"Low","override_hint":None}

    if score>=threshold+5 and coverage>=70:
        return {"decision":"Strong Hire","confidence":90,
                "reason":f"{coverage:.1f}% coverage, {score}% overall.",
                "next_step":"Fast-track technical interview","offer_likelihood":"High","override_hint":None}
    elif score>=threshold and coverage>=60:
        return {"decision":"Hire","confidence":75,
                "reason":f"{coverage:.1f}% coverage, {score}% overall.",
                "next_step":"Technical interview","offer_likelihood":"Medium","override_hint":None}
    elif score>=threshold-10:
        return {"decision":"Hold","confidence":65,
                "reason":f"Borderline — {coverage:.1f}% coverage.",
                "next_step":"Exploratory interview","offer_likelihood":"Medium","override_hint":None}
    return {"decision":"Reject","confidence":75,
            "reason":f"Score {score}% and {coverage:.1f}% below dynamic threshold ({threshold}%).",
            "next_step":"Decline","offer_likelihood":"Low","override_hint":None}


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(resume_text: str, resume_data: Dict, jd_text: str) -> Dict:
    r={}
    with st.status("🔍 JD Intelligence...",expanded=False):
        r["jd_data"]=jd_intelligence_agent(jd_text)
        jd=r["jd_data"]
        st.write(f"✅ {jd.get('role_title')} | {jd.get('seniority')} | "
                 f"Exp: {jd.get('experience_required_years',0)}–{jd.get('experience_max_years','?')} yrs")
    with st.status("⚡ Pre-checks...",expanded=False):
        r["jd_quality"] = assess_jd_quality(jd_text, r["jd_data"])
        r["jd_intent"]  = detect_jd_intent(jd_text, r["jd_data"])
        # FIX B: pass jd_text for context scoring; FIX C: pass resume_text
        r["exp_check"]  = check_overqualification(resume_data.get("experience_years",0),
                                                   r["jd_data"], r["jd_quality"], jd_text)
        r["role_fit"]   = assess_role_fit(resume_data, r["jd_data"], r["jd_quality"], resume_text)
        sim_txt   = f"{r['role_fit']['similarity']:.0%}" if r['role_fit']['similarity'] else "skipped"
        boost_txt = (f" (+{r['role_fit']['title_boost']} cluster/title/FDE)"
                     if r['role_fit'].get('title_boost',0)>0 else "")
        appeal_txt = (f" | Role appeal: {r['exp_check']['overqual_context'].get('role_appeal','')}"
                      if r['exp_check'].get('overqual_context') else "")
        st.write(f"✅ JD: {r['jd_quality']['quality']} | Intent: {r['jd_intent']['intent']} | "
                 f"Exp: {r['exp_check']['flag']} | Role: {sim_txt}{boost_txt}{appeal_txt}")
    with st.status("📊 Matching + Screening...",expanded=False):
        r["match"]     = matching_agent(resume_data, resume_text, r["jd_data"], jd_text,
                                         r["exp_check"], r["jd_quality"], r["role_fit"], r["jd_intent"])
        r["screening"] = screening_agent(resume_data, r["jd_data"], r["exp_check"], r["jd_quality"],
                                          r["match"]["matched_with_credit"], r["match"]["missing_skills"])
        impl_count = len(resume_data.get("implicit_skill_map",{}))
        plat_note  = " | Platform penalty reduced ✅" if r["match"]["score_audit"].get("platform_learnability") else ""
        st.write(f"✅ Score: {r['match']['final_score']}% | Coverage: {r['screening']['must_have_coverage_pct']}% | "
                 f"Implicit: {impl_count} | Impact: +{r['match']['impact']['impact_score']}{plat_note}")
    with st.status("🧠 Evaluation...",expanded=False):
        r["evaluation"] = evaluation_agent(resume_data, r["jd_data"], r["match"],
                                            r["exp_check"], r["jd_quality"], r["role_fit"])
        st.write(f"✅ Risk: {r['evaluation'].get('risk_level','N/A')}")
    with st.status("🎯 Decision...",expanded=False):
        r["level"]         = determine_level(resume_data.get("experience_years",0),
                                              resume_data.get("job_titles",[]),
                                              resume_data.get("skills",[]))
        r["logic_decision"]= decision_logic_layer(r["match"], r["screening"],
                                                   r["exp_check"], r["jd_quality"], r["role_fit"])
        r["skill_recs"]    = generate_recommendations_v15(r["match"]["missing_skills"],
                                                           r["match"]["matched_with_credit"],
                                                           r["match"]["final_score"],
                                                           r["jd_intent"].get("intent","general"),
                                                           resume_data.get("skills",[]))
        st.write(f"✅ {r['logic_decision']['decision']} ({r['logic_decision']['confidence']}%)")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sc(s): return "🟢" if s>=75 else "🟡" if s>=50 else "🔴"
def dec_fn(d):
    dl=d.lower()
    if "strong hire" in dl: return st.success
    if "hire" in dl:        return st.success
    if "hold" in dl:        return st.warning
    return st.error
def scr_fn(s): return {"Pass":st.success,"Borderline":st.warning,"Reject":st.error}.get(s,st.info)
def si(s):     return {"Pass":"✅","Borderline":"⚠️","Reject":"❌"}.get(s,"❓")


# ══════════════════════════════════════════════════════════════════════════════
# UI — RESUME TAB
# ══════════════════════════════════════════════════════════════════════════════

def display_resume_tab():
    st.markdown("### 📄 Upload Your Resume")
    uploaded=st.file_uploader("Resume PDF",type="pdf",label_visibility="collapsed")
    if uploaded:
        if st.session_state.file_processed!=uploaded.name:
            st.session_state.file_processed=uploaded.name
            for k in ['resume_data','resume_text','pipeline_results']: st.session_state[k]=None
        fb=uploaded.read()
        with st.spinner("Reading PDF..."): rt=extract_text_from_pdf(fb)
        if not rt: st.error("❌ Failed to extract text."); return
        st.session_state.resume_text=rt
        st.success(f"✅ PDF loaded — {len(rt)} characters extracted")
        with st.expander("🔍 View Extracted Text"):
            st.text(rt[:2000]+("..." if len(rt)>2000 else ""))
        if st.button("🔎 Parse Resume",type="primary",use_container_width=True):
            with st.spinner("Parsing + inferring skills + implicit signal scan..."):
                st.session_state.resume_data=resume_parser_agent(rt)
                st.session_state.pipeline_results=None
            st.rerun()

    if not st.session_state.resume_data: return
    data=st.session_state.resume_data
    li=determine_level(data.get("experience_years",0),data.get("job_titles",[]),data.get("skills",[]))

    st.divider(); st.markdown("### 📊 Parsed Resume")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Experience",f"{data.get('experience_years',0)} yrs")
    with c2: st.metric("Skills",len(data.get("skills",[])))
    with c3: st.metric("Companies",len(data.get("companies",[])))
    with c4: st.metric("Level",li["level"])

    if data.get("inferred_skills"):
        parts=[f"{i['skill']} ({int(i['confidence']*100)}%)" for i in data["inferred_skills"]]
        st.caption(f"🔍 **Inferred:** {', '.join(parts)}")
    impl=data.get("implicit_skill_map",{})
    if impl:
        high=[f"{k} ({v['tier']})" for k,v in impl.items() if v["confidence"]>=0.75][:8]
        if high: st.caption(f"💡 **Implicit signals** (from resume context): {', '.join(high)}")
    if "Technical Leadership" in data.get("skills",[]):
        st.caption("🏅 **Leadership detected** from resume text")
    # FIX A: show platform learnability
    if has_platform_learnability(data.get("skills",[])):
        st.caption("🔧 **Platform learnability detected** — Power Platform / Copilot Studio penalties reduced")

    if data.get("summary") and data["summary"]!="Not Available":
        st.divider(); st.markdown("#### 🧑‍💼 Professional Summary"); st.info(data["summary"])

    st.divider(); st.markdown("#### 👤 Personal Information")
    c1,c2=st.columns(2)
    with c1:
        st.write(f"**Name:** {data.get('name','N/A')}")
        st.write(f"**Email:** {data.get('email','N/A')}")
    with c2:
        st.write(f"**Phone:** {data.get('phone','N/A')}")
        st.write(f"**Location:** {data.get('location','N/A')}")
    if data.get("linkedin_url") and data["linkedin_url"]!="Not Available":
        st.write(f"**LinkedIn:** {data['linkedin_url']}")

    st.divider(); st.markdown("#### 💼 Current Position")
    st.write(f"**Role:** {data.get('current_role','N/A')}")
    st.write(f"**Company:** {data.get('current_company','N/A')}")
    if data.get("domains"): st.write(f"**Domains:** {', '.join(data['domains'])}")

    st.divider(); st.markdown("#### 🛠️ Skills")
    for s in data.get("skills",[]): st.markdown(f":blue-badge[{s}]")

    st.divider(); st.markdown("#### 🎓 Education")
    st.write(data.get("education","N/A"))

    st.divider(); st.markdown("#### 🏢 Work History")
    cos=data.get("companies",[]); ts=data.get("job_titles",[])
    for i,co in enumerate(cos): st.write(f"• **{ts[i] if i<len(ts) else 'Unknown'}** at {co}")

    if data.get("certifications") and data["certifications"][0]!="Not Available":
        st.divider(); st.markdown("#### 📜 Certifications")
        for c in data["certifications"]: st.write(f"• {c}")

    with st.expander("View Raw JSON"): st.json(data)
    st.divider()
    if st.button("🔄 Clear & Start Over",use_container_width=True):
        for k in ['resume_data','resume_text','file_processed','pipeline_results','last_override']:
            st.session_state[k]=None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# UI — JD MATCH TAB
# ══════════════════════════════════════════════════════════════════════════════

def display_jd_tab():
    if not st.session_state.resume_data:
        st.info("👈 Upload and parse a resume first."); return

    st.markdown("### 📋 Paste Job Description")
    jd_text=st.text_area("JD",height=220,placeholder="Paste the full job description...",
                          label_visibility="collapsed",key="jd_input")
    if st.button("⚡ Run Full AI Pipeline",type="primary",use_container_width=True,
                 disabled=not jd_text.strip()):
        st.session_state.pipeline_results=run_pipeline(
            st.session_state.resume_text,st.session_state.resume_data,jd_text)
        st.session_state.last_override=None; st.rerun()
    if not st.session_state.pipeline_results: return

    r=st.session_state.pipeline_results
    jd_data=r["jd_data"]; screening=r["screening"]; match=r["match"]
    evaluation=r["evaluation"]; logic_dec=r["logic_decision"]
    level_info=r["level"]; exp_check=r["exp_check"]
    jd_quality=r["jd_quality"]; role_fit=r["role_fit"]
    jd_intent=r["jd_intent"]; skill_recs=r.get("skill_recs",[])
    audit=match.get("score_audit",{}); impact=match.get("impact",{})

    # Banners
    jq=jd_quality["quality"]
    if jq=="Low":    st.warning(f"⚠️ **JD Quality: Low** — {jd_quality['label']}. Threshold: {jd_quality['screening_threshold']}%.")
    elif jq=="Medium": st.info(f"ℹ️ **JD Quality: Medium** — {jd_quality['label']}.")
    intent_icon={"genai":"🤖","engineering":"⚙️","research":"🔬","platform":"🏗️","product":"📦","fde":"🚀"}.get(jd_intent.get("intent",""),"🎯")
    st.info(f"{intent_icon} **JD Intent: {jd_intent.get('intent','general').title()}** — {jd_intent.get('label','')}. Weights adapted.")
    if role_fit["mismatch"]: st.error(f"🚫 **Role Mismatch** — {role_fit['note']}")
    elif role_fit.get("similarity"): st.success(f"✅ **Role Fit** — {role_fit['note']}")

    if exp_check.get("is_overqualified"):
        overq_ctx = exp_check.get("overqual_context",{})
        appeal    = overq_ctx.get("role_appeal","")
        appeal_icon = {"High":"🟢","Medium":"🟡","Low":"🔴"}.get(appeal,"⚪")
        st.warning(f"⚠️ **Overqualification Signal** — {exp_check['detail']}"
                   +(f" | Role appeal: {appeal_icon} {appeal}" if appeal else ""))
        if exp_check.get("discussion_points"):
            st.caption("Discussion: "+"; ".join(exp_check["discussion_points"]))
        # FIX B: show tech depth breakdown
        if overq_ctx:
            with st.expander("📊 Role Appeal Breakdown (for overqualified candidate)"):
                oc1,oc2,oc3=st.columns(3)
                with oc1: st.metric("Tech Depth",f"{overq_ctx.get('tech_depth_score',0)}/10")
                with oc2: st.metric("Ownership Scope",f"{overq_ctx.get('ownership_score',0)}/10")
                with oc3: st.metric("Learning Opportunity",f"{overq_ctx.get('learning_score',0)}/10")

    # FIX A: Platform learnability banner
    if audit.get("platform_learnability"):
        st.success("🔧 **Platform Learnability Active** — missing platform tools (Power Platform, Copilot Studio) "
                   "down-weighted: candidate has cloud/API/LLM foundation to learn them quickly.")

    impl_used=audit.get("implicit_skills_used",[])
    if impl_used:
        st.info(f"💡 **Implicit inference active** — {len(impl_used)} skill signals inferred: "
                f"{', '.join(impl_used[:6])}")

    with st.expander("📋 JD Intelligence Summary"):
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Role",jd_data.get("role_title","N/A"))
        with c2: st.metric("Seniority",jd_data.get("seniority","N/A"))
        with c3: st.metric("Exp",f"{jd_data.get('experience_required_years',0)}–{jd_data.get('experience_max_years','?')} yrs")
        st.markdown("**Must-Have:**")
        for s in jd_data.get("must_have_skills",[]): st.markdown(f":red-badge[{s}]")
        if jd_data.get("good_to_have_skills"):
            st.markdown("**Nice-to-Have:**")
            for s in jd_data["good_to_have_skills"]: st.markdown(f":blue-badge[{s}]")

    st.divider()

    # Screening
    st.markdown("### 🚦 HR Screening")
    status=screening.get("status","?")
    scr_fn(status)(f"{si(status)} **{status}** — {screening.get('reason','')}")
    c1,c2=st.columns(2)
    with c1:
        mhc=screening.get("must_have_coverage_pct",0); thr=screening.get("threshold_used",70)
        st.metric("Weighted Coverage",f"{mhc:.1f}%",
                  help=f"Threshold: {thr}%. Platform tools down-weighted if learnable. Optional excluded.")
        st.progress(int(mhc)/100)
    with c2:
        st.metric("Experience",screening.get("overqualification_flag","N/A"),
                  help="No score penalty. Flagged for discussion only when JD has explicit max years.")
        st.progress(min(int(screening.get("experience_coverage_pct",100)),100)/100)
    if screening.get("borderline_notes"): st.info(f"💡 {screening['borderline_notes']}")
    if screening.get("met_requirements"):
        st.markdown("**✅ Met:** "+" · ".join(screening["met_requirements"][:8]))
    if screening.get("missing_hard_requirements"):
        st.markdown("**❌ Missing:** "+" · ".join(screening["missing_hard_requirements"]))

    st.divider()

    # Scores + audit
    st.markdown("### 📊 Match Scores & Audit Trail")
    final=match["final_score"]
    st.markdown(f"## {sc(final)} Overall Match: **{final}%**")
    st.progress(int(final)/100)
    if audit.get("formula"):
        alignment_note="⚠️ Coverage-alignment applied" if audit.get("coverage_alignment_applied") else "✅ No alignment needed"
        title_note=f" | Cluster/title/FDE boost: +{audit.get('title_boost',0)}" if audit.get("title_boost",0)>0 else ""
        st.code(f"Adaptive weights [{jd_intent.get('intent','general')} intent]: "
                f"Skill {audit.get('skill_weight',35)}% · Exp {audit.get('exp_weight',25)}% · "
                f"Role {audit.get('role_weight',30)}% · AI {audit.get('ai_weight',10)}%\n"
                f"{audit['formula']}\n{alignment_note}{title_note}",language=None)
    if impact.get("hits"):
        st.success(f"📈 **Impact signals** (+{audit.get('impact_boost',0)}% boost): "
                   f"{', '.join(impact['hits'][:5])}"
                   +(f" | {impact['quantified_count']} quantified" if impact.get("quantified_count",0)>0 else ""))
    if audit.get("ai_verdict_reason"):
        with st.expander("🤖 Structured AI Assessment"):
            col1,col2=st.columns(2)
            with col1:
                st.markdown("**Key Strengths:**")
                for s in audit.get("ai_key_strengths",[]): st.write(f"✓ {s}")
            with col2:
                st.markdown("**Key Risks:**")
                for r2 in audit.get("ai_key_risks",[]): st.write(f"⚠ {r2}")
            st.markdown(f"**Verdict:** _{audit['ai_verdict_reason']}_")

    b1,b2,b3,b4=st.columns(4)
    with b1: st.metric(f"🧠 Skill ({audit.get('skill_weight',35)}%)",f"{match['skill_score']}%",
                        help="7-tier: Exact→Group→Cluster→Concept→Implicit. Platform tools down-weighted.")
    with b2: st.metric(f"📅 Exp ({audit.get('exp_weight',25)}%)",f"{match['experience_score']}%",
                        help="No overqual penalty. Role appeal evaluated separately.")
    with b3: st.metric(f"🎯 Role ({audit.get('role_weight',30)}%)",f"{match['role_score']}%",
                        help="Embedding + role archetype cluster + FDE signal detection.")
    with b4: st.metric(f"🤖 AI ({audit.get('ai_weight',10)}%)",f"{match['ai_score']}%",
                        help="Platform learnability + agentic alias + FDE signals all factored in.")

    if audit.get("skill_breakdown"):
        with st.expander("🔬 Skill Score Breakdown"):
            for skill,info in audit["skill_breakdown"].items():
                cr=info.get("credit",1.0); conf=info.get("confidence",1.0)
                reason=info.get("reason","exact")
                icon={"exact":"📄","group":"🔗","cluster":"🔗","concept":"💡","substring":"↔",
                       "programming-concept":"🖥️","implicit (Explicit/Strong)":"⚡",
                       "implicit (Strong Inferred)":"💡","implicit (Weak Inferred)":"〰️"}.get(reason,"📄")
                if "implicit" in reason: icon="💡"
                src=reason.replace("implicit ","Implicit ")
                st.write(f"• **{skill}** ← _{info.get('matched_by','')}_ | {icon} {src} | "
                         f"w:{info['weight']} × c:{cr} × conf:{conf} = **{info['effective']}**")

    st.divider()

    # Level
    st.markdown("### 🏅 Candidate Level")
    lc1,lc2,lc3=st.columns(3)
    with lc1: st.metric("Level",level_info.get("level","N/A"))
    with lc2: st.metric("Band",level_info.get("band","N/A"))
    with lc3: st.metric("Leadership","Yes ✅" if level_info.get("has_leadership") else "No")

    st.divider()

    # Section matching
    st.markdown("### 🔍 Resume Section Matching")
    st.caption("v15: Experience section embedding weight 75%. FDE signals detected in Experience section.")
    for sec,score in sorted(match.get("section_scores",{}).items(),key=lambda x:x[1],reverse=True):
        cl,cr=st.columns([4,1])
        with cl: st.markdown(f"**{sc(score)} {sec}**"); st.progress(int(score)/100)
        with cr: st.markdown(f"### {score}%")
        if sec in match.get("sections",{}):
            with st.expander(f"View {sec}"):
                st.write(match["sections"][sec][:600]+("..." if len(match["sections"][sec])>600 else ""))

    st.divider()

    # Skill analysis
    st.markdown("### 🛠️ Skill Analysis")
    st.caption("v15: AI Agents = Agentic AI (direct). Platform tools learnable if foundation exists. "
               "FDE signals detected. 7-tier matching.")
    ca,cb,cc=st.columns(3)
    with ca:
        st.markdown("#### ✅ Matched")
        for m in match.get("matched_with_credit",[]) or [{"skill":"None","credit":1.0,"reason":""}]:
            credit=m.get("credit",1.0); reason=m.get("reason","")
            icon={"exact":"📄","group":"🔗","cluster":"🔗","concept":"💡","substring":"↔",
                   "programming-concept":"🖥️"}.get(reason.split(" ")[0],"📄")
            if "implicit" in reason: icon="💡"
            badge=":green-badge" if credit>=0.95 else (":blue-badge" if credit>=0.75 else ":orange-badge")
            label=m["skill"]+(f" ← {m.get('matched_by','')}" if m.get("matched_by","")!=m["skill"] and credit<0.95 else "")
            st.markdown(f"{badge}[{label}]")
            if credit<0.95: st.caption(f"  {icon} {reason} ({int(credit*100)}% credit)")
    with cb:
        st.markdown("#### ❌ Missing")
        missing=match.get("missing_skills",[])
        resume_skills_local = st.session_state.resume_data.get("skills",[]) if st.session_state.resume_data else []
        if missing:
            for s in missing:
                tier=skill_tier(s); head=cluster_head(s)
                cn=f" → cluster: {head}" if head else ""
                is_low      = s.lower() in LOW_CODE_TOOLS or s.lower() in COMMODITY_TOOLS
                is_platform = s.lower() in PLATFORM_SPECIFIC_TOOLS
                learnable   = is_platform and has_platform_learnability(resume_skills_local)
                if tier=="optional":
                    st.markdown(f":gray-badge[{s}]"); st.caption("  Optional — no impact")
                elif is_low:
                    st.markdown(f":gray-badge[{s}]"); st.caption("  Low-code — learnable quickly")
                elif learnable:
                    st.markdown(f":blue-badge[{s}]"); st.caption("  🔧 Platform tool — learnable (foundation exists)")
                elif tier=="critical":
                    st.markdown(f":red-badge[{s}]"); st.caption(f"  Critical{cn}")
                else:
                    st.markdown(f":orange-badge[{s}]"); st.caption(f"  Important{cn}")
        else: st.write("None 🎉")
    with cc:
        st.markdown("#### ⭐ Bonus")
        for s in match.get("bonus_skills",[]) or ["None"]: st.markdown(f":blue-badge[{s}]")

    if skill_recs:
        st.divider(); st.markdown("### 📚 Contextual Skill Gap Recommendations")
        st.caption(f"Tailored for **{jd_intent.get('intent','general')} role**. "
                   "Platform tools shown as learnable gaps, not blockers.")
        for rec in skill_recs:
            ti={"critical":"🔴","important":"🟡"}.get(rec["tier"],"⚪")
            learnable_tag = " 🔧 *learnable*" if rec.get("learnable") else ""
            c1,c2=st.columns([3,1])
            with c1:
                st.markdown(f"**{ti} {rec['skill']}**{learnable_tag}")
                if rec.get("context"): st.caption(f"  _{rec['context']}_")
                st.markdown(f"  → _{rec['resource']}_")
            with c2: st.metric("Score boost",f"+{rec['impact_pct']}%",help=f"Est. new score: {rec['new_score']}%")

    st.divider()

    # Technical evaluation
    st.markdown("### 🧠 Technical Evaluation")
    risk=evaluation.get("risk_level","N/A")
    ri={"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk,"⚪")
    st.markdown(f"**Risk:** {ri} {risk} — {evaluation.get('risk_reason','')}")
    cs,cc2=st.columns(2)
    with cs:
        st.markdown("**💪 Strengths**")
        for s in evaluation.get("strengths",[]): st.write(f"✓ {s}")
    with cc2:
        st.markdown("**⚠️ Concerns**")
        for c in evaluation.get("concerns",[]): st.write(f"⚠ {c}")
    st.markdown("**🎤 Interview Questions**")
    for q in evaluation.get("interview_questions",[]):
        with st.expander(f"[{q.get('category','?')}] {q.get('question','')}"):
            st.write(f"**Why:** {q.get('rationale','')}")

    st.divider()

    # Decision
    st.markdown("### 🎯 Final Hiring Decision")
    path=[]
    if jd_quality["quality"]=="Low":       path.append(f"Low JD (thr:{jd_quality['screening_threshold']}%)")
    if role_fit["mismatch"]:               path.append("Role Mismatch ❌")
    if exp_check.get("is_overqualified"):
        appeal = exp_check.get("overqual_context",{}).get("role_appeal","?")
        path.append(f"Overqualified (appeal:{appeal}) ⚠️")
    path.append(f"Score {final}%")
    if audit.get("coverage_alignment_applied"): path.append("Align↓")
    if audit.get("impact_boost",0)>0:      path.append(f"Impact+{audit['impact_boost']}")
    if audit.get("platform_learnability"): path.append("Platform✅")
    st.caption(f"🔀 Path: {' → '.join(path)} → **{logic_dec['decision']}**")
    dec=logic_dec.get("decision","N/A"); conf=logic_dec.get("confidence",0)
    dec_fn(dec)(f"{dec} — Confidence: {conf}%")
    st.progress(int(conf)/100)
    dc1,dc2=st.columns(2)
    with dc1:
        st.write(f"**Next Step:** {logic_dec.get('next_step','N/A')}")
        st.write(f"**Offer Likelihood:** {logic_dec.get('offer_likelihood','N/A')}")
    with dc2:
        if logic_dec.get("override_hint"): st.info(f"💡 **Recruiter hint:** {logic_dec['override_hint']}")
    st.info(logic_dec.get("reason",""))

    st.divider()

    # Recruiter override
    st.markdown("### 👤 Recruiter Override")
    st.caption("AI recommends — recruiter decides. Both stored with full audit trail.")
    if st.session_state.get("last_override"):
        lo=st.session_state.last_override
        (st.success if "hire" in lo["decision"].lower() else st.warning)(
            f"✅ **On record:** {lo['decision']} {'(overridden)' if lo.get('override') else '(AI accepted)'}")
    with st.form("override_form"):
        override_dec=st.selectbox("Override",["— Accept AI Decision —","Strong Hire","Hire","Hold","Reject"])
        override_note=st.text_area("Notes",height=80,placeholder="Reason for override...")
        submitted=st.form_submit_button("💾 Save to Memory",type="primary")
    if submitted:
        mem=load_memory(); override=None
        if override_dec!="— Accept AI Decision —":
            override={"decision":override_dec,"note":override_note}
            st.success(f"✅ Overridden to: **{override_dec}**")
        else: st.success("✅ AI decision accepted and saved.")
        store_candidate(mem,st.session_state.resume_data,jd_data.get("role_title","N/A"),
                        match,logic_dec,exp_check,jd_quality,audit,override)
        st.session_state.last_override={
            "decision":override_dec if override else logic_dec.get("decision","N/A"),
            "override":override is not None}


# ══════════════════════════════════════════════════════════════════════════════
# UI — CANDIDATES TAB
# ══════════════════════════════════════════════════════════════════════════════

def display_memory_tab():
    st.markdown("### 🗂️ Candidate Memory & Ranking")
    mem=load_memory(); candidates=mem.get("candidates",[])
    if not candidates: st.info("No candidates saved yet."); return

    fc1,fc2,fc3=st.columns(3)
    with fc1:
        roles=[""]+sorted(set(c.get("role","") for c in candidates if c.get("role")))
        sel_role=st.selectbox("Role / JD",roles,format_func=lambda x:"All Roles" if x=="" else x)
    with fc2:
        decs=[""]+sorted(set(c.get("final_decision","") for c in candidates))
        sel_dec=st.selectbox("Decision",decs,format_func=lambda x:"All" if x=="" else x)
    with fc3:
        sort_by=st.selectbox("Sort",["Score ↓","Score ↑","Date ↓","Impact ↓"])

    filtered=candidates
    if sel_role: filtered=[c for c in filtered if c.get("role")==sel_role]
    if sel_dec:  filtered=[c for c in filtered if c.get("final_decision")==sel_dec]
    if sort_by=="Score ↓":    filtered=sorted(filtered,key=lambda x:x.get("final_score",0),reverse=True)
    elif sort_by=="Score ↑":  filtered=sorted(filtered,key=lambda x:x.get("final_score",0))
    elif sort_by=="Impact ↓": filtered=sorted(filtered,key=lambda x:x.get("impact_score",0),reverse=True)
    else:                     filtered=sorted(filtered,key=lambda x:x.get("timestamp",""),reverse=True)

    if sel_role:
        st.markdown(f"#### 🏆 Leaderboard — {sel_role}")
        top=sorted(filtered,key=lambda x:x.get("final_score",0),reverse=True)[:10]
        for i,c in enumerate(top):
            score=c.get("final_score",0); medal=["🥇","🥈","🥉"][i] if i<3 else f"#{i+1}"
            dec=c.get("final_decision","?"); ob=" 🔁" if c.get("override") else ""
            imp_b=f" 📈+{c.get('impact_score',0)}" if c.get("impact_score",0)>0 else ""
            st.write(f"{medal} **{c.get('name','?')}** — {sc(score)} {score}% | {dec}{ob}{imp_b} | "
                     f"{c.get('experience_years',0)} yrs | {c.get('exp_flag','?')}")
        st.divider()
    else:
        st.caption("⚠️ Scores across JDs are not comparable. Filter by Role for per-JD ranking.")

    st.markdown(f"#### All Candidates ({len(filtered)})")
    for i,c in enumerate(filtered):
        score=c.get("final_score",0); dec=c.get("final_decision","?")
        ai_dec=c.get("ai_decision","?"); ob=" 🔁" if c.get("override") else ""
        with st.expander(f"#{i+1} — {sc(score)} {c.get('name','?')} | {score}% | {dec}{ob}"):
            col1,col2,col3=st.columns(3)
            with col1:
                st.write(f"**Role:** {c.get('role','?')}")
                st.write(f"**Exp:** {c.get('experience_years',0)} yrs ({c.get('exp_flag','?')})")
                st.write(f"**JD Quality:** {c.get('jd_quality','?')} | Impact: {c.get('impact_score',0)}")
            with col2:
                st.write(f"**Score:** {score}% | **Conf:** {c.get('confidence',0)}%")
                st.write(f"**AI:** {ai_dec} → **Final:** **{dec}**")
                st.write(f"**Date:** {c.get('timestamp','?')[:10]}")
            with col3:
                if c.get("override_note"): st.markdown(f"**Note:** {c['override_note']}")
                audit=c.get("score_audit",{})
                if audit.get("formula"): st.caption(audit["formula"][:120])
            if c.get("matched_skills"):
                st.markdown("**Matched:** "+" · ".join(c["matched_skills"][:6]))
            if c.get("missing_skills"):
                st.markdown("**Missing:** "+" · ".join(c["missing_skills"][:6]))

    if mem.get("patterns"):
        st.divider(); st.markdown("### 📈 Skill Pattern Insights")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Most commonly missing:**")
            for s,n in sorted(mem["patterns"]["missing"].items(),key=lambda x:x[1],reverse=True)[:8]:
                tier=skill_tier(s)
                imp={"critical":"🔴","important":"🟡","optional":"⚪"}.get(tier,"")
                st.write(f"• {s} ({n}) {imp}")
        with c2:
            st.markdown("**Most commonly matched:**")
            for s,n in sorted(mem["patterns"]["matched"].items(),key=lambda x:x[1],reverse=True)[:8]:
                st.write(f"• {s} ({n})")

    st.divider()
    if st.button("🗑️ Clear All Memory",type="secondary"):
        MEMORY_FILE.write_text(json.dumps({"candidates":[],"patterns":{"missing":{},"matched":{}}}))
        st.success("Memory cleared."); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="AI Talent Acquisition System",page_icon="🤖",layout="wide")
    st.title("🤖 AI Talent Acquisition System")
    st.caption("v15 · Platform Tool Learnability · Smart Overqualification · "
               "FDE Archetype Detection · Agentic AI ≈ AI Agents Alias · "
               "Expanded Role Clustering · Role Appeal Scoring")

    for k in ['resume_data','resume_text','file_processed','pipeline_results','last_override']:
        if k not in st.session_state: st.session_state[k]=None

    if st.session_state.resume_data:
        name=st.session_state.resume_data.get("name","Candidate")
        st.caption(f"✅ Resume loaded: **{name}**  |  Go to **Job Match** tab →")

    t1,t2,t3=st.tabs(["📄 Resume","📋 Job Match","🗂️ Candidates"])
    with t1: display_resume_tab()
    with t2: display_jd_tab()
    with t3: display_memory_tab()


if __name__=="__main__":
    main()