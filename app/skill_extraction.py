"""
Skill extraction via keyword matching.
Add or extend SKILLS_DB with domain-specific terms as needed.
"""

SKILLS_DB = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "go", "rust", "kotlin", "swift", "php", "ruby", "matlab", "perl", "haskell",
    "elixir", "clojure", "groovy", "dart", "lua", "cobol", "fortran", "assembly",
    "vba", "julia", "solidity", "abap",

    # Web & frameworks
    "html", "css", "react", "angular", "vue", "node", "flask", "django",
    "fastapi", "spring", "express", "next.js", "nuxt", "svelte", "gatsby",
    "graphql", "rest api", "soap", "webpack", "vite", "tailwind", "bootstrap",
    "jquery", "redux", "rxjs", "storybook", "jest", "cypress", "playwright",
    "selenium", "laravel", "symfony", "rails", "asp.net", "blazor",

    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm",
    "catboost", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "tfidf", "cosine similarity", "bert", "transformers", "gpt", "llm",
    "reinforcement learning", "time series", "forecasting", "anomaly detection",
    "recommendation systems", "classification", "regression", "clustering",
    "dimensionality reduction", "pca", "random forest", "gradient boosting",
    "neural networks", "cnn", "rnn", "lstm", "gan", "diffusion models",
    "embeddings", "vector search", "rag", "langchain", "llama", "openai",
    "hugging face", "mlflow", "weights and biases", "optuna", "a/b testing",
    "statistics", "probability", "hypothesis testing", "bayesian inference",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "nosql", "cassandra", "dynamodb", "firestore",
    "neo4j", "influxdb", "clickhouse", "snowflake", "bigquery", "redshift",
    "hive", "hbase", "couchdb", "mariadb", "supabase", "prisma",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd",
    "git", "github", "gitlab", "linux", "bash", "terraform", "ansible",
    "jenkins", "github actions", "circleci", "argocd", "helm", "istio",
    "prometheus", "grafana", "datadog", "splunk", "elk stack", "nginx",
    "apache", "serverless", "lambda", "cloud functions", "cloudformation",
    "pulumi", "vagrant", "packer", "vault", "consul",

    # BI & Analytics
    "tableau", "power bi", "excel", "looker", "data studio", "qlik",
    "metabase", "superset", "dbt", "alteryx", "sas", "spss",
    "google analytics", "mixpanel", "amplitude", "segment",

    # Data Engineering
    "airflow", "kafka", "spark", "hadoop", "flink", "dask", "ray",
    "etl", "data pipeline", "data warehouse", "data lake", "data lakehouse",
    "delta lake", "iceberg", "dbt", "fivetran", "stitch", "airbyte",
    "nifi", "luigi", "prefect", "dagster",

    # Other tools & platforms
    "spacy", "nltk", "streamlit", "opencv", "pillow", "ffmpeg",
    "celery", "rabbitmq", "grpc", "protobuf", "websockets",
    "jira", "confluence", "notion", "trello", "slack", "figma",
    "postman", "swagger", "openapi", "sentry", "new relic",

    # Security & Networking
    "cybersecurity", "penetration testing", "owasp", "sso", "oauth",
    "jwt", "tls", "ssl", "firewalls", "vpn", "zero trust", "siem",
    "vulnerability assessment", "cryptography",

    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",
    "swift ui", "jetpack compose", "expo",

    # AI / Emerging
    "generative ai", "prompt engineering", "fine-tuning", "rlhf",
    "vector database", "pinecone", "weaviate", "chroma", "faiss",
    "stable diffusion", "midjourney", "whisper", "text-to-speech",

    # Soft / general
    "data analysis", "data visualization", "feature engineering",
    "model deployment", "api", "agile", "scrum", "kanban",
    "product management", "technical writing", "code review",
    "system design", "microservices", "event-driven architecture",
    "object oriented programming", "functional programming",
    "test driven development", "design patterns", "algorithms",
    "data structures", "problem solving", "communication",
    "project management", "stakeholder management", "leadership",
]


def extract_skills(text: str) -> set:
    """Return a set of skills found in the given text (case-insensitive)."""
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_DB:
        # Match whole-word occurrences to avoid false positives
        import re
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def compare_skills(resume_text: str, jd_text: str) -> dict:
    """
    Compare skills between resume and job description.

    Returns:
        matched  : skills present in both
        missing  : skills required by JD but absent in resume
        extra    : skills in resume not mentioned in JD
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra = resume_skills - jd_skills

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }
