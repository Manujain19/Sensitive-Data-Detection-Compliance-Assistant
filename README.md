# Sensitive Data Detection & Compliance Assistant

An AI-assisted Streamlit prototype for uploading PDF, TXT, and CSV documents, detecting sensitive information, classifying document risk, generating a compliance/security summary, redacting detected values, and answering questions about the uploaded content.

## Features

- Upload PDF, TXT, and CSV files.
- Centered credential-page login and registration with JWT-backed sessions.
- Password reset and change-password flows.
- Google OAuth sign-in/sign-up when Google Client ID, Client Secret, and redirect URI are configured.
- Analyze one or more documents in the same session.
- OCR fallback for scanned PDFs when OCR dependencies are available.
- Detect Aadhaar numbers, PAN numbers, email addresses, Indian phone numbers, credit card numbers, bank details, API keys/passwords, employee IDs, and confidential business terms.
- Classify each document as Low Risk, Medium Risk, or High Risk.
- Generate compliance observations, security risks, and suggested remediation steps using a private AI configuration loaded from environment variables.
- Ask natural-language questions about counts, sensitive data types, summary, and compliance risk.
- Download findings, summaries, QA answers, action plans, NLP insights, redacted text, and document text as TXT, JSON, HTML, or PDF.
- Show additional NLP insights and semantic risk signals when optional enrichment keys are configured.
- AI Engineering Console with ten AI/ML capabilities, reviewer feedback labels, model-call traces, monitoring metrics, policy controls, and prompt-version visibility.
- Compliance framework mapping for DPDP India, PCI-style payment safety, ISO 27001-style access controls, confidentiality/NDA controls, and AI prompt-safety guardrails.
- Risk heatmap by document/category, severity explanations, reviewer workflow, policy upload/check, before/after redaction preview, evidence-backed QA, and a final compliance report generator.
- Demo mode with sample sensitive documents for quick evaluation.
- Keep a lightweight local audit log in `audit_log.jsonl`.
- Store user-specific document history and role-aware audit logs in SQLite.
- Admin dashboard for users, document activity, high-risk documents, and audit events.
- Docker, Render, and Streamlit deployment configuration.
- Includes sample TXT/CSV files and unit tests.

## Setup Instructions

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

OCR note: local OCR for scanned PDFs requires Tesseract. The Dockerfile and `packages.txt` include `tesseract-ocr` for supported deployment environments.

Optional AI setup:

```bash
copy .env.example .env
```

Add your private AI keys to `.env`:

```text
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
GROQ_MODEL=qwen/qwen3-32b
GEMINI_MODEL=gemini-2.5-flash
AI_TEMPERATURE=0.1
JWT_SECRET=replace_with_a_long_random_secret
ADMIN_EMAIL=admin@example.com
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8501/
HUGGINGFACE_API_KEY=your_huggingface_api_key
HUGGINGFACE_ZERO_SHOT_MODEL=facebook/bart-large-mnli
SPACY_MODEL=en_core_web_sm
```

For Google login, create an OAuth 2.0 Web Client in Google Cloud Console and add the redirect URI used by the app. For local development, use:

```text
http://localhost:8501/
```

The Streamlit screen does not expose provider names, model names, or API keys. If both generation keys are present, the app uses an internal ensemble and merges the strongest summary/QA output. The semantic classifier key enriches risk signals, and tracing keys are only for debugging/monitoring.

To run tests:

```bash
pytest
```

## Docker Setup

```bash
docker build -t sensitive-data-assistant .
docker run -p 8501:8501 sensitive-data-assistant
```

## Deployment

Included deployment assets:

- `Dockerfile` for container deployment.
- `render.yaml` for Render web service deployment.
- `packages.txt` for Streamlit Community Cloud system package installation.
- `.streamlit/config.toml` for Streamlit server/theme settings.

For Streamlit Community Cloud, add private API keys in the app secrets/settings rather than committing `.env`.

## Architecture Overview

```text
User Upload
   |
   v
document_loader.py
   - PDF extraction with pypdf
   - OCR fallback for scanned PDFs
   - TXT decoding
   - CSV row-to-text conversion
   |
   v
sensitive_detector.py
   - Regex and validation rules
   - Luhn validation for card numbers
   - Confidential keyword detection
   - Risk scoring
   - Redaction/masking
   |
   v
advanced_nlp.py
   - NLP entity and phrase enrichment
   - Optional semantic risk classification
   - Document chunking and local retrieval
   - Vector-style context retrieval for better QA
   |
   v
auth_store.py
   - SQLite users, document history, and audit events
   - Password hashing
   - JWT session token creation and validation
   - Password reset tokens and change-password workflow
   - Admin dashboard data
   |
   v
ai_graph.py
   - AI workflow orchestration
   - Privacy-first prompt context with retrieved sections
   - Private model selection from environment variables
   - LLM compliance summary
   - LLM document question answering
   - Rules fallback if key/package/API fails
   |
   v
app.py
   - Landing page, login, registration, Google OAuth callback, JWT session handling
   - Streamlit upload UI
   - Multi-document analysis
   - Provider details hidden from the user interface
   - Interactive dashboard, filters, charts, and quick questions
   - Findings table
   - Summary and QA tabs
   - Redacted download
   - Audit logging
```

## Bonus Features Implemented

- **OCR support:** scanned/empty PDFs trigger OCR fallback when Tesseract and OCR packages are available.
- **Data masking/redaction:** detected sensitive values are masked in findings and can be downloaded as redacted text.
- **RAG implementation:** document text is chunked and relevant sections are retrieved before QA prompts.
- **Multi-document support:** users can upload multiple PDF/TXT/CSV files in one run.
- **Dashboard/UI improvements:** risk gauge, category chart, filters, quick questions, NLP insights, and tabbed workflow.
- **Dockerization:** Dockerfile included with OCR system dependency.
- **Deployment:** Render, Streamlit config, and package files included.
- **Audit logging:** local JSONL audit log records file metadata, counts, OCR status, categories, and risk level without raw sensitive values.
- **JWT authentication:** registered users get signed session tokens.
- **Password management:** users can reset forgotten passwords and change passwords after login.
- **User-specific document history:** each analyzed document is recorded per user.
- **Admin dashboard:** admin users can review user counts, document activity, high-risk documents, and audit events.

## AI/ML Approach Used

This prototype uses an AI/ML-first compliance architecture with deterministic guardrails:

- **AI/ML sensitive-data extraction:** spaCy NER and contextual entity detection identify people, organizations, locations, DOBs, confidential project names, business-sensitive phrases, and contextual PII.
- **Structured identifier guardrails:** validators still protect high-precision identifiers such as Aadhaar, PAN, email, phone, cards, bank details, credentials, IFSC, and employee IDs.
- **Semantic classification:** HuggingFace zero-shot classification can label document context as personal data, financial data, credentials/secrets, confidential business information, or public information.
- **AI/ML risk scoring:** severity, frequency, category diversity, contextual NER findings, credential/financial/personal-data presence, and document profile signals are combined into Low, Medium, or High Risk.
- **AI guardrails:** prompt-injection and jailbreak-like document text is detected as a high-risk category and surfaced in the action plan.
- **Workflow orchestration:** document analysis flows through graph-style nodes for context preparation, AI summary generation, and QA.
- **Private LLM ensemble:** when multiple generation keys are provided through `.env`, the app queries both internally and merges the strongest compliance summary and QA output.
- **NLP enrichment:** an NLP layer extracts additional context signals and supports optional semantic classification through a private inference key.
- **Retrieval-augmented QA:** uploaded text is chunked and relevant sections are retrieved before the QA prompt, improving answers on larger documents.
- **Privacy-first prompting:** raw detected values are masked and the document context is redacted before being sent to the LLM.
- **AI governance trace:** the app exposes a model/pipeline trace with ingestion, entity detection, risk scoring, RAG, LLM reasoning, and guardrail status.
- **Guardrail fallback:** if an API call fails, the app still produces explainable results from local ML/contextual detection and validators.

This is stronger than an LLM-only design because contextual AI/ML detection handles semantic content while deterministic guardrails reduce false positives for regulated identifiers.

## Risk Classification Logic

- **Low Risk:** No sensitive data or only minimal low-impact indicators.
- **Medium Risk:** At least one high-severity item, several medium-severity items, or multiple sensitive categories.
- **High Risk:** Multiple high-severity identifiers, credentials, financial data, or a high aggregate risk score.

## Challenges Faced

- Avoiding false positives for long numeric strings by validating credit cards with the Luhn algorithm.
- Handling mixed document formats while keeping the prototype simple and explainable.
- Balancing AI/ML contextual detection with validator guardrails for regulated identifiers.
- Designing QA responses that are useful even without a remote LLM dependency.
- Preventing unnecessary sensitive-data exposure by sending redacted text and masked findings to LLM providers.
- Keeping deterministic fallback logic so the prototype remains demoable even if an API key is unavailable.
- Adding retrieval and semantic enrichment without making them hard requirements for a local demo.
- Combining multiple model responses without exposing provider details in the app interface.

## Future Improvements

- Extend the current retrieval layer to persistent multi-document collections.
- Add configurable vector backends for larger document sets.
- Add role-based access control, encrypted storage, and stronger audit trails.
- Add configurable compliance frameworks such as DPDP Act, GDPR, PCI DSS, and HIPAA.
- Add deployment automation for Streamlit Community Cloud, Render, or Azure App Service.

## Demo Video

Suggested 2-5 minute demo flow:

1. Start the app and upload `sample_data/sample_sensitive_document.txt`.
2. Show the risk score and findings table.
3. Open the summary tab and explain compliance/security observations.
4. Ask: "How many email addresses are present?"
5. Ask: "What compliance risks are identified?"
6. Show the redacted preview and download option.

## Working Prototype Deployment Link

```text
Deployment: https://sensitive-data-detection-compliance-assistant-2vuhkzrcmtbrcbtv.streamlit.app/
```

The prototype is deployed on Streamlit Community Cloud. Runtime secrets such as Gemini, Groq, HuggingFace, LangSmith, JWT, and Google OAuth credentials are configured through Streamlit app secrets and are not committed to the repository.

## GitHub Repository Link

```text
GitHub: https://github.com/Manujain19/Sensitive-Data-Detection-Compliance-Assistant
```

## Security Notes

- Uploaded files are processed in memory by Streamlit during the session.
- The prototype writes only metadata counts to `audit_log.jsonl`; it does not log raw sensitive values.
- Redacted output should be manually reviewed before external sharing.
- Production deployments should add authentication, encryption, retention controls, malware scanning, and secure secret management.
