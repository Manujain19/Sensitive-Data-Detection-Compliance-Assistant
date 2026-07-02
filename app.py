from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import hmac
import secrets
import time
import textwrap
from html import escape
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
import streamlit as st

from advanced_nlp import NLPReport, analyze_document_nlp, retrieve_relevant_context
from ai_graph import AIConfig, answer_question_with_ai, generate_ai_summary
from auth_store import (
    User,
    admin_dashboard_data,
    authenticate_user,
    change_password,
    create_access_token,
    create_or_update_oauth_user,
    create_user,
    get_user_from_token,
    init_db,
    list_user_audits,
    list_user_documents,
    list_ai_call_traces,
    list_detection_feedback,
    log_audit,
    request_password_reset,
    reset_password,
    save_ai_call_trace,
    save_detection_feedback,
    save_document_history,
)
from document_loader import load_document
from document_loader import _tesseract_command
from sensitive_detector import (
    classify_risk,
    detect_sensitive_data,
    mask_value,
    redact_text,
)


APP_DIR = Path(__file__).parent
AUDIT_LOG = APP_DIR / "audit_log.jsonl"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
OAUTH_STATE_TTL_SECONDS = 10 * 60


@dataclass(frozen=True)
class AnalysisDocument:
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, str | int]

try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / ".env")
except ImportError:
    pass


def load_streamlit_secrets_to_env() -> None:
    possible_secret_files = (
        Path.home() / ".streamlit" / "secrets.toml",
        APP_DIR / ".streamlit" / "secrets.toml",
    )
    if not any(path.exists() for path in possible_secret_files):
        return

    secret_names = [
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_MODEL",
        "GEMINI_MODEL",
        "AI_TEMPERATURE",
        "JWT_SECRET",
        "ADMIN_EMAIL",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REDIRECT_URI",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_ZERO_SHOT_MODEL",
        "SPACY_MODEL",
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
    ]
    try:
        for name in secret_names:
            if not os.getenv(name) and name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except Exception:
        pass


st.set_page_config(
    page_title="Sensitive Data Detection & Compliance Assistant",
    page_icon="lock",
    layout="wide",
)

load_streamlit_secrets_to_env()


def main() -> None:
    init_db()

    user = current_user()
    apply_theme(auth_mode=not bool(user))
    if not user:
        render_auth_screen()
        return

    render_header(user)
    page_options = ["Analyze Documents", "AI Engineering Console", "Document History", "Audit Logs"]
    if user.role == "admin":
        page_options.append("Admin Dashboard")
    nav_col, content_col = st.columns([1.15, 4.85], gap="large")
    with nav_col:
        st.markdown(
            """
            <div class="workspace-nav-card">
                <span class="console-kicker">Workspace</span>
                <h3>Compliance Console</h3>
                <p>Select an operation and keep every scan tied to an audit trail.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", page_options, label_visibility="collapsed")
        if st.button("Logout", use_container_width=True):
            log_audit(user.id, "logout", {"email": user.email})
            st.session_state.pop("access_token", None)
            st.rerun()

    with content_col:
        if page == "Document History":
            render_document_history(user)
            return
        if page == "AI Engineering Console":
            render_ai_engineering_console(user)
            return
        if page == "Audit Logs":
            render_audit_logs(user)
            return
        if page == "Admin Dashboard":
            render_admin_dashboard()
            return

        ai_config = render_ai_settings()
        render_ai_ml_stack(ai_config)
        render_capability_status()
        render_feature_diagnostics()
        uploaded_files = st.file_uploader("Upload document(s)", type=["pdf", "txt", "csv"], accept_multiple_files=True)
        demo_col, clear_demo_col = st.columns([1, 4])
        with demo_col:
            if st.button("Load demo document", use_container_width=True):
                st.session_state.demo_payloads = demo_file_payloads()
                st.rerun()
        with clear_demo_col:
            if st.session_state.get("demo_payloads") and st.button("Clear demo", use_container_width=True):
                st.session_state.pop("demo_payloads", None)
                st.rerun()

        if uploaded_files:
            file_payloads = uploaded_file_payloads(uploaded_files)
            st.session_state.pop("demo_payloads", None)
        else:
            file_payloads = tuple(st.session_state.get("demo_payloads", ()))

        if not file_payloads:
            render_empty_state()
            return

        upload_id = upload_fingerprint(user, file_payloads)
        try:
            documents = load_uploaded_documents_cached(file_payloads)
        except Exception as exc:
            st.error(str(exc))
            return

        document = combine_documents(documents)
        source_ranges = build_source_ranges(documents)
        detection_sources = map_detection_sources(detect_sensitive_data_cached(document.text), source_ranges)
        detections = list(detection_sources.keys())
        risk = classify_risk_cached(detections, document.text)
        redacted_text = redact_text(document.text, detections)
        nlp_report = analyze_document_nlp_cached(redacted_text)
        analysis_id = analysis_fingerprint(user, documents)
        if st.session_state.get("active_analysis_id") != analysis_id:
            st.session_state.active_analysis_id = analysis_id
            st.session_state.active_question = ""
            st.session_state.qa_history = []

        summary, ai_status = cached_ai_summary(user, analysis_id, upload_id, document.text, detections, risk, ai_config)
        if st.session_state.get("last_persisted_analysis") != analysis_id:
            persist_analysis(user, document, documents, detections, risk)
            write_audit_event(user, documents, detections, risk.level)
            st.session_state.last_persisted_analysis = analysis_id

        render_metrics(document, detections, risk, ai_status)
        render_intelligence_brief(document, detections, risk, nlp_report, ai_status)
        render_batch_dashboard(documents, detections, detection_sources, risk)
        render_analysis_overview(detections, risk)

        tabs = st.tabs(
            [
                "Findings",
                "Summary",
                "Compliance Map",
                "Ask",
                "Action Plan",
                "Review",
                "NLP Insights",
                "AI Governance",
                "Policy Check",
                "Redaction",
                "Document Text",
                "Final Report",
            ]
        )
        with tabs[0]:
            render_findings(user, analysis_id, detections, detection_sources)
        with tabs[1]:
            render_summary(summary, ai_status)
        with tabs[2]:
            render_compliance_map(detections, risk, detection_sources)
        with tabs[3]:
            render_question_answering(user, analysis_id, document.text, detections, risk, ai_config)
        with tabs[4]:
            render_action_plan(detections, risk, nlp_report)
        with tabs[5]:
            render_reviewer_workflow(user, analysis_id, detections, risk, nlp_report)
        with tabs[6]:
            render_nlp_insights(nlp_report)
        with tabs[7]:
            render_ai_governance(document, detections, risk, nlp_report, ai_status, ai_config)
        with tabs[8]:
            render_policy_check(document.text, detections, risk)
        with tabs[9]:
            render_redaction(document.text, detections, document.file_name)
        with tabs[10]:
            header_col, export_col = st.columns([6, 2])
            with header_col:
                st.markdown('<div class="section-kicker">Extracted document text</div>', unsafe_allow_html=True)
            with export_col:
                render_export_buttons(
                    "document_text_export",
                    f"{Path(document.file_name).stem}_document_text",
                    document.text,
                    {"file_name": document.file_name, "text": document.text},
                )
            st.text_area("Extracted text", document.text, height=420, label_visibility="collapsed")
        with tabs[11]:
            render_final_report(document, detections, risk, summary, nlp_report, detection_sources, ai_status)


def current_user() -> User | None:
    token = st.session_state.get("access_token")
    if not token:
        return None
    user = get_user_from_token(token)
    if not user:
        st.session_state.pop("access_token", None)
    return user


def set_auth_mode(mode: str) -> None:
    st.session_state.auth_mode = mode
    st.query_params["auth"] = mode


def render_public_topbar(user: User | None = None) -> None:
    if not user:
        brand_col, login_col, signup_col = st.columns([7.8, 1.1, 1.1])
        with brand_col:
            st.markdown(
                """
                <div class="topbar-inline">
                    <div class="top-brand"><span class="brand-pulse">~</span><span>SecureSight</span></div>
                    <div class="top-product">Sensitive Data Detection & Compliance Console</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with login_col:
            if st.button("Log in", key="top_login", use_container_width=True):
                set_auth_mode("login")
                st.rerun()
        with signup_col:
            if st.button("Get started", key="top_signup", type="primary", use_container_width=True):
                set_auth_mode("signup")
                st.rerun()
        return

    account_html = ""
    if user:
        display_name = user.full_name or user.email
        account_html = (
            f'<span class="top-user">{escape(display_name)} &middot; {escape(user.role)}</span>'
            '<a class="top-link" href="?nav=dashboard">Dashboard</a>'
        )
    else:
        account_html = (
            '<span class="top-spacer"></span>'
        )
    st.markdown(
        f"""
        <div class="app-topbar">
            <div class="top-brand"><span class="brand-pulse">~</span><span>SecureSight</span></div>
            <div class="top-product">Sensitive Data Detection & Compliance Console</div>
            <div class="top-actions">{account_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not user:
        top_left, top_login, top_signup = st.columns([8, 1, 1])
        with top_login:
            if st.button("Log in", key="top_login", use_container_width=True):
                set_auth_mode("login")
                st.rerun()
        with top_signup:
            if st.button("Get started", key="top_signup", type="primary", use_container_width=True):
                set_auth_mode("signup")
                st.rerun()


def render_public_landing() -> None:
    demo_profiles = {
        "HR onboarding": {
            "risk": "High Risk",
            "score": "92/100",
            "findings": "PAN, Aadhaar, email, phone, employee ID",
            "action": "Mask identifiers before sharing with payroll vendors.",
        },
        "Finance export": {
            "risk": "High Risk",
            "score": "96/100",
            "findings": "Bank account, IFSC, card-like numbers, email",
            "action": "Restrict access and generate a redacted copy.",
        },
        "Business memo": {
            "risk": "Medium Risk",
            "score": "64/100",
            "findings": "Confidential terms, strategy language, internal contacts",
            "action": "Review classification and retention policy.",
        },
    }
    if "landing_demo" not in st.session_state:
        st.session_state.landing_demo = "Finance export"

    st.markdown(
        """
        <section class="launch-grid">
            <div class="launch-copy">
                <div class="launch-badge">LangGraph + RAG compliance intelligence</div>
                <h1>Turn sensitive documents into audit-ready decisions.</h1>
                <p>
                    Upload PDFs, TXT, or CSV files. The assistant detects regulated data, classifies risk,
                    retrieves evidence, writes remediation guidance, and lets teams ask safe questions over
                    redacted context.
                </p>
                <div class="trust-row">
                    <span>JWT access</span><span>Google OAuth</span><span>Audit logs</span><span>Redaction</span>
                </div>
            </div>
            <div class="live-console-card">
                <div class="console-topline"><span>Live scan preview</span><strong>Enhanced</strong></div>
                <div class="scan-meter"><span style="width: 92%"></span></div>
                <div class="console-stats">
                    <div><span>Risk</span><strong>High</strong></div>
                    <div><span>Findings</span><strong>18</strong></div>
                    <div><span>High severity</span><strong>7</strong></div>
                </div>
                <div class="workflow-graph">
                    <span>Extract</span><i></i><span>Detect</span><i></i><span>Classify</span><i></i><span>Answer</span>
                </div>
                <p>Prompts use masked findings and redacted chunks, so responses stay useful without exposing raw secrets.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, start_col, login_col, right = st.columns([3.2, 1.2, 1.2, 3.2])
    with start_col:
        if st.button("Get started ->", key="hero_signup", type="primary", use_container_width=True):
            set_auth_mode("signup")
            st.rerun()
    with login_col:
        if st.button("Log in", key="hero_login", use_container_width=True):
            set_auth_mode("login")
            st.rerun()

    st.markdown('<div class="section-kicker">Try the intelligence preview</div>', unsafe_allow_html=True)
    profile_cols = st.columns(3)
    for index, name in enumerate(demo_profiles):
        if profile_cols[index].button(name, key=f"landing_profile_{index}", use_container_width=True):
            st.session_state.landing_demo = name
    active_profile = demo_profiles[st.session_state.landing_demo]
    st.markdown(
        f"""
        <div class="interactive-preview">
            <div><span>Scenario</span><strong>{st.session_state.landing_demo}</strong></div>
            <div><span>Predicted risk</span><strong>{active_profile["risk"]}</strong></div>
            <div><span>Score</span><strong>{active_profile["score"]}</strong></div>
            <div><span>Detected signals</span><p>{active_profile["findings"]}</p></div>
            <div><span>Suggested action</span><p>{active_profile["action"]}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-kicker">How it works</div>
        <div class="workflow-board">
            <div class="dark-card"><span>01</span><h3>Document enters</h3><p>Upload one or many PDFs, TXT files, or CSV sheets. OCR fallback handles scanned PDFs when available.</p></div>
            <div class="dark-card"><span>02</span><h3>AI/ML detectors inspect</h3><p>spaCy NER, contextual entity detection, semantic classifiers, and validator guardrails identify PII, secrets, and confidential text.</p></div>
            <div class="dark-card"><span>03</span><h3>Risk is scored</h3><p>Deterministic severity weights classify Low, Medium, or High Risk with clear score drivers.</p></div>
            <div class="dark-card"><span>04</span><h3>Analyst acts</h3><p>Generate summaries, ask RAG questions, download findings, redact sensitive values, and preserve audit logs.</p></div>
        </div>
        <div class="section-kicker">AI/ML approach used</div>
        <div class="feature-grid">
            <div class="dark-card"><span>AI</span><h3>Private provider orchestration</h3><p>Gemini and Groq can run together behind the scenes; the UI exposes only enhanced or standard analysis.</p></div>
            <div class="dark-card"><span>LG</span><h3>LangGraph workflow</h3><p>Summary and QA are routed through a controlled graph that uses redacted text and masked findings.</p></div>
            <div class="dark-card"><span>RAG</span><h3>Document question answering</h3><p>Retrieval uses protected chunks so users can ask compliance questions without dumping raw secrets into prompts.</p></div>
            <div class="dark-card"><span>SEC</span><h3>JWT, history, audit</h3><p>Each user gets document history, audit events, password controls, and admin visibility.</p></div>
            <div class="dark-card"><span>MASK</span><h3>Redaction workflow</h3><p>Masked previews and redacted downloads support safe sharing and remediation.</p></div>
            <div class="dark-card"><span>OPS</span><h3>Deployment ready</h3><p>Docker and Render config are included for prototype deployment and demo review.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def google_login_href() -> str:
    return build_google_auth_url() or "?auth=google-missing"


def render_google_auth_button(label: str, key: str) -> None:
    auth_url = build_google_auth_url()
    if auth_url:
        st.markdown(
            f'<a class="google-same-tab" href="{escape(auth_url, quote=True)}" target="_self">{label} with Google</a>',
            unsafe_allow_html=True,
        )
        return
    if st.button(f"{label} with Google", key=key, use_container_width=True):
        st.warning("Google login needs GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in .env.")


def build_google_auth_url() -> str | None:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    redirect_uri = google_redirect_uri()
    if not client_id or not os.getenv("GOOGLE_CLIENT_SECRET", "").strip():
        return None
    state = create_google_oauth_state()
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
        "access_type": "online",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def handle_google_oauth_callback() -> bool:
    code = st.query_params.get("code")
    state = st.query_params.get("state")
    if isinstance(code, list):
        code = code[0] if code else None
    if isinstance(state, list):
        state = state[0] if state else None
    if not code:
        return False
    if not validate_google_oauth_state(state or ""):
        st.session_state.google_auth_error = "Google sign-in state check failed. Please try again."
        st.session_state.auth_mode = "login"
        st.query_params.clear()
        st.rerun()
        return True

    try:
        profile = fetch_google_profile(code)
        if not profile.get("email_verified", False):
            raise ValueError("Google account email is not verified.")
        user = create_or_update_oauth_user(
            email=profile["email"],
            full_name=profile.get("name", ""),
            provider="google",
            provider_subject=profile["sub"],
        )
        st.session_state.access_token = create_access_token(user)
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.session_state.google_auth_error = f"Google sign-in failed: {exc}"
        st.session_state.auth_mode = "login"
        st.query_params.clear()
        st.rerun()
    return True


def fetch_google_profile(code: str) -> dict:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = google_redirect_uri()
    if not client_id or not client_secret:
        raise ValueError("Google OAuth credentials are not configured.")
    token_response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]
    profile_response = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    profile_response.raise_for_status()
    return profile_response.json()


def google_redirect_uri() -> str:
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/").strip()
    parsed = urlparse(redirect_uri)
    if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.path:
        return f"{redirect_uri}/"
    return redirect_uri


def create_google_oauth_state() -> str:
    payload = {
        "nonce": secrets.token_urlsafe(24),
        "iat": int(time.time()),
        "exp": int(time.time()) + OAUTH_STATE_TTL_SECONDS,
    }
    payload_text = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_part = _b64url_encode(payload_text)
    signature = hmac.new(_oauth_state_key(), payload_part.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_part}.{_b64url_encode(signature)}"


def validate_google_oauth_state(state: str) -> bool:
    try:
        payload_part, signature_part = state.split(".", 1)
        expected_signature = hmac.new(_oauth_state_key(), payload_part.encode("ascii"), hashlib.sha256).digest()
        supplied_signature = _b64url_decode(signature_part)
        if not hmac.compare_digest(expected_signature, supplied_signature):
            return False
        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        return int(payload.get("exp", 0)) >= int(time.time())
    except Exception:
        return False


def _oauth_state_key() -> bytes:
    secret = os.getenv("JWT_SECRET", "change-this-development-secret").encode("utf-8")
    return hashlib.sha256(secret).digest()


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def render_auth_screen() -> None:
    query_mode = st.query_params.get("auth", None)
    if isinstance(query_mode, list):
        query_mode = query_mode[0] if query_mode else None
    if handle_google_oauth_callback():
        return
    if query_mode == "google-missing":
        st.session_state.google_auth_warning = True
        query_mode = st.session_state.get("auth_mode", "login")
    if query_mode in {"signup", "login", "reset"}:
        st.session_state.auth_mode = query_mode
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "home"

    mode = st.session_state.auth_mode
    render_public_topbar()
    if mode == "home":
        render_public_landing()
        return

    left, center, right = st.columns([1.0, 0.72, 1.0])
    with center:
        title = {
            "signup": "Create account",
            "login": "Sign in",
            "reset": "Reset / change password",
        }[mode]
        subtitle = {
            "signup": "Start a secure compliance workspace.",
            "login": "Use your registered account.",
            "reset": "Verify your email and create a new password.",
        }[mode]
        st.markdown(
            f"""
            <div class="auth-panel">
                <div class="auth-panel-title">{title}</div>
                <div class="auth-panel-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.pop("google_auth_warning", False):
            st.warning(
                "Google login needs GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in .env."
            )
        if st.session_state.get("google_auth_error"):
            st.error(st.session_state.pop("google_auth_error"))

        if mode == "login":
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="name@example.com", key="login_email")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                user = authenticate_user(email, password)
                if user:
                    st.session_state.access_token = create_access_token(user)
                    st.success("Signed in successfully.")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
            render_google_auth_button("Sign in", "google_login")
            auth_link_cols = st.columns(2)
            with auth_link_cols[0]:
                if st.button("Forgot / change password", key="login_to_reset", use_container_width=True):
                    set_auth_mode("reset")
                    st.rerun()
            with auth_link_cols[1]:
                if st.button("Create account", key="login_to_signup", use_container_width=True):
                    set_auth_mode("signup")
                    st.rerun()

        elif mode == "signup":
            with st.form("register_form"):
                full_name = st.text_input("Full name", placeholder="Your name", key="register_name")
                email = st.text_input("Email", placeholder="name@example.com", key="register_email")
                password = st.text_input("Password (min 8 chars)", type="password", placeholder="Create a password", key="register_password")
                submitted = st.form_submit_button("Sign up", use_container_width=True)
            if submitted:
                try:
                    user = create_user(email, password, full_name=full_name)
                    st.session_state.access_token = create_access_token(user)
                    st.success("Account created.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            render_google_auth_button("Sign up", "google_signup")
            auth_link_cols = st.columns(2)
            with auth_link_cols[0]:
                if st.button("Reset password", key="signup_to_reset", use_container_width=True):
                    set_auth_mode("reset")
                    st.rerun()
            with auth_link_cols[1]:
                if st.button("Log in", key="signup_to_login", use_container_width=True):
                    set_auth_mode("login")
                    st.rerun()

        else:
            with st.form("request_reset_form"):
                reset_email = st.text_input("Account email", placeholder="name@example.com", key="reset_email")
                requested = st.form_submit_button("Generate reset token", use_container_width=True)
            if requested:
                try:
                    token = request_password_reset(reset_email)
                    if token:
                        st.session_state.reset_token_demo = token
                        st.success("Reset token generated.")
                    else:
                        st.info("If an account exists for that email, a reset token will be created.")
                except ValueError as exc:
                    st.error(str(exc))

            if st.session_state.get("reset_token_demo"):
                st.code(st.session_state.reset_token_demo, language="text")

            with st.form("complete_reset_form"):
                token = st.text_input("Reset token", type="password", key="reset_token")
                new_password = st.text_input("New password", type="password", key="reset_new_password")
                confirm_password = st.text_input("Confirm new password", type="password", key="reset_confirm_password")
                completed = st.form_submit_button("Reset password", use_container_width=True)
            if completed:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif reset_password(token, new_password):
                    st.session_state.pop("reset_token_demo", None)
                    st.success("Password reset complete. You can sign in with the new password.")
                else:
                    st.error("Invalid or expired reset token.")
            if st.button("Back to sign in", key="reset_to_login", use_container_width=True):
                set_auth_mode("login")
                st.rerun()


def render_user_bar(user: User) -> None:
    col1, col2 = st.columns([4, 1])
    col1.caption(f"Signed in as {user.email} | Role: {user.role}")
    if col2.button("Logout", use_container_width=True):
        log_audit(user.id, "logout", {"email": user.email})
        st.session_state.pop("access_token", None)
        st.rerun()


def render_change_password(user: User) -> None:
    st.markdown(
        """
        <div class="console-panel">
            <span class="console-kicker">Account security</span>
            <h3>Change password</h3>
            <p>Update your local email/password credential. Google sign-in users can continue using OAuth without a local password.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 1])
    with left:
        with st.form("change_password_form"):
            current_password = st.text_input("Current password", type="password")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Update password", use_container_width=True)
    with right:
        st.markdown(
            f"""
            <div class="summary-card">
                <h3>Signed-in account</h3>
                <ul>
                    <li>{escape(user.email)}</li>
                    <li>Role: {escape(user.role)}</li>
                    <li>Use at least 8 characters for stronger local credentials.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        elif change_password(user.id, current_password, new_password):
            st.success("Password updated successfully.")
        else:
            st.error("Current password is incorrect.")


def render_ai_settings() -> AIConfig:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    temperature = float(os.getenv("AI_TEMPERATURE", "0.1"))

    if gemini_key and groq_key:
        return AIConfig(
            provider="gemini",
            api_key=gemini_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            secondary_provider="groq",
            secondary_api_key=groq_key,
            secondary_model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
            fast_responses=True,
        )

    if gemini_key:
        return AIConfig(
            provider="gemini",
            api_key=gemini_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            fast_responses=True,
        )

    if groq_key:
        return AIConfig(
            provider="groq",
            api_key=groq_key,
            model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
            temperature=temperature,
            fast_responses=True,
        )

    return AIConfig(fast_responses=True)


def uploaded_file_payloads(uploaded_files) -> tuple[tuple[str, bytes], ...]:
    return tuple((file.name, file.getvalue()) for file in uploaded_files)


def demo_file_payloads() -> tuple[tuple[str, bytes], ...]:
    preferred = [
        APP_DIR / "sample_data" / "sample_sensitive_document.txt",
        APP_DIR / "sample_data" / "sample_contacts.csv",
        APP_DIR / "sample_data" / "demo_security_credentials.txt",
    ]
    payloads = []
    for path in preferred:
        if path.exists():
            payloads.append((path.name, path.read_bytes()))
    if payloads:
        return tuple(payloads)

    fallback = (
        "Confidential finance review\n"
        "Employee ID EMP-AX91\n"
        "PAN ABCDE1234F\n"
        "Email priya@example.com\n"
        "Phone 9876543210\n"
        "Bank account number: 123456789012\n"
        "IFSC HDFC0001234\n"
        "api_key=demo_token_1234567890abcdef\n"
        "Recommended scanner behavior: classify risk, redact values, and create an audit report."
    )
    return (("demo_sensitive_document.txt", fallback.encode("utf-8")),)


def upload_fingerprint(user: User, payloads: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(str(user.id).encode("utf-8"))
    for file_name, data in payloads:
        digest.update(file_name.encode("utf-8", errors="ignore"))
        digest.update(str(len(data)).encode("ascii"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def load_uploaded_documents_cached(payloads: tuple[tuple[str, bytes], ...]):
    return [load_document(file_name, data) for file_name, data in payloads]


@st.cache_data(show_spinner=False)
def detect_sensitive_data_cached(text: str):
    return detect_sensitive_data(text)


@st.cache_data(show_spinner=False)
def classify_risk_cached(detections, text: str):
    return classify_risk(list(detections), text)


@st.cache_data(show_spinner=False)
def analyze_document_nlp_cached(redacted_text: str) -> NLPReport:
    return analyze_document_nlp(redacted_text)


def cached_ai_summary(user: User, analysis_id: str, upload_id: str, text: str, detections, risk, ai_config: AIConfig):
    cache_key = f"{upload_id}:{ai_config_signature(ai_config)}"
    summary_cache = st.session_state.setdefault("summary_cache", {})
    if cache_key not in summary_cache:
        start = time.perf_counter()
        with st.spinner("Generating compliance summary once for this upload..."):
            summary_cache[cache_key] = generate_ai_summary(text, detections, risk, ai_config)
        summary, status = summary_cache[cache_key]
        save_ai_trace_safely(
            user,
            analysis_id,
            feature="summary",
            ai_config=ai_config,
            status=status,
            latency_ms=int((time.perf_counter() - start) * 1000),
            output=json.dumps(summary, sort_keys=True),
            metadata={"detections": len(detections), "risk": risk.level},
        )
    return summary_cache[cache_key]


def ai_config_signature(ai_config: AIConfig) -> str:
    parts = [
        ai_config.provider,
        ai_config.model,
        ai_config.secondary_provider,
        ai_config.secondary_model,
        str(ai_config.temperature),
        "primary" if ai_config.api_key else "no-primary",
        "secondary" if ai_config.secondary_api_key else "no-secondary",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def save_ai_trace_safely(
    user: User,
    analysis_id: str,
    feature: str,
    ai_config: AIConfig,
    status: str,
    latency_ms: int,
    output: str,
    metadata: dict,
) -> None:
    provider = ai_config.provider if ai_config.api_key else "local"
    model_name = ai_config.model if ai_config.api_key else "local-ai-ml-fallback"
    try:
        save_ai_call_trace(
            user.id,
            analysis_id,
            feature,
            provider,
            model_name,
            "compliance-v2",
            status,
            latency_ms,
            hashlib.sha256(output.encode("utf-8", errors="ignore")).hexdigest()[:16],
            metadata,
        )
    except Exception:
        pass


def combine_documents(documents) -> AnalysisDocument:
    if len(documents) == 1:
        item = documents[0]
        return AnalysisDocument(item.file_name, item.file_type, item.text, item.metadata)

    sections = []
    total_characters = 0
    total_pages = 0
    file_types = set()
    ocr_statuses = []
    for item in documents:
        sections.append(f"\n\n===== Document: {item.file_name} ({item.file_type}) =====\n{item.text}")
        total_characters += len(item.text)
        total_pages += int(item.metadata.get("pages", 0) or 0)
        file_types.add(item.file_type)
        if "ocr_status" in item.metadata:
            ocr_statuses.append(f"{item.file_name}: {item.metadata['ocr_status']}")

    metadata: dict[str, str | int] = {
        "files": len(documents),
        "file_types": ", ".join(sorted(file_types)),
        "characters": total_characters,
        "pages": total_pages,
        "ocr_status": "; ".join(ocr_statuses) if ocr_statuses else "not_applicable",
    }
    return AnalysisDocument("multi_document_analysis", "MULTI", "".join(sections), metadata)


def build_source_ranges(documents) -> list[tuple[int, int, str]]:
    ranges = []
    cursor = 0
    for item in documents:
        prefix = f"\n\n===== Document: {item.file_name} ({item.file_type}) =====\n" if len(documents) > 1 else ""
        start = cursor + len(prefix)
        end = start + len(item.text)
        ranges.append((start, end, item.file_name))
        cursor = end
    return ranges


def map_detection_sources(detections, source_ranges: list[tuple[int, int, str]]) -> dict:
    mapping = {}
    for detection in detections:
        source = "Unknown"
        for start, end, file_name in source_ranges:
            if start <= detection.start <= end:
                source = file_name
                break
        mapping[detection] = source
    return mapping


def persist_analysis(user: User, combined_document, source_documents, detections, risk) -> None:
    categories = dict(Counter(item.category for item in detections))
    high_severity = sum(1 for item in detections if item.severity == "High")

    for item in source_documents:
        save_document_history(
            user_id=user.id,
            file_name=item.file_name,
            file_type=item.file_type,
            risk_level=risk.level,
            risk_score=risk.score,
            detections=len(detections),
            high_severity_detections=high_severity,
            categories=categories,
            metadata=item.metadata,
        )

    log_audit(
        user.id,
        "document_analyzed",
        {
            "file_count": len(source_documents),
            "file_names": [item.file_name for item in source_documents],
            "combined_file_name": combined_document.file_name,
            "risk_level": risk.level,
            "risk_score": risk.score,
            "detections": len(detections),
            "categories": categories,
        },
    )


def analysis_fingerprint(user: User, documents) -> str:
    digest = hashlib.sha256()
    digest.update(str(user.id).encode("utf-8"))
    for item in documents:
        digest.update(item.file_name.encode("utf-8"))
        digest.update(str(len(item.text)).encode("utf-8"))
        digest.update(item.text[:500].encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _history_rows(records: list[dict]) -> list[dict]:
    return [
        {
            "Date": item["created_at"],
            "Document": item["file_name"],
            "Type": item["file_type"],
            "Risk": item["risk_level"],
            "Score": item["risk_score"],
            "Findings": item["detections"],
            "High severity": item["high_severity_detections"],
            "Categories": ", ".join(f"{k}: {v}" for k, v in item["categories"].items()),
            "OCR": item["metadata"].get("ocr_status", "not_applicable"),
        }
        for item in records
    ]


def _audit_rows(records: list[dict]) -> list[dict]:
    return [
        {
            "Date": item["created_at"],
            "Event": item["event_type"],
            "Details": json.dumps(item["details"], default=str),
        }
        for item in records
    ]


def render_document_history(user: User) -> None:
    records = list_user_documents(user.id)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Document history</div>', unsafe_allow_html=True)
        st.subheader("Previous scans")
    with export_col:
        render_export_buttons(
            "history_export",
            "document_history",
            _table_rows_to_text("Document History", _history_rows(records)),
            records,
        )
    if not records:
        st.info("No document analysis history yet.")
        return

    rows = _history_rows(records)
    high_docs = sum(1 for item in records if item["risk_level"] == "High Risk")
    col1, col2, col3 = st.columns(3)
    col1.metric("Scans", len(records))
    col2.metric("High risk", high_docs)
    col3.metric("Latest score", records[0]["risk_score"] if records else 0)

    search = st.text_input("Search documents", placeholder="Filter by file name or category")
    risk_options = sorted({row["Risk"] for row in rows})
    selected_risks = st.multiselect("Risk filter", risk_options, default=risk_options)
    filtered = [
        row
        for row in rows
        if row["Risk"] in selected_risks
        and (not search or search.lower() in json.dumps(row).lower())
    ]
    render_dark_table(filtered)


def render_audit_logs(user: User) -> None:
    records = list_user_audits(user.id)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Audit logs</div>', unsafe_allow_html=True)
        st.subheader("Account activity")
    with export_col:
        render_export_buttons(
            "audit_export",
            "audit_logs",
            _table_rows_to_text("Audit Logs", _audit_rows(records)),
            records,
        )
    if not records:
        st.info("No audit events yet.")
        return

    rows = _audit_rows(records)
    event_options = sorted({row["Event"] for row in rows})
    col1, col2, col3 = st.columns(3)
    col1.metric("Events", len(records))
    col2.metric("Event types", len(event_options))
    col3.metric("Latest event", rows[0]["Event"] if rows else "-")
    selected_events = st.multiselect("Event filter", event_options, default=event_options)
    filtered = [row for row in rows if row["Event"] in selected_events]
    render_dark_table(filtered)


def render_admin_dashboard() -> None:
    data = admin_dashboard_data()
    totals = data["totals"]
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Admin dashboard</div>', unsafe_allow_html=True)
        st.subheader("System overview")
    with export_col:
        render_export_buttons(
            "admin_export",
            "admin_dashboard",
            json.dumps(data, indent=2, default=str),
            data,
        )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", totals["users"])
    col2.metric("Documents", totals["documents"])
    col3.metric("High risk docs", totals["high_risk_documents"])
    col4.metric("Audit events", totals["audits"])

    tab1, tab2, tab3 = st.tabs(["Users", "Documents", "Audit Events"])
    with tab1:
        render_dark_table(data["users"])
    with tab2:
        render_dark_table(data["documents"])
    with tab3:
        render_dark_table(data["audits"])


def render_ai_engineering_console(user: User) -> None:
    feedback = list_detection_feedback(user.id if user.role != "admin" else None)
    traces = list_ai_call_traces(user.id if user.role != "admin" else None)
    metrics = evaluation_metrics_from_feedback(feedback)
    capability_rows = ai_engineering_capability_rows()
    policy_rows = policy_engine_rows()
    monitoring_rows = monitoring_rows_from_traces(traces)

    st.markdown('<div class="section-kicker">AI engineering console</div>', unsafe_allow_html=True)
    st.subheader("Evaluation, feedback, monitoring, and policy controls")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Feedback labels", len(feedback))
    col2.metric("Estimated precision", f"{metrics['precision']:.0%}")
    col3.metric("False positives", metrics["false_positives"])
    col4.metric("AI traces", len(traces))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["AI Capabilities", "Evaluation", "Feedback Dataset", "Model Monitoring", "Policy Engine", "Prompt Versions"]
    )
    with tab1:
        render_dark_table(capability_rows)
    with tab2:
        render_dark_table(
            [
                {"Metric": "Correct labels", "Value": metrics["correct"]},
                {"Metric": "False positives", "Value": metrics["false_positives"]},
                {"Metric": "Missed sensitive data", "Value": metrics["missed"]},
                {"Metric": "Estimated precision", "Value": f"{metrics['precision']:.0%}"},
                {"Metric": "Review coverage", "Value": f"{metrics['coverage']:.0%}"},
            ]
        )
    with tab3:
        rows = [
            {
                "Date": item["created_at"],
                "Category": item["category"],
                "Masked value": item["masked_value"],
                "Verdict": item["verdict"],
                "Note": item["note"],
            }
            for item in feedback
        ]
        render_dark_table(rows)
        render_export_buttons("feedback_dataset_export", "active_learning_feedback", _table_rows_to_text("Feedback Dataset", rows), rows)
    with tab4:
        render_dark_table(monitoring_rows)
    with tab5:
        render_dark_table(policy_rows)
    with tab6:
        trace_rows = [
            {
                "Date": item["created_at"],
                "Feature": item["feature"],
                "Provider": item["model_provider"],
                "Model": item["model_name"],
                "Prompt": item["prompt_version"],
                "Status": item["status"],
                "Latency ms": item["latency_ms"],
                "Output hash": item["output_hash"],
            }
            for item in traces
        ]
        render_dark_table(trace_rows)


def ai_engineering_capability_rows() -> list[dict[str, str]]:
    return [
        {
            "No": "1",
            "Capability": "Hybrid AI/ML PII detection",
            "Implementation": "Regex validators plus contextual AI/ML-style entity rules and optional spaCy NER.",
            "Status": "Active",
        },
        {
            "No": "2",
            "Capability": "Semantic NLP enrichment",
            "Implementation": "Keyword/entity signals, optional HuggingFace zero-shot classifier, and confidential context scoring.",
            "Status": "Active",
        },
        {
            "No": "3",
            "Capability": "Risk classification",
            "Implementation": "Risk score combines sensitivity category, severity, confidence, secrets, finance, identity, and business context.",
            "Status": "Active",
        },
        {
            "No": "4",
            "Capability": "LangGraph AI workflow",
            "Implementation": "Context preparation, summary generation, question answering, fallback control, and provider ensemble orchestration.",
            "Status": "Active",
        },
        {
            "No": "5",
            "Capability": "RAG document QA",
            "Implementation": "Redacted chunk retrieval with local hash-vector ranking before AI answers.",
            "Status": "Active",
        },
        {
            "No": "6",
            "Capability": "Gemini + Groq ensemble",
            "Implementation": "Private environment keys drive dual-model summary and QA merging when both providers are configured.",
            "Status": "Active",
        },
        {
            "No": "7",
            "Capability": "Prompt and data safety",
            "Implementation": "Raw secrets are masked/redacted before prompts; prompt-injection text is detected as a security signal.",
            "Status": "Active",
        },
        {
            "No": "8",
            "Capability": "Human feedback loop",
            "Implementation": "Reviewers label findings as correct, false positive, or missed data for active-learning evaluation.",
            "Status": "Active",
        },
        {
            "No": "9",
            "Capability": "AI observability",
            "Implementation": "Summary and QA calls record provider, model, prompt version, latency, status, and output hash.",
            "Status": "Active",
        },
        {
            "No": "10",
            "Capability": "Policy control engine",
            "Implementation": "Findings map to DPDP, PCI-style payment safety, credential exposure, and confidential-business controls.",
            "Status": "Active",
        },
    ]


def evaluation_metrics_from_feedback(feedback: list[dict]) -> dict[str, float | int]:
    total = len(feedback)
    correct = sum(1 for item in feedback if item["verdict"] == "Correct")
    false_positives = sum(1 for item in feedback if item["verdict"] == "False positive")
    missed = sum(1 for item in feedback if item["verdict"] == "Missed sensitive data")
    reviewed_predictions = correct + false_positives
    precision = correct / reviewed_predictions if reviewed_predictions else 0.0
    coverage = total / max(1, total + 25)
    return {
        "correct": correct,
        "false_positives": false_positives,
        "missed": missed,
        "precision": precision,
        "coverage": coverage,
    }


def monitoring_rows_from_traces(traces: list[dict]) -> list[dict]:
    if not traces:
        return [{"Metric": "No AI traces yet", "Value": "Ask a question or generate a summary to populate monitoring."}]
    latencies = [int(item["latency_ms"]) for item in traces]
    fallback_count = sum(1 for item in traces if "fallback" in item["status"].lower() or item["model_provider"] == "local")
    return [
        {"Metric": "Total AI calls", "Value": len(traces)},
        {"Metric": "Average latency", "Value": f"{sum(latencies) // max(1, len(latencies))} ms"},
        {"Metric": "Max latency", "Value": f"{max(latencies)} ms"},
        {"Metric": "Fallback/local rate", "Value": f"{fallback_count / len(traces):.0%}"},
        {"Metric": "Prompt version", "Value": "compliance-v2"},
    ]


def policy_engine_rows() -> list[dict[str, str]]:
    return [
        {"Policy": "DPDP India", "Trigger": "Aadhaar, PAN, personal data, DOB, address", "Control": "Purpose, consent, minimization, retention review"},
        {"Policy": "PCI-style payment safety", "Trigger": "Credit card or payment account data", "Control": "Redact payment data and restrict storage"},
        {"Policy": "Credential exposure", "Trigger": "API keys, passwords, tokens", "Control": "Rotate credentials and review access logs"},
        {"Policy": "Confidential business", "Trigger": "NDA, trade secret, board, pricing, project codename", "Control": "Classify and restrict external sharing"},
        {"Policy": "AI prompt-injection guardrail", "Trigger": "Jailbreak or instruction override text", "Control": "Treat as untrusted content and isolate from prompts"},
    ]


def apply_theme(auth_mode: bool = False) -> None:
    block_padding = "0 0 56px 0" if auth_mode else "0 28px 56px"
    topbar_margin = "0 0 56px" if auth_mode else "0 -28px 56px"
    st.markdown(
        f"""
        <style>
        header[data-testid="stHeader"] {{
            display: none;
        }}
        .stApp {{
            background: #0d1018;
            color: #e5e7eb;
        }}
        .block-container {{
            padding: {block_padding};
            max-width: 100%;
        }}
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {{
            display: none;
        }}
        section[data-testid="stSidebar"] + div {{
            margin-left: 0;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: inherit;
        }}
        .app-topbar {{
            height: 54px;
            border-bottom: 1px solid #242938;
            display: flex;
            align-items: center;
            gap: 18px;
            margin: {topbar_margin};
            padding: 0 22px;
            background: #11141d;
            position: sticky;
            top: 0;
            z-index: 20;
        }}
        .top-brand {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #f8fafc;
            font-weight: 760;
            white-space: nowrap;
        }}
        .brand-pulse {{
            color: #fb923c;
            font-weight: 900;
        }}
        .top-product {{
            color: #64748b;
            font-size: 0.86rem;
            flex: 1;
        }}
        .topbar-inline {{
            height: 54px;
            display: flex;
            align-items: center;
            gap: 18px;
            border-bottom: 1px solid #242938;
            margin: 0 0 56px;
            padding-left: 16px;
            background: #11141d;
        }}
        .topbar-inline + div {{
            margin-top: 0;
        }}
        .top-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .top-link,
        .top-primary {{
            color: #9db2d1 !important;
            text-decoration: none;
            font-weight: 650;
            font-size: 0.92rem;
        }}
        .top-primary {{
            color: #ffffff !important;
            background: #4c9be8;
            border: 1px solid #4c9be8;
            border-radius: 8px;
            padding: 8px 12px;
        }}
        .top-user {{
            color: #9db2d1;
            border: 1px solid #242938;
            background: #141923;
            border-radius: 6px;
            padding: 5px 9px;
            font-size: 0.84rem;
        }}
        .launch-hero,
        .dashboard-hero {{
            max-width: 860px;
            text-align: center;
            margin: 0 auto 86px;
        }}
        .launch-grid {{
            max-width: 1120px;
            margin: 0 auto 28px;
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(340px, 0.95fr);
            gap: 22px;
            align-items: center;
        }}
        .launch-copy {{
            padding: 24px 0;
        }}
        .launch-copy h1 {{
            color: #f8fafc;
            font-size: 2.65rem;
            line-height: 1.08;
            margin: 0 0 16px;
            font-weight: 820;
        }}
        .launch-copy p {{
            color: #9db2d1;
            line-height: 1.65;
            font-size: 1.03rem;
            max-width: 650px;
        }}
        .trust-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 20px;
        }}
        .trust-row span {{
            border: 1px solid #2a374a;
            background: #121824;
            color: #dbeafe;
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 0.84rem;
        }}
        .live-console-card {{
            background: #111722;
            border: 1px solid #2a374a;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.25);
        }}
        .console-topline {{
            display: flex;
            justify-content: space-between;
            color: #9db2d1;
            margin-bottom: 14px;
        }}
        .console-topline strong {{
            color: #22c55e;
        }}
        .scan-meter {{
            height: 9px;
            background: #0b1220;
            border: 1px solid #263244;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 16px;
        }}
        .scan-meter span {{
            display: block;
            height: 100%;
            background: linear-gradient(90deg, #4c9be8, #fb7185);
        }}
        .console-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .console-stats div,
        .interactive-preview div {{
            background: #151a23;
            border: 1px solid #273244;
            border-radius: 8px;
            padding: 12px;
        }}
        .console-stats span,
        .interactive-preview span {{
            color: #7aa7dc;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
        }}
        .console-stats strong,
        .interactive-preview strong {{
            display: block;
            color: #f8fafc;
            margin-top: 6px;
            font-size: 1.2rem;
        }}
        .workflow-graph {{
            display: grid;
            grid-template-columns: 1fr 18px 1fr 18px 1fr 18px 1fr;
            gap: 8px;
            align-items: center;
            margin: 16px 0;
        }}
        .workflow-graph span {{
            text-align: center;
            background: #0f1724;
            border: 1px solid #2a374a;
            border-radius: 8px;
            padding: 9px 6px;
            color: #dbeafe;
            font-size: 0.82rem;
        }}
        .workflow-graph i {{
            height: 2px;
            background: #4c9be8;
        }}
        .live-console-card p {{
            color: #94a3b8;
            margin: 0;
            line-height: 1.45;
            font-size: 0.9rem;
        }}
        .interactive-preview {{
            max-width: 1120px;
            margin: 0 auto 58px;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }}
        .interactive-preview p {{
            color: #a8bad4;
            margin: 6px 0 0;
            line-height: 1.35;
            font-size: 0.9rem;
        }}
        .workflow-board {{
            max-width: 1120px;
            margin: 0 auto 78px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }}
        .dashboard-hero {{
            margin-bottom: 26px;
        }}
        .launch-badge {{
            display: inline-flex;
            border: 1px solid #26364d;
            background: #151b26;
            color: #9db2d1;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 18px;
        }}
        .launch-hero h1,
        .dashboard-hero h1 {{
            color: #f8fafc;
            font-size: 2.45rem;
            line-height: 1.12;
            margin: 0 0 16px;
            font-weight: 800;
        }}
        .launch-hero p,
        .dashboard-hero p {{
            color: #94a3b8;
            max-width: 720px;
            margin: 0 auto;
            line-height: 1.65;
            font-size: 1.02rem;
        }}
        .launch-actions,
        .hero-chip-row {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 26px;
        }}
        .launch-primary,
        .launch-secondary {{
            min-width: 132px;
            min-height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 720;
        }}
        .launch-primary {{
            color: #ffffff !important;
            background: #4c9be8;
            border: 1px solid #4c9be8;
        }}
        .launch-secondary {{
            color: #cbd5e1 !important;
            background: #10141d;
            border: 1px solid #273142;
        }}
        .section-kicker {{
            text-align: center;
            text-transform: uppercase;
            color: #7aa7dc;
            font-weight: 760;
            font-size: 0.82rem;
            margin: 0 0 20px;
        }}
        .step-grid,
        .feature-grid {{
            max-width: 1120px;
            margin: 0 auto 78px;
            display: grid;
            gap: 12px;
        }}
        .step-grid {{
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }}
        .feature-grid {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }}
        .dark-card {{
            background: #151a23;
            border: 1px solid #242c3a;
            border-radius: 8px;
            padding: 18px;
            min-height: 128px;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.15);
        }}
        .dark-card span {{
            color: #4c9be8;
            font-weight: 800;
            font-size: 0.82rem;
        }}
        .dark-card h3 {{
            color: #f8fafc;
            margin: 10px 0 8px;
            font-size: 1.02rem;
        }}
        .dark-card p {{
            color: #94a3b8;
            margin: 0;
            line-height: 1.45;
            font-size: 0.92rem;
        }}
        .console-panel {{
            max-width: 1120px;
            margin: 0 auto 22px;
            background: #111722;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 18px;
        }}
        .console-panel h3 {{
            color: #f8fafc;
            margin: 6px 0 6px;
            font-size: 1.15rem;
        }}
        .console-panel p {{
            color: #9db2d1;
            margin: 0;
            line-height: 1.5;
        }}
        .console-kicker {{
            color: #4c9be8;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }}
        .workspace-nav-card {{
            background: #111722;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }}
        .workspace-nav-card h3 {{
            color: #f8fafc;
            margin: 8px 0 6px;
            font-size: 1rem;
        }}
        .workspace-nav-card p {{
            color: #9db2d1;
            margin: 0;
            line-height: 1.45;
            font-size: 0.88rem;
        }}
        .pipeline-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }}
        .pipeline-row span {{
            border: 1px solid #2d3a4d;
            background: #151d2a;
            color: #dbeafe;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
        }}
        .mini-console {{
            max-width: 1120px;
            margin: 0 auto 22px;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }}
        .capability-grid {{
            max-width: 1120px;
            margin: -8px auto 22px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }}
        .capability-card {{
            background: #101823;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 12px;
            min-height: 118px;
        }}
        .capability-card span {{
            display: inline-flex;
            align-items: center;
            border: 1px solid #24405f;
            border-radius: 999px;
            background: #10213a;
            color: #7cc3ff;
            padding: 4px 8px;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }}
        .capability-card h3 {{
            color: #f8fafc;
            margin: 10px 0 6px;
            font-size: 0.98rem;
        }}
        .capability-card p {{
            color: #9db2d1;
            margin: 0;
            line-height: 1.42;
            font-size: 0.88rem;
        }}
        .diagnostic-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }}
        .diagnostic-card {{
            background: #101823;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 12px;
        }}
        .diagnostic-card.ready {{
            border-color: #14532d;
        }}
        .diagnostic-card.warn {{
            border-color: #854d0e;
        }}
        .diagnostic-card span {{
            color: #9db2d1;
            display: block;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .diagnostic-card strong {{
            color: #f8fafc;
            display: block;
            margin-bottom: 6px;
        }}
        .diagnostic-card.ready strong {{
            color: #86efac;
        }}
        .diagnostic-card.warn strong {{
            color: #fbbf24;
        }}
        .diagnostic-card p {{
            color: #9db2d1;
            margin: 0;
            line-height: 1.4;
            font-size: 0.86rem;
        }}
        .governance-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 18px;
        }}
        .governance-card {{
            background: #101823;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 14px;
            min-height: 130px;
        }}
        .governance-card.ready {{
            border-color: #14532d;
        }}
        .governance-card.warn {{
            border-color: #854d0e;
        }}
        .governance-card span {{
            color: #60a5fa;
            display: block;
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .governance-card strong {{
            color: #f8fafc;
            display: block;
            font-size: 1.25rem;
            margin-bottom: 8px;
        }}
        .governance-card p {{
            color: #9db2d1;
            margin: 0;
            line-height: 1.42;
            font-size: 0.9rem;
        }}
        .mini-card,
        .summary-card,
        .answer-card {{
            background: #151a23;
            border: 1px solid #242c3a;
            border-radius: 8px;
            padding: 14px;
        }}
        .mini-card span {{
            color: #4c9be8;
            font-size: 0.72rem;
            font-weight: 800;
        }}
        .mini-card strong {{
            color: #f8fafc;
            display: block;
            margin: 6px 0;
        }}
        .mini-card p,
        .summary-card li,
        .answer-card {{
            color: #a8bad4;
            line-height: 1.45;
        }}
        .summary-card {{
            margin-bottom: 12px;
        }}
        .summary-card h3 {{
            color: #f8fafc;
            margin: 0 0 8px;
            font-size: 1.05rem;
        }}
        .summary-card ul {{
            margin: 0;
            padding-left: 18px;
        }}
        .risk-card,
        .distribution-card,
        .chat-shell,
        .action-card,
        .dark-table-wrap {{
            background: #111722;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.16);
        }}
        .risk-card h3 {{
            color: #f8fafc;
            margin: 14px 0 10px;
            font-size: 1.2rem;
        }}
        .risk-card ul {{
            color: #a8bad4;
            margin: 14px 0 0;
            padding-left: 18px;
            line-height: 1.5;
        }}
        .risk-topline {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            color: #dbeafe;
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 12px;
        }}
        .risk-topline strong {{
            color: #60a5fa;
        }}
        .risk-meter {{
            height: 10px;
            overflow: hidden;
            border-radius: 999px;
            background: #1c2738;
            border: 1px solid #2b3950;
        }}
        .risk-meter span {{
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #60a5fa, #fb7185);
        }}
        .risk-stat-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }}
        .risk-stat-row div {{
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 10px;
            background: #151d2a;
        }}
        .risk-stat-row span,
        .dist-count,
        .dist-label {{
            color: #9db2d1;
            font-size: 0.84rem;
        }}
        .risk-stat-row strong {{
            display: block;
            color: #f8fafc;
            font-size: 1.35rem;
            margin-top: 4px;
        }}
        .distribution-card {{
            min-height: 260px;
        }}
        .dist-row {{
            display: grid;
            grid-template-columns: minmax(140px, 230px) 1fr 42px;
            gap: 12px;
            align-items: center;
            margin: 12px 0;
        }}
        .dist-track {{
            height: 28px;
            overflow: hidden;
            border-radius: 7px;
            background: #1b2534;
            border: 1px solid #2c3748;
        }}
        .dist-track span {{
            display: block;
            height: 100%;
            min-width: 5%;
            border-radius: inherit;
            background: linear-gradient(90deg, #2563eb, #38bdf8);
        }}
        .dist-count {{
            color: #f8fafc;
            font-weight: 800;
            text-align: right;
        }}
        .dark-table-wrap {{
            padding: 0;
            overflow: auto;
            margin: 12px 0 18px;
        }}
        .dark-table {{
            width: 100%;
            border-collapse: collapse;
            color: #dbeafe;
            min-width: 720px;
        }}
        .dark-table th {{
            color: #9db2d1;
            text-align: left;
            font-size: 0.82rem;
            font-weight: 760;
            background: #151d2a;
            border-bottom: 1px solid #263244;
            padding: 11px 12px;
        }}
        .dark-table td {{
            border-bottom: 1px solid #202b3d;
            padding: 11px 12px;
            vertical-align: top;
            line-height: 1.45;
        }}
        .dark-table tr:hover td {{
            background: #151d2a;
        }}
        .chat-shell {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            background: linear-gradient(135deg, #101827, #132033);
        }}
        .assistant-visual {{
            display: grid;
            grid-template-columns: 92px 1fr;
            gap: 16px;
            align-items: center;
            background: #101823;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 14px;
            margin: 0 0 16px;
        }}
        .ai-cube {{
            width: 62px;
            height: 62px;
            margin: 0 auto;
            position: relative;
            transform-style: preserve-3d;
            animation: cubeSpin 5s linear infinite;
        }}
        .ai-cube span {{
            position: absolute;
            inset: 0;
            border: 1px solid #60a5fa;
            background: rgba(37, 99, 235, 0.16);
            box-shadow: inset 0 0 18px rgba(96, 165, 250, 0.2);
        }}
        .ai-cube span:nth-child(1) {{ transform: translateZ(31px); }}
        .ai-cube span:nth-child(2) {{ transform: rotateY(90deg) translateZ(31px); }}
        .ai-cube span:nth-child(3) {{ transform: rotateX(90deg) translateZ(31px); }}
        .assistant-visual h3 {{
            color: #f8fafc;
            margin: 0 0 6px;
            font-size: 1.05rem;
        }}
        .assistant-visual p {{
            color: #9db2d1;
            margin: 0;
            line-height: 1.45;
        }}
        .signal-strip {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 7px;
            margin-top: 10px;
        }}
        .signal-strip span {{
            height: 7px;
            border-radius: 999px;
            background: #1d2c42;
            overflow: hidden;
            position: relative;
        }}
        .signal-strip span::after {{
            content: "";
            position: absolute;
            inset: 0;
            width: 45%;
            background: #60a5fa;
            border-radius: inherit;
            animation: signalFlow 1.8s ease-in-out infinite;
        }}
        .signal-strip span:nth-child(2)::after {{ animation-delay: 0.2s; }}
        .signal-strip span:nth-child(3)::after {{ animation-delay: 0.4s; }}
        .signal-strip span:nth-child(4)::after {{ animation-delay: 0.6s; }}
        @keyframes cubeSpin {{
            from {{ transform: rotateX(-18deg) rotateY(0deg); }}
            to {{ transform: rotateX(-18deg) rotateY(360deg); }}
        }}
        @keyframes signalFlow {{
            0% {{ transform: translateX(-120%); opacity: 0.4; }}
            50% {{ opacity: 1; }}
            100% {{ transform: translateX(230%); opacity: 0.4; }}
        }}
        .chat-shell h3 {{
            color: #f8fafc;
            margin: 6px 0;
            font-size: 1.25rem;
        }}
        .chat-shell p,
        .chat-turn p,
        .action-card p {{
            color: #a8bad4;
            margin: 0;
            line-height: 1.5;
        }}
        .chat-turn {{
            max-width: 960px;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 12px 14px;
            margin: 10px 0;
        }}
        .chat-turn span,
        .action-card span {{
            display: block;
            color: #60a5fa;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 6px;
            text-transform: uppercase;
        }}
        .chat-turn.user {{
            margin-left: auto;
            background: #172033;
        }}
        .chat-turn.assistant {{
            background: #101823;
        }}
        .action-card {{
            margin: 10px 0;
        }}
        .action-card h3 {{
            color: #f8fafc;
            margin: 0 0 6px;
            font-size: 1rem;
        }}
        .export-row {{
            margin-top: 10px;
        }}
        div[data-testid="stPopover"] button {{
            justify-content: center;
            background: #121a27 !important;
            border-color: #33435a !important;
            color: #e5e7eb !important;
        }}
        div[data-testid="stPopover"] button:hover {{
            background: #2563eb !important;
            border-color: #60a5fa !important;
        }}
        div[data-testid="stPopoverBody"],
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div {{
            background: #101823 !important;
            color: #e5e7eb !important;
            border: 1px solid #2f4058 !important;
            border-radius: 10px !important;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45) !important;
        }}
        div[data-testid="stPopoverBody"] p,
        div[data-baseweb="popover"] p {{
            color: #9db2d1 !important;
        }}
        .export-menu-title {{
            border: 1px solid #263244;
            background: #111722;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
        }}
        .export-menu-title strong {{
            display: block;
            color: #f8fafc;
            margin-bottom: 4px;
        }}
        .export-menu-title span {{
            color: #9db2d1;
            font-size: 0.82rem;
        }}
        div[data-testid="stPopoverBody"] div[data-testid="stDownloadButton"] button,
        div[data-baseweb="popover"] div[data-testid="stDownloadButton"] button {{
            background: #151d2a !important;
            color: #f8fafc !important;
            border: 1px solid #33435a !important;
            margin: 3px 0 !important;
        }}
        div[data-testid="stPopoverBody"] div[data-testid="stDownloadButton"] button:hover,
        div[data-baseweb="popover"] div[data-testid="stDownloadButton"] button:hover {{
            background: #2563eb !important;
            border-color: #60a5fa !important;
        }}
        .auth-panel {{
            width: 100%;
            margin-bottom: 18px;
        }}
        .auth-panel-title {{
            color: #f8fafc;
            font-size: 1.55rem;
            font-weight: 760;
            margin-bottom: 8px;
        }}
        .auth-panel-subtitle,
        .auth-inline {{
            color: #94a3b8;
            font-size: 0.96rem;
        }}
        .auth-inline {{
            text-align: center;
            font-size: 0.9rem;
            margin: 8px 0 0;
        }}
        .auth-inline a,
        .auth-link-row a,
        .auth-text-link {{
            color: #60a5fa !important;
            text-decoration: none;
            font-weight: 650;
        }}
        .auth-link-row {{
            text-align: center;
            margin: 8px 0 0;
        }}
        .google-auth {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
            min-height: 44px;
            border-radius: 8px;
            background: #ffffff;
            color: #111827 !important;
            border: 1px solid #e5e7eb;
            text-decoration: none;
            font-weight: 650;
            margin: 10px 0 0;
        }}
        .google-same-tab {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            min-height: 44px;
            border-radius: 8px;
            background: #ffffff;
            color: #111827 !important;
            border: 1px solid #e5e7eb;
            text-decoration: none;
            font-weight: 700;
            margin: 10px 0 0;
        }}
        .google-same-tab:hover {{
            background: #f8fafc;
            color: #111827 !important;
            text-decoration: none;
        }}
        .google-mark {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            color: #4285f4;
            font-weight: 800;
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }}
        .status-chip {{
            display: inline-block;
            border: 1px solid #26364d;
            border-radius: 999px;
            padding: 6px 11px;
            color: #b6c6df;
            background: #121824;
            font-size: 0.84rem;
        }}
        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stFileUploader"] section {{
            background: #151a23 !important;
            border: 1px solid #242c3a !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }}
        div[data-testid="stMetric"] {{
            padding: 14px 16px;
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {{
            color: #e5e7eb !important;
        }}
        div[data-testid="stRadio"] {{
            background: #11141d;
            border: 1px solid #242c3a;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 18px;
        }}
        div[data-testid="stRadio"] [role="radiogroup"] {{
            gap: 8px;
        }}
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span,
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] label p,
        div[data-testid="stFileUploader"] section p,
        div[data-testid="stFileUploader"] section span {{
            color: #dbeafe !important;
        }}
        div[data-testid="stFileUploader"] small {{
            color: #94a3b8 !important;
        }}
        div[data-testid="stTabs"] {{
            margin-top: 18px;
        }}
        button,
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {{
            border-radius: 8px !important;
            min-height: 40px;
            font-weight: 650 !important;
            background: #151a23 !important;
            color: #e5e7eb !important;
            border: 1px solid #2c3748 !important;
        }}
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stButton"] button:hover {{
            background: #4c9be8 !important;
            color: #ffffff !important;
            border-color: #4c9be8 !important;
        }}
        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stButton"] button p,
        div[data-testid="stDownloadButton"] button p {{
            color: inherit !important;
        }}
        div[data-testid="stForm"],
        div[data-testid="stAlert"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stJson"] {{
            background: #11141d;
            border-color: #242c3a;
            color: #e5e7eb;
        }}
        div[data-testid="stForm"] {{
            border: 0;
            border-radius: 8px;
            padding: 0;
            background: transparent;
        }}
        div[data-testid="stForm"] label {{
            color: #9db2d1;
            font-weight: 560;
        }}
        div[data-testid="stTextInput-RootElement"],
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: #11141d !important;
            background-color: #11141d !important;
            border-color: #2c3748 !important;
            color: #e5e7eb !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }}
        textarea,
        input {{
            color: #e5e7eb !important;
        }}
        input::placeholder,
        textarea::placeholder {{
            color: #566176 !important;
            opacity: 1;
        }}
        div[data-baseweb="input"] button {{
            background: transparent !important;
            border: 0 !important;
            color: #94a3b8 !important;
            min-width: 40px !important;
            padding: 0 12px !important;
        }}
        div[data-baseweb="input"] button svg {{
            fill: #8aa0bf !important;
        }}
        .stProgress > div > div > div > div {{
            background-color: #4c9be8;
        }}
        @media (max-width: 1000px) {{
            .step-grid,
            .workflow-board,
            .feature-grid,
            .capability-grid,
            .diagnostic-grid,
            .governance-grid,
            .mini-console,
            .interactive-preview,
            .launch-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .launch-hero h1,
            .dashboard-hero h1 {{
                font-size: 1.9rem;
            }}
        }}
        @media (max-width: 720px) {{
            .app-topbar {{
                margin-bottom: 34px;
            }}
            .top-product {{
                display: none;
            }}
            .step-grid,
            .workflow-board,
            .feature-grid,
            .capability-grid,
            .diagnostic-grid,
            .governance-grid,
            .mini-console,
            .interactive-preview,
            .launch-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(user: User) -> None:
    st.markdown(
        f"""
        <div class="app-topbar">
            <div class="top-brand"><span class="brand-pulse">~</span><span>SecureSight</span></div>
            <div class="top-product">Sensitive Data Detection & Compliance Console</div>
            <div class="top-actions">
                <span class="top-user">{escape(user.email)} &middot; {escape(user.role)}</span>
            </div>
        </div>
        <section class="dashboard-hero">
            <div class="launch-badge">Secure compliance workspace</div>
            <h1>Scan documents, explain risk, and act from one audit trail.</h1>
            <p>
                Detect Aadhaar, PAN, email, phone, card, bank, password, API key, employee ID,
                and confidential business information across uploaded documents.
            </p>
            <div class="hero-chip-row">
                <span class="status-chip">PDF / TXT / CSV</span>
                <span class="status-chip">Multi-document</span>
                <span class="status-chip">OCR fallback</span>
                <span class="status-chip">LangGraph</span>
                <span class="status-chip">RAG QA</span>
                <span class="status-chip">Audit logs</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="section-kicker">Ready for analysis</div>
        <div class="console-panel">
            <div>
                <span class="console-kicker">Live workflow</span>
                <h3>Drop documents to launch the AI compliance pipeline</h3>
                <p>The scanner will extract text, detect sensitive values, classify risk, enrich context with NLP, retrieve evidence chunks, and prepare remediation guidance.</p>
            </div>
            <div class="pipeline-row">
                <span>Extract</span><span>Detect</span><span>Classify</span><span>Retrieve</span><span>Summarize</span><span>Redact</span>
            </div>
        </div>
        <div class="feature-grid">
            <div class="dark-card"><span>DETECT</span><h3>Regulated identifiers</h3><p>Aadhaar, PAN, email, phone, card, bank, employee ID, password, API key, and confidential terms.</p></div>
            <div class="dark-card"><span>CLASSIFY</span><h3>Risk score drivers</h3><p>Severity, confidence, category counts, and business-sensitive language drive Low, Medium, or High Risk.</p></div>
            <div class="dark-card"><span>ACT</span><h3>Summary, QA, redaction</h3><p>Generate compliance observations, ask questions with RAG, download findings, and create redacted text.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_ml_stack(ai_config: AIConfig) -> None:
    llm_status = "Dual model ensemble" if ai_config.secondary_api_key else ("Private LLM active" if ai_config.api_key else "Rules fallback")
    hf_status = "Semantic classifier ready" if os.getenv("HUGGINGFACE_API_KEY", "").strip() else "Semantic classifier optional"
    spacy_status = os.getenv("SPACY_MODEL", "en_core_web_sm")
    st.markdown(
        f"""
        <div class="mini-console">
            <div class="mini-card"><span>PII ENGINE</span><strong>AI/ML entity ensemble</strong><p>spaCy/contextual NER detects people, locations, DOBs, business context, secrets, and structured IDs with validator guardrails.</p></div>
            <div class="mini-card"><span>GRAPH</span><strong>LangGraph workflow</strong><p>Context prep, summary generation, QA, and fallback control in one graph.</p></div>
            <div class="mini-card"><span>RAG</span><strong>Chunk retrieval</strong><p>LlamaIndex splitting with Chroma/hash-vector retrieval for grounded answers.</p></div>
            <div class="mini-card"><span>NLP</span><strong>{spacy_status}</strong><p>spaCy/entity-ruler signals plus confidential-business phrase detection.</p></div>
            <div class="mini-card"><span>LLM</span><strong>{llm_status}</strong><p>{hf_status}; prompts use redacted text and masked findings only.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_capability_status() -> None:
    capabilities = [
        ("OCR", "OCR fallback", "Scanned PDFs can fall back to OCR extraction when the local OCR stack is installed."),
        ("MASK", "Data masking/redaction", "Findings are masked in tables and can be exported as redacted text."),
        ("RAG", "Document chat", "Questions use retrieved chunks, redacted context, and masked findings."),
        ("MULTI", "Multi-document scans", "Upload multiple PDF, TXT, or CSV files and map findings back to source documents."),
        ("UI", "Dashboard controls", "History, audit, admin, filters, dark tables, and export menus are built into the console."),
        ("DOCKER", "Dockerization", "Dockerfile, packages, and dependency files are included for reproducible runs."),
        ("DEPLOY", "Deployment config", "Render deployment configuration is included for prototype publishing."),
        ("AUDIT", "Audit logging", "Login, scan, password, and document events are persisted for review."),
    ]
    cards = "".join(
        f"""
        <div class="capability-card">
            <span>{escape(code)}</span>
            <h3>{escape(title)}</h3>
            <p>{escape(body)}</p>
        </div>
        """
        for code, title, body in capabilities
    )
    st.markdown(
        f"""
        <div class="section-kicker">Implemented system capabilities</div>
        <div class="capability-grid">{cards}</div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_diagnostics() -> None:
    diagnostics = feature_diagnostics()
    cards = "".join(
        f"""
        <div class="diagnostic-card {'ready' if item['ready'] else 'warn'}">
            <span>{escape(item['name'])}</span>
            <strong>{escape(item['status'])}</strong>
            <p>{escape(item['detail'])}</p>
        </div>
        """
        for item in diagnostics
    )
    with st.expander("Feature diagnostics", expanded=False):
        render_html_block(f'<div class="diagnostic-grid">{cards}</div>')


def feature_diagnostics() -> list[dict[str, str | bool]]:
    pdf_renderer_ready = _module_available("pypdfium2") or _module_available("fitz")
    ocr_python_ready = pdf_renderer_ready and _module_available("pytesseract") and _module_available("PIL")
    tesseract_path = _tesseract_command() or shutil.which("tesseract")
    rag_ready = bool(retrieve_relevant_context("Confidential pricing and password reset process.", "password pricing", top_k=1))
    docker_ready = (APP_DIR / "Dockerfile").exists() and (APP_DIR / "requirements.txt").exists()
    deployment_ready = (APP_DIR / "render.yaml").exists()
    audit_ready = True
    try:
        init_db()
    except Exception:
        audit_ready = False

    return [
        {
            "name": "OCR support",
            "ready": bool(ocr_python_ready and tesseract_path),
            "status": "Ready" if ocr_python_ready and tesseract_path else "Dependency needed",
            "detail": "Tesseract and PDF renderer found." if ocr_python_ready and tesseract_path else "Install Tesseract and pypdfium2 locally or run with Docker/Render packages.",
        },
        {
            "name": "Masking/redaction",
            "ready": True,
            "status": "Ready",
            "detail": "Findings are masked and redacted previews are generated from detected spans.",
        },
        {
            "name": "RAG retrieval",
            "ready": rag_ready,
            "status": "Ready" if rag_ready else "Fallback unavailable",
            "detail": "Local hash-vector retrieval is active; Chroma can be enabled with ENABLE_CHROMA_RAG=true.",
        },
        {
            "name": "Multi-document",
            "ready": True,
            "status": "Ready",
            "detail": "Uploader accepts multiple files and maps findings back to source documents.",
        },
        {
            "name": "Dockerization",
            "ready": docker_ready,
            "status": "Ready" if docker_ready else "Missing files",
            "detail": "Dockerfile and requirements are present." if docker_ready else "Dockerfile or requirements.txt missing.",
        },
        {
            "name": "Deployment",
            "ready": deployment_ready,
            "status": "Ready" if deployment_ready else "Missing config",
            "detail": "render.yaml is present." if deployment_ready else "render.yaml missing.",
        },
        {
            "name": "Audit logging",
            "ready": audit_ready,
            "status": "Ready" if audit_ready else "Database error",
            "detail": "SQLite audit events and JSONL scan logs are available.",
        },
    ]


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def render_metrics(document, detections, risk, ai_status: str) -> None:
    counts = Counter(item.category for item in detections)
    high_count = sum(1 for item in detections if item.severity == "High")
    assistant_status = "Enhanced" if ai_status == "Enhanced analysis" else "Standard"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risk level", risk.level)
    col2.metric("Risk score", f"{risk.score}/100")
    col3.metric("Sensitive items", len(detections))
    col4.metric("High severity", high_count)
    col5.metric("Assistant", assistant_status)

    with st.expander("Document metadata", expanded=False):
        st.json(
            {
                "file_name": document.file_name,
                "file_type": document.file_type,
                **document.metadata,
                "detected_categories": dict(counts),
            }
        )


def render_intelligence_brief(document, detections, risk, report: NLPReport, ai_status: str) -> None:
    counts = Counter(item.category for item in detections)
    top_category = counts.most_common(1)[0][0] if counts else "No sensitive category"
    high_count = sum(1 for item in detections if item.severity == "High")
    semantic_status = "Semantic signals found" if report.semantic_signals else "Semantic classifier fallback"
    st.markdown(
        f"""
        <div class="console-panel">
            <div>
                <span class="console-kicker">Analysis cockpit</span>
                <h3>{risk.level} - {risk.score}/100 - {len(detections)} finding(s)</h3>
                <p>Primary exposure: {top_category}. High severity findings: {high_count}. Assistant mode: {ai_status}. NLP mode: {report.status}; {semantic_status}.</p>
            </div>
            <div class="pipeline-row">
                <span>{document.file_type}</span><span>{document.metadata.get("characters", 0)} chars</span><span>{len(counts)} categories</span><span>{high_count} high</span><span>{report.status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_overview(detections, risk) -> None:
    counts = Counter(item.category for item in detections)
    high_count = sum(1 for item in detections if item.severity == "High")
    medium_count = sum(1 for item in detections if item.severity == "Medium")
    left, right = st.columns([1, 2])

    with left:
        reasons_html = "".join(f"<li>{escape(reason)}</li>" for reason in risk.reasons)
        render_html_block(
            f"""
            <div class="risk-card">
                <div class="risk-topline"><span>Risk gauge</span><strong>{risk.score}/100</strong></div>
                <div class="risk-meter"><span style="width: {min(max(risk.score, 0), 100)}%"></span></div>
                <h3>{escape(risk.level)}</h3>
                <div class="risk-stat-row">
                    <div><span>Total</span><strong>{len(detections)}</strong></div>
                    <div><span>High</span><strong>{high_count}</strong></div>
                    <div><span>Medium</span><strong>{medium_count}</strong></div>
                </div>
                <ul>{reasons_html}</ul>
            </div>
            """
        )

    with right:
        if counts:
            max_count = max(counts.values())
            st.markdown(
                f"""
                <div class="distribution-card">
                    <div class="risk-topline"><span>Sensitive data distribution</span><strong>{len(counts)} categories</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for category, count in counts.most_common():
                ratio = count / max_count if max_count else 0
                st.progress(ratio, text=f"{category}: {count}")
        else:
            st.success("No configured sensitive-data categories were detected.")


def render_batch_dashboard(documents, detections, detection_sources: dict, risk) -> None:
    if not documents:
        return

    source_counts = Counter(detection_sources.get(item, "Current document") for item in detections)
    rows = []
    for document in documents:
        count = source_counts.get(document.file_name, 0)
        rows.append(
            {
                "Document": document.file_name,
                "Type": document.file_type,
                "Findings": count,
                "Characters": document.metadata.get("characters", len(document.text)),
                "Priority": "Review first" if count == max(source_counts.values() or [0]) and count else "Normal review",
            }
        )

    with st.expander("Batch scan dashboard", expanded=len(documents) > 1):
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", len(documents))
        col2.metric("Total findings", len(detections))
        col3.metric("Overall risk", risk.level)
        render_dark_table(rows)


def render_compliance_map(detections, risk, detection_sources: dict | None = None) -> None:
    framework_rows = compliance_framework_rows(detections, risk)
    heatmap_rows = risk_heatmap_rows(detections, detection_sources or {})
    severity_rows = severity_explanation_rows(detections)

    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Compliance framework mapping</div>', unsafe_allow_html=True)
        st.subheader("Controls, obligations, and risk evidence")
    with export_col:
        payload = {
            "framework_map": framework_rows,
            "risk_heatmap": heatmap_rows,
            "severity_explanations": severity_rows,
        }
        render_export_buttons(
            "compliance_map_export",
            "compliance_framework_map",
            _table_rows_to_text("Compliance Framework Map", framework_rows)
            + "\n\n"
            + _table_rows_to_text("Risk Heatmap", heatmap_rows)
            + "\n\n"
            + _table_rows_to_text("Severity Explanations", severity_rows),
            payload,
        )

    st.subheader("Framework mapping")
    render_dark_table(framework_rows)
    st.subheader("Risk heatmap by document")
    render_dark_table(heatmap_rows)
    st.subheader("Why each severity was assigned")
    render_dark_table(severity_rows)


def compliance_framework_rows(detections, risk) -> list[dict[str, str]]:
    categories = {item.category for item in detections}
    rows: list[dict[str, str]] = []
    rules = [
        (
            "DPDP Act India",
            {"Aadhaar Number", "PAN Number", "Email Address", "Phone Number", "Person Name", "Date of Birth", "Address / Location"},
            "Personal data processing",
            "Verify purpose, consent/notice, minimization, retention, and access controls.",
        ),
        (
            "PCI-DSS style payment safety",
            {"Credit Card Number", "Bank Account Number", "IFSC Code"},
            "Payment or account data",
            "Restrict access, avoid storage, redact payment/account fields, and monitor exports.",
        ),
        (
            "ISO 27001 A.5/A.8",
            {"API Key", "Password", "Employee ID"},
            "Identity, access, and asset protection",
            "Rotate credentials, review access, classify assets, and record remediation evidence.",
        ),
        (
            "Confidentiality / NDA",
            {"Confidential Business Information", "Confidential Project Name", "Organization Name"},
            "Business confidential information",
            "Apply confidential classification and limit sharing to approved reviewers.",
        ),
        (
            "AI Safety Guardrail",
            {"Prompt Injection Attempt"},
            "Untrusted document instructions",
            "Do not follow embedded instructions; isolate content from system prompts.",
        ),
    ]
    for framework, trigger_categories, obligation, control in rules:
        matched = sorted(categories & trigger_categories)
        rows.append(
            {
                "Framework": framework,
                "Matched": ", ".join(matched) if matched else "No direct match",
                "Obligation": obligation,
                "Recommended control": control,
                "Priority": "High" if matched and risk.level == "High Risk" else "Medium" if matched else "Monitor",
            }
        )
    return rows


def risk_heatmap_rows(detections, detection_sources: dict) -> list[dict[str, str | int]]:
    groups = {
        "Identity": {"Aadhaar Number", "PAN Number", "Person Name", "Date of Birth", "Address / Location"},
        "Financial": {"Credit Card Number", "Bank Account Number", "IFSC Code"},
        "Credentials": {"API Key", "Password", "Employee ID"},
        "Confidential": {"Confidential Business Information", "Confidential Project Name", "Organization Name"},
        "AI Guardrail": {"Prompt Injection Attempt"},
    }
    documents = sorted({detection_sources.get(item, "Current document") for item in detections}) or ["Current document"]
    rows = []
    for document in documents:
        document_items = [item for item in detections if detection_sources.get(item, "Current document") == document]
        row: dict[str, str | int] = {"Document": document}
        for group, categories in groups.items():
            row[group] = sum(1 for item in document_items if item.category in categories)
        row["Total"] = len(document_items)
        row["Risk signal"] = "Critical" if row["Credentials"] or row["Financial"] else "High" if row["Identity"] or row["Confidential"] else "Monitor"
        rows.append(row)
    return rows


def severity_explanation_rows(detections) -> list[dict[str, str]]:
    rows = []
    for item in detections[:80]:
        if item.severity == "High":
            why = "High because this category can directly enable identity fraud, account takeover, payment misuse, or AI prompt abuse."
        elif item.severity == "Medium":
            why = "Medium because this is personal, employee, organizational, or business-sensitive context that needs controlled handling."
        else:
            why = "Low because it is a contextual signal that should be reviewed with surrounding evidence."
        rows.append(
            {
                "Category": item.category,
                "Masked value": mask_value(item.value),
                "Severity": item.severity,
                "Confidence": f"{item.confidence:.0%}",
                "Explanation": why,
                "Next action": recommended_action_for_category(item.category),
            }
        )
    return rows


def recommended_action_for_category(category: str) -> str:
    if category in {"API Key", "Password"}:
        return "Rotate/revoke credential and investigate access logs."
    if category in {"Aadhaar Number", "PAN Number", "Credit Card Number", "Bank Account Number", "IFSC Code"}:
        return "Redact before sharing and restrict document access."
    if category == "Prompt Injection Attempt":
        return "Treat text as untrusted and exclude it from control prompts."
    if category in {"Confidential Business Information", "Confidential Project Name"}:
        return "Apply confidential classification and review sharing permissions."
    return "Review, minimize, and keep only if business purpose is valid."


def render_findings(user: User, analysis_id: str, detections, detection_sources: dict | None = None) -> None:
    if not detections:
        st.success("No configured sensitive-data indicators were detected.")
        return

    rows = [
        {
            "Document": (detection_sources or {}).get(item, "Current document"),
            "Category": item.category,
            "Masked value": mask_value(item.value),
            "Severity": item.severity,
            "Confidence": f"{item.confidence:.0%}",
            "Reason": item.reason,
        }
        for item in detections
    ]
    categories = sorted({row["Category"] for row in rows})
    severities = sorted({row["Severity"] for row in rows})
    documents = sorted({row["Document"] for row in rows})
    col1, col2, col3 = st.columns(3)
    selected_categories = col1.multiselect("Filter by category", categories, default=categories)
    selected_severities = col2.multiselect("Filter by severity", severities, default=severities)
    selected_documents = col3.multiselect("Filter by document", documents, default=documents)
    filtered_rows = [
        row
        for row in rows
        if row["Category"] in selected_categories
        and row["Severity"] in selected_severities
        and row["Document"] in selected_documents
    ]
    export_payload = {
        "title": "Sensitive Data Findings",
        "rows": filtered_rows,
    }
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Detected sensitive items</div>', unsafe_allow_html=True)
    with export_col:
        render_export_buttons(
            "findings_export",
            "sensitive_data_findings",
            _table_rows_to_text("Sensitive Data Findings", filtered_rows),
            export_payload,
        )
    render_dark_table(filtered_rows)
    render_feedback_loop(user, analysis_id, filtered_rows)


def render_summary(summary: dict[str, list[str]], ai_status: str) -> None:
    summary_text = _summary_to_text(summary, ai_status)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Compliance summary</div>', unsafe_allow_html=True)
        st.caption(f"Generated with: {'Enhanced analysis' if ai_status == 'Enhanced analysis' else 'Standard analysis'}")
    with export_col:
        render_export_buttons(
            "summary_export",
            "compliance_summary",
            summary_text,
            {"status": ai_status, "summary": summary},
        )
    for section, items in summary.items():
        body = "".join(f"<li>{escape(item)}</li>" for item in items)
        st.markdown(
            f"""
            <div class="summary-card">
                <h3>{escape(section)}</h3>
                <ul>{body}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feedback_loop(user: User, analysis_id: str, rows: list[dict]) -> None:
    st.markdown('<div class="section-kicker">Human feedback loop</div>', unsafe_allow_html=True)
    if not rows:
        st.info("No finding is available for feedback.")
        return

    options = [f"{index + 1}. {row['Category']} | {row['Masked value']}" for index, row in enumerate(rows[:50])]
    with st.form(f"feedback_form_{analysis_id}"):
        selected = st.selectbox("Finding", options)
        verdict = st.radio("Feedback", ["Correct", "False positive", "Missed sensitive data"], horizontal=True)
        note = st.text_input("Reviewer note", placeholder="Optional evidence, correction, or missing data type")
        submitted = st.form_submit_button("Save feedback", use_container_width=True)
    if submitted:
        row = rows[options.index(selected)]
        save_detection_feedback(
            user.id,
            analysis_id,
            row["Category"],
            row["Masked value"],
            verdict,
            note,
        )
        st.success("Feedback saved for model evaluation and active learning.")


def render_question_answering(user: User, analysis_id: str, text: str, detections, risk, ai_config: AIConfig) -> None:
    examples = [
        "What sensitive data exists in the document?",
        "How many email addresses are present?",
        "Summarize this document.",
        "What compliance risks are identified?",
        "Which items should be remediated first?",
        "Create a short audit note for this scan.",
    ]
    if "active_question" not in st.session_state:
        st.session_state.active_question = ""
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    st.markdown(
        """
        <div class="chat-shell">
            <div>
                <span class="console-kicker">AI document chat</span>
                <h3>Ask over the uploaded document without exposing raw secrets</h3>
                <p>Answers use masked findings, redacted text, risk context, and retrieved chunks from the document.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode_label = "Instant scan answers + AI/RAG for open questions" if ai_config.enabled else "Instant scan answers"
    st.markdown(
        f"""
        <div class="assistant-visual">
            <div class="ai-cube"><span></span><span></span><span></span></div>
            <div>
                <h3>SecureSight Copilot</h3>
                <p>{escape(mode_label)}. Common compliance questions answer immediately from this upload; deeper questions use retrieved redacted context.</p>
                <div class="signal-strip"><span></span><span></span><span></span><span></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for index, example in enumerate(examples):
        if cols[index % 3].button(example, use_container_width=True):
            st.session_state.active_question = example

    custom_question = st.text_input("Ask a question", value=st.session_state.active_question, placeholder="Example: What should be remediated first?")
    ask_col, clear_col = st.columns([5, 1])
    with ask_col:
        submitted = st.button("Ask AI", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.active_question = ""
            st.session_state.qa_history = []
            st.rerun()
    active_question = custom_question.strip()

    if active_question and submitted:
        start = time.perf_counter()
        answer, ai_status = answer_question_with_ai(active_question, text, detections, risk, ai_config)
        evidence_chunks = retrieve_relevant_context(redact_text(text, detections), active_question, top_k=3)
        latency_ms = int((time.perf_counter() - start) * 1000)
        save_ai_trace_safely(
            user,
            analysis_id,
            feature="qa",
            ai_config=ai_config,
            status=ai_status,
            latency_ms=latency_ms,
            output=answer,
            metadata={"question": active_question, "detections": len(detections), "risk": risk.level},
        )
        st.caption(f"Answered with: {'Enhanced analysis' if ai_status == 'Enhanced analysis' else 'Standard analysis'}")
        st.session_state.qa_history.append(
            {"question": active_question, "answer": answer, "status": ai_status, "evidence": evidence_chunks}
        )

    if st.session_state.qa_history:
        latest = st.session_state.qa_history[-1]
        qa_text = f"Question: {latest['question']}\n\nAnswer:\n{latest['answer']}"
        header_col, export_col = st.columns([6, 2])
        with header_col:
            st.markdown('<div class="section-kicker">Conversation</div>', unsafe_allow_html=True)
        with export_col:
            render_export_buttons(
                "qa_export",
                "document_question_answer",
                qa_text,
                latest,
            )
        for item in st.session_state.qa_history[-6:]:
            evidence_html = "".join(
                f"<li>{escape(chunk[:280])}</li>"
                for chunk in item.get("evidence", [])[:3]
            )
            st.markdown(
                f"""
                <div class="chat-turn user"><span>You</span><p>{escape(item['question'])}</p></div>
                <div class="chat-turn assistant"><span>Assistant &middot; {escape(item.get('status', 'Standard analysis'))}</span><p>{escape(item['answer'])}</p><ul>{evidence_html}</ul></div>
                """,
                unsafe_allow_html=True,
            )


def render_action_plan(detections, risk, report: NLPReport) -> None:
    action_rows = build_action_plan_rows(detections, risk, report)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Recommended response plan</div>', unsafe_allow_html=True)
    with export_col:
        render_export_buttons(
            "action_plan_export",
            "remediation_action_plan",
            _table_rows_to_text("Recommended Response Plan", action_rows),
            {"risk": risk.level, "actions": action_rows},
        )
    st.markdown(
        "".join(
            f"""
            <div class="action-card">
                <span>{escape(row['Priority'])}</span>
                <h3>{escape(row['Action'])}</h3>
                <p>{escape(row['Evidence'])}</p>
            </div>
            """
            for row in action_rows
        ),
        unsafe_allow_html=True,
    )


def render_reviewer_workflow(user: User, analysis_id: str, detections, risk, report: NLPReport) -> None:
    review_rows = reviewer_workflow_rows(detections, risk, report)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Reviewer workflow</div>', unsafe_allow_html=True)
        st.subheader("Assign, track, and record remediation decisions")
    with export_col:
        render_export_buttons(
            "reviewer_workflow_export",
            "reviewer_workflow",
            _table_rows_to_text("Reviewer Workflow", review_rows),
            {"analysis_id": analysis_id, "items": review_rows},
        )

    render_dark_table(review_rows)
    with st.form(f"reviewer_decision_{analysis_id}"):
        status = st.selectbox("Remediation status", ["Open", "In progress", "Fixed", "Accepted risk"])
        owner = st.text_input("Owner", placeholder="Example: Compliance reviewer")
        note = st.text_area("Reviewer decision note", placeholder="What changed, what remains, or why risk was accepted")
        submitted = st.form_submit_button("Record reviewer decision", use_container_width=True)
    if submitted:
        log_audit(
            user.id,
            "reviewer_decision_recorded",
            {
                "analysis_id": analysis_id,
                "status": status,
                "owner": owner,
                "note": note,
                "risk": risk.level,
                "detections": len(detections),
            },
        )
        st.success("Reviewer decision saved to audit logs.")


def reviewer_workflow_rows(detections, risk, report: NLPReport) -> list[dict[str, str]]:
    categories = sorted({item.category for item in detections})
    rows = []
    if any(category in {"API Key", "Password"} for category in categories):
        rows.append({"Owner": "Security", "Task": "Rotate exposed credentials and invalidate old tokens.", "Status": "Open", "Priority": "P0", "Evidence": "Credential-like findings detected."})
    if any(category in {"Aadhaar Number", "PAN Number", "Date of Birth", "Person Name"} for category in categories):
        rows.append({"Owner": "Privacy", "Task": "Validate purpose, retention, and access basis for personal data.", "Status": "Open", "Priority": "P1", "Evidence": "Personal-data categories detected."})
    if any(category in {"Credit Card Number", "Bank Account Number", "IFSC Code"} for category in categories):
        rows.append({"Owner": "Finance", "Task": "Create finance-safe redacted copy and restrict account data.", "Status": "Open", "Priority": "P1", "Evidence": "Payment or bank-data categories detected."})
    if any(category in {"Confidential Business Information", "Confidential Project Name"} for category in categories):
        rows.append({"Owner": "Legal/Business", "Task": "Apply confidential classification and review external sharing.", "Status": "Open", "Priority": "P1", "Evidence": "Business-confidential context detected."})
    rows.append({"Owner": "Compliance", "Task": "Attach final report and reviewer decision to audit record.", "Status": "Open", "Priority": "P2" if risk.level != "High Risk" else "P1", "Evidence": f"{risk.level}; NLP mode {report.status}."})
    return rows


def render_policy_check(text: str, detections, risk) -> None:
    st.markdown('<div class="section-kicker">Policy-aware compliance check</div>', unsafe_allow_html=True)
    st.subheader("Compare findings against built-in or uploaded policy text")
    policy_file = st.file_uploader("Upload internal policy TXT/CSV/PDF", type=["txt", "csv", "pdf"], key="policy_upload")
    policy_text = ""
    if policy_file:
        try:
            policy_text = load_document(policy_file.name, policy_file.getvalue()).text
        except Exception as exc:
            st.error(f"Policy upload could not be read: {exc}")
            policy_text = ""
    rows = policy_check_rows(policy_text, detections, risk)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.caption("Uses uploaded policy text when available; otherwise uses the built-in compliance baseline.")
    with export_col:
        render_export_buttons(
            "policy_check_export",
            "policy_compliance_check",
            _table_rows_to_text("Policy Compliance Check", rows),
            {"risk": risk.level, "policy_uploaded": bool(policy_text), "rows": rows},
        )
    render_dark_table(rows)
    if policy_text:
        st.subheader("Relevant policy evidence")
        evidence = retrieve_relevant_context(policy_text, "redact retention access consent credentials payment confidential", top_k=3)
        for index, chunk in enumerate(evidence, start=1):
            st.markdown(f'<div class="answer-card"><strong>Policy excerpt {index}</strong><p>{escape(chunk[:900])}</p></div>', unsafe_allow_html=True)


def policy_check_rows(policy_text: str, detections, risk) -> list[dict[str, str]]:
    lowered_policy = policy_text.lower()
    categories = {item.category for item in detections}
    baseline_controls = [
        ("Access control", {"API Key", "Password", "Employee ID", "Confidential Business Information"}, "Restrict access to least privilege."),
        ("Redaction", {"Aadhaar Number", "PAN Number", "Credit Card Number", "Bank Account Number", "IFSC Code", "Email Address", "Phone Number"}, "Mask or redact sensitive fields before sharing."),
        ("Retention", set(categories), "Store documents only as long as required for review."),
        ("Incident response", {"API Key", "Password", "Prompt Injection Attempt"}, "Escalate credentials or hostile prompts to security review."),
        ("Data subject/privacy review", {"Aadhaar Number", "PAN Number", "Person Name", "Date of Birth", "Address / Location"}, "Verify purpose and legal basis for personal data."),
    ]
    rows = []
    for control, trigger_categories, recommendation in baseline_controls:
        matched = sorted(categories & trigger_categories)
        if policy_text:
            keywords = [word for word in control.lower().split() + recommendation.lower().split() if len(word) > 5]
            policy_match = any(keyword in lowered_policy for keyword in keywords)
        else:
            policy_match = True
        rows.append(
            {
                "Control": control,
                "Triggered by": ", ".join(matched) if matched else "No direct trigger",
                "Policy evidence": "Matched uploaded policy" if policy_text and policy_match else "Built-in baseline" if not policy_text else "No clear policy match",
                "Recommendation": recommendation,
                "Priority": "High" if matched and risk.level == "High Risk" else "Medium" if matched else "Monitor",
            }
        )
    return rows


def render_nlp_insights(report: NLPReport) -> None:
    nlp_payload = {
        "status": report.status,
        "entities": [item.__dict__ for item in report.entities],
        "semantic_signals": [item.__dict__ for item in report.semantic_signals],
    }
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">NLP intelligence</div>', unsafe_allow_html=True)
        st.caption(f"Insight mode: {report.status}")
    with export_col:
        render_export_buttons("nlp_export", "nlp_insights", json.dumps(nlp_payload, indent=2), nlp_payload)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Extracted context signals")
        if report.entities:
            rows = [
                {"Signal": item.label, "Text": item.text, "Count": item.count}
                for item in report.entities
            ]
            render_dark_table(rows)
        else:
            st.info("No additional NLP signals were found.")

    with col2:
        st.subheader("Semantic risk signals")
        if report.semantic_signals:
            rows = [
                {"Signal": item.label, "Score": f"{item.score:.0%}"}
                for item in report.semantic_signals
            ]
            render_dark_table(rows)
        else:
            st.info("Semantic enrichment is available when a private classifier key is configured.")


def render_ai_governance(document, detections, risk, report: NLPReport, ai_status: str, ai_config: AIConfig) -> None:
    guardrail_count = sum(1 for item in detections if item.category == "Prompt Injection Attempt")
    ml_count = sum(1 for item in detections if _is_ml_detection(item))
    validator_count = len(detections) - ml_count
    avg_confidence = int((sum(item.confidence for item in detections) / max(1, len(detections))) * 100)
    model_mode = "AI/ML ensemble active" if ai_config.enabled else "Local AI/ML fallback"
    trace_rows = build_ai_trace_rows(document, detections, risk, report, ai_status, ai_config)
    governance_text = _table_rows_to_text("AI Governance Trace", trace_rows)

    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">AI governance and model trace</div>', unsafe_allow_html=True)
    with export_col:
        render_export_buttons(
            "ai_governance_export",
            "ai_governance_trace",
            governance_text,
            {
                "model_mode": model_mode,
                "average_confidence": avg_confidence,
                "guardrail_count": guardrail_count,
                "trace": trace_rows,
            },
        )

    st.markdown(
        f"""
        <div class="governance-grid">
            <div class="governance-card"><span>MODE</span><strong>{escape(model_mode)}</strong><p>LLM calls use redacted context and masked findings; instant answers use local scan evidence.</p></div>
            <div class="governance-card"><span>CONFIDENCE</span><strong>{avg_confidence}%</strong><p>Average detector confidence across validator and AI/ML findings.</p></div>
            <div class="governance-card"><span>AI/ML ENTITIES</span><strong>{ml_count}</strong><p>Contextual NER and semantic entity findings.</p></div>
            <div class="governance-card {'warn' if guardrail_count else 'ready'}"><span>GUARDRAILS</span><strong>{guardrail_count} signal(s)</strong><p>{'Prompt-injection text detected and treated as untrusted content.' if guardrail_count else 'No prompt-injection text detected.'}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_dark_table(trace_rows)


def build_ai_trace_rows(document, detections, risk, report: NLPReport, ai_status: str, ai_config: AIConfig) -> list[dict[str, str]]:
    guardrail_count = sum(1 for item in detections if item.category == "Prompt Injection Attempt")
    return [
        {
            "Stage": "1. Document ingestion",
            "AI/ML role": "OCR/text extraction prepares machine-readable context.",
            "Evidence": f"{document.file_type}; {document.metadata.get('characters', 0)} chars; OCR={document.metadata.get('ocr_status', 'not_applicable')}",
            "Status": "Ready",
        },
        {
            "Stage": "2. Entity detection",
            "AI/ML role": "spaCy/contextual NER plus validator guardrails detect sensitive entities.",
            "Evidence": f"{len(detections)} finding(s), {len(set(item.category for item in detections))} categories.",
            "Status": "Ready",
        },
        {
            "Stage": "3. Risk model",
            "AI/ML role": "Weighted risk scoring combines severity, frequency, categories, and contextual entities.",
            "Evidence": f"{risk.level}; score {risk.score}/100.",
            "Status": "Ready",
        },
        {
            "Stage": "4. RAG retrieval",
            "AI/ML role": "Local hash-vector retrieval selects relevant redacted chunks for QA.",
            "Evidence": "RAG enabled with local fallback; Chroma optional.",
            "Status": "Ready",
        },
        {
            "Stage": "5. LLM reasoning",
            "AI/ML role": "Private model ensemble generates summaries and open-ended answers when configured.",
            "Evidence": ai_status,
            "Status": "Ready" if ai_config.enabled else "Local fallback",
        },
        {
            "Stage": "6. Guardrails",
            "AI/ML role": "Prompt-injection detector flags adversarial document instructions.",
            "Evidence": f"{guardrail_count} guardrail signal(s).",
            "Status": "Review" if guardrail_count else "Clear",
        },
    ]


def _is_ml_detection(item) -> bool:
    return (
        "AI/ML contextual" in item.reason
        or "spaCy NER" in item.reason
        or item.category in {"Person Name", "Date of Birth", "Address / Location", "Organization Name", "Confidential Project Name"}
    )


def render_redaction(text: str, detections, file_name: str) -> None:
    redacted = redact_text(text, detections)
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Before/after redaction preview</div>', unsafe_allow_html=True)
    with export_col:
        render_export_buttons(
            "redacted_export",
            f"{Path(file_name).stem}_redacted",
            redacted,
            {"file_name": file_name, "redacted_text": redacted},
        )
    before_col, after_col = st.columns(2)
    with before_col:
        st.caption("Original extracted text")
        st.text_area("Original extracted text", text, height=420, label_visibility="collapsed")
    with after_col:
        st.caption("Safe redacted version")
        st.text_area("Safe redacted version", redacted, height=420, label_visibility="collapsed")


def render_final_report(document, detections, risk, summary: dict[str, list[str]], report: NLPReport, detection_sources: dict, ai_status: str) -> None:
    framework_rows = compliance_framework_rows(detections, risk)
    action_rows = build_action_plan_rows(detections, risk, report)
    review_rows = reviewer_workflow_rows(detections, risk, report)
    policy_rows = policy_check_rows("", detections, risk)
    report_text = build_final_report_text(
        document,
        detections,
        risk,
        summary,
        report,
        framework_rows,
        action_rows,
        review_rows,
        policy_rows,
        detection_sources,
        ai_status,
    )
    header_col, export_col = st.columns([6, 2])
    with header_col:
        st.markdown('<div class="section-kicker">Final compliance report</div>', unsafe_allow_html=True)
        st.subheader("Submission-ready security and compliance package")
    with export_col:
        render_export_buttons(
            "final_report_export",
            f"{Path(document.file_name).stem}_compliance_report",
            report_text,
            {
                "document": document.file_name,
                "risk": risk.level,
                "score": risk.score,
                "summary": summary,
                "framework_map": framework_rows,
                "actions": action_rows,
                "review_workflow": review_rows,
                "policy_check": policy_rows,
            },
        )
    st.text_area("Report preview", report_text, height=520, label_visibility="collapsed")


def build_final_report_text(
    document,
    detections,
    risk,
    summary: dict[str, list[str]],
    report: NLPReport,
    framework_rows: list[dict],
    action_rows: list[dict],
    review_rows: list[dict],
    policy_rows: list[dict],
    detection_sources: dict,
    ai_status: str,
) -> str:
    lines = [
        "Sensitive Data Detection & Compliance Assistant - Final Report",
        "",
        "Executive Summary",
        f"Document: {document.file_name}",
        f"File type: {document.file_type}",
        f"Risk: {risk.level} ({risk.score}/100)",
        f"Findings: {len(detections)} sensitive item(s)",
        f"AI mode: {ai_status}",
        f"NLP mode: {report.status}",
        "",
        "Detected Categories",
    ]
    for category, count in Counter(item.category for item in detections).most_common():
        lines.append(f"- {category}: {count}")
    if not detections:
        lines.append("- No configured sensitive data detected.")

    lines.extend(["", "Compliance Observations"])
    for item in summary.get("Compliance observations", []):
        lines.append(f"- {item}")

    lines.extend(["", "Security Risks"])
    for item in summary.get("Security risks", []):
        lines.append(f"- {item}")

    lines.extend(["", "Suggested Remediation"])
    for item in summary.get("Suggested remediation steps", []):
        lines.append(f"- {item}")

    lines.extend(["", "Framework Mapping"])
    for row in framework_rows:
        lines.append(f"- {row['Framework']}: {row['Matched']} | {row['Recommended control']}")

    lines.extend(["", "Action Plan"])
    for row in action_rows:
        lines.append(f"- {row['Priority']}: {row['Action']} Evidence: {row['Evidence']}")

    lines.extend(["", "Reviewer Workflow"])
    for row in review_rows:
        lines.append(f"- {row['Priority']} / {row['Owner']}: {row['Task']} ({row['Status']})")

    lines.extend(["", "Policy Check"])
    for row in policy_rows:
        lines.append(f"- {row['Control']}: {row['Triggered by']} | {row['Recommendation']}")

    lines.extend(["", "Finding Evidence"])
    for item in detections[:30]:
        source = detection_sources.get(item, "Current document")
        lines.append(f"- {source}: {item.category} | {mask_value(item.value)} | {item.severity} | {item.reason}")

    lines.extend(
        [
            "",
            "Audit Note",
            "This report uses masked findings and redacted context. Raw sensitive values should not be shared externally.",
        ]
    )
    return "\n".join(lines)


def render_dark_table(rows: list[dict]) -> None:
    if not rows:
        st.info("No rows match the current filters.")
        return

    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{escape(str(row.get(header, '')))}</td>" for header in headers)
        body_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <div class="dark-table-wrap">
            <table class="dark-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_block(html: str) -> None:
    cleaned = textwrap.dedent(html).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


def build_action_plan_rows(detections, risk, report: NLPReport) -> list[dict[str, str]]:
    categories = sorted({item.category for item in detections})
    high_categories = sorted({item.category for item in detections if item.severity == "High"})
    rows: list[dict[str, str]] = []

    if high_categories:
        rows.append(
            {
                "Priority": "P0",
                "Action": "Quarantine sharing until the data owner approves a redacted version.",
                "Evidence": f"High-severity exposure: {', '.join(high_categories)}.",
            }
        )

    if any(item.category == "Prompt Injection Attempt" for item in detections):
        rows.append(
            {
                "Priority": "P0",
                "Action": "Treat embedded AI instructions as hostile document content and keep them out of system/developer prompts.",
                "Evidence": "Prompt-injection or jailbreak language was detected.",
            }
        )

    if any(item.category in {"API Key", "Password"} for item in detections):
        rows.append(
            {
                "Priority": "P0",
                "Action": "Rotate exposed credentials, revoke old tokens, and check recent access logs.",
                "Evidence": "Credential-like values were detected.",
            }
        )

    if any(item.category in {"Credit Card Number", "Bank Account Number", "IFSC Code"} for item in detections):
        rows.append(
            {
                "Priority": "P1",
                "Action": "Create a finance-safe copy with account, card, and IFSC fields redacted.",
                "Evidence": "Financial identifiers were detected.",
            }
        )

    if any(item.category in {"Aadhaar Number", "PAN Number"} for item in detections):
        rows.append(
            {
                "Priority": "P1",
                "Action": "Verify legal basis and retention period before storing or forwarding the document.",
                "Evidence": "Indian national identifiers were detected.",
            }
        )

    if any(item.category == "Confidential Business Information" for item in detections):
        rows.append(
            {
                "Priority": "P1",
                "Action": "Apply confidential classification and limit external distribution.",
                "Evidence": "Business-sensitive terms were detected.",
            }
        )

    if categories:
        rows.append(
            {
                "Priority": "P2",
                "Action": "Export and attach the redacted scan result to the review ticket.",
                "Evidence": f"Detected categories: {', '.join(categories)}.",
            }
        )

    semantic_labels = ", ".join(item.label for item in report.semantic_signals[:3]) or report.status
    rows.append(
        {
            "Priority": "P2" if risk.level != "High Risk" else "P1",
            "Action": "Record reviewer decision, remediation owner, and retention outcome in audit logs.",
            "Evidence": f"NLP signal: {semantic_labels}.",
        }
    )
    deduped = []
    seen = set()
    for row in rows:
        key = (row["Priority"], row["Action"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def render_export_buttons(key_prefix: str, base_name: str, text_content: str, json_payload: object) -> None:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in base_name).strip("_") or "export"
    formats = {
        "PDF": (_simple_pdf_bytes(base_name.replace("_", " ").title(), text_content), f"{safe_name}.pdf", "application/pdf"),
        "TXT": (text_content, f"{safe_name}.txt", "text/plain"),
        "JSON": (json.dumps(json_payload, indent=2, default=str), f"{safe_name}.json", "application/json"),
        "HTML": (_html_report(base_name.replace("_", " ").title(), text_content), f"{safe_name}.html", "text/html"),
    }
    selected_format = st.selectbox(
        "Export format",
        list(formats),
        key=f"{key_prefix}_format",
        label_visibility="collapsed",
    )
    data, file_name, mime = formats[selected_format]
    st.download_button(
        "Download",
        data,
        file_name=file_name,
        mime=mime,
        key=f"{key_prefix}_download",
        use_container_width=True,
        help=f"Download {selected_format}",
    )


def _table_rows_to_text(title: str, rows: list[dict]) -> str:
    lines = [title, ""]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}.")
        for key, value in row.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    return "\n".join(lines).strip()


def _summary_to_text(summary: dict[str, list[str]], ai_status: str) -> str:
    lines = [f"Generated with: {ai_status}", ""]
    for section, items in summary.items():
        lines.append(section)
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _html_report(title: str, text_content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #0f172a; line-height: 1.5; }}
    h1 {{ font-size: 24px; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #e2e8f0; padding: 18px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <pre>{escape(text_content)}</pre>
</body>
</html>"""


def _simple_pdf_bytes(title: str, text_content: str) -> bytes:
    raw_lines = [title, ""] + text_content.replace("\r", "").split("\n")
    wrapped_lines: list[str] = []
    for line in raw_lines:
        wrapped = textwrap.wrap(line, width=88) or [""]
        wrapped_lines.extend(wrapped)

    pages = [wrapped_lines[index : index + 48] for index in range(0, len(wrapped_lines), 48)] or [[""]]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_refs: list[int] = []

    for page_lines in pages:
        stream_lines = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        stream_lines.extend(f"({_pdf_escape(line)}) Tj T*" for line in page_lines)
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        content_ref = len(objects) + 1
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
        page_ref = len(objects) + 1
        page_refs.append(page_ref)
        page = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            + b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_ref).encode("ascii")
            + b" 0 R >>"
        )
        objects.append(page)

    kids = b" ".join(f"{ref} 0 R".encode("ascii") for ref in page_refs)
    objects[1] = b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_refs)).encode("ascii") + b" >>"

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, content in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(content)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("ascii")
    )
    return bytes(output)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_audit_event(user: User, documents, detections, risk_level: str) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user_id": user.id,
        "user_email": user.email,
        "file_count": len(documents),
        "file_names": [item.file_name for item in documents],
        "file_types": sorted({item.file_type for item in documents}),
        "detections": len(detections),
        "high_severity_detections": sum(1 for item in detections if item.severity == "High"),
        "categories": dict(Counter(item.category for item in detections)),
        "risk_level": risk_level,
        "ocr_status": {item.file_name: item.metadata.get("ocr_status", "not_applicable") for item in documents},
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    main()
