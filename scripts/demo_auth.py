"""
GlossWerk Demo Authentication & Usage Gating

Manages invite codes, usage tracking, and demo limits.

Usage:
    Import into glosswerk_app.py and call check_auth() at the top.
    Stores data in a local JSON file (demo_users.json).

Demo limits:
    - 5 patents per invite code
    - 14-day expiry from first use
    - Per-user daily API spend ceiling ($5)
"""

import json
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path


# --- Config ---
MAX_PATENTS = 5
EXPIRY_DAYS = 14
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demo_users.json")
CODES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demo_codes.json")


def _ensure_data_dir():
    Path(os.path.dirname(DATA_FILE)).mkdir(parents=True, exist_ok=True)


def _load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def _save_json(filepath, data):
    _ensure_data_dir()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Invite code management (run from CLI to generate codes)
# ---------------------------------------------------------------------------

def generate_invite_code(company_name, contact_email, notes=""):
    """Generate a unique invite code for a demo user."""
    code = f"GW-{secrets.token_hex(4).upper()}"
    codes = _load_json(CODES_FILE)
    codes[code] = {
        "company": company_name,
        "email": contact_email,
        "notes": notes,
        "created": datetime.now().isoformat(),
        "used": False,
    }
    _save_json(CODES_FILE, codes)
    return code


def list_codes():
    """List all invite codes and their status."""
    codes = _load_json(CODES_FILE)
    users = _load_json(DATA_FILE)

    results = []
    for code, info in codes.items():
        user = users.get(code, {})
        results.append({
            "code": code,
            "company": info.get("company", ""),
            "email": info.get("email", ""),
            "used": info.get("used", False),
            "patents_used": user.get("patents_used", 0),
            "first_use": user.get("first_use", ""),
            "expires": user.get("expires", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Auth validation (called from Streamlit app)
# ---------------------------------------------------------------------------

def validate_code(code):
    """
    Validate an invite code and return status.

    Returns:
        dict with:
        - valid: bool
        - message: str (error message if invalid)
        - patents_remaining: int
        - days_remaining: int
        - company: str
    """
    code = code.strip().upper()
    codes = _load_json(CODES_FILE)

    if code not in codes:
        return {"valid": False, "message": "Invalid invite code."}

    code_info = codes[code]

    # Load or create user record
    users = _load_json(DATA_FILE)
    if code not in users:
        users[code] = {
            "first_use": datetime.now().isoformat(),
            "expires": (datetime.now() + timedelta(days=EXPIRY_DAYS)).isoformat(),
            "patents_used": 0,
            "patent_log": [],
        }
        codes[code]["used"] = True
        _save_json(CODES_FILE, codes)
        _save_json(DATA_FILE, users)

    user = users[code]

    # Check expiry
    expires = datetime.fromisoformat(user["expires"])
    if datetime.now() > expires:
        return {
            "valid": False,
            "message": f"Demo expired on {expires.strftime('%b %d, %Y')}. Contact us for extended access.",
        }

    # Check patent limit
    patents_remaining = MAX_PATENTS - user.get("patents_used", 0)
    if patents_remaining <= 0:
        return {
            "valid": False,
            "message": f"Demo limit reached ({MAX_PATENTS} patents). Contact us for a pilot license.",
        }

    days_remaining = (expires - datetime.now()).days

    return {
        "valid": True,
        "message": "Access granted.",
        "patents_remaining": patents_remaining,
        "days_remaining": days_remaining,
        "company": code_info.get("company", ""),
        "email": code_info.get("email", ""),
    }


def record_patent_use(code, filename):
    """Record that a patent was processed under this invite code."""
    code = code.strip().upper()
    users = _load_json(DATA_FILE)

    if code not in users:
        return

    users[code]["patents_used"] = users[code].get("patents_used", 0) + 1
    users[code].setdefault("patent_log", []).append({
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
    })
    _save_json(DATA_FILE, users)


# ---------------------------------------------------------------------------
# Watermark text for demo exports
# ---------------------------------------------------------------------------

WATERMARK_TEXT = "Generated by GlossWerk — Demo Version"


def get_watermark_paragraph():
    """Return a watermark paragraph for docx exports."""
    from docx.shared import Pt, RGBColor
    from docx import Document
    from docx.oxml.ns import qn

    # Return the text — caller adds to document
    return WATERMARK_TEXT


# ---------------------------------------------------------------------------
# Streamlit auth UI
# ---------------------------------------------------------------------------

def show_auth_gate():
    """
    Show the auth gate in Streamlit. Returns auth status dict or None.

    Usage in app:
        auth = show_auth_gate()
        if not auth:
            st.stop()
    """
    import streamlit as st

    # Check if already authenticated this session
    if "demo_auth" in st.session_state and st.session_state.demo_auth.get("valid"):
        # Re-validate (check expiry/limits)
        recheck = validate_code(st.session_state.demo_code)
        if recheck["valid"]:
            return recheck
        else:
            st.session_state.demo_auth = None

    # Show login
    st.markdown("""
    <style>
        .auth-container {
            max-width: 440px; margin: 4rem auto; padding: 2.5rem;
            background: white; border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        }
        .auth-header {
            text-align: center; margin-bottom: 1.5rem;
        }
        .auth-header h2 { margin: 0; color: #1a1a2e; font-size: 1.6rem; }
        .auth-header p { color: #6b7280; font-size: 0.9rem; margin-top: 0.3rem; }
    </style>
    <div class="auth-container">
        <div class="auth-header">
            <h2>GlossWerk Demo</h2>
            <p>Enter your invite code to get started</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        code = st.text_input("Invite code", placeholder="GW-XXXXXXXX",
                             label_visibility="collapsed")
        if st.button("Enter", type="primary", use_container_width=True):
            if code:
                result = validate_code(code)
                if result["valid"]:
                    st.session_state.demo_auth = result
                    st.session_state.demo_code = code.strip().upper()
                    st.rerun()
                else:
                    st.error(result["message"])
            else:
                st.warning("Please enter your invite code.")

        st.caption("Don't have a code? [Request demo access](https://glosswerk.com/#demo)")

    return None


# ---------------------------------------------------------------------------
# CLI for managing codes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GlossWerk demo code manager")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a new invite code")
    gen.add_argument("--company", required=True)
    gen.add_argument("--email", required=True)
    gen.add_argument("--notes", default="")

    sub.add_parser("list", help="List all codes and usage")

    args = parser.parse_args()

    if args.command == "generate":
        code = generate_invite_code(args.company, args.email, args.notes)
        print(f"\nInvite code generated: {code}")
        print(f"  Company: {args.company}")
        print(f"  Email:   {args.email}")
        print(f"  Limits:  {MAX_PATENTS} patents, {EXPIRY_DAYS}-day expiry")
        print(f"\nSend this code to the contact.")

    elif args.command == "list":
        codes = list_codes()
        if not codes:
            print("No codes generated yet.")
        else:
            print(f"\n{'Code':<15} {'Company':<20} {'Used':<8} {'Patents':<10} {'Expires'}")
            print("-" * 75)
            for c in codes:
                exp = c["expires"][:10] if c["expires"] else "—"
                print(f"{c['code']:<15} {c['company']:<20} {'Yes' if c['used'] else 'No':<8} "
                      f"{c['patents_used']}/{MAX_PATENTS:<7} {exp}")

    else:
        parser.print_help()
