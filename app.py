"""
AI Sommelier Assistant
-----------------------
Streamlit web app για υπαλλήλους εστιατορίου/κάβας.
Εισάγεις το όνομα ενός κρασιού και παίρνεις δομημένη ανάλυση:
τύπο, ποικιλία, προέλευση, γευστικά χαρακτηριστικά, ταιριάσματα
με φαγητό και εναλλακτικές προτάσεις.

Αυτή η έκδοση συνδέεται με το Google Gemini API (google-genai SDK,
μοντέλο gemini-2.5-flash) και ζητά structured JSON output που
αντιστοιχεί απευθείας στο Pydantic schema `WineAnalysis`.
"""

import os
from typing import List, Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError


# ---------------------------------------------------------------------------
# 1. PYDANTIC MODELS — Structured Output Schema
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    """Μία εναλλακτική πρόταση φιάλης."""
    name: str = Field(..., description="Όνομα προτεινόμενης φιάλης")
    reasoning: str = Field(..., description="Γιατί προτείνεται αυτή η εναλλακτική")
    profile: str = Field(..., description="Σύντομο γευστικό προφίλ της εναλλακτικής")


class WineAnalysis(BaseModel):
    """Πλήρης δομημένη ανάλυση ενός κρασιού."""
    requested_wine: str = Field(..., description="Το όνομα του κρασιού που ζητήθηκε")
    wine_type: str = Field(..., description="Τύπος κρασιού: Λευκό, Κόκκινο, Ροζέ, Αφρώδες κ.λπ.")
    variety: str = Field(..., description="Ποικιλία / ποικιλίες σταφυλιού")
    origin: str = Field(..., description="Περιοχή και χώρα προέλευσης")
    characteristics: List[str] = Field(
        default_factory=list,
        description="Γευστικά χαρακτηριστικά: σώμα, οξύτητα, τανίνες, αρωματικές νότες",
    )
    food_pairing: List[str] = Field(
        default_factory=list,
        description="Προτεινόμενα ταιριάσματα με φαγητό",
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Τουλάχιστον 2 εναλλακτικές προτάσεις φιάλης",
    )


# ---------------------------------------------------------------------------
# 2. GEMINI API LAYER
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"

BASE_SYSTEM_INSTRUCTION = (
    "Είσαι ένας κορυφαίος Sommelier. Ανάλυσε το κρασί που ζητάει ο χρήστης "
    "και πρότεινε τουλάχιστον 2 εναλλακτικές φιάλες με παρόμοιο γευστικό "
    "προφίλ ή ποικιλία. Όλες οι απαντήσεις πρέπει να είναι στα Ελληνικά."
)

# Προεπιλεγμένη (demo) λίστα κρασιών καταστήματος — χρησιμοποιείται όταν
# ο χρήστης ενεργοποιήσει το toggle "Χρήση προεπιλεγμένης λίστας" χωρίς
# να ανεβάσει δικό του αρχείο.
DEFAULT_WINE_LIST: List[str] = [
    "Κτήμα Γεροβασιλείου Ξινόμαυρο",
    "Κτήμα Καρυδά Ξινόμαυρο",
    "Αγιωργίτικο Νεμέα, Παπαϊωάννου",
    "Ασύρτικο Σαντορίνης, Sigalas",
    "Μοσχοφίλερο, Τσέλεπος",
    "Μαλαγουζιά, Κτήμα Βιβλία Χώρα",
    "Cabernet Sauvignon, Κτήμα Λαζαρίδη",
    "Merlot, Κτήμα Στροφιλιά",
    "Νεμέα Ροζέ, Domaine Skouras",
    "Αφρώδες Ξινόμαυρο Brut, Κτήμα Ράψανη",
]


def build_system_instruction(wine_list: Optional[List[str]]) -> str:
    """
    Χτίζει το τελικό System Instruction ανάλογα με το ενεργό Mode:

    MODE A (wine_list μη κενή): οι αντιπροτάσεις πρέπει να προέρχονται
    ΑΠΟΚΛΕΙΣΤΙΚΑ από τη διαθέσιμη λίστα του μαγαζιού.

    MODE B (χωρίς λίστα): ελεύθερη αναζήτηση στον παγκόσμιο αμπελώνα.
    """
    if wine_list:
        list_str = ", ".join(wine_list)
        mode_instruction = (
            "Επέλεξε τις αντιπροτάσεις ΑΠΟΚΛΕΙΣΤΙΚΑ από την παρακάτω "
            f"διαθέσιμη λίστα κρασιών του μαγαζιού: [{list_str}]. "
            "Μην προτείνεις καμία φιάλη που δεν περιλαμβάνεται σε αυτή τη "
            "λίστα, ακόμα κι αν υπάρχει καλύτερη επιλογή αλλού."
        )
    else:
        mode_instruction = (
            "Πρότεινε τις καλύτερες εναλλακτικές φιάλες από τον παγκόσμιο "
            "αμπελώνα με βάση το γευστικό προφίλ."
        )
    return f"{BASE_SYSTEM_INSTRUCTION}\n\n{mode_instruction}"


def parse_wine_list_file(uploaded_file) -> List[str]:
    """
    Διαβάζει ένα ανεβασμένο CSV ή Excel αρχείο και επιστρέφει μια επίπεδη
    λίστα με ονόματα κρασιών (strings).

    Αναζητά στήλη με προφανές όνομα (wine/name/κρασί/όνομα/label) και,
    αν δεν βρεθεί, χρησιμοποιεί την πρώτη στήλη του αρχείου.
    Ρίχνει ValueError με κατανοητό μήνυμα αν κάτι πάει στραβά.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(
            "Μη υποστηριζόμενος τύπος αρχείου. Χρησιμοποίησε .csv, .xlsx ή .xls."
        )

    if df.empty or len(df.columns) == 0:
        raise ValueError("Το αρχείο δεν περιέχει δεδομένα.")

    candidate_names = {"wine", "name", "wine_name", "κρασί", "κρασι", "όνομα", "ονομα", "label"}
    candidate_cols = [c for c in df.columns if str(c).strip().lower() in candidate_names]
    target_col = candidate_cols[0] if candidate_cols else df.columns[0]

    wines = (
        df[target_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    wines = [w for w in wines.tolist() if w]

    if not wines:
        raise ValueError("Δεν εντοπίστηκαν ονόματα κρασιών στο αρχείο.")

    return wines


def get_api_key() -> str:
    """
    Επιστρέφει το Gemini API key.

    Προτεραιότητα:
    1. Environment variable GEMINI_API_KEY
    2. Πεδίο εισαγωγής στο Sidebar (st.session_state["gemini_api_key"])
    """
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    return st.session_state.get("gemini_api_key", "").strip()


def get_wine_analysis(wine_name: str, wine_list: Optional[List[str]] = None) -> WineAnalysis:
    """
    Καλεί το Gemini API (gemini-2.5-flash) ζητώντας structured JSON output
    που αντιστοιχεί στο WineAnalysis schema, και επιστρέφει το
    αποτέλεσμα ως πλήρως τυποποιημένο Pydantic object.

    Αν δοθεί `wine_list`, ενεργοποιείται το MODE A (αντιπροτάσεις
    αποκλειστικά από τη λίστα)· διαφορετικά ενεργοποιείται το MODE B
    (ελεύθερη αναζήτηση στον παγκόσμιο αμπελώνα).

    Ρίχνει εξαίρεση (RuntimeError / ValueError / APIError) αν κάτι πάει
    στραβά — ο caller είναι υπεύθυνος για το try/except γύρω από το UI.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Δεν βρέθηκε Gemini API Key. Όρισε το environment variable "
            "GEMINI_API_KEY ή συμπλήρωσέ το στο πεδίο της Sidebar."
        )

    client = genai.Client(api_key=api_key)
    system_instruction = build_system_instruction(wine_list)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Ανάλυσε το εξής κρασί: {wine_name}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=WineAnalysis,
        ),
    )

    # Το SDK παρέχει ήδη το parsed αντικείμενο όταν δίνεται response_schema.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, WineAnalysis):
        return parsed

    # Fallback: αν το .parsed δεν είναι διαθέσιμο για κάποιο λόγο,
    # κάνουμε validation πάνω στο raw JSON text.
    if not response.text:
        raise ValueError("Το μοντέλο επέστρεψε κενή απάντηση.")
    return WineAnalysis.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# 3. UI CONFIG & STYLING
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Sommelier Assistant",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* -------- Global dark, wine-themed palette -------- */
    :root {
        --wine-bg: #1a0f13;
        --wine-panel: #2a161c;
        --wine-panel-alt: #331b22;
        --wine-accent: #a8324a;
        --wine-accent-light: #d4657d;
        --wine-gold: #c9a86a;
        --wine-text: #f2e9e4;
        --wine-text-dim: #c9b8b8;
    }

    .stApp {
        background: linear-gradient(180deg, var(--wine-bg) 0%, #14090c 100%);
        color: var(--wine-text);
    }

    /* Hide default Streamlit chrome for a cleaner mobile look */
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 640px;
    }

    /* -------- Header -------- */
    .sommelier-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sommelier-header h1 {
        font-size: 1.9rem;
        margin-bottom: 0.2rem;
        color: var(--wine-text);
    }
    .sommelier-header p {
        color: var(--wine-text-dim);
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* -------- Input row -------- */
    div[data-testid="stTextInput"] input {
        background-color: var(--wine-panel);
        color: var(--wine-text);
        border: 1px solid #4a2a33;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--wine-gold);
        box-shadow: 0 0 0 1px var(--wine-gold);
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--wine-accent) 0%, #7a2138 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(168, 50, 74, 0.45);
        color: #fff;
        border: none;
    }

    /* -------- Result cards -------- */
    .wine-card {
        background: var(--wine-panel);
        border: 1px solid #4a2a33;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
    }
    .wine-card h3 {
        margin-top: 0;
        color: var(--wine-gold);
        font-size: 1.05rem;
        border-bottom: 1px solid #4a2a33;
        padding-bottom: 0.5rem;
        margin-bottom: 0.7rem;
    }
    .wine-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--wine-text);
        margin-bottom: 0.1rem;
    }
    .wine-subtitle {
        color: var(--wine-accent-light);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .badge {
        background: var(--wine-panel-alt);
        color: var(--wine-gold);
        border: 1px solid #4a2a33;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .char-item, .pairing-item {
        padding: 0.35rem 0;
        border-bottom: 1px dashed #3a2229;
        color: var(--wine-text-dim);
    }
    .char-item:last-child, .pairing-item:last-child {
        border-bottom: none;
    }
    .rec-name {
        font-weight: 700;
        color: var(--wine-text);
        font-size: 1.02rem;
    }
    .rec-profile {
        color: var(--wine-gold);
        font-size: 0.85rem;
        font-style: italic;
        margin: 0.2rem 0 0.4rem 0;
    }
    .rec-reason {
        color: var(--wine-text-dim);
        font-size: 0.92rem;
        margin-bottom: 0.2rem;
    }
    .rec-block {
        padding: 0.8rem 0;
        border-bottom: 1px solid #3a2229;
    }
    .rec-block:last-child {
        border-bottom: none;
    }

    hr {
        border-color: #3a2229;
    }

    /* -------- Dynamic Mode indicator -------- */
    .mode-badge-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 1.2rem;
    }
    .mode-badge {
        border-radius: 999px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .mode-badge.mode-a {
        background: rgba(64, 176, 108, 0.15);
        border-color: rgba(64, 176, 108, 0.5);
        color: #7fe0a0;
    }
    .mode-badge.mode-b {
        background: rgba(88, 149, 214, 0.15);
        border-color: rgba(88, 149, 214, 0.5);
        color: #8fc2f0;
    }

    /* -------- st.metric — Τύπος / Ποικιλία / Περιοχή -------- */
    div[data-testid="stMetric"] {
        background: var(--wine-panel);
        border: 1px solid #4a2a33;
        border-radius: 12px;
        padding: 0.7rem 0.5rem;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        justify-content: center;
        color: var(--wine-gold) !important;
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetricValue"] {
        justify-content: center;
        color: var(--wine-text) !important;
        font-size: 1.05rem !important;
        word-break: break-word;
    }

    /* -------- Characteristic pills -------- */
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.3rem;
    }
    .pill {
        background: var(--wine-panel-alt);
        color: var(--wine-text);
        border: 1px solid #4a2a33;
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.85rem;
        line-height: 1.2;
    }
    .pill.pill-taste {
        border-color: rgba(201, 168, 106, 0.5);
        color: var(--wine-gold);
    }

    /* -------- Recommendation cards -------- */
    .rec-card {
        background: var(--wine-panel);
        border: 1px solid #4a2a33;
        border-left: 4px solid var(--wine-accent);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .rec-card .rec-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--wine-text);
        margin-bottom: 0.3rem;
    }
    .rec-card .rec-profile-pill {
        display: inline-block;
        background: var(--wine-panel-alt);
        color: var(--wine-gold);
        border: 1px solid #4a2a33;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        margin-bottom: 0.6rem;
    }
    .rec-why-box {
        background: rgba(168, 50, 74, 0.12);
        border: 1px solid rgba(168, 50, 74, 0.4);
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
    }
    .rec-why-label {
        color: var(--wine-accent-light);
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        display: block;
        margin-bottom: 0.2rem;
    }
    .rec-why-text {
        color: var(--wine-text);
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.35;
    }

    /* -------- Reset / clear button (secondary) -------- */
    button[kind="secondary"] {
        background: transparent !important;
        color: var(--wine-text-dim) !important;
        border: 1px solid #4a2a33 !important;
        font-weight: 600;
    }
    button[kind="secondary"]:hover {
        border-color: var(--wine-gold) !important;
        color: var(--wine-gold) !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3b. SIDEBAR — API KEY CONFIGURATION
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Ρυθμίσεις API")
    env_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())

    if env_key_present:
        st.success("Το GEMINI_API_KEY βρέθηκε ως environment variable.")
    else:
        st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Επικόλλησε εδώ το API key σου",
            help=(
                "Εναλλακτικά, όρισε το environment variable GEMINI_API_KEY "
                "ώστε να μη χρειάζεται να το εισάγεις εδώ."
            ),
            key="gemini_api_key",
        )
        st.caption(
            "Το key χρησιμοποιείται μόνο κατά τη διάρκεια αυτής της συνεδρίας "
            "και δεν αποθηκεύεται μόνιμα."
        )

    st.markdown("---")
    st.markdown("### 📋 Λίστα Κρασιών Καταστήματος")
    st.caption(
        "Ανέβασε τη λίστα κρασιών του μαγαζιού σου για να περιοριστούν οι "
        "αντιπροτάσεις σε ό,τι έχεις πραγματικά διαθέσιμο."
    )

    uploaded_wine_file = st.file_uploader(
        "Αρχείο CSV ή Excel",
        type=["csv", "xlsx", "xls"],
        help="Μία στήλη με τα ονόματα των κρασιών (π.χ. 'Όνομα' ή 'Wine').",
    )

    use_default_list = st.toggle(
        "Χρήση προεπιλεγμένης demo λίστας",
        value=False,
        disabled=uploaded_wine_file is not None,
        help="Ενεργοποίησέ το αν θέλεις να δοκιμάσεις το Mode A χωρίς να ανεβάσεις δικό σου αρχείο.",
    )

    store_wine_list: List[str] = []
    list_parse_error: Optional[str] = None

    if uploaded_wine_file is not None:
        try:
            store_wine_list = parse_wine_list_file(uploaded_wine_file)
            st.success(f"Φορτώθηκαν {len(store_wine_list)} κρασιά από το αρχείο.")
        except Exception as e:
            list_parse_error = str(e)
            st.error(f"⚠️ Πρόβλημα στο αρχείο: {list_parse_error}")
    elif use_default_list:
        store_wine_list = DEFAULT_WINE_LIST

    if store_wine_list:
        with st.expander(f"Δες τη λίστα ({len(store_wine_list)} κρασιά)"):
            for w in store_wine_list:
                st.markdown(f"- {w}")


# ---------------------------------------------------------------------------
# 4. HEADER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="sommelier-header">
        <h1>🍷 AI Sommelier Assistant</h1>
        <p>Γρήγορη ανάλυση κρασιού για την ομάδα σέρβις &amp; κάβας</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Dynamic Mode indicator ---
if store_wine_list:
    mode_badge_html = (
        '<div class="mode-badge-wrap">'
        f'<span class="mode-badge mode-a">🟢 Αναζήτηση από Λίστα Καταστήματος ({len(store_wine_list)} κρασιά)</span>'
        "</div>"
    )
else:
    mode_badge_html = (
        '<div class="mode-badge-wrap">'
        '<span class="mode-badge mode-b">🌐 Ελεύθερη Αναζήτηση (Παγκόσμιος Αμπελώνας)</span>'
        "</div>"
    )
st.markdown(mode_badge_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5. INPUT AREA
# ---------------------------------------------------------------------------

with st.form(key="wine_search_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1.3])
    with col1:
        wine_name_input = st.text_input(
            "Όνομα κρασιού",
            placeholder="π.χ. Κτήμα Γεροβασιλείου Ξινόμαυρο 2021",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.form_submit_button("🔍 Αναζήτηση")

if "last_analysis" in st.session_state:
    if st.button("🧹 Καθαρισμός / Νέα Αναζήτηση", type="secondary", use_container_width=True):
        st.session_state.pop("last_analysis", None)
        st.rerun()


# ---------------------------------------------------------------------------
# 6. RESULTS RENDERING
# ---------------------------------------------------------------------------

def render_analysis(analysis: WineAnalysis) -> None:
    # --- Title ---
    st.markdown(
        f"""
        <div class="wine-card" style="margin-bottom: 0.8rem;">
            <div class="wine-title">{analysis.requested_wine}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Βασικά στοιχεία: Τύπος / Ποικιλία / Περιοχή (st.metric, γρήγορη ματιά) ---
    m1, m2, m3 = st.columns(3)
    m1.metric("🍷 Τύπος", analysis.wine_type)
    m2.metric("🍇 Ποικιλία", analysis.variety)
    m3.metric("🌍 Περιοχή", analysis.origin)

    # --- Χαρακτηριστικά (pills) ---
    char_pills = "".join(
        f'<span class="pill pill-taste">{c}</span>' for c in analysis.characteristics
    )
    st.markdown(
        f"""
        <div class="wine-card">
            <h3>👅 Γευστικά Χαρακτηριστικά</h3>
            <div class="pill-row">{char_pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Ταίριασμα με φαγητό (pills) ---
    pairing_pills = "".join(
        f'<span class="pill">🍽️ {p}</span>' for p in analysis.food_pairing
    )
    st.markdown(
        f"""
        <div class="wine-card">
            <h3>🍽️ Ταίριασμα με Φαγητό</h3>
            <div class="pill-row">{pairing_pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Εναλλακτικές προτάσεις: ξεχωριστή κάρτα ανά πρόταση ---
    st.markdown("#### 💡 Εναλλακτικές Προτάσεις")
    for rec in analysis.recommendations:
        st.markdown(
            f"""
            <div class="rec-card">
                <div class="rec-name">🔄 {rec.name}</div>
                <span class="rec-profile-pill">{rec.profile}</span>
                <div class="rec-why-box">
                    <span class="rec-why-label">🔑 Γιατί προτείνεται</span>
                    <span class="rec-why-text">{rec.reasoning}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


search_failed = False

if search_clicked:
    if not wine_name_input.strip():
        st.warning("Παρακαλώ γράψε το όνομα ενός κρασιού για αναζήτηση.")
    else:
        try:
            with st.spinner("Ο AI Sommelier αναλύει το κρασί..."):
                result = get_wine_analysis(
                    wine_name_input.strip(),
                    wine_list=store_wine_list or None,
                )
            st.session_state["last_analysis"] = result
        except RuntimeError as e:
            # Κυρίως: λείπει το API key
            search_failed = True
            st.error(f"⚠️ {e}")
        except APIError as e:
            # Σφάλματα από το ίδιο το Gemini API (auth, quota, μοντέλο κ.λπ.)
            search_failed = True
            st.error(f"⚠️ Σφάλμα κατά την κλήση στο Gemini API: {e}")
        except ValueError as e:
            # Πρόβλημα στο parsing/validation του JSON response
            search_failed = True
            st.error(f"⚠️ Το αποτέλεσμα δεν ήταν έγκυρο: {e}")
        except Exception as e:
            # Γενικό δίχτυ ασφαλείας για οτιδήποτε απρόβλεπτο
            search_failed = True
            st.error(f"⚠️ Απρόσμενο σφάλμα: {e}")

# Εμφάνιση αποτελέσματος: είτε μετά από νέα επιτυχή αναζήτηση, είτε
# διατηρώντας το προηγούμενο αποτέλεσμα σε ένα απλό rerun χωρίς νέο κλικ.
if not search_failed and "last_analysis" in st.session_state and (
    not search_clicked or wine_name_input.strip()
):
    render_analysis(st.session_state["last_analysis"])

if "last_analysis" not in st.session_state:
    st.markdown(
        """
        <div style="text-align:center; color:#c9b8b8; margin-top:2rem; font-size:0.9rem;">
            Πληκτρολόγησε το όνομα ενός κρασιού παραπάνω και πάτησε "Αναζήτηση"
            για να πάρεις μια ανάλυση από τον AI Sommelier (Gemini).
        </div>
        """,
        unsafe_allow_html=True,
    )
