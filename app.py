"""
Streamlit frontend for the Chuka University GraphRAG Assistant.
Manages the chat interface, session states, and PDF/Voice uploads.
"""

import streamlit as st
import uuid
import io
import pandas as pd
from src.database import (
    DATABASE_STATUS_MESSAGE,
    USING_FALLBACK_DATABASE,
    SessionLocal,
    UserProfile,
    clear_chat_history,
    get_chat_history,
    get_or_create_user,
    get_user_sessions,
    log_chat_history,
    save_user_profile,
)
from src.pdf_handler import parse_chuka_document

st.set_page_config(
    page_title="Chuka University GraphRAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Resource Initialization
@st.cache_resource(show_spinner=False)
def get_assistant():
    """Cache assistant startup results without letting initialization exceptions crash the app."""
    from src.chuka_graphrag_pipeline import GraphRAGAssistant

    try:
        return GraphRAGAssistant(), None
    except Exception as exc:
        return None, str(exc)


def initialize_assistant_state():
    """Initialise the assistant once and keep startup errors visible in the UI."""
    if "assistant_ready" not in st.session_state:
        from src.chuka_graphrag_pipeline import get_missing_runtime_config

        st.session_state.assistant_ready = False
        st.session_state.assistant_error = None
        st.session_state.assistant = None

        missing_config = get_missing_runtime_config()
        if missing_config:
            st.session_state.assistant_error = (
                "Missing required configuration: " + ", ".join(missing_config)
            )
            return

        assistant, error = get_assistant()
        st.session_state.assistant = assistant
        st.session_state.assistant_error = error
        st.session_state.assistant_ready = assistant is not None and error is None

# UI Styling
st.markdown("""
<style>
/* 1. Force light mode on main content area */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .main .block-container {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
}
[data-testid="stAppViewContainer"] > section:nth-child(2) {
    background-color: #ffffff !important;
}
/* 2. Sidebar styling */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #1e2330 !important;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
[data-testid="stSidebar"] hr { border-color: #2e3650 !important; }

/* sidebar buttons (new chat / log out) */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #3a4460 !important;
    color: #c9d1e0 !important;
    border-radius: 8px !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    width: 100% !important;
    text-align: left !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #2b3352 !important;
    border-color: #5a6aaa !important;
}

/* 3. Main container spacing */
.block-container {
    padding-top: 3.5rem !important;
    background: #ffffff !important;
}

/* 4. Chat messages UI cleanup */
[data-testid="stChatMessage"] {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 12px 0 !important;
}
/* 5. Chat input bar styling */
[data-testid="stChatInput"] > div {
    background: #f4f6fb !important;
    border: 1px solid #dce1ef !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #1a1a2e !important;
}

/* 6. Onboarding form fields */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #1a1a2e !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] li { color: #1a1a2e !important; }
input[type="text"] {
    background: #ffffff !important;
    color: #1a1a2e !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Hide Streamlit default header right-side elements (Deploy button etc) but keep sidebar toggle */
[data-testid="stHeaderActionElements"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Force zero top padding on the main block to push header flush to the top */
[data-testid="stMainBlockContainer"], .main .block-container {
    padding-top: 15px !important;
    padding-bottom: 120px !important;
}

/* Header refinement */
.custom-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 40px 20px 40px;
    border-bottom: 1px solid #e2e8f0;
    margin: 0 -40px 30px -40px; /* Stretch to edges of container */
}
.header-icon {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #7b61ff;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}
.header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
}

/* 5. Chat Input Styling */
/* Native Streamlit positioning works best; just style it */
[data-testid="stChatInput"] {
    border-radius: 28px !important;
    background: white !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    border: 1px solid #e2e8f0 !important;
}

/* Ensure padding-bottom so messages aren't hidden under bar */
.main .block-container {
    padding-bottom: 120px !important;
}

/* Responsive adjustments for Mobile/Tablet */
@media screen and (max-width: 768px) {
    .custom-header {
        padding: 15px 20px !important;
        margin: 0 -20px 20px -20px !important;
    }
    .header-title {
        font-size: 1.3rem !important;
    }
    .main .block-container {
        padding-top: 10px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Force Streamlit columns to stack cleanly on smaller screens */
    [data-testid="column"] {
        min-width: 100% !important;
        margin-bottom: 0.5rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

# Session Identity Management
# Uses URL-based device tokens for anonymous session tracking.
# replace with SSO/OAuth in future.
if "device_token" not in st.session_state:
    query_params = st.query_params
    if "token" in query_params:
        st.session_state.device_token = query_params["token"]
    else:
        st.session_state.device_token = str(uuid.uuid4())
        st.query_params["token"] = st.session_state.device_token

user = get_or_create_user(device_token=st.session_state.device_token)
st.session_state.user_id = user["user_id"]

if "assistant" in st.session_state:
    if st.session_state.assistant is not None and not hasattr(st.session_state.assistant, "get_personalized_timetable"):
        del st.session_state["assistant"]
        st.cache_resource.clear()


# State Initialization
initialize_assistant_state()

if "mapped_programmes" not in st.session_state:
    if not st.session_state.assistant_ready:
        st.session_state.mapped_programmes = []
    else:
        st.session_state.mapped_programmes = st.session_state.assistant.get_mapped_programmes()

if "user_profile" not in st.session_state:
    db = SessionLocal()
    try:
        p = db.query(UserProfile).filter(UserProfile.user_id == user["user_id"]).first()
        st.session_state.user_profile = (
            {"faculty": p.faculty, "department": p.department, "program": p.program,
             "year": str(p.year_of_study), "semester": str(p.semester)}
            if p and p.faculty else None
        )
    finally:
        db.close()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    rows = get_chat_history(user["user_id"], session_id=st.session_state.current_session_id)
    st.session_state.chat_history = []
    for r in rows:
        st.session_state.chat_history += [
            {"role": "user", "content": r.query_text},
            {"role": "assistant", "content": r.response_text},
        ]

if "extra_context" not in st.session_state:
    st.session_state.extra_context = ""

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# UI Components: Onboarding
def onboarding_screen():
    """Initial screening to capture academic context used to filter Cypher queries."""
    _, col, _ = st.columns([1, 2.5, 1])
    with col:
        if not st.session_state.assistant_ready:
            st.error(
                "The AI assistant is not configured for this deployment yet. "
                "Add `GEMINI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in Streamlit secrets."
            )
            if st.session_state.assistant_error:
                st.caption(f"Startup detail: {st.session_state.assistant_error}")

        st.markdown("""
        <div style="text-align:center;margin-bottom:20px;margin-top:40px;">
            <div style="display:inline-flex;align-items:center;justify-content:center;
                        background:linear-gradient(135deg,#000000,#333333);
                        color:white;font-size:2em;margin-bottom:14px;"></div>
            <h2 style="color:#000000;font-weight:700;margin:0 0 4px;">Chuka University</h2>
            <p style="color:#64748b;font-size:0.9em;margin:0 0 28px;">Academic Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        mapped_progs = st.session_state.get('mapped_programmes', [])
        
        # 1. Faculty Selection
        if not mapped_progs:
            faculties = ["Select Faculty", "Faculty of Science & Technology", "Faculty of Business Studies"]
        else:
            db_faculties = sorted(list(set([p['faculty'] for p in mapped_progs if p.get('faculty')])))
            faculties = ["Select Faculty"] + db_faculties

        faculty = st.selectbox("Faculty", faculties)
        
        # 2. Filtered Program Selection
        if faculty == "Select Faculty":
            display_options = ["Select Program"]
        else:
            filtered = [p for p in mapped_progs if p.get('faculty') == faculty]
            if not filtered:
                display_options = ["Select Program"]
            else:
                display_options = ["Select Program"] + filtered

        program_select = st.selectbox(
            "Program", 
            display_options,
            format_func=lambda x: x['name'] if isinstance(x, dict) else x
        )
        
        program = program_select['name'] if isinstance(program_select, dict) else program_select
        department = program_select['department'] if isinstance(program_select, dict) and 'department' in program_select else None
        
        year = st.selectbox("Year of Study", ["Select Year", "1", "2", "3", "4", "5"])
        semester = st.selectbox("Semester", ["Select Semester", "1", "2", "3"])

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Continue", use_container_width=True):
            if "Select" in faculty or "Select" in program or "Select" in year or "Select" in semester:
                st.error("Please complete all fields.")
            else:
                st.session_state.user_profile = {
                    "faculty": faculty, "department": department, "program": program, "year": year, "semester": semester
                }
                save_user_profile(st.session_state.user_id, faculty, department, program, year, semester)
                st.rerun()

    st.markdown("""
    <style>
        .stButton > button {
            background: #176BFF !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover { background: #0e52c9 !important; }
    </style>
    """, unsafe_allow_html=True)

def course_explorer_view():
    import pandas as pd
    import os
    
    st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;
                    padding-bottom:16px;border-bottom:1px solid #e8ecf4;margin-bottom:20px;">
            <div style="width:42px;height:42px;border-radius:50%;
                        background:linear-gradient(135deg,#10b981,#059669);
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-size:1.1em;flex-shrink:0;"></div>
            <span style="font-size:1.5em;font-weight:700;color:#0f172a;">Course Explorer</span>
        </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "data", "curricular_mapping.csv")
        df = pd.read_csv(data_path)
        return df.fillna("Unknown")

    try:
        df = load_data()
        mapped_progs = st.session_state.get('mapped_programmes', [])
        if mapped_progs:
            mapped_names = [p['name'] for p in mapped_progs]
            df = df[df['programme'].isin(mapped_names)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            faculties = ["All"] + sorted(df['faculty'].unique().tolist())
            sel_fac = st.selectbox("Filter by Faculty", faculties)
        
        with col2:
            if sel_fac != "All":
                depts = ["All"] + sorted(df[df['faculty'] == sel_fac]['department'].unique().tolist())
            else:
                depts = ["All"] + sorted(df['department'].unique().tolist())
            sel_dept = st.selectbox("Filter by Department", depts)

        with col3:
            prog_filter_df = df
            if sel_fac != "All":
                prog_filter_df = prog_filter_df[prog_filter_df['faculty'] == sel_fac]
            if sel_dept != "All":
                prog_filter_df = prog_filter_df[prog_filter_df['department'] == sel_dept]
                
            prog_options = ["All"]
            if mapped_progs:
                current_progs = sorted(prog_filter_df['programme'].unique().tolist())
                for p in mapped_progs:
                    if p['name'] in current_progs:
                        prog_options.append(p)
            else:
                prog_options += sorted(prog_filter_df['programme'].unique().tolist())

            sel_prog_raw = st.selectbox(
                "Filter by Programme", 
                prog_options,
                format_func=lambda x: f"{x['name']} ({x['count']} units)" if isinstance(x, dict) else x
            )
            sel_prog = sel_prog_raw['name'] if isinstance(sel_prog_raw, dict) else sel_prog_raw
            
        filtered_df = df
        if sel_fac != "All":
            filtered_df = filtered_df[filtered_df['faculty'] == sel_fac]
        if sel_dept != "All":
            filtered_df = filtered_df[filtered_df['department'] == sel_dept]
        if sel_prog != "All":
            filtered_df = filtered_df[filtered_df['programme'] == sel_prog]
            
        st.markdown(f"**Showing {len(filtered_df)} units**")
        st.dataframe(
            filtered_df[['course_code', 'course_name', 'department', 'Academic_Level', 'semester', 'year']], 
            use_container_width=True, 
            hide_index=True
        )
    except Exception as e:
        st.error(f"Could not load course data: {e}")

# UI Components: Main Chat Interface
def main_chat():
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"

    if not st.session_state.assistant_ready:
        st.error(
            "The AI assistant is not configured for this deployment yet. "
            "Add `GEMINI_API_KEY` and the Neo4j credentials in Streamlit secrets."
        )
        if st.session_state.assistant_error:
            st.caption(f"Startup detail: {st.session_state.assistant_error}")
        return

    # Pre-fetch data to avoid sidebar blocking
    tt_data = []
    if st.session_state.user_profile:
        try:
            tt_data = st.session_state.assistant.get_personalized_timetable(st.session_state.user_profile)
        except Exception as e:
            st.sidebar.error("Could not load timetable data.")

    # SIDEBAR 
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.current_view = "chat"
            st.rerun()

        if USING_FALLBACK_DATABASE:
            st.warning(DATABASE_STATUS_MESSAGE)
            
        if st.button("Course Explorer", use_container_width=True, key="open_explorer"):
            st.session_state.current_view = "explorer"
            st.rerun()

        if st.button("Clear History", use_container_width=True, key="clear_hist"):
            clear_chat_history(st.session_state.user_id)
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Personalized Tools**", unsafe_allow_html=True)
        
        # Export logic
        if tt_data:
            # Grid-Based PDF export
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import landscape, letter
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                
                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=landscape(letter), topMargin=30)
                styles = getSampleStyleSheet()
                elements = []

                # Header
                title_style = ParagraphStyle(
                    'TitleStyle', parent=styles['Heading1'], alignment=1, spaceAfter=10
                )
                elements.append(Paragraph("Chuka University", title_style))
                elements.append(Paragraph(f"Academic Timetable Matrix", styles['Heading2']))
                
                profile = st.session_state.user_profile
                elements.append(Paragraph(
                    f"<b>Program:</b> {profile['program']} | <b>Level:</b> Year {profile['year']}, Sem {profile['semester']}",
                    styles['Normal']
                ))
                elements.append(Spacer(1, 15))

                # Grid Matrix
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                time_slots = ["7:00AM-10:00AM", "10:00AM-1:00PM", "1:00PM-4:00PM", "4:00PM-7:00PM"]
                
                # Header row
                grid_data = [["", "7:00AM - 10:00AM", "10:00AM - 1:00PM", "1:00PM - 4:00PM", "4:00PM - 7:00PM"]]
                
                for day in days:
                    row_content = [day[:3]]
                    for slot in time_slots:
                        cell_units = []
                        for item in tt_data:
                            if item.get('day') == day and item.get('time'):
                                clean_time = item['time'].replace(" ", "").upper()
                                clean_slot = slot.replace(" ", "").upper()
                                if clean_time == clean_slot:
                                    venue = f" ({item['room']})" if item.get('room') and item['room'] != "None" else ""
                                    cell_units.append(f"{item['code']}{venue}")
                        row_content.append("\n".join(cell_units) if cell_units else "-")
                    grid_data.append(row_content)

                # Create Table
                t = Table(grid_data, colWidths=[60, 160, 160, 160, 160])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (1, 1), (-1, -1), colors.white),
                    ('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#f8fafc")),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor("#000000")),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ]))
                elements.append(t)
                
                elements.append(Spacer(1, 20))
                elements.append(Paragraph(
                    "<font color='grey' size='8'><i>Legend: UNIT_CODE (ROOM) | Generated by Chuka Virtual Assistant</i></font>",
                    styles['Normal']
                ))

                doc.build(elements)
                buf.seek(0)
                st.download_button(
                    label="Download PDF Schedule",
                    data=buf,
                    file_name=f"Chuka_Timetable_{str(st.session_state.user_id)}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Export error: {e}")

        # History list
        sessions = get_user_sessions(st.session_state.user_id)
        
        st.markdown("<div style='margin-top:15px;margin-bottom:8px;font-size:0.85em;color:#64748b;font-weight:600;'>History</div>", unsafe_allow_html=True)
        
        if sessions:
            for s in sessions[:15]:  # Limit to 15 recent sessions
                col1, col2 = st.columns([0.85, 0.15], gap="small")
                btn_title = s['title']
                
                with col1:
                    if st.button(f"{btn_title}", key=f"sess_{s['session_id']}", use_container_width=True):
                        st.session_state.current_session_id = s["session_id"]
                        if "chat_history" in st.session_state:
                             del st.session_state["chat_history"]
                        st.rerun()
                with col2:
                    if st.button("×", key=f"del_{s['session_id']}", use_container_width=True):
                        clear_chat_history(st.session_state.user_id, session_id=s['session_id'])
                        if st.session_state.current_session_id == s['session_id']:
                            st.session_state.current_session_id = str(uuid.uuid4())
                            if "chat_history" in st.session_state:
                                 del st.session_state["chat_history"]
                        st.rerun()
        else:
            st.markdown("<div style='font-size:0.8em;opacity:0.5;padding-left:10px;'>No previous chats</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:auto;padding:20px 0 0 0;">
            <hr style="border-color:#2e3650;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;font-size:.9em;cursor:pointer;">
                Library
            </div>
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;font-size:.9em;cursor:pointer;">
                Settings
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("New Identity / Logout", use_container_width=True, key="logout"):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

    # MAIN AREA 
    if st.session_state.current_view == "explorer":
        course_explorer_view()
    else:
        # Header
        st.markdown("""
            <div class="custom-header">
                <div class="header-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
                </div>
                <div class="header-title">AI Assistant</div>
            </div>
        """, unsafe_allow_html=True)

        # Output existing chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                avatar = "https://ui-avatars.com/api/?name=C&background=4B0082&color=fff&rounded=true"
            else:
                avatar = "https://ui-avatars.com/api/?name=U&background=6c757d&color=fff&rounded=true"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                
        # Access the globally cached assistant
        assistant = get_assistant()

        #CHAT INPUT WITH FILE/AUDIO 

        prompt = st.chat_input(
            "Ask the University Assistant...", 
            accept_file=True, 
            accept_audio=True
        )

        if prompt:
            final_prompt = ""
            
            # Handle text input
            if getattr(prompt, "text", None):
                final_prompt = prompt.text
            
            # Handle audio upload
            if getattr(prompt, "audio", None):
                with st.spinner("Transcribing audio..."):
                    audio_transcription = assistant.transcribe_audio(prompt.audio.read())
                    if audio_transcription:
                        if final_prompt: 
                            final_prompt += f"\n[Voice Transcription]: {audio_transcription}"
                        else:
                            final_prompt = audio_transcription
            
            # Handle file upload
            if getattr(prompt, "files", None) and len(prompt.files) > 0:
                with st.spinner(f"Processing document {prompt.files[0].name}..."):
                    context = parse_chuka_document(prompt.files[0].name, prompt.files[0].read())
                    st.session_state.extra_context = context
                    st.session_state.uploaded_file_name = prompt.files[0].name
                    st.success(f"Loaded: {prompt.files[0].name}")
                    
            if final_prompt:
                st.session_state.chat_history.append({"role": "user", "content": final_prompt})
                with st.chat_message("user", avatar="https://ui-avatars.com/api/?name=U&background=6c757d&color=fff&rounded=true"):
                    st.markdown(final_prompt)

            with st.chat_message("assistant", avatar="https://ui-avatars.com/api/?name=C&background=4B0082&color=fff&rounded=true"):
                with st.spinner(""):
                    try:
                        response = assistant.generate_response(
                            final_prompt,
                            st.session_state.user_profile,
                            extra_context=st.session_state.get("extra_context", "")
                        )
                    except Exception as e:
                        response = f"Sorry, I ran into an error: {e}"
                    st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            log_chat_history(st.session_state.user_id, st.session_state.current_session_id, final_prompt, response)
            st.rerun()

# Router
if not st.session_state.user_profile:
    onboarding_screen()
else:
    main_chat()
