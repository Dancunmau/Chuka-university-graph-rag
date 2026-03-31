"""
Streamlit frontend for the Chuka University GraphRAG Assistant.
Manages the chat interface, session states, and PDF/Voice uploads.
"""

import streamlit as st
import uuid
import io
import os
import logging
import base64
import pandas as pd
from src.chuka_graphrag_pipeline import GraphRAGAssistant
from src.database import get_or_create_user, save_user_profile, log_chat_history, get_chat_history, clear_chat_history, get_user_sessions, SessionLocal, UserProfile, update_chat_feedback
from src.pdf_handler import parse_chuka_document

@st.cache_data
def get_base64_image(image_file="download.jpeg"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, image_file)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

st.set_page_config(
    page_title="Chuka University Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Resource Initialization
@st.cache_resource(show_spinner=False)
def get_assistant():
    """Caches the GraphRAG pipeline globally to share the Neo4j connection pool and FAISS index in memory across user sessions."""
    from src.chuka_graphrag_pipeline import GraphRAGAssistant
    return GraphRAGAssistant()

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

/* 7. New: Vibrant Blue 'Continue' Button styling */
div.stButton > button:not([data-testid="stSidebar"] *) {
    background-color: #1b5cfc !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

div.stButton > button:not([data-testid="stSidebar"] *):hover {
    background-color: #0d47d1 !important;
    box-shadow: 0 10px 15px -3px rgba(27, 92, 252, 0.3), 0 4px 6px -2px rgba(27, 92, 252, 0.1) !important;
    transform: translateY(-1px) !important;
}

div.stButton > button:not([data-testid="stSidebar"] *):active {
    transform: translateY(0px) !important;
}

/* 8. Input fields focus styling */
div[data-baseweb="select"] > div:focus-within {
    border-color: #1b5cfc !important;
    box-shadow: 0 0 0 1px #1b5cfc !important;
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
    if not hasattr(st.session_state.assistant, "get_personalized_timetable"):
        del st.session_state["assistant"]
        st.cache_resource.clear()


# State Initialization
if "assistant" not in st.session_state:
    st.session_state.assistant = get_assistant()

if "mapped_programmes" not in st.session_state:
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
            {"role": "assistant", "content": r.response_text, "id": r.history_id, "feedback": r.feedback},
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
        logo_b64 = get_base64_image("download.jpeg")
        logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" style="width:110px; max-height:110px; margin-bottom:15px; border-radius:8px; object-fit:contain; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">' if logo_b64 else '<div style="display:inline-flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#000000,#333333);color:white;font-size:2em;margin-bottom:14px;width:100px;height:100px;border-radius:10px;"></div>'

        st.markdown(f"""
        <div style="text-align:center;margin-bottom:20px;margin-top:20px;">
            {logo_html}
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
            
        # Metric Cards
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("📚 Total Units Found", len(filtered_df))
        m2.metric("🏛️ Faculties", len(filtered_df['faculty'].unique()))
        m3.metric("🎓 Programmes", len(filtered_df['programme'].unique()))
        st.markdown("<hr style='border-color:#e2e8f0; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        # Dataframe
        st.dataframe(
            filtered_df[['course_code', 'course_name', 'department', 'Academic_Level', 'semester', 'year']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "course_code": st.column_config.TextColumn("Unit Code", width="small"),
                "course_name": st.column_config.TextColumn("Unit Title", width="medium"),
                "department": st.column_config.TextColumn("Department", width="medium"),
                "Academic_Level": "Level",
                "semester": st.column_config.NumberColumn("Sem", format="%d"),
                "year": st.column_config.NumberColumn("Year", format="%d")
            }
        )
    except Exception as e:
        logging.error(f"Course Explorer error: {e}")
        st.error("We couldn't load the course data right now. Please try again later.")

# UI Components: Main Chat Interface
def main_chat():
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"

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
        
        # Student Profile ID Card
        profile = st.session_state.user_profile
        prog_name = profile['program'] if len(profile['program']) < 35 else profile['program'][:32] + "..."
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 20px;">
            <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px;">Student Profile</div>
            <div style="color: #f1f5f9; font-weight: 500; font-size: 0.9em; line-height: 1.3; margin-bottom: 8px;">{prog_name}</div>
            <div style="display: flex; gap: 8px; font-size: 0.8em; color: #cbd5e1;">
                <span style="background: #334155; padding: 2px 8px; border-radius: 4px;">Year {profile['year']}</span>
                <span style="background: #334155; padding: 2px 8px; border-radius: 4px;">Sem {profile['semester']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.current_view = "chat"
            st.rerun()
            
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
                logging.error(f"PDF Export error: {e}")
                st.error("We couldn't generate your PDF schedule right now. Please try again later.")

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
            <a href="https://library.chuka.ac.ke/" target="_blank" style="text-decoration:none; color:#c9d1e0;">
                <div style="display:flex;align-items:center;gap:10px;padding:8px 0;font-size:.9em;cursor:pointer;">
                    Library
                </div>
            </a>
            <a href="https://repository.chuka.ac.ke/" target="_blank" style="text-decoration:none; color:#c9d1e0;">
                <div style="display:flex;align-items:center;gap:10px;padding:8px 0;font-size:.9em;cursor:pointer;">
                    Repository
                </div>
            </a>
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
        logo_b64 = get_base64_image("download.jpeg")
        header_icon = f'<img src="data:image/jpeg;base64,{logo_b64}" style="width:100%; height:100%; object-fit:contain; border-radius: 5px;">' if logo_b64 else '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>'

        st.markdown(f"""
            <div class="custom-header">
                <div class="header-icon" style="background:transparent; padding:0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {header_icon}
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
                content = msg["content"]
                if msg["role"] == "assistant" and "|||CONTEXT|||" in content:
                    main_text, ctx_str = content.split("|||CONTEXT|||", 1)
                    st.markdown(main_text)
                    
                    graph_ctx = ""
                    faiss_ctx = ""
                    if "|||FAISS|||" in ctx_str:
                        graph_ctx, faiss_ctx = ctx_str.split("|||FAISS|||", 1)
                    else:
                        graph_ctx = ctx_str
                        
                    if graph_ctx.strip() or faiss_ctx.strip():
                        with st.expander("Sources & Database Records Used"):
                            if faiss_ctx.strip():
                                st.markdown(f"**Policy / Handbook Context**\n\n```text\n{faiss_ctx.strip()}\n```")
                            if graph_ctx.strip():
                                st.markdown(f"**Neo4j Reference Data**\n\n```text\n{graph_ctx.strip()}\n```")
                else:
                    st.markdown(content)
                    
                # Render thumbs up/down
                if msg["role"] == "assistant" and "id" in msg:
                    fb_key = f"fb_{msg['id']}"
                    if fb_key not in st.session_state and msg.get("feedback") is not None:
                        st.session_state[fb_key] = msg["feedback"]
                    st.feedback("thumbs", key=fb_key, on_change=lambda mid=msg['id'], k=fb_key: update_chat_feedback(mid, st.session_state[k]))
                
        # Access the globally cached assistant
        assistant = get_assistant()

        # Quick Welcome Prompts for Empty Chat
        if not st.session_state.chat_history:
            profile = st.session_state.user_profile
            prog = profile.get('program', 'your program')
            
            st.markdown(f"""
            <div style="text-align:center; padding: 20px 0 30px 0;">
                <h3 style="color:#1e293b; font-weight:600;">Welcome to your Academic Assistant</h3>
                <p style="color:#64748b; font-size: 0.95em;">I'm here to help you navigate {prog}. Choose a quick action or type a message below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                if st.button("📅 What is my schedule tomorrow?", use_container_width=True, key="p1"):
                    st.session_state.triggered_prompt = "What is my schedule tomorrow?"
                    st.rerun()
                if st.button("💰 Track my program fees", use_container_width=True, key="p2"):
                    st.session_state.triggered_prompt = "What are the fee structures for my program?"
                    st.rerun()
            with p_col2:
                if st.button("📚 Show my course units", use_container_width=True, key="p3"):
                    st.session_state.triggered_prompt = "Show me my course units for this semester"
                    st.rerun()
                if st.button("📜 Handbook exam rules", use_container_width=True, key="p4"):
                    st.session_state.triggered_prompt = "What are the examination rules and regulations?"
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)

        # CHAT INPUT WITH FILE/AUDIO 

        prompt = st.chat_input(
            "Ask the University Assistant...", 
            accept_file=True, 
            accept_audio=True
        )

        final_prompt = ""
        has_media = False
        
        if prompt:
            # Handle text input natively from chat bar
            if getattr(prompt, "text", None):
                final_prompt = prompt.text
            elif isinstance(prompt, str):
                final_prompt = prompt
                
            # Handle audio upload
            if getattr(prompt, "audio", None):
                has_media = True
                with st.spinner("Transcribing audio..."):
                    audio_transcription = assistant.transcribe_audio(prompt.audio.read())
                    if audio_transcription:
                        if final_prompt: 
                            final_prompt += f"\n[Voice Transcription]: {audio_transcription}"
                        else:
                            final_prompt = audio_transcription
            
            # Handle file upload
            if getattr(prompt, "files", None) and len(prompt.files) > 0:
                has_media = True
                with st.spinner(f"Processing document {prompt.files[0].name}..."):
                    from src.pdf_handler import parse_chuka_document
                    context = parse_chuka_document(prompt.files[0].name, prompt.files[0].read())
                    st.session_state.extra_context = context
                    st.session_state.uploaded_file_name = prompt.files[0].name
                    st.success(f"Loaded: {prompt.files[0].name}")
                    
        # Intercept Quick Prompts from buttons
        if st.session_state.get("triggered_prompt"):
            final_prompt = st.session_state.triggered_prompt
            del st.session_state["triggered_prompt"]
            
        if final_prompt or has_media:
            st.session_state.chat_history.append({"role": "user", "content": final_prompt})
            with st.chat_message("user", avatar="https://ui-avatars.com/api/?name=U&background=6c757d&color=fff&rounded=true"):
                st.markdown(final_prompt)

            with st.chat_message("assistant", avatar="https://ui-avatars.com/api/?name=C&background=4B0082&color=fff&rounded=true"):
                with st.spinner(""):
                    try:
                        ctx_container = {}
                        stream_gen = assistant.generate_response_stream(
                            final_prompt,
                            st.session_state.user_profile,
                            extra_context=st.session_state.get("extra_context", ""),
                            context_container=ctx_container
                        )
                        # st.write_stream handles the typewriter effect seamlessly and returns the complete joined text
                        response = st.write_stream(stream_gen)
                        
                        graph_ctx = ctx_container.get("graph_nodes", "")
                        faiss_ctx = ctx_container.get("faiss_results", "")
                        
                        if graph_ctx.strip() or faiss_ctx.strip():
                            with st.expander("Sources & Database Records Used"):
                                if faiss_ctx.strip():
                                    st.markdown(f"**Policy / Handbook Context**\n\n```text\n{faiss_ctx.strip()}\n```")
                                if graph_ctx.strip():
                                    st.markdown(f"**Neo4j Reference Data**\n\n```text\n{graph_ctx.strip()}\n```")
                                    
                            # Append implicitly to save to history without double printing
                            response += f"|||CONTEXT|||{graph_ctx}|||FAISS|||{faiss_ctx}"
                            
                    except Exception as e:
                        logging.error(f"Chat error: {e}")
                        try:
                            if hasattr(e, 'last_attempt') and hasattr(e.last_attempt, 'exception'):
                                cause = str(e.last_attempt.exception())
                            else:
                                cause = str(e)
                            clean_cause = cause.split("[")[0].strip()
                            response = f"Sorry, the AI service encountered an error: **{clean_cause}**"
                        except Exception:
                            response = "Sorry, I ran into an unexpected error. Please try again later."
                        st.markdown(response)

            msg_id = log_chat_history(st.session_state.user_id, st.session_state.current_session_id, final_prompt, response)
            st.session_state.chat_history.append({"role": "assistant", "content": response, "id": msg_id, "feedback": None})
            st.rerun()

# Router
if not st.session_state.user_profile:
    onboarding_screen()
else:
    main_chat()
