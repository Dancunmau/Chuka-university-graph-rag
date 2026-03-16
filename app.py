import streamlit as st
import uuid
import io
import pandas as pd
from src.chuka_graphrag_pipeline import GraphRAGAssistant
from src.database import get_or_create_user, save_user_profile, log_chat_history, get_chat_history, clear_chat_history, SessionLocal, UserProfile

st.set_page_config(
    page_title="Chuka University GraphRAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS 
st.markdown("""
<style>
/* ── 1. Force light mode on everything except sidebar ─────── */
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
/* ── 2. SIDEBAR – dark navy ───────────────────────────────── */
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

/* ── 3. MAIN — white background, no padding artefacts ─────── */
.block-container {
    padding-top: 3.5rem !important;
    background: #ffffff !important;
}

/* ── 4. CHAT messages — no background bubbles ─────────────── */
[data-testid="stChatMessage"] {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 12px 0 !important;
}
/* thin separator between messages */
[data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
    border-top: 1px solid #f1f3f9 !important;
}

/* ── 5. Chat input bar ─────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: #f4f6fb !important;
    border: 1px solid #dce1ef !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #1a1a2e !important;
}

/* ── 6. Onboarding — inputs must be white ─────────────────── */
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
</style>
""", unsafe_allow_html=True)

# ── 1. Device Token Identity (Se─
if "device_token" not in st.session_state:
    query_params = st.query_params
    if "token" in query_params:
        st.session_state.device_token = query_params["token"]
    else:
        st.session_state.device_token = str(uuid.uuid4())
        st.query_params["token"] = st.session_state.device_token

user = get_or_create_user(device_token=st.session_state.device_token)
st.session_state.user_id = user["user_id"]

@st.cache_resource(show_spinner=False)
def load_assistant():
    return GraphRAGAssistant()

if "assistant" in st.session_state:
    if not hasattr(st.session_state.assistant, "get_personalized_timetable"):
        del st.session_state["assistant"]
        st.cache_resource.clear()

if "assistant" not in st.session_state:
    st.session_state.assistant = load_assistant()

if "mapped_programmes" not in st.session_state:
    st.session_state.mapped_programmes = st.session_state.assistant.get_mapped_programmes()

if "user_profile" not in st.session_state:
    db = SessionLocal()
    try:
        p = db.query(UserProfile).filter(UserProfile.user_id == user["user_id"]).first()
        st.session_state.user_profile = (
            {"faculty": p.faculty, "program": p.program,
             "year": str(p.year_of_study), "semester": str(p.semester)}
            if p and p.faculty else None
        )
    finally:
        db.close()

if "chat_history" not in st.session_state:
    rows = get_chat_history(user["user_id"])
    st.session_state.chat_history = []
    for r in rows:
        st.session_state.chat_history += [
            {"role": "user", "content": r.query_text},
            {"role": "assistant", "content": r.response_text},
        ]

#  ONBOARDING SCREEN
def onboarding_screen():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;margin-bottom:20px;margin-top:40px;">
            <div style="display:inline-flex;align-items:center;justify-content:center;
                        width:72px;height:72px;border-radius:50%;
                        background:linear-gradient(135deg,#0b2d71,#176BFF);
                        color:white;font-size:2em;margin-bottom:14px;">🎓</div>
            <h2 style="color:#0b2d71;font-weight:700;margin:0 0 4px;">Chuka University</h2>
            <p style="color:#64748b;font-size:0.9em;margin:0 0 28px;">Academic Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        faculty = st.selectbox("Faculty", [
            "Select Faculty",
            "Faculty of Science",
            "Faculty of Education",
            "Faculty of Business",
            "Faculty of Environment",
            "Faculty of Agriculture",
            "Faculty of Engineering",
            "Faculty of Nursing",
            "Faculty of Arts",
            "Faculty of Law",
        ])
        mapped_progs = st.session_state.get('mapped_programmes', [])
        if not mapped_progs:
            display_options = ["Select Program", "BSc Computer Science", "BCom", "BEd Science"]
        else:
            display_options = ["Select Program"] + mapped_progs

        program_select = st.selectbox(
            "Program", 
            display_options,
            format_func=lambda x: f"{x['name']} ({x['count']} units)" if isinstance(x, dict) else x
        )
        
        program = program_select['name'] if isinstance(program_select, dict) else program_select
        year = st.selectbox("Year of Study", ["Select Year", "1", "2", "3", "4", "5", "6"])
        semester = st.selectbox("Semester", ["Select Semester", "1", "2", "3"])

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Continue", use_container_width=True):
            if "Select" in faculty or "Select" in program or "Select" in year or "Select" in semester:
                st.error("Please complete all fields.")
            else:
                st.session_state.user_profile = {
                    "faculty": faculty, "program": program, "year": year, "semester": semester
                }
                save_user_profile(st.session_state.user_id, faculty, program, year, semester)
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
                        color:white;font-size:1.1em;flex-shrink:0;">📚</div>
            <span style="font-size:1.5em;font-weight:700;color:#0f172a;">Course Explorer</span>
        </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv("d:/Jupyter notebook/Graph rag/data/curricular_mapping.csv")
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

#  MAIN APP INTERFACE
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
        if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
            st.session_state.chat_history = []
            st.session_state.current_view = "chat"
            st.rerun()
            
        if st.button("Course Explorer", use_container_width=True, key="open_explorer"):
            st.session_state.current_view = "explorer"
            st.rerun()

        if st.button("Clear History", use_container_width=True, key="clear_hist"):
            clear_chat_history(st.session_state.user_id)
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Personalized Tools**", unsafe_allow_html=True)
        
        # Export logic
        if tt_data:
            # Grid-Based professional PDF export
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
                elements.append(Paragraph("Chuka University 🎓", title_style))
                elements.append(Paragraph(f"Academic Timetable Matrix", styles['Heading2']))
                
                profile = st.session_state.user_profile
                elements.append(Paragraph(
                    f"<b>Program:</b> {profile['program']} | <b>Level:</b> Year {profile['year']}, Sem {profile['semester']}",
                    styles['Normal']
                ))
                elements.append(Spacer(1, 15))

                # Define Grid Matrix
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
                    ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor("#0b2d71")),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ]))
                elements.append(t)
                
                elements.append(Spacer(1, 20))
                elements.append(Paragraph(
                    "<font color='grey' size='8'><i>Legend: UNIT_CODE (ROOM) | Generated by Chuka AI Expert System</i></font>",
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

        st.markdown("<br>", unsafe_allow_html=True)

        # History list
        past_queries = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
        if past_queries:
            for q in reversed(past_queries[-8:]):
                label = q[:35] + "…" if len(q) > 37 else q
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 4px;'
                    f'border-radius:6px;cursor:pointer;font-size:0.88em;">'
                    f'<span style="opacity:.5;">☐</span>{label}</div>',
                    unsafe_allow_html=True
                )
        else:
            samples = [
                "Explain course registration...",
                "What are office hours for...",
                "How to apply for scholarships",
                "Campus map directions to...",
                "Graduation requirements",
                "Library resources for...",
                "Student housing options",
                "Add/drop course deadline",
            ]
            for s in samples:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 4px;'
                    f'border-radius:6px;cursor:pointer;font-size:0.88em;opacity:.7;">'
                    f'<span>☐</span>{s}</div>',
                    unsafe_allow_html=True
                )

        st.markdown("""
        <div style="margin-top:auto;padding:20px 0 0 0;">
            <hr style="border-color:#2e3650;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;font-size:.9em;cursor:pointer;">
                <span>📚</span> Library
            </div>
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;font-size:.9em;cursor:pointer;">
                <span>⚙️</span> Settings
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("↩  New Identity / Logout", use_container_width=True, key="logout"):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

    # MAIN AREA 
    if st.session_state.current_view == "explorer":
        course_explorer_view()
    else:
        # Header
        st.markdown("""
            <div style="display:flex;align-items:center;gap:14px;
                        padding-bottom:16px;border-bottom:1px solid #e8ecf4;margin-bottom:10px;">
                <div style="width:42px;height:42px;border-radius:50%;
                            background:linear-gradient(135deg,#7b61ff,#2563eb);
                            display:flex;align-items:center;justify-content:center;
                            color:white;font-size:1.1em;flex-shrink:0;">🎓</div>
                <span style="font-size:1.5em;font-weight:700;color:#0f172a;">AI Assistant</span>
            </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask the University Assistant..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        response = st.session_state.assistant.generate_response(
                            prompt, st.session_state.user_profile
                        )
                    except Exception as e:
                        response = f"Error: {e}"
                    st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            log_chat_history(st.session_state.user_id, prompt, response)
            st.rerun()

#  Router
if not st.session_state.user_profile:
    onboarding_screen()
else:
    main_chat()
