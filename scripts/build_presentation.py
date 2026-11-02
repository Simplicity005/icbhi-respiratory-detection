"""
Generate Professional Widescreen PowerPoint Presentation for ICBHI Respiratory Sound Detection Project.
Includes simple plain-English explanations, structured slides, embedded figures, and future work.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    figures_dir = os.path.join(project_root, "reports", "figures")
    output_pptx = os.path.join(project_root, "ICBHI_Respiratory_Detection_Progress.pptx")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5) # 16:9 widescreen
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    NAVY = RGBColor(26, 54, 93)      # #1A365D Primary dark
    BLUE = RGBColor(43, 108, 176)    # #2B6CB0 Accent blue
    TEAL = RGBColor(49, 151, 149)    # #319795 Accent teal
    CHARCOAL = RGBColor(45, 55, 72)  # #2D3748 Body text
    LIGHT_BG = RGBColor(247, 250, 252) # #F7FAFC Slide bg
    WHITE = RGBColor(255, 255, 255)
    GREEN = RGBColor(47, 133, 90)    # #2F855A Success green
    CARD_BG = RGBColor(237, 242, 247) # #EDF2F7 Card background
    CARD_BORDER = RGBColor(226, 232, 240)

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, category, title, subtitle=None):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = TEAL

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

        # Optional Subtitle
        if subtitle:
            p_sub = tf_title.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.size = Pt(13)
            p_sub.font.color.rgb = CHARCOAL

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # Accent decorative bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.15), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    # Title text box
    tb_title = slide1.shapes.add_textbox(Inches(1.4), Inches(1.4), Inches(10.8), Inches(4.0))
    tf1 = tb_title.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "MACHINE LEARNING PROJECT PRESENTATION"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = TEAL

    p1 = tf1.add_paragraph()
    p1.text = "AI-Powered Respiratory Sound Detection"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Automated Classification of Abnormal Lung Sounds (Crackles & Wheezes) from Stethoscope Audio"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(226, 232, 240)
    p2.space_before = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Based on the ICBHI 2017 International Scientific Challenge Database (126 Patients | 6,898 Breath Cycles)"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(160, 174, 192)
    p3.space_before = Pt(16)

    # Footer Card
    fcard = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.4), Inches(5.8), Inches(10.5), Inches(0.9))
    fcard.fill.solid()
    fcard.fill.fore_color.rgb = RGBColor(43, 108, 176)
    fcard.line.fill.background()
    tf_fcard = fcard.text_frame
    tf_fcard.word_wrap = True
    pf = tf_fcard.paragraphs[0]
    pf.text = "Current Status: 80% Complete (Phases 1 to 4 Fully Implemented & Benchmarked)"
    pf.font.size = Pt(14)
    pf.font.bold = True
    pf.font.color.rgb = WHITE
    pf.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: What We Are Doing & The Medical Problem
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Introduction & Motivation", "What Are We Doing & Why Does It Matter?",
               "Bridging the gap between physician auscultation and objective machine learning.")

    # Left Container: The Problem
    card_prob = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card_prob.fill.solid()
    card_prob.fill.fore_color.rgb = WHITE
    card_prob.line.color.rgb = CARD_BORDER
    tf_prob = card_prob.text_frame
    tf_prob.word_wrap = True

    p = tf_prob.paragraphs[0]
    p.text = "⚠️ The Clinical Challenge in Simple Words"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(197, 48, 48)

    bullets_prob = [
        "How doctors check lungs: They listen with a stethoscope for wheezing (whistling) or crackles (popping).",
        "The problem with human ears: Interpretation is highly subjective. Two doctors often disagree on the exact sound.",
        "Noisy clinic rooms: Ambient hospital noise and quiet, shallow breathing make faint disease sounds hard to hear.",
        "Late diagnosis: Chronic lung diseases like COPD and Pneumonia are often caught late, increasing mortality."
    ]
    for b in bullets_prob:
        bp = tf_prob.add_paragraph()
        bp.text = "• " + b
        bp.font.size = Pt(13)
        bp.font.color.rgb = CHARCOAL
        bp.space_before = Pt(8)

    # Right Container: Our Solution
    card_sol = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8))
    card_sol.fill.solid()
    card_sol.fill.fore_color.rgb = WHITE
    card_sol.line.color.rgb = CARD_BORDER
    tf_sol = card_sol.text_frame
    tf_sol.word_wrap = True

    p = tf_sol.paragraphs[0]
    p.text = "💡 Our Automated Machine Learning Solution"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = GREEN

    bullets_sol = [
        "Digital stethoscope recording: Capture the breath sound digitally from the chest wall.",
        "Automatic noise cleaning: Remove low-frequency heart thuds and high-frequency microphone hiss.",
        "Breath-by-breath analysis: Chop continuous audio into individual breath cycles (6,898 cycles in our study).",
        "Acoustic AI detection: Convert each breath into 307 acoustic numbers and use machine learning to detect disease sounds accurately."
    ]
    for b in bullets_sol:
        bp = tf_sol.add_paragraph()
        bp.text = "• " + b
        bp.font.size = Pt(13)
        bp.font.color.rgb = CHARCOAL
        bp.space_before = Pt(8)

    # =========================================================================
    # SLIDE 3: The 4 Lung Sounds Explained in Everyday Words
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "Clinical Fundamentals", "The 4 Respiratory Sounds We Are Detecting",
               "Every breath belongs to one of four sound classes annotated by medical specialists.")

    classes_info = [
        ("Normal Breathing (52.8%)", "Smooth, clean breath sound with no obstructions or rattles.",
         "Vesicular sound of air moving smoothly through healthy bronchial passages.", RGBColor(47, 133, 90)),
        ("Crackles (27.0%)", "Short, explosive popping clicks (like bubble wrap or peeling Velcro).",
         "Caused by collapsed small airways suddenly snapping open when fluid is present (e.g., Pneumonia).", RGBColor(221, 107, 32)),
        ("Wheezes (12.8%)", "Continuous, high-pitched musical whistling or squeaking.",
         "Caused by air forced at high speed through narrowed, inflamed airways (e.g., Asthma, COPD).", RGBColor(49, 130, 206)),
        ("Both (7.3%)", "Severe breaths containing both crackles and wheezes together.",
         "Indicates advanced, multi-symptom lung disease; the rarest and hardest to classify.", RGBColor(128, 90, 213))
    ]

    for idx, (c_name, c_simple, c_medical, c_color) in enumerate(classes_info):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8 + col * 6.0)
        top = Inches(1.8 + row * 2.5)

        card = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.7), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = c_color
        card.line.width = Pt(1.5)

        tf = card.text_frame
        tf.word_wrap = True
        p_title = tf.paragraphs[0]
        p_title.text = c_name
        p_title.font.size = Pt(15)
        p_title.font.bold = True
        p_title.font.color.rgb = c_color

        p_desc = tf.add_paragraph()
        p_desc.text = "In Plain English: " + c_simple
        p_desc.font.size = Pt(12)
        p_desc.font.bold = True
        p_desc.font.color.rgb = CHARCOAL
        p_desc.space_before = Pt(4)

        p_med = tf.add_paragraph()
        p_med.text = "Clinical Cause: " + c_medical
        p_med.font.size = Pt(11)
        p_med.font.color.rgb = RGBColor(113, 128, 150)
        p_med.space_before = Pt(4)

    # =========================================================================
    # SLIDE 4: Phase 1 — Exploratory Data Analysis (EDA) Results
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Phase 1: Exploratory Data Analysis", "What Did We Learn from the ICBHI Dataset?",
               "Deep analysis of 126 patients, 920 recordings, and 6,898 breath cycles.")

    # Left bullet card
    card_eda = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    card_eda.fill.solid()
    card_eda.fill.fore_color.rgb = WHITE
    card_eda.line.color.rgb = CARD_BORDER
    tf_eda = card_eda.text_frame
    tf_eda.word_wrap = True

    p = tf_eda.paragraphs[0]
    p.text = "Key Findings in Plain Words:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    findings = [
        "126 Patients Examined: Ages ranged from infants (0.2 years) to elderly adults (93 years).",
        "Disease Imbalance: COPD was by far the most frequent diagnosis (>70% of sick patients), reflecting real hospital admissions.",
        "Hardware Variation: Audio was captured using 4 different stethoscope devices with sampling rates ranging from 4 kHz to 44.1 kHz.",
        "Sound Class Imbalance: Healthy normal breaths account for 52.8%, crackles 27.0%, wheezes 12.8%, and severe 'both' cycles only 7.3%.",
        "Official Partition: 79 patients (4,142 cycles) assigned to Training, and 49 patients (2,756 cycles) reserved for Testing."
    ]
    for f in findings:
        fp = tf_eda.add_paragraph()
        fp.text = "• " + f
        fp.font.size = Pt(12)
        fp.font.color.rgb = CHARCOAL
        fp.space_before = Pt(6)

    # Right side: Embed Figure 4 (Sound classes distribution)
    fig4_path = os.path.join(figures_dir, "fig4_respiratory_cycles_adventitious_sounds.png")
    if os.path.exists(fig4_path):
        slide4.shapes.add_picture(fig4_path, Inches(6.7), Inches(1.8), width=Inches(5.8))

    # =========================================================================
    # SLIDE 5: Project Roadmap & 5-Phase Plan
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Engineering Roadmap", "Project Architecture: 5 Core Phases",
               "Structured medical machine learning pipeline from raw audio to clinical prediction.")

    phases = [
        ("Phase 1: Exploratory Data Analysis", "COMPLETED",
         "Survey 126 patients, audit sampling rates, demographics, and class distributions.", GREEN),
        ("Phase 2: Signal Preprocessing & Segmentation", "COMPLETED",
         "16 kHz resampling, 50-2000 Hz Butterworth bandpass filtering, slicing 6,898 breath cycles.", GREEN),
        ("Phase 3: Acoustic Feature Engineering", "COMPLETED",
         "Extract 307 acoustic features per cycle (MFCCs, spectral rolloff, pitch, temporal dynamics).", GREEN),
        ("Phase 4: ML Model Benchmarking", "COMPLETED",
         "Train Random Forest, LightGBM, SVM, and Logistic Regression with 5-fold patient-aware CV.", GREEN),
        ("Phase 5: Deep Learning & Deployment", "NEXT UP",
         "Train 2D Spectrogram CNNs (ResNet, AST) to preserve time-frequency details and boost score.", RGBColor(183, 121, 31))
    ]

    for idx, (p_title, p_stat, p_desc, p_color) in enumerate(phases):
        top_y = Inches(1.7 + idx * 1.05)
        pcard = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_y, Inches(11.7), Inches(0.92))
        pcard.fill.solid()
        pcard.fill.fore_color.rgb = WHITE
        pcard.line.color.rgb = p_color
        pcard.line.width = Pt(1.5)

        tf = pcard.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = f"{p_title}   [{p_stat}]"
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = p_color

        pd = tf.add_paragraph()
        pd.text = p_desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = CHARCOAL

    # =========================================================================
    # SLIDE 6: Phase 2 — Signal Preprocessing & Audio Slicing
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Phase 2: Audio Preprocessing", "Cleaning Raw Audio & Slicing Breath Cycles",
               "Turning noisy, multi-speed stethoscope recordings into clean standardized breath cycles.")

    # Left text box
    card_p2 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    card_p2.fill.solid()
    card_p2.fill.fore_color.rgb = WHITE
    card_p2.line.color.rgb = CARD_BORDER
    tf_p2 = card_p2.text_frame
    tf_p2.word_wrap = True

    p = tf_p2.paragraphs[0]
    p.text = "3 Crucial Cleaning Steps in Plain English:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2_points = [
        "1. Standardize Audio Speed (16 kHz Resampling): Different stethoscopes recorded at different speeds (4,000 to 44,100 samples/sec). We converted everything to 16,000 samples/second so models see consistent audio.",
        "2. Heartbeat & Static Filtering (50 to 2,000 Hz Bandpass): Heartbeats are low-frequency (<50 Hz thuds) that drown out lung sounds. We designed a zero-phase Butterworth bandpass filter that removes heart sounds below 50 Hz and microphone hissing above 2,000 Hz.",
        "3. Breath Cycle Slicing: Doctors marked the exact start and end seconds of each breath. We automatically sliced continuous recordings into 6,898 discrete breath audio files."
    ]
    for pt in p2_points:
        p_pt = tf_p2.add_paragraph()
        p_pt.text = "• " + pt
        p_pt.font.size = Pt(11.5)
        p_pt.font.color.rgb = CHARCOAL
        p_pt.space_before = Pt(8)

    # Right side: Embed Figure 7 (Filtering & Segmentation)
    fig7_path = os.path.join(figures_dir, "fig7_preprocessing_and_segmentation.png")
    if os.path.exists(fig7_path):
        slide6.shapes.add_picture(fig7_path, Inches(6.7), Inches(1.8), width=Inches(5.8))

    # =========================================================================
    # SLIDE 7: Phase 3 — Acoustic Feature Engineering
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Phase 3: Feature Engineering", "Converting Breath Sounds into 307 Numbers",
               "Translating raw audio vibrations into meaningful mathematical biomarkers for ML.")

    # Left text card
    card_p3 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    card_p3.fill.solid()
    card_p3.fill.fore_color.rgb = WHITE
    card_p3.line.color.rgb = CARD_BORDER
    tf_p3 = card_p3.text_frame
    tf_p3.word_wrap = True

    p = tf_p3.paragraphs[0]
    p.text = "What Do the 307 Numbers Measure?"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    feats = [
        "1. Sound Texture / Timbre (MFCCs - 240 features): Captures the unique vocal profile and texture of the breath over time (like telling a guitar apart from a piano).",
        "2. Pitch & Brightness (Spectral - 38 features): Measures pitch centroid, rolloff, and bandwidth. Tells the model if the sound is high-pitched (whistling wheeze) or low-pitched (normal).",
        "3. Sudden Pops & Energy (Temporal - 9 features): Catches sharp explosive clicks and energy bursts lasting <20 ms (crackles).",
        "4. Musical Harmonics (Chroma - 24 features): Wheezes have musical notes with harmonic overtones that chroma features detect.",
        "Output Datasets: Exported ready-to-train X_train.csv (4,142 rows) and X_test.csv (2,756 rows)."
    ]
    for f in feats:
        p_f = tf_p3.add_paragraph()
        p_f.text = "• " + f
        p_f.font.size = Pt(11.5)
        p_f.font.color.rgb = CHARCOAL
        p_f.space_before = Pt(6)

    # Right side: Embed Figure 8 (Feature distribution boxplots)
    fig8_path = os.path.join(figures_dir, "fig8_acoustic_features_distribution.png")
    if os.path.exists(fig8_path):
        slide7.shapes.add_picture(fig8_path, Inches(6.7), Inches(1.8), width=Inches(5.8))

    # =========================================================================
    # SLIDE 8: Phase 4 — Machine Learning & Patient-Aware CV
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Phase 4: ML Methodology", "Patient-Aware Cross-Validation (No Cheating!)",
               "Ensuring algorithms learn general medical disease patterns, not individual patient voices.")

    # Left Container: Patient Leakage Explanation
    card_gkf = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card_gkf.fill.solid()
    card_gkf.fill.fore_color.rgb = WHITE
    card_gkf.line.color.rgb = CARD_BORDER
    tf_gkf = card_gkf.text_frame
    tf_gkf.word_wrap = True

    p = tf_gkf.paragraphs[0]
    p.text = "⚠️ Why Standard Train/Test Splitting Fails in Healthcare"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RGBColor(197, 48, 48)

    gkf_points = [
        "The Data Leakage Trap: Each patient contributes multiple breath cycles. If you randomly split breaths, Cycle 1 of Patient 107 is in training, and Cycle 2 of Patient 107 is in testing.",
        "Cheating without knowing it: The model simply memorizes Patient 107's unique stethoscope noise and tone, getting false high test scores.",
        "Our Solution (5-Fold GroupKFold): We strictly group by patient_id. Every breath from a patient is either 100% in training or 100% in validation. Zero overlap!",
        "Final Unseen Test Cohort: All models are finally evaluated on 49 completely new patients (2,756 cycles) who were never seen during training."
    ]
    for pt in gkf_points:
        p_pt = tf_gkf.add_paragraph()
        p_pt.text = "• " + pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = CHARCOAL
        p_pt.space_before = Pt(8)

    # Right Container: The 4 Classifiers
    card_clfs = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8))
    card_clfs.fill.solid()
    card_clfs.fill.fore_color.rgb = WHITE
    card_clfs.line.color.rgb = CARD_BORDER
    tf_clfs = card_clfs.text_frame
    tf_clfs.word_wrap = True

    p = tf_clfs.paragraphs[0]
    p.text = "🌲 The 4 Machine Learning Classifiers Tested"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE

    clfs = [
        "1. Random Forest (RF): 200 decision trees voting together. Naturally robust to noisy features with balanced class weights.",
        "2. LightGBM (LGB): Fast gradient-boosted decision trees. Outstanding at finding subtle splits in tabular feature data.",
        "3. Support Vector Machine (SVM): RBF non-linear kernel finding high-dimensional decision boundaries.",
        "4. Logistic Regression (LR): Standard linear baseline with L2 regularization to establish an interpretable floor.",
        "Official Metric: ICBHI Score = (Sensitivity + Specificity) / 2. Gives fair 50/50 balance between catching sick breaths and avoiding false alarms."
    ]
    for c in clfs:
        p_c = tf_clfs.add_paragraph()
        p_c.text = "• " + c
        p_c.font.size = Pt(12)
        p_c.font.color.rgb = CHARCOAL
        p_c.space_before = Pt(6)

    # =========================================================================
    # SLIDE 9: Phase 4 — Benchmark Results Comparison
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Phase 4: Results & Benchmarks", "Official Test Set Performance Comparison",
               "Evaluated on 2,756 respiratory cycles across 49 completely unseen patients.")

    # Left side: Embed Figure 11 (Benchmark comparison)
    fig11_path = os.path.join(figures_dir, "fig11_model_benchmark_comparison.png")
    if os.path.exists(fig11_path):
        slide9.shapes.add_picture(fig11_path, Inches(0.8), Inches(1.8), width=Inches(6.2))

    # Right side: Summary table & takeaway
    card_res = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.1))
    card_res.fill.solid()
    card_res.fill.fore_color.rgb = WHITE
    card_res.line.color.rgb = CARD_BORDER
    tf_res = card_res.text_frame
    tf_res.word_wrap = True

    p = tf_res.paragraphs[0]
    p.text = "🏆 Key Results & Winner:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = GREEN

    res_bullets = [
        "LightGBM Won Both Tasks: Achieved the highest overall ICBHI score among all tested models.",
        "4-Class Task (Normal, Crackle, Wheeze, Both):",
        "  • LightGBM: 42.85% Score (55.98% Sp, 47.71% Acc)",
        "  • Random Forest: 39.78% Score (48.13% Sp)",
        "  • SVM (RBF): 39.40% Score (47.37% Sp)",
        "  • Logistic Regression: 36.92% Score",
        "2-Class Binary Task (Healthy vs. Sick):",
        "  • LightGBM: 55.70% Score (59.22% Se, 52.18% Sp)",
        "  • SVM (RBF): 55.52% Score (56.24% Sp)",
        "  • Random Forest: 55.00% Score (58.45% Se)",
        "Matches Published Literature: Standard classical ML models typically hit 40-45% on 4-Class and 55-58% on Binary under patient-independent isolation."
    ]
    for rb in res_bullets:
        p_rb = tf_res.add_paragraph()
        p_rb.text = rb
        p_rb.font.size = Pt(11)
        p_rb.font.color.rgb = CHARCOAL
        p_rb.space_before = Pt(3)

    # =========================================================================
    # SLIDE 10: Where Do Models Make Mistakes & What Matters?
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Phase 4: Deep Insights", "Confusion Matrices & Top Acoustic Biomarkers",
               "Analyzing where errors occur and which sound features matter most.")

    # Left: Embed Figure 12 (Confusion matrix)
    fig12_path = os.path.join(figures_dir, "fig12_confusion_matrices.png")
    if os.path.exists(fig12_path):
        slide10.shapes.add_picture(fig12_path, Inches(0.8), Inches(1.8), width=Inches(5.7))

    # Right: Embed Figure 13 (Feature importances)
    fig13_path = os.path.join(figures_dir, "fig13_feature_importances.png")
    if os.path.exists(fig13_path):
        slide10.shapes.add_picture(fig13_path, Inches(6.8), Inches(1.8), width=Inches(5.7))

    # Bottom takeaway bar
    bbar = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.9), Inches(11.7), Inches(0.95))
    bbar.fill.solid()
    bbar.fill.fore_color.rgb = WHITE
    bbar.line.color.rgb = TEAL
    tf_bbar = bbar.text_frame
    tf_bbar.word_wrap = True
    pb = tf_bbar.paragraphs[0]
    pb.text = "💡 Clinical Insights: (1) Subtle crackles are sometimes mistaken for normal breathing because they last <20ms. (2) Wheezes have musical whistle harmonics that models detect reliably. (3) MFCC timbre dynamics & spectral centroid are the most predictive biomarkers."
    pb.font.size = Pt(11)
    pb.font.bold = True
    pb.font.color.rgb = NAVY

    # =========================================================================
    # SLIDE 11: Future Work — Phase 5 (Deep Learning & CNNs)
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "Future Work & Next Steps", "Phase 5: Deep Learning on Spectrograms",
               "Why we are moving from tabular numbers to 2D Audio Computer Vision.")

    card_why = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card_why.fill.solid()
    card_why.fill.fore_color.rgb = WHITE
    card_why.line.color.rgb = CARD_BORDER
    tf_why = card_why.text_frame
    tf_why.word_wrap = True

    p = tf_why.paragraphs[0]
    p.text = "🔍 Why Classical ML Hits a Ceiling"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RGBColor(197, 48, 48)

    why_points = [
        "Information Loss: When we average 307 numbers across a 2.5-second breath (mean, std, skew), we lose the exact time when a crackle clicked.",
        "Crackles are tiny: A crackle is a 20 ms pop inside a 2,500 ms breath. In statistical averages, that tiny spike gets washed out.",
        "Wheezes vary over time: Patients wheeze during late exhalation, not during inhalation. Tabular models don't know which phase of breathing the whistle happened in.",
        "Classical ML Cap: This is why classical models across the literature plateau around 42–45% on 4-Class."
    ]
    for wp in why_points:
        p_wp = tf_why.add_paragraph()
        p_wp.text = "• " + wp
        p_wp.font.size = Pt(12)
        p_wp.font.color.rgb = CHARCOAL
        p_wp.space_before = Pt(8)

    card_dl = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8))
    card_dl.fill.solid()
    card_dl.fill.fore_color.rgb = WHITE
    card_dl.line.color.rgb = CARD_BORDER
    tf_dl = card_dl.text_frame
    tf_dl.word_wrap = True

    p = tf_dl.paragraphs[0]
    p.text = "🚀 Our Phase 5 Deep Learning Plan"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = GREEN

    dl_points = [
        "1. Mel-Spectrogram Images: Convert audio into 2D heatmaps where X-axis is time and Y-axis is pitch (frequency).",
        "2. 2D Convolutional Neural Networks (CNNs / ResNet): Treat breath spectrograms like images. The CNN's convolutional filters will look for the exact visual shape of crackle spikes and wheeze ribbons.",
        "3. Audio Spectrogram Transformers (AST): Use self-attention across time patches to capture long-range breathing patterns.",
        "4. Data Augmentation (SpecAugment / Mixup): Synthesize realistic pitch shifts and time masks to address class imbalance.",
        "Goal: Push ICBHI Score from 55% to 65%+ on unseen patients!"
    ]
    for dp in dl_points:
        p_dp = tf_dl.add_paragraph()
        p_dp.text = "• " + dp
        p_dp.font.size = Pt(12)
        p_dp.font.color.rgb = CHARCOAL
        p_dp.space_before = Pt(6)

    # =========================================================================
    # SLIDE 12: Summary & Conclusion
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    bg12 = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = NAVY
    bg12.line.fill.background()

    tb_end = slide12.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.3), Inches(5.0))
    tf12 = tb_end.text_frame
    tf12.word_wrap = True

    pe0 = tf12.paragraphs[0]
    pe0.text = "SUMMARY & CONCLUSION"
    pe0.font.size = Pt(14)
    pe0.font.bold = True
    pe0.font.color.rgb = TEAL

    pe1 = tf12.add_paragraph()
    pe1.text = "What We Have Accomplished So Far"
    pe1.font.size = Pt(32)
    pe1.font.bold = True
    pe1.font.color.rgb = WHITE
    pe1.space_before = Pt(8)

    achievements = [
        "✓ 4 Complete Phases: Exploratory analysis, audio filtering & cycle segmentation, 307-feature extraction, and ML benchmarking.",
        "✓ 6,898 Real Breath Cycles Analyzed: Filtered from 50 to 2,000 Hz, zero-phase Butterworth filtering removing heartbeats.",
        "✓ Zero Patient Leakage: 5-Fold GroupKFold and official 49-patient unseen test cohort validation.",
        "✓ Rigorous Benchmarking: LightGBM established as the strongest classical baseline (42.85% 4-Class, 55.70% Binary).",
        "✓ Publication-Ready Artifacts: 13 high-resolution figures, 4 interactive notebooks, trained model weights, and clean documentation.",
        "✓ Clear Path to Phase 5: Moving to 2D Mel-Spectrogram CNNs to break through tabular feature limits."
    ]
    for a in achievements:
        pa = tf12.add_paragraph()
        pa.text = a
        pa.font.size = Pt(14)
        pa.font.color.rgb = RGBColor(226, 232, 240)
        pa.space_before = Pt(8)

    pe_q = tf12.add_paragraph()
    pe_q.text = "Thank You! Questions & Discussion"
    pe_q.font.size = Pt(20)
    pe_q.font.bold = True
    pe_q.font.color.rgb = TEAL
    pe_q.space_before = Pt(20)
    pe_q.alignment = PP_ALIGN.CENTER

    # Save presentation
    prs.save(output_pptx)
    print(f"[OK] Presentation successfully generated: {output_pptx}")
    return output_pptx

if __name__ == "__main__":
    create_presentation()
