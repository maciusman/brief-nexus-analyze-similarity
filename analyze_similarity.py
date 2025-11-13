import os
import re
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
from typing import List, Dict

# ============== CONFIG ==============
SIMILARITY_THRESHOLDS = {
    'critical': 0.90,
    'warning': 0.75,
    'info': 0.60
}

# ============== PARSER MARKDOWN ==============
def parse_markdown_brief(file_path: str) -> List[Dict]:
    """Parse markdown brief do struktury z heading/knowledge/keywords"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    article_name = Path(file_path).stem
    sections = []
    
    # Split content by ## headings (any level - ##, ###, ####)
    # Pattern: finds ## X\. Heading (markdown files have literal backslash before dot)
    parts = re.split(r'\n(##+ \d+(?:\\\.\d+)*\\..*?)\n', content)
    
    # Process pairs: heading + content
    for i in range(1, len(parts), 2):
        if i+1 >= len(parts):
            break
            
        heading_raw = parts[i].strip()
        section_content = parts[i+1]
        
        # Extract section number from heading (e.g., "## 1\." -> "1" or "### 4\." -> "4")
        section_num_match = re.search(r'##+ (\d+(?:\\\.\d+)*)', heading_raw)
        if not section_num_match:
            continue
        section_num = section_num_match.group(1)

        # Clean heading (remove ##, number, and backslash-dot)
        heading_clean = re.sub(r'^##+ \d+(?:\\\.\d+)*\\\.\s*', '', heading_raw)
        heading_clean = re.sub(r'[\*\_]', '', heading_clean).strip()
        
        # Extract Wiedza (from **Wiedza:** to **Keywords:** or **Słowa kluczowe:**)
        # Note: Content may start with \n, and there may be spaces after colons
        knowledge_match = re.search(r'\*\*Wiedza:\*\*\s+\n(.*?)\n\*\*(?:Keywords|Słowa kluczowe):\*\*', section_content, re.DOTALL)
        if not knowledge_match:
            # Try alternative format without extra spaces
            knowledge_match = re.search(r'\*\*Wiedza:\*\*\s*\n(.*?)\n\*\*(?:Keywords|Słowa kluczowe):\*\*', section_content, re.DOTALL)
        knowledge = knowledge_match.group(1).strip() if knowledge_match else ""

        # Extract Keywords (from **Keywords:** or **Słowa kluczowe:** to end or next ##)
        keywords_match = re.search(r'\*\*(?:Keywords|Słowa kluczowe):\*\*\s+\n(.*?)(?=\n##|$)', section_content, re.DOTALL)
        if not keywords_match:
            # Try alternative format
            keywords_match = re.search(r'\*\*(?:Keywords|Słowa kluczowe):\*\*\s*\n(.*?)(?=\n##|$)', section_content, re.DOTALL)
        keywords = keywords_match.group(1).strip() if keywords_match else ""
        
        if knowledge:  # Only add if we found knowledge
            sections.append({
                'id': f"{article_name}_sec{section_num}",
                'article': article_name,
                'section_num': section_num,
                'heading': heading_clean,
                'knowledge': knowledge,
                'keywords': keywords,
                'combined_text': f"{heading_clean}\n\n{knowledge}\n\nKeywords: {keywords}"
            })
    
    return sections
# ============== MAIN ANALYSIS ==============
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def analyze_similarities(all_sections: List[Dict], model, threshold: float = 0.60):
    """Analyze similarities between sections"""
    
    # Generate embeddings
    progress_bar = st.progress(0, text="🧠 Generowanie embeddingów...")
    
    texts = [s['combined_text'] for s in all_sections]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    progress_bar.progress(50, text="🔍 Obliczanie podobieństw...")
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    progress_bar.progress(75, text="📊 Wyszukiwanie par...")
    
    # Find pairs above threshold
    pairs = []
    n = len(all_sections)
    
    for i in range(n):
        for j in range(i+1, n):
            sim = similarity_matrix[i][j]
            
            if sim >= threshold:
                pairs.append({
                    'section_1': all_sections[i],
                    'section_2': all_sections[j],
                    'similarity': sim,
                    'same_article': all_sections[i]['article'] == all_sections[j]['article']
                })
    
    # Sort by similarity
    pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    progress_bar.progress(100, text="✅ Gotowe!")
    
    return pairs, similarity_matrix

# ============== STREAMLIT UI ==============
def main():
    st.set_page_config(page_title="Content Similarity Analyzer", layout="wide")
    
    st.title("📊 Content Similarity Analyzer")
    st.markdown("Analiza podobieństwa sekcji briefów - detekcja kanibalizacji treści")
    
    # Sidebar - Config
    st.sidebar.header("⚙️ Konfiguracja")
    
    folder_path = st.sidebar.text_input("Ścieżka do folderu z briefami:", value="./briefy")
    threshold = st.sidebar.slider("Próg podobieństwa (INFO):", 0.0, 1.0, 0.60, 0.05)
    
    show_same_article = st.sidebar.checkbox("Pokaż podobieństwa w tym samym artykule", value=True)
    show_diff_article = st.sidebar.checkbox("Pokaż podobieństwa między artykułami", value=True)
    
    if st.sidebar.button("🚀 URUCHOM ANALIZĘ", type="primary"):
        
        # Load files
        st.header("📂 Ładowanie plików...")
        
        folder = Path(folder_path)
        if not folder.exists():
            st.error(f"❌ Folder nie istnieje: {folder_path}")
            return
            
        md_files = list(folder.glob("*.md"))
        
        if not md_files:
            st.error(f"❌ Nie znaleziono plików .md w folderze: {folder_path}")
            return
        
        st.success(f"✅ Znaleziono {len(md_files)} plików")
        
        # Parse files
        all_sections = []
        with st.expander("📄 Przetwarzanie plików", expanded=True):
            for md_file in md_files:
                try:
                    sections = parse_markdown_brief(md_file)
                    all_sections.extend(sections)
                    
                    if sections:
                        st.write(f"✅ {md_file.name}: **{len(sections)}** sekcji")
                    else:
                        st.warning(f"⚠️ {md_file.name}: 0 sekcji (sprawdź format)")
                        
                except Exception as e:
                    st.error(f"❌ Błąd w {md_file.name}: {e}")
        
        if not all_sections:
            st.error("❌ Nie załadowano żadnych sekcji! Sprawdź format plików.")
            
            with st.expander("🔍 Debug - Przykład oczekiwanego formatu"):
                st.code("""
## 1. Nagłówek pierwszej sekcji
**Wiedza:**
Treść wiedzy dla pierwszej sekcji...

**Słowa kluczowe:**
słowo1, słowo2, słowo3

## 2. Nagłówek drugiej sekcji
**Wiedza:**
Treść wiedzy dla drugiej sekcji...

**Słowa kluczowe:**
słowo4, słowo5, słowo6
                """, language="markdown")
            return
        
        st.success(f"📊 **Łącznie: {len(all_sections)} sekcji** z {len(set([s['article'] for s in all_sections]))} artykułów")
        
        # Load model and analyze
        with st.spinner("Ładowanie modelu embeddingowego..."):
            model = load_model()
        
        pairs, similarity_matrix = analyze_similarities(all_sections, model, threshold)
        
        # Filter pairs based on checkboxes
        filtered_pairs = []
        for pair in pairs:
            if pair['same_article'] and show_same_article:
                filtered_pairs.append(pair)
            elif not pair['same_article'] and show_diff_article:
                filtered_pairs.append(pair)
        
        # Statistics
        st.header("📈 Statystyki")
        
        col1, col2, col3, col4 = st.columns(4)
        
        critical = sum(1 for p in filtered_pairs if p['similarity'] >= 0.90)
        warning = sum(1 for p in filtered_pairs if 0.75 <= p['similarity'] < 0.90)
        info = sum(1 for p in filtered_pairs if 0.60 <= p['similarity'] < 0.75)
        
        col1.metric("🔴 CRITICAL (≥90%)", critical)
        col2.metric("🟡 WARNING (75-90%)", warning)
        col3.metric("🔵 INFO (60-75%)", info)
        col4.metric("📝 Razem", len(filtered_pairs))
        
        # Display pairs
        st.header("🔍 Wykryte podobieństwa")
        
        if not filtered_pairs:
            st.success("✅ Brak podobieństw powyżej progu!")
            return
        
        # Filter options
        severity_filter = st.selectbox(
            "Filtruj po poziomie:",
            ["Wszystkie", "🔴 Critical", "🟡 Warning", "🔵 Info"]
        )
        
        displayed_count = 0
        
        for idx, pair in enumerate(filtered_pairs):
            sim = pair['similarity']
            
            # Determine severity
            if sim >= 0.90:
                badge = "🔴 CRITICAL"
                color = "#e74c3c"
            elif sim >= 0.75:
                badge = "🟡 WARNING"
                color = "#f39c12"
            else:
                badge = "🔵 INFO"
                color = "#3498db"
            
            # Apply filter
            if severity_filter == "🔴 Critical" and sim < 0.90:
                continue
            elif severity_filter == "🟡 Warning" and (sim < 0.75 or sim >= 0.90):
                continue
            elif severity_filter == "🔵 Info" and (sim < 0.60 or sim >= 0.75):
                continue
            
            displayed_count += 1
            
            # Display pair
            with st.container():
                st.markdown(f"### Para #{displayed_count}")
                
                col_badge, col_sim, col_status = st.columns([2, 2, 3])
                
                with col_badge:
                    st.markdown(f"<h3 style='color:{color};'>{badge}</h3>", unsafe_allow_html=True)
                with col_sim:
                    st.markdown(f"**Podobieństwo: {sim:.1%}**")
                with col_status:
                    if pair['same_article']:
                        st.success("✓ Ten sam artykuł")
                    else:
                        st.error("⚠ Różne artykuły")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📄 Artykuł 1**")
                    st.info(f"**{pair['section_1']['article']}**")
                    st.markdown(f"**Sekcja:** {pair['section_1']['section_num']}")
                    st.markdown(f"**Nagłówek:** {pair['section_1']['heading']}")
                    
                    with st.expander("📖 Wiedza", expanded=False):
                        st.text_area("", pair['section_1']['knowledge'], height=200, key=f"knowledge1_{idx}", disabled=True)
                    
                    with st.expander("🏷️ Keywords", expanded=False):
                        st.text(pair['section_1']['keywords'])
                
                with col2:
                    st.markdown(f"**📄 Artykuł 2**")
                    st.info(f"**{pair['section_2']['article']}**")
                    st.markdown(f"**Sekcja:** {pair['section_2']['section_num']}")
                    st.markdown(f"**Nagłówek:** {pair['section_2']['heading']}")
                    
                    with st.expander("📖 Wiedza", expanded=False):
                        st.text_area("", pair['section_2']['knowledge'], height=200, key=f"knowledge2_{idx}", disabled=True)
                    
                    with st.expander("🏷️ Keywords", expanded=False):
                        st.text(pair['section_2']['keywords'])
                
                st.divider()
        
        # Export to CSV
        st.header("💾 Export")

        # Prepare CSV data
        df_export = []
        for pair in filtered_pairs:
            df_export.append({
                'Podobieństwo': f"{pair['similarity']:.3f}",
                'Artykuł 1': pair['section_1']['article'],
                'Sekcja 1': pair['section_1']['section_num'],
                'Nagłówek 1': pair['section_1']['heading'],
                'Artykuł 2': pair['section_2']['article'],
                'Sekcja 2': pair['section_2']['section_num'],
                'Nagłówek 2': pair['section_2']['heading'],
                'Ten sam artykuł': 'TAK' if pair['same_article'] else 'NIE'
            })

        df = pd.DataFrame(df_export)
        csv = df.to_csv(index=False, encoding='utf-8-sig')

        # Direct download button
        st.download_button(
            label="📥 Pobierz raport CSV",
            data=csv,
            file_name="similarity_report.csv",
            mime="text/csv",
            use_container_width=False
        )

if __name__ == "__main__":
    main()