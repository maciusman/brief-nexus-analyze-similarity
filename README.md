# BRIEF: Content Similarity Analyzer

Aplikacja Streamlit do analizy podobieństwa sekcji w briefach contentowych i wykrywania potencjalnej kanibalizacji treści SEO.

## Funkcje

- **Parser markdown** - Automatyczne wyciąganie sekcji z plików .md (obsługuje nagłówki ##, ###, ####)
- **Analiza podobieństwa** - Wykorzystuje embeddingi (sentence-transformers) i cosine similarity
- **Progi podobieństwa**:
  - 🔴 CRITICAL: ≥90%
  - 🟡 WARNING: 75-90%
  - 🔵 INFO: 60-75%
- **Interfejs Streamlit** - Przejrzysty GUI z filtrowaniem i możliwością exportu do CSV
- **Filtrowanie wyników** - Możliwość filtrowania podobieństw w tym samym artykule vs między artykułami

## Instalacja

```bash
# Instalacja zależności
pip install -r requirements.txt
```

## Uruchomienie

```bash
# Z głównego katalogu projektu
python -m streamlit run analyze_similarity.py

# Lub bezpośrednio
streamlit run analyze_similarity.py
```

Aplikacja uruchomi się na `http://localhost:8501`

## Struktura projektu

```
brief-nexus-analyze-similarity/
├── analyze_similarity.py   # Główna aplikacja Streamlit
├── requirements.txt        # Zależności Python
├── README.md              # Ten plik
└── briefy/                # Folder z plikami markdown
    ├── domena.pl - Brief - Tytuł....md
    ├── domena.pl - Brief - Tytuł.....md
    └── domena.pl - Brief - Tytuł.....md
```

## Format plików markdown

Pliki w folderze `./briefy/` powinny mieć strukturę:

```markdown
## 1\. Nagłówek pierwszej sekcji

**Wiedza:**
Treść wiedzy dla pierwszej sekcji...

**Keywords:**
słowo1, słowo2, słowo3

## 2\. Nagłówek drugiej sekcji

**Wiedza:**
Treść wiedzy dla drugiej sekcji...

**Keywords:**
słowo4, słowo5, słowo6
```

**Uwagi:**
- Nagłówki mogą mieć różne poziomy: `##`, `###`, `####`
- Numeracja sekcji: `## 1\.`, `### 2\.1`, etc.
- Parser obsługuje zarówno `**Keywords:**` jak i `**Słowa kluczowe:**`

## Użycie

1. Umieść pliki markdown w folderze `./briefy/`
2. Uruchom aplikację
3. W sidebarze:
   - Ustaw ścieżkę do folderu (domyślnie `./briefy`)
   - Wybierz próg podobieństwa (domyślnie 0.60)
   - Zaznacz opcje filtrowania
4. Kliknij **🚀 URUCHOM ANALIZĘ**
5. Przejrzyj wyniki i wyeksportuj do CSV

## Technologie

- **Streamlit** - Framework UI
- **sentence-transformers** - Model embeddingowy `paraphrase-multilingual-MiniLM-L12-v2`
- **scikit-learn** - Obliczanie cosine similarity
- **pandas** - Przetwarzanie danych i export CSV

## Rozwiązane problemy

Parser został zaktualizowany, aby obsługiwał:
- Backslash-escaped kropki w nagłówkach markdown (`## 1\.`)
- Różne poziomy nagłówków (`##`, `###`, `####`)
- Oba warianty keywords: `**Keywords:**` i `**Słowa kluczowe:**`
- Dodatkowe spacje po nagłówkach sekcji

## Wydajność

- Embedding ~46 sekcji: ~15-25 sekund
- Similarity matrix 46x46: ~1 sekunda
- **Total: < 30 sekund na pełną analizę**
