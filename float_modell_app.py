import os
import streamlit as st
import google.generativeai as genai
from datetime import datetime

##############################
# Helper functions
##############################

def _configure_api(api_key: str | None):
    """Configure Gemini API with given key or env var.
    Raises a friendly Streamlit error if missing.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Gemini API anahtarı bulunamadı. Sidebar'dan veya GEMINI_API_KEY env değişkeninden verin.")
        st.stop()
    genai.configure(api_key=api_key)


def _pick_supported_model(preferred: str = "gemini-1.5-pro") -> str:
    """Return a model name that exists *and* supports generateContent.
    Falls back to gemini-1.5-flash or the first suitable model.
    """
    try:
        models = list(genai.list_models())
    except Exception as e:
        # If listing fails (rare), assume preferred is fine and let the request handle it
        st.warning(f"Model listesi alınamadı: {e}")
        return preferred

    def supports_text(m):
        methods = getattr(m, "supported_generation_methods", []) or []
        return any("generateContent" in x or "generate_content" in x for x in methods)

    names = [m.name for m in models if supports_text(m)]

    # Exact preferred available
    if preferred in names:
        return preferred

    # Reasonable fallbacks
    for cand in ("gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"):
        if cand in names:
            st.info(f"'{preferred}' bulunamadı. '{cand}' modeline düşüldü.")
            return cand

    # Any other suitable
    if names:
        st.info(f"'{preferred}' bulunamadı. '{names[0]}' modeline düşüldü.")
        return names[0]

    # If we really couldn't find anything, keep preferred; a later error will explain
    return preferred


def generate_plan(model_name: str, persona: str, goal: str, tone: str, language: str) -> str:
    """Call Gemini and return the generated action plan text."""
    # Ensure the model actually exists (or pick a fallback)
    model_name = _pick_supported_model(model_name)
    model = genai.GenerativeModel(model_name)

    system_msg = (
        "Sen üst düzey bir ürün ve büyüme danışmanısın. Kısa, uygulanabilir, net maddeler yaz."
    )

    prompt = f"""
SİSTEM: {system_msg}

KİŞİ:
{persona}

HEDEF:
{goal}

İSTENEN TON: {tone}
DİL: {language}

LÜTFEN çıktıyı aşağıdaki başlıklarla ver:
1) Özet (3-4 cümle)
2) 30-60-90 Gün Eylem Planı (madde madde)
3) Riskler ve Karşı Önlemler
4) Ölçülecek KPI'lar
5) Hemen Şimdi (ilk 3 adım)
"""

    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
    except Exception as e:
        # Handle common 404 model errors explicitly and offer guidance
        msg = str(e)
        if "404" in msg and ("not found" in msg.lower() or "not supported" in msg.lower()):
            st.error(
                "Seçilen model bu API sürümünde bulunamadı veya desteklenmiyor. Lütfen 'Model' alanından başka bir model seçin (örn. gemini-1.5-pro / gemini-1.5-flash)."
            )
        raise


##############################
# Streamlit App
##############################

st.set_page_config(page_title="Copilot (Gemini) Analizi", page_icon="🤖", layout="wide")

st.title("🤖 Copilot (Gemini) Analizi")

# Sidebar – API key & model
with st.sidebar:
    st.subheader("Ayarlar")
    api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    _configure_api(api_key)

    # Build model options dynamically when possible
    model_default = "gemini-1.5-pro"
    model_options = [model_default, "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
    try:
        available = [m.name for m in genai.list_models()]
        for n in available:
            if n not in model_options:
                model_options.append(n)
    except Exception:
        pass

    selected_model = st.selectbox("Model", options=model_options, index=0)

    if st.button("Model Desteklerini Göster"):
        try:
            rows = []
            for m in genai.list_models():
                methods = ", ".join(m.supported_generation_methods or [])
                rows.append(f"{m.name}  →  {methods}")
            st.code("\n".join(rows))
        except Exception as e:
            st.warning(f"Listeleme hatası: {e}")

# Inputs
col1, col2 = st.columns([1,1])
with col1:
    persona = st.text_area(
        "Kişi/Profil (ör. Arda Turan – rol, güçlü yönler, zorluklar)",
        value="Arda Turan – Pazarlama Direktörü; güçlü yön: iletişim, marka; zorluk: veri odaklı büyüme",
        height=160,
    )
    tone = st.selectbox("Ton", ["Resmi", "Samimi", "Nokta Atışı"], index=2)

with col2:
    goal = st.text_area(
        "Hedef (örn. 90 günde %20 MRR artışı)",
        value="90 gün içinde inbound MQL'lerde %30 artış ve deneme→ücretli dönüşümde +3 puan",
        height=160,
    )
    language = st.selectbox("Dil", ["Türkçe", "English"], index=0)

# Generate
if st.button("Eylem Planı Oluştur", type="primary"):
    with st.spinner("Gemini ile oluşturuluyor…"):
        try:
            text = generate_plan(selected_model, persona, goal, tone, language)
        except Exception as e:
            st.exception(e)
        else:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success(f"Hazır! ({ts})")
            st.markdown(text)

st.caption("Not: Model adı 404 verirse, model açılır listesinden başka bir sürüm seçin. Bu uygulama, 1.5 ailesiyle uyumlu olacak şekilde güncellendi.")
