import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Grafik kütüphanesi
import io # Excel/CSV işlemleri için
import joblib # MODELİ YÜKLEMEK İÇİN YENİ KÜTÜPHANE

# ---------------------------------------------------------------------
# 0. SAYFA YAPILANDIRMASI VE BAŞLIK
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Opet Pay 'Akıllı' Dashboard")

st.title("Opet Pay 'Akıllı Strateji' Dashboardu 🚀")
st.markdown("Bu dashboard, net kârlılığı analiz eder, **gerçek ML modeliyle** churn riskini tahmin eder ve müşteri arayüzünü simüle eder.")

# ---------------------------------------------------------------------
# 1. ÇEKİRDEK HESAPLAMA MANTIĞI (Net Kâr)
# ---------------------------------------------------------------------

@st.cache_data
def calculate_net_profitability(
    musteri_sayisi, aylik_yukleme, bakiye_tutma_suresi, 
    faiz_orani, cashback_payi_yuzde, 
    islem_maliyeti_yuzde, op_maliyet_tl
):
    """ Net kârlılığı hesaplar: (Gelirler) - (Tüm Giderler) """
    bakiye_tutma_suresi = min(bakiye_tutma_suresi, 30)
    gunluk_ortalama_bakiye = (aylik_yukleme * bakiye_tutma_suresi) / 30
    toplam_yatirilabilir_float = gunluk_ortalama_bakiye * musteri_sayisi
    
    toplam_aylik_brut_gelir = (toplam_yatirilabilir_float * (faiz_orani / 100)) / 12
    
    toplam_aylik_yukleme = aylik_yukleme * musteri_sayisi
    toplam_islem_maliyeti = toplam_aylik_yukleme * (islem_maliyeti_yuzde / 100)
    toplam_op_maliyeti = op_maliyet_tl * musteri_sayisi
    toplam_cashback_maliyeti = toplam_aylik_brut_gelir * (cashback_payi_yuzde / 100)
    
    toplam_net_kar = toplam_aylik_brut_gelir - toplam_islem_maliyeti - toplam_op_maliyeti - toplam_cashback_maliyeti
    
    return {
        "toplam_aylik_brut_gelir": toplam_aylik_brut_gelir,
        "toplam_islem_maliyeti": toplam_islem_maliyeti,
        "toplam_op_maliyeti": toplam_op_maliyeti,
        "toplam_cashback_maliyeti": toplam_cashback_maliyeti,
        "toplam_net_kar": toplam_net_kar
    }

# ---------------------------------------------------------------------
# 2. GERÇEK ML MODELİNİ YÜKLEME
# ---------------------------------------------------------------------

@st.cache_resource # Modeli hafızada tutmak için _resource kullanılır
def load_model():
    """ 'churn_model.pkl' ve 'model_columns.pkl' dosyalarını yükler. """
    try:
        model = joblib.load("churn_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        print("ML Modeli ve Kolonları başarıyla yüklendi.")
        return model, model_columns
    except FileNotFoundError:
        st.error("HATA: 'churn_model.pkl' veya 'model_columns.pkl' dosyaları bulunamadı!")
        st.warning("Lütfen 'churn_model.pkl' ve 'model_columns.pkl' dosyalarınızın GitHub deponuzda olduğundan emin olun.")
        return None, None
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None, None

model, model_columns = load_model()

# ---------------------------------------------------------------------
# 3. STREAMLIT ARAYÜZÜ - SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.header("Genel Simülasyon Parametreleri ⚙️")
st.sidebar.caption("Buradaki ayarlar, 'Hipotetik' analiz sekmelerinin temelini oluşturur.")
musteri_sayisi = st.sidebar.slider("Toplam Müşteri Sayısı (Hipotetik)", 1000, 1000000, 50000, step=1000, format="%d kullanıcı")
aylik_yukleme = st.sidebar.slider("Ort. Aylık Yükleme (Hipotetik)", 500, 10000, 3000, step=100, format="%d TL")
bakiye_tutma_suresi = st.sidebar.slider("Ort. Bakiye Tutma Süresi (Hipotetik)", 1, 30, 10)

st.sidebar.header("Finansal Model Ayarları 💰")
st.sidebar.caption("Bu ayarlar TÜM hesaplamaları (yüklenen veri dahil) etkiler.")
faiz_orani = st.sidebar.slider("Yıllık Mevduat/Fon Getirisi (%)", 5.0, 50.0, 35.0, step=0.5)
cashback_payi_yuzde = st.sidebar.slider("Müşteriye Verilecek Ortalama Cashback Oranı (%)", 0, 100, 50)

st.sidebar.header("Maliyet Girdileri (Net Kâr için) 💸")
islem_maliyeti_yuzde = st.sidebar.slider("İşlem Maliyeti (%)", 0.0, 5.0, 2.5, step=0.1)
op_maliyet_tl = st.sidebar.slider("Müşteri Başı Aylık Operasyonel Maliyet (TL)", 0.0, 10.0, 1.0, step=0.5)

# ---------------------------------------------------------------------
# 4. ANA HESAPLAMALAR VE DASHBOARD SEKMELERİ
# ---------------------------------------------------------------------

results = calculate_net_profitability(
    musteri_sayisi, aylik_yukleme, bakiye_tutma_suresi, 
    faiz_orani, cashback_payi_yuzde, 
    islem_maliyeti_yuzde, op_maliyet_tl
)

if 'df_loaded' not in st.session_state:
    st.session_state['df_loaded'] = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ana Dashboard (Hipotetik) 📈", 
    "Net Kârlılık Dağılımı 📊", 
    "Hipotetik Segmentasyon 🎯",
    "Veri Yükle & Akıllı Analiz 🧠",
    "Müşteri Simülasyonu 📱"
])

# ----------------------------------
# TAB 1, 2, 3 - Değişiklik Yok
# ----------------------------------
with tab1:
    st.header("Genel Proje Kârlılığı (Hipotetik / Aylık)")
    st.info(f"Bu hesaplamalar, soldaki ayarlara göre **{musteri_sayisi:,}** adet 'ortalama' müşteriye ve maliyetlere dayanmaktadır.")
    col1, col2 = st.columns(2); col3, col4 = st.columns(2)
    with col1: st.metric("💸 Toplam Aylık Brüt Gelir (Faizden)", f"{results['toplam_aylik_brut_gelir']:,.0f} TL")
    with col2: st.metric("🏦 Opet'e Kalan Aylık NET KÂR", f"{results['toplam_net_kar']:,.0f} TL")
    st.divider(); st.subheader("Aylık Maliyet Dağılımı (Giderler)"); col_cb, col_islem, col_op = st.columns(3)
    with col_cb: st.metric("🎁 Müşteri Cashback Maliyeti", f"{results['toplam_cashback_maliyeti']:,.0f} TL", delta_color="inverse")
    with col_islem: st.metric("💳 İşlem Maliyeti (Yükleme)", f"{results['toplam_islem_maliyeti']:,.0f} TL", delta_color="inverse")
    with col_op: st.metric("⚙️ Operasyonel Maliyet (Sabit)", f"{results['toplam_op_maliyeti']:,.0f} TL", delta_color="inverse")
    st.divider(); st.header("🤖 Proje Asistanı Yorumu (Hipotetik)");
    with st.container(border=True):
        if results['toplam_net_kar'] > 0:
            st.success(f"**Proje Sağlığı: POZİTİF**\nMevcut ayarlarla, proje ayda **{results['toplam_net_kar']:,.0f} TL Net Kâr** üretiyor.")
        else:
            st.error(f"**Proje Sağlığı: NEGATİF**\nMevcut ayarlarla, proje ayda **{results['toplam_net_kar']:,.0f} TL Net ZARAR** üretiyor. Kâra geçmek için maliyetleri düşürün veya 'bakiye tutma süresini' artırın.")

with tab2:
    st.header("Görsel Net Kârlılık Dağılımı"); col1_chart, col2_chart = st.columns(2)
    with col1_chart:
        st.subheader("Aylık Brüt Gelir Dağılımı")
        if results['toplam_aylik_brut_gelir'] > 0:
            labels = ["Opet'e Kalan (Net Kâr)", "Müşteriye Giden (Cashback)", "Bankaya Giden (İşlem Maliyeti)", "Giderler (Operasyonel)"]
            net_kar_size = max(0, results['toplam_net_kar']); diger_maliyetler_toplami = results['toplam_cashback_maliyeti'] + results['toplam_islem_maliyeti'] + results['toplam_op_maliyeti']
            if results['toplam_net_kar'] < 0:
                 labels = ["Maliyetler (Geliri Aştı)"]; sizes = [diger_maliyetler_toplami]; colors = ['#FF4B4B']
            else:
                 sizes = [net_kar_size, results['toplam_cashback_maliyeti'], results['toplam_islem_maliyeti'], results['toplam_op_maliyeti']]; colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
            fig, ax = plt.subplots(); fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
            wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=colors, wedgeprops={'width':0.4})
            for text in autotexts: text.set_color('white')
            total_profit_text = f"Brüt Gelir\n{results['toplam_aylik_brut_gelir']:,.0f} TL"; ax.text(0, 0, total_profit_text, ha='center', va='center', fontsize=12, color='white')
            legend = ax.legend(wedges, labels, title="Gelir Dağılımı", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), facecolor='#222') 
            plt.setp(legend.get_title(), color='white'); plt.setp(legend.get_texts(), color='white'); ax.axis('equal')  
            st.pyplot(fig, use_container_width=True)
        else: st.warning("Kârlılık için lütfen parametreleri ayarlayın.")
    scenario_data = []
    with col2_chart:
        st.subheader("Net Kârın 'Tutma Süresine' Göre Değişimi")
        for gun in [1, 5, 10, 15, 20, 25, 30]:
            res = calculate_net_profitability(musteri_sayisi, aylik_yukleme, gun, faiz_orani, cashback_payi_yuzde, islem_maliyeti_yuzde, op_maliyet_tl)
            scenario_data.append({ "gun": gun, "label": f"{gun} Gün", "value": res['toplam_net_kar'] })
        scenario_df = pd.DataFrame(scenario_data); fig, ax = plt.subplots(); fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        bar_colors = ['#FF4B4B' if v < 0 else '#1f77b4' for v in scenario_df["value"]]; bars = ax.bar(scenario_df["label"], scenario_df["value"], color=bar_colors)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white'); ax.set_ylabel("Aylık NET Kâr (TL)", color='white')
        for bar in bars: yval = bar.get_height(); ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,.0f}', va='bottom', ha='center', color='white', fontsize=9)
        st.pyplot(fig, use_container_width=True)
    st.divider(); st.header("🤖 Finansal Analist Asistanı Yorumu");
    with st.container(border=True):
        st.subheader("Gelir Dağılımı Analizi")
        if results['toplam_aylik_brut_gelir'] > 0:
            net_kar_yuzde = (results['toplam_net_kar'] / results['toplam_aylik_brut_gelir']) * 100; islem_maliyet_yuzde = (results['toplam_islem_maliyeti'] / results['toplam_aylik_brut_gelir']) * 100
            if net_kar_yuzde > 20: st.success(f"**Net Kâr Marjı: %{net_kar_yuzde:.1f} (Güçlü)**\nFaizden gelen her 100 TL'nin {net_kar_yuzde:.1f} TL'si Opet'e net kâr olarak kalıyor.")
            elif net_kar_yuzde > 0: st.warning(f"**Net Kâr Marjı: %{net_kar_yuzde:.1f} (Zayıf)**\nProje kârlı, ancak net kâr marjı düşük. En büyük maliyet kalemi %{islem_maliyet_yuzde:.1f} ile 'İşlem Maliyeti' gibi görünüyor.")
            else: st.error(f"**Net Kâr Marjı: %{net_kar_yuzde:.1f} (Negatif)**\nModel şu an zarar ediyor. Maliyetler (özellikle 'İşlem Maliyeti': %{islem_maliyet_yuzde:.1f}), faizden gelen brüt gelirden daha yüksek.")
        else: st.info("Brüt Gelir 0 olduğu için marj hesaplanamıyor.")
        st.subheader("Başa Baş (Break-Even) Analizi"); basa_bas_gunu = None
        for item in scenario_data:
            if item['value'] > 0: basa_bas_gunu = item['gun']; break
        if basa_bas_gunu: st.success(f"**Başa Baş Noktası: {basa_bas_gunu} Gün**\nBu ayarlarla, 'ortalama' bir müşterinin bize net kâr getirmeye başlaması için, parasını sistemde en az **{basa_bas_gunu} gün** tutması gerekiyor.")
        else: st.error("**Başa Baş Noktası BULUNAMADI**\nMevcut maliyet yapısıyla, müşteri parasını 30 gün tutsa bile bu model net kâr üretemiyor.")

with tab3:
    st.header("Hipotetik Segmentasyon & Kampanya Motoru 🎯")
    st.info("Bu bölüm, 'ortalama' müşteri verisine dayalı 5 varsayımsal segmenti *net kârlılık* bazında analiz eder.")
    st.subheader("Müşteri Segmentasyonu ve NET Kârlılık (Kullanıcı Başına)"); cols = st.columns(5)
    segments = {"Kayıp": {"yukleme": aylik_yukleme * 0.5, "sure": 2, "cb_stratejisi_yuzde": 0, "emoji": "💔"},"Geçici": {"yukleme": aylik_yukleme * 2.0, "sure": 3, "cb_stratejisi_yuzde": 40, "emoji": "💨"},"Standart": {"yukleme": aylik_yukleme, "sure": bakiye_tutma_suresi, "cb_stratejisi_yuzde": cashback_payi_yuzde, "emoji": "👤"},"Sadık": {"yukleme": aylik_yukleme * 0.8, "sure": 25, "cb_stratejisi_yuzde": 60, "emoji": "💖"},"Altın": {"yukleme": aylik_yukleme * 2.5, "sure": 28, "cb_stratejisi_yuzde": 75, "emoji": "🌟"}}
    segment_results_net = {}
    for i, (segment_name, params) in enumerate(segments.items()):
        with cols[i]:
            st.markdown(f"#### {params['emoji']} {segment_name}")
            res = calculate_net_profitability(1, params['yukleme'], params['sure'], faiz_orani, params['cb_stratejisi_yuzde'], islem_maliyeti_yuzde, op_maliyet_tl)
            segment_results_net[segment_name] = {"brut_gelir": res['toplam_aylik_brut_gelir'], "net_kar": res['toplam_net_kar'], "maliyet_islem": res['toplam_islem_maliyeti'], "maliyet_cb": res['toplam_cashback_maliyeti']}
            st.metric("Müşteri Başı Aylık NET KÂR", f"{res['toplam_net_kar']:,.2f} TL")
            st.metric("Müşteri Başı Brüt Gelir (Faiz)", f"{res['toplam_aylik_brut_gelir']:,.2f} TL")
            st.metric("Ort. Aylık Yükleme", f"{params['yukleme']:,.0f} TL")
            st.metric("Ort. Bakiye Tutma Süresi", f"{params['sure']} Gün")
            st.metric(f"İşlem Maliyeti (%{islem_maliyeti_yuzde})", f"{res['toplam_islem_maliyeti']:,.2f} TL")
            
    st.divider(); st.header("🤖 Detaylı Kampanya Asistanı (Hipotetik / Maliyet-Odaklı)");
    try:
        with st.container(border=True):
            st.subheader("Strateji 1: 'Geçici' 💨 Müşteriyi Dönüştürme"); gecici_net_kar = segment_results_net['Geçici']['net_kar']; gecici_islem_maliyet = segment_results_net['Geçici']['maliyet_islem']
            if gecici_net_kar < 0: st.error(f"**KRİTİK ANALİZ:** 'Geçici' segment şu anda **net {gecici_net_kar:,.2f} TL ZARAR** üretiyor. **Neden?** Yüksek işlem maliyeti ({gecici_islem_maliyet:,.2f} TL), düşük faiz gelirinden fazla. **Strateji:** Bu segmente ASLA cashback vermeyin. Tek hedef, 'bakiye tutma süresini' uzatmaktır.")
            else: st.warning("Geçici segment şu an kârlı, ancak işlem maliyetlerine dikkat edilmeli.")
        with st.container(border=True):
            st.subheader("Strateji 2: 'Sadık' 💖 Müşteriyi Büyütme"); sadik_net_kar = segment_results_net['Sadık']['net_kar']; sadik_islem_maliyet = segment_results_net['Sadık']['maliyet_islem']
            st.info(f"**ANALİZ:** 'Sadık' segment **net {sadik_net_kar:,.2f} TL KÂR** üretiyor. **Güçlü Yön:** İşlem maliyetleri ({sadik_islem_maliyet:,.2f} TL) çok düşük, faiz geliri harika. **Strateji:** Bu segmentin 'aylık yükleme tutarını' artırmalıyız.")
    except Exception as e: st.error(f"Hipotetik asistan yüklenirken bir hata oluştu: {e}")

# ----------------------------------
# TAB 4: Veri Yükle & Akıllı Analiz (KEYERROR İÇİN DÜZELTİLDİ)
# ----------------------------------
with tab4:
    st.header("Veri Yükle & Akıllı Segmentasyon (Gerçek ML Modeli) 🧠")
    st.info("Kendi müşteri verinizi yükleyerek *net kârlılık* ve *gerçek ML modeliyle* churn (terk) riski analizi yapın. Soldaki TÜM faiz ve maliyet ayarları bu hesaplama için kullanılacaktır.")

    st.subheader("1. Adım: Şablonu İndirin")
    sample_data = {
        'musteri_id': ['M-1001', 'M-1002', 'M-1003'],
        'ad_soyad': ['Ali Veli (Riskli)', 'Ayşe Yılmaz (Sadık)', 'Mehmet Öztürk (Zarar)'],
        'ortalama_aylik_yukleme_tl': [8000, 2000, 15000],
        'ortalama_bakiye_tutma_suresi_gun': [25, 28, 3],
        'aylik_yukleme_sikligi': [2, 1, 4],
        'aylik_harcama_sikligi': [5, 2, 8],
        'son_islem_uzerinden_gecen_gun': [35, 2, 1], 
        'harcama_trendi_yuzde': [-25, 10, 5]
    }
    df_sample = pd.DataFrame(sample_data)
    
    @st.cache_data
    def to_excel_v2(df): 
        output = io.BytesIO();
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Musteri_Verisi')
        return output.getvalue()

    excel_data = to_excel_v2(df_sample)
    st.download_button(label="📥 Yeni Akıllı Şablonu İndir (.xlsx)", data=excel_data, file_name='opet_pay_akilli_sablon.xlsx')
    
    st.subheader("2. Adım: Veri Dosyasını Yükleyin")
    uploaded_file = st.file_uploader("Doldurduğunuz yeni şablonu (Excel/CSV) buraya yükleyin:", type=["xlsx", "csv"], key="file_uploader")

    st.subheader("3. Adım: Dinamik Net Kârlılık ve Churn Analizi")
    
    # Modelin yüklenip yüklenmediğini kontrol et
    if model is None or model_columns is None:
        st.error("ML Modeli yüklenemedi. Lütfen 'churn_model.pkl' ve 'model_columns.pkl' dosyalarının ana kodla aynı klasörde olduğundan ve GitHub'a yüklendiğinden emin olun.")
    
    elif uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # ----- DÜZELTME BURADA BAŞLIYOR (KeyError Kontrolü) -----
            
            # 1. Modelin ihtiyaç duyduğu kolonlar (model_columns) ile yüklenen dosyanın kolonlarını (df.columns) karşılaştır
            required_cols_set = set(model_columns)
            uploaded_cols_set = set(df.columns)
            
            # Eksik kolonları bul
            missing_cols = required_cols_set - uploaded_cols_set
            
            # ML için Gerekli Kolonlar
            ml_ready = not bool(missing_cols) # Eğer eksik kolon yoksa (True)
            
            # Finansal Analiz için Gerekli Kolonlar
            financial_cols_ok = 'ortalama_aylik_yukleme_tl' in df.columns and 'ortalama_bakiye_tutma_suresi_gun' in df.columns
            
            # ----- DÜZELTME BURADA BİTİYOR -----
            
            if not financial_cols_ok:
                st.error("HATA: Yüklediğiniz dosyada 'ortalama_aylik_yukleme_tl' ve 'ortalama_bakiye_tutma_suresi_gun' kolonları bulunamadı. Temel analiz yapılamıyor.")
                if 'df_loaded' in st.session_state: del st.session_state['df_loaded']
            
            else:
                # Finansal kolonlar TAMAM, en azından Net Kâr analizi yapabiliriz
                g_faiz_orani = faiz_orani; g_islem_maliyeti_yuzde = islem_maliyeti_yuzde; g_op_maliyet_tl = op_maliyet_tl
                def calculate_customer_net_profit(row):
                    res = calculate_net_profitability(1, row['ortalama_aylik_yukleme_tl'], row['ortalama_bakiye_tutma_suresi_gun'], g_faiz_orani, 0, g_islem_maliyeti_yuzde, g_op_maliyet_tl)
                    return res['toplam_aylik_brut_gelir'], res['toplam_net_kar'], res['toplam_islem_maliyeti']
                df[['Aylık Brüt Gelir (Faiz)', 'Aylık NET Kâr (CB Hariç)', 'Aylık İşlem Maliyeti']] = df.apply(calculate_customer_net_profit, axis=1, result_type='expand')
                
                df.loc[df['Aylık NET Kâr (CB Hariç)'] <= 0, 'Segment'] = 'Kayıp (Zarar)'
                karlilar = df[df['Aylık NET Kâr (CB Hariç)'] > 0]
                if not karlilar.empty:
                    try:
                        karlilar['Segment'] = pd.qcut(karlilar['Aylık NET Kâr (CB Hariç)'], 4, labels=['Bronz', 'Gümüş', 'Altın', 'Platin'], duplicates='drop')
                        df.update(karlilar)
                    except ValueError: karlilar['Segment'] = 'Altın'; df.update(karlilar)

                # ----- ML MODELİNİ ÇALIŞTIRMA (Sadece Mümkünse) -----
                if ml_ready:
                    # ML Modeli için gerekli tüm kolonlar var
                    df_for_model = df[model_columns].fillna(0)
                    churn_probabilities = model.predict_proba(df_for_model)[:, 1]
                    df['Churn Riski (%)'] = (churn_probabilities * 100).round(0)
                    
                    def set_risk_level(row):
                        score = row['Churn Riski (%)']; segment = row['Segment']
                        seviye = "Düşük"
                        if score > 75: seviye = "KRİTİK"
                        elif score > 50: seviye = "Yüksek"
                        elif score > 20: seviye = "Orta"
                        if seviye in ["Yüksek", "Orta"] and segment in ['Platin', 'Altın']:
                            seviye = "KRİTİK"
                        return seviye
                    df['Risk Seviyesi'] = df.apply(set_risk_level, axis=1)
                    st.success(f"{len(df)} adet müşteri verisi başarıyla işlendi ve GERÇEK ML MODELİ ile churn tahmini tamamlandı!")
                
                else:
                    # ML Modeli için kolonlar EKSİK
                    st.warning(f"ML Modeli için gerekli kolonlar bulunamadı: {', '.join(missing_cols)}. Churn tahmini (Risk Seviyesi) atlanıyor.")
                    df['Churn Riski (%)'] = 0
                    df['Risk Seviyesi'] = 'Veri Eksik'
                    st.success(f"{len(df)} adet müşteri verisi için Net Kâr analizi tamamlandı (ML tahmini atlandı).")

                # ----- VERİYİ HAFIZAYA KAYDET -----
                st.session_state['df_loaded'] = df

                # ----- SONUÇLARI GÖSTER -----
                st.header("🚨 Acil Eylem Raporu (Churn Riski)");
                if ml_ready:
                    churn_summary = df.groupby('Risk Seviyesi')['Aylık NET Kâr (CB Hariç)'].agg(['count', 'sum']).rename(columns={'count': 'Müşteri Sayısı', 'sum': 'Risk Altındaki NET Kâr (Aylık)'})
                    st.dataframe(churn_summary.style.format({'Müşteri Sayısı': '{:,.0f}', 'Risk Altındaki NET Kâr (Aylık)': '{:,.2f} TL'}))
                else:
                    st.info("ML tahmini yapılmadığı için Churn Raporu oluşturulamadı. Lütfen 8 kolonlu 'Akıllı Şablonu' yükleyin.")
                
                st.header("🤖 Akıllı Kampanya Asistanı (Veriye Dayalı)"); 
                if ml_ready:
                    df_kritik = df[df['Risk Seviyesi'] == 'KRİTİK'].sort_values(by='Aylık NET Kâr (CB Hariç)', ascending=False)
                    if not df_kritik.empty:
                        st.error(f"**ACİL EYLEM GEREKİYOR!** {len(df_kritik)} adet YÜKSEK DEĞERLİ ve 'KRİTİK' riskli müşteri tespit edildi.")
                        with st.container(border=True):
                            for index, musteri in df_kritik.head(3).iterrows(): 
                                musteri_adi = musteri.get('ad_soyad', musteri['musteri_id']); st.warning(f"**Müşteri: {musteri_adi} (Segment: {musteri['Segment']})**")
                                st.markdown(f"  - **Model Tahmini:** %{musteri['Churn Riski (%)']:.0f} Terk Etme Riski.")
                                st.markdown(f"  - **Kaybedilmekte Olan Kâr:** Aylık **{musteri['Aylık NET Kâr (CB Hariç)']:,.2f} TL**.")
                    else: st.success("Harika! 'KRİTİK' seviyede risk taşıyan yüksek değerli müşteriniz bulunmuyor.")
                else:
                    st.info("ML tahmini yapılmadığı için Akıllı Asistan önerileri sınırlıdır.")

                st.header("📝 Detaylı Müşteri Listesi (Net Kâr, Segment ve Risk)")
                st.dataframe(df.sort_values(by='Churn Riski (%)', ascending=False), use_container_width=True)
        
        except KeyError as e:
            st.error(f"HATA: Yüklediğiniz dosyada '{e}' kolonu bulunamadı. Lütfen 'Akıllı Şablon' formatını kullandığınızdan emin olun.")
            if 'df_loaded' in st.session_state: del st.session_state['df_loaded']
        except Exception as e:
            st.error(f"Dosya okunurken veya ML modeli çalışırken bir hata oluştu: {e}")
            if 'df_loaded' in st.session_state: del st.session_state['df_loaded']

# ----------------------------------
# TAB 5: Müşteri Simülasyonu 📱 (ATTRIBUTEERROR İÇİN DÜZELTİLDİ)
# ----------------------------------
with tab5:
    st.header("Müşteri Arayüzü Simülasyonu 📱")
    st.info("Bu simülasyon, 'Akıllı Analiz' sekmesinde yüklediğiniz ve ML modeli tarafından skorlanmış veriyi kullanır.")

    if st.session_state.get('df_loaded') is None:
        st.warning("Simülasyonu başlatmak için lütfen önce 'Veri Yükle & Akıllı Analiz 🧠' sekmesinden bir müşteri veri dosyası yükleyin.")
    else:
        df_loaded = st.session_state['df_loaded']
        
        display_column = 'ad_soyad' if 'ad_soyad' in df_loaded.columns else 'musteri_id'
        customer_list = df_loaded[display_column].tolist()
        
        selected_customer_name = st.selectbox("Simülasyon için bir müşteri seçin:", customer_list, index=None, placeholder="Bir müşteri seçin...")

        if selected_customer_name:
            
            # ----- DÜZELTME BURADA BAŞLIYOR (AttributeError Kontrolü) -----
            customer_data = None
            try:
                filtered_df = df_loaded[df_loaded[display_column] == selected_customer_name]
                if not filtered_df.empty:
                    customer_data = filtered_df.iloc[0] 
                else:
                    st.error(f"HATA: '{selected_customer_name}' adlı müşteri için veri bulunamadı.")
            except Exception as e:
                st.error(f"Müşteri verisi alınırken beklenmedik bir hata oluştu: {e}")
            
            # ----- DÜZELTME BURADA BİTİYOR -----

            if customer_data is not None:
                segment = customer_data.get('Segment', 'Kayıp (Zarar)'); brut_gelir = customer_data.get('Aylık Brüt Gelir (Faiz)', 0)
                segment_cb_map = {'Platin': 0.75, 'Altın': 0.60, 'Gümüş': 0.40, 'Bronz': 0.20, 'Kayıp (Zarar)': 0.0}
                cb_orani = segment_cb_map.get(segment, 0.0); tahmini_kazanc_tl = brut_gelir * cb_orani
                
                st.markdown("---"); col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with st.container(border=True):
                        st.markdown(f"<h3 style='text-align: center;'>Opet Pay</h3>", unsafe_allow_html=True); st.markdown(f"Merhaba, **{selected_customer_name}**"); st.divider()
                        st.markdown(f"Mevcut Segmentiniz")
                        if segment == "Platin": st.markdown(f"<h2 style='color: #8A2BE2;'>Platin 🌟</h2>", unsafe_allow_html=True)
                        elif segment == "Altın": st.markdown(f"<h2 style='color: #FFD700;'>Altın 🥇</h2>", unsafe_allow_html=True)
                        elif segment == "Gümüş": st.markdown(f"<h2 style='color: #C0C0C0;'>Gümüş 🥈</h2>", unsafe_allow_html=True)
                        elif segment == "Bronz": st.markdown(f"<h2 style='color: #CD7F32;'>Bronz 🥉</h2>", unsafe_allow_html=True)
                        else: st.markdown(f"<h2>Kayıp (Zarar) 💔</h2>", unsafe_allow_html=True)
                        st.markdown(f"Bu Ayki Tahmini Kazancınız (Cashback)"); st.markdown(f"<h1 style='color: #2ca02c;'>{tahmini_kazanc_tl:,.2f} TL 💸</h1>", unsafe_allow_html=True)
                        st.caption(f"Bu kazanç, {brut_gelir:,.2f} TL'lik faiz geliriniz üzerinden {cb_orani:.0%} oranında hesaplanan payınızdır.")
                        st.divider(); st.subheader("Akıllı Asistanınız Diyor ki:")
                        
                        risk_seviyesi = customer_data.get('Risk Seviyesi', 'Düşük'); churn_riski_yuzde = customer_data.get('Churn Riski (%)', 0)
                        
                        if risk_seviyesi == "KRİTİK":
                            kayip_kar = customer_data['Aylık NET Kâr (CB Hariç)']; bonus = max(50, kayip_kar * 0.5)
                            st.error(f"**Sizi Özledik!**\nML Modelimiz, %{churn_riski_yuzde:.0f} ihtimalle sizi kaybetmek üzere olduğumuzu tahmin ediyor. Lütfen geri dönün, size özel **{bonus:,.0f} TL**'lik yakıt puanı anında cüzdanınızda!")
                        elif risk_seviyesi == "Veri Eksik":
                             st.info("Davranışsal verileriniz (örn: son işlem tarihi) eksik olduğu için size özel bir risk analizi yapamıyoruz, ancak standart tekliflerimizden yararlanabilirsiniz.")
                        elif segment == "Kayıp (Zarar)":
                            yukleme = customer_data['ortalama_aylik_yukleme_tl']
                            st.warning(f"**Yeni Fırsat!**\nYüksek yükleme ({yukleme:,.0f} TL) yaptığınızı görüyoruz. Bu parayı 15 gün 'Kazandıran Bakiye' olarak ayırın, işlem ücreti maliyetinizin yarısını puan olarak iade edelim!")
                        elif segment == "Bronz":
                            st.info("**Daha Çok Kazanın!**\nBu ay yüklemenizi 5.000 TL'ye tamamlayın, 'Gümüş' segmente geçin ve cashback oranınızı ikiye katlayın!")
                        else: 
                            st.success(f"**Sadakatinizle Kazandırıyorsunuz!**\n{segment} segmentinde olduğunuz için teşekkür ederiz. Opet Pay'i kullandığınız sürece atıl bakiyeniz sizin için çalışmaya devam edecek.")

# ---------------------------------------------------------------------
# YASAL UYARI (Her zaman en altta)
# ---------------------------------------------------------------------
st.header("⚖️ Yasal Sorumluluk Reddi (Önemli)")
st.warning("""
**Bu uygulama yalnızca bir Proof of Concept (PoC) çalışmasıdır ve yatırım taviyesi değildir.**
Bahsi geçen 'faize yatırma', 'nemalandırma' ve 'mevdat' benzeri faaliyetler, Türkiye Cumhuriyeti'nde
**BDDK (Bankacılık Düzenleme ve Denetleme Kurumu)** ve **TCMB (Türkiye Cumhuriyet Merkez Bankası)** regülasyonlarına tabidir.
...
""")