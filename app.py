import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Deret Waktu Saham",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih baik
st.markdown("""
    <style>
    /* Styling untuk header utama */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    /* Styling untuk metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Styling untuk section headers */
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Styling untuk sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Styling untuk selectbox */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Styling untuk date input */
    .stDateInput > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Info box styling */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan mempersiapkan data
def load_data(ticker):
    # Ganti dengan path dataset Anda
    file_path = 'World-Stock-Prices-Dataset.csv'  # Pastikan path dataset benar
    df = pd.read_csv(file_path)

    # Mengubah kolom 'Date' menjadi datetime dan memastikan bahwa 'Date' tidak memiliki zona waktu
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

    # Memeriksa apakah ada nilai yang tidak terkonversi menjadi datetime (NaT)
    if df['Date'].isnull().any():
        st.error("Ada nilai yang tidak dapat dikonversi menjadi datetime di kolom 'Date'.")
        st.write(df[df['Date'].isnull()])  # Menampilkan baris yang gagal konversi
        return None

    # Memfilter data untuk hanya sesuai dengan ticker yang dipilih
    df_ticker = df[df['Ticker'] == ticker]

    # Memeriksa apakah data cukup untuk dekomposisi
    if len(df_ticker) < 2:
        st.error(f"Data yang tersedia untuk ticker {ticker} tidak cukup untuk analisis dekomposisi.")
        return None

    # Mengatur kolom 'Date' sebagai indeks dan memastikan indeksnya menjadi DatetimeIndex
    df_ticker.set_index('Date', inplace=True)

    # Memastikan bahwa indeks sudah dalam format DatetimeIndex
    if not isinstance(df_ticker.index, pd.DatetimeIndex):
        st.error("Indeks 'Date' tidak terkonversi menjadi DatetimeIndex dengan benar.")
        return None

    # Mengisi missing values dengan interpolasi linear
    df_ticker['Close'] = df_ticker['Close'].interpolate(method='linear')

    # Resampling data untuk rata-rata bulanan
    df_monthly = df_ticker['Close'].resample('M').mean()

    # Menetapkan frekuensi pada indeks setelah resampling
    df_monthly = df_monthly.asfreq('M')  # Menambahkan frekuensi bulanan

    # Memeriksa apakah data cukup untuk dekomposisi
    if len(df_monthly) < 2:
        st.error(f"Data yang tersedia untuk ticker {ticker} tidak cukup untuk analisis dekomposisi.")
        return None

    return df_monthly


# Fungsi untuk dekomposisi deret waktu
def decompose_data(df_monthly):
    try:
        decomposition = seasonal_decompose(df_monthly, model='additive')
        return decomposition
    except Exception as e:
        st.error(f"Terjadi kesalahan saat dekomposisi: {e}")
        return None

# Header utama dengan styling
st.markdown("""
    <div class="main-header">
        <h1>📈 Analisis Deret Waktu Saham</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">Platform Analisis Time Series untuk Prediksi Harga Saham</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar untuk input
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Analisis")
    st.markdown("---")
    
    # Input untuk memilih ticker saham
    ticker = st.selectbox(
        '📊 Pilih Ticker Saham',
        ['AAPL', 'AMZN', 'TSLA'],
        help="Pilih saham yang ingin dianalisis"
    )
    
    st.markdown("---")
    
    # Input untuk memilih rentang tanggal
    st.markdown("### 📅 Rentang Waktu")
    start_date = st.date_input(
        "Tanggal Mulai",
        pd.to_datetime('2018-01-01'),
        help="Pilih tanggal mulai analisis"
    )
    end_date = st.date_input(
        "Tanggal Akhir",
        pd.to_datetime('2022-12-31'),
        help="Pilih tanggal akhir analisis"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.info("Aplikasi ini menganalisis data harga saham menggunakan metode dekomposisi time series untuk mengidentifikasi trend, seasonality, dan residual.")

# Mengonversi start_date dan end_date menjadi objek datetime dengan waktu default
start_date_dt = datetime.combine(start_date, datetime.min.time())
end_date_dt = datetime.combine(end_date, datetime.min.time())

# Menambahkan zona waktu UTC pada start_date dan end_date
start_date_utc = pd.Timestamp(start_date_dt).tz_localize('UTC')
end_date_utc = pd.Timestamp(end_date_dt).tz_localize('UTC')

# Memuat data
df_monthly = load_data(ticker)

# Jika data berhasil dimuat
if df_monthly is not None:
    # Menyaring data sesuai dengan rentang tanggal yang dipilih
    df_filtered = df_monthly.loc[start_date_utc:end_date_utc]
    
    # Menampilkan statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Jumlah Data",
            value=len(df_filtered),
            help="Total data point dalam rentang waktu yang dipilih"
        )
    
    with col2:
        min_price = df_filtered.min()
        st.metric(
            label="📉 Harga Terendah",
            value=f"${min_price:.2f}",
            help="Harga penutupan terendah dalam periode"
        )
    
    with col3:
        max_price = df_filtered.max()
        st.metric(
            label="📈 Harga Tertinggi",
            value=f"${max_price:.2f}",
            help="Harga penutupan tertinggi dalam periode"
        )
    
    with col4:
        avg_price = df_filtered.mean()
        st.metric(
            label="📊 Rata-rata Harga",
            value=f"${avg_price:.2f}",
            help="Rata-rata harga penutupan dalam periode"
        )
    
    st.markdown("---")
    
    # Menampilkan grafik harga saham dengan Plotly
    st.markdown("### 📈 Grafik Harga Penutupan Saham")
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered.values,
        mode='lines',
        name='Harga Penutupan',
        line=dict(color='#667eea', width=2),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig_price.update_layout(
        title=f'Harga Penutupan Saham {ticker}',
        xaxis_title='Tanggal',
        yaxis_title='Harga Penutupan (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_price, use_container_width=True)

    # Dekomposisi dan menampilkan hasilnya
    st.markdown("---")
    st.markdown("### 🔍 Dekomposisi Time Series")
    st.markdown("Dekomposisi data menjadi komponen Observed, Trend, Seasonality, dan Residual")
    
    decomposition = decompose_data(df_filtered)

    # Jika dekomposisi berhasil
    if decomposition:
        # Menggunakan Plotly untuk dekomposisi yang lebih interaktif
        fig_decomp = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed (Data Asli)', 'Trend', 'Seasonality', 'Residual'),
            vertical_spacing=0.08,
            row_heights=[0.3, 0.3, 0.2, 0.2]
        )
        
        # Observed
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values,
                      mode='lines', name='Observed', line=dict(color='#667eea')),
            row=1, col=1
        )
        
        # Trend
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values,
                      mode='lines', name='Trend', line=dict(color='#f093fb')),
            row=2, col=1
        )
        
        # Seasonal
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values,
                      mode='lines', name='Seasonal', line=dict(color='#4facfe')),
            row=3, col=1
        )
        
        # Residual
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values,
                      mode='lines', name='Residual', line=dict(color='#f5576c')),
            row=4, col=1
        )
        
        fig_decomp.update_layout(
            height=800,
            showlegend=False,
            template='plotly_white',
            title_text=f"Dekomposisi Time Series - {ticker}"
        )
        
        fig_decomp.update_xaxes(title_text="Tanggal", row=4, col=1)
        fig_decomp.update_yaxes(title_text="Nilai", row=2, col=1)
        
        st.plotly_chart(fig_decomp, use_container_width=True)

        # Menampilkan Rolling Mean dan Rolling Std
        st.markdown("---")
        st.markdown("### 📊 Analisis Rolling Statistics")
        st.markdown("Rolling Mean dan Rolling Standard Deviation dengan window 12 bulan")
        
        rolling_mean = df_filtered.rolling(window=12).mean()
        rolling_std = df_filtered.rolling(window=12).std()

        # Menampilkan grafik Rolling Mean dan Rolling Std dengan Plotly
        fig_rolling = go.Figure()
        
        fig_rolling.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered.values,
            mode='lines',
            name='Harga Penutupan',
            line=dict(color='#667eea', width=2)
        ))
        
        fig_rolling.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            mode='lines',
            name='Rolling Mean (12 bulan)',
            line=dict(color='#f5576c', width=2, dash='dash')
        ))
        
        fig_rolling.add_trace(go.Scatter(
            x=rolling_std.index,
            y=rolling_std.values,
            mode='lines',
            name='Rolling Std (12 bulan)',
            line=dict(color='#43e97b', width=2, dash='dot')
        ))
        
        fig_rolling.update_layout(
            title=f'Rolling Mean dan Rolling Std - {ticker}',
            xaxis_title='Tanggal',
            yaxis_title='Harga Penutupan (USD)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_rolling, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>📊 Analisis Deret Waktu Saham | Dibuat dengan Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
