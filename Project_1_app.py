import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import json
import datetime

img_logo = Image.open("page_icon.png")
img_logo = img_logo.resize((150, 150))
img_banner = Image.open("banner.png")

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Smart Home Pricing",
    page_icon=img_logo,
    layout="wide"
)

# =========================
# ANOMALY DETECTOR CLASS
# =========================
class AnomalyDetector:
    def __init__(
        self,
        iso_model,
        scaler,
        iso_feature_cols_num,
        iso_feature_cols_cat,
        iso_dummy_columns,
        minmax_stats,
        p_stats,
        threshold,
        ml_min,
        ml_max,
        weights
    ):
        self.iso_model = iso_model
        self.scaler = scaler
        self.iso_feature_cols_num = iso_feature_cols_num
        self.iso_feature_cols_cat = iso_feature_cols_cat
        self.iso_dummy_columns = iso_dummy_columns
        self.minmax_stats = minmax_stats
        self.p_stats = p_stats
        self.threshold = threshold
        self.ml_min = ml_min
        self.ml_max = ml_max
        self.weights = weights

    def _build_iso_input(self, form_data):
        row = {}

        for col in self.iso_feature_cols_num:
            row[col] = form_data.get(col, 0)

        for col in self.iso_feature_cols_cat:
            val = str(form_data.get(col, "Không rõ"))
            row[f"{col}_{val}"] = 1

        df = pd.DataFrame([row])

        for col in self.iso_dummy_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.iso_dummy_columns]
        return df

    def _compute_ml_score(self, iso_input):
        scaled = self.scaler.transform(iso_input)
        raw_score = self.iso_model.decision_function(scaled)[0]

        s_ml = (self.ml_max - raw_score) / (self.ml_max - self.ml_min + 1e-6)
        s_ml = max(0, min(1, s_ml))
        return s_ml

    def _find_threshold(self, table, form_data, lower_col, upper_col=None):
        if table is None:
            return None

        temp = table.copy()

        if "quan" in temp.columns:
            temp = temp[temp["quan"] == form_data.get("quan")]

        if "loai_hinh" in temp.columns:
            match = temp[temp["loai_hinh"] == form_data.get("loai_hinh")]
            if not match.empty:
                temp = match

        if temp.empty:
            return None

        row = temp.iloc[0]

        if upper_col:
            return row[lower_col], row[upper_col]
        return row[lower_col]

    def predict(self, form_data, suggested_price):
        posted_price = form_data.get("gia_dang_vnd", 0)
        area = max(form_data.get("dien_tich", 1), 1e-6)

        gia_m2 = posted_price / area
        residual = posted_price - suggested_price
        residual_ratio = abs(residual) / max(suggested_price, 1)

        s_resid = 0
        if residual_ratio >= 0.35:
            s_resid = 1
        elif residual_ratio >= 0.2:
            s_resid = 0.5

        s_minmax = 0
        mm = self._find_threshold(self.minmax_stats, form_data, "min_threshold", "max_threshold")
        if mm:
            min_th, max_th = mm
            min_th *= 1e9
            max_th *= 1e9
            if gia_m2 < min_th or gia_m2 > max_th:
                s_minmax = 1

        s_percentile = 0
        p = self._find_threshold(self.p_stats, form_data, "P10", "P90")
        if p:
            p10, p90 = p
            p10 *= 1e9
            p90 *= 1e9
            if gia_m2 < p10 or gia_m2 > p90:
                s_percentile = 1

        iso_input = self._build_iso_input(form_data)
        s_ml = self._compute_ml_score(iso_input)

        w1 = self.weights["w1"]
        w2 = self.weights["w2"]
        w3 = self.weights["w3"]
        w4 = self.weights["w4"]

        total_score = (
            w1 * s_resid +
            w2 * s_minmax +
            w3 * s_percentile +
            w4 * s_ml
        )

        score = total_score * 100
        is_anomaly = total_score >= self.threshold

        reasons = []
        if s_resid > 0:
            reasons.append("Sai lệch giá so với đề xuất")
        if s_minmax > 0:
            reasons.append("Vi phạm ngưỡng Min-Max")
        if s_percentile > 0:
            reasons.append("Ngoài vùng P10-P90")
        if s_ml > 0.5:
            reasons.append("Bất thường theo ML")

        if not reasons:
            reasons.append("Bình thường")

        return {
            "score": score,
            "is_anomaly": is_anomaly,
            "reasons": reasons
        }

# =========================
# LOAD DATA / MODEL
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("output_data/final_master.csv")

@st.cache_data
def load_data_original():
    return pd.read_csv("output_data/original_data.csv")

@st.cache_resource
def load_artifacts():
    price_artifact = None
    anomaly_artifact = None

    if os.path.exists("model/price_artifact.pkl"):
        price_artifact = joblib.load("model/price_artifact.pkl")

    if os.path.exists("model/Anomaly_artifact.pkl"):
        anomaly_artifact = joblib.load("model/Anomaly_artifact.pkl")

    return price_artifact, anomaly_artifact

df = load_data()
price_artifact, anomaly_artifact = load_artifacts()

DATA_FILE = "data_posts.json"

# =========================
# SESSION STATE
# =========================
def save_posts(posts):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2, default=str)

def load_posts():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            return []
    return []

if "submitted_posts" not in st.session_state:
    st.session_state.submitted_posts = load_posts()

if "last_suggested_price" not in st.session_state:
    st.session_state.last_suggested_price = None

# CSV bulk states
if "bulk_result_df" not in st.session_state:
    st.session_state.bulk_result_df = None
if "bulk_posts_to_submit" not in st.session_state:
    st.session_state.bulk_posts_to_submit = []

# =========================
# HELPERS
# =========================
def format_vnd(value):
    if value is None:
        return "-"
    return f"{int(round(value)):,.0f} VND".replace(",", ".")

def build_model_input(form_data, feature_cols_list):
    """
    Tạo DataFrame input đúng cấu trúc với feature_cols
    """
    row = {}

    numeric_defaults = {
        "dien_tich": form_data.get("dien_tich", 0),
        "dien_tich_dat": form_data.get("dien_tich_dat", 0),
        "dien_tich_su_dung": form_data.get("dien_tich_su_dung", form_data.get("dien_tich", 0)),
        "chieu_ngang": form_data.get("chieu_ngang", 0),
        "chieu_dai": form_data.get("chieu_dai", 0),
        "so_phong_ngu": form_data.get("so_phong_ngu", 0),
        "so_phong_ve_sinh": form_data.get("so_phong_ve_sinh", 0),
        "tong_so_tang": form_data.get("tong_so_tang", 0),
    }

    cat_defaults = {
        "quan": form_data.get("quan", "Không rõ"),
        "loai_hinh": form_data.get("loai_hinh", "Không rõ"),
        "tinh_trang_noi_that": form_data.get("tinh_trang_noi_that", "Không rõ"),
        "giay_to_phap_ly": form_data.get("giay_to_phap_ly", "Không rõ"),
        "dac_diem": form_data.get("dac_diem", "Không rõ"),
        "huong_cua_chinh": form_data.get("huong_cua_chinh", "Không rõ"),
    }

    for col in feature_cols_list:
        if col in numeric_defaults:
            row[col] = numeric_defaults[col]
        elif col in cat_defaults:
            row[col] = cat_defaults[col]
        else:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                row[col] = 0
            else:
                row[col] = "Không rõ"

    return pd.DataFrame([row])

def predict_price(form_data):
    """
    User nhập VND >> chuyển đổi qua tỷ >> chạy model >> chuyển VND
    >> Output cuối cùng hiển thị VND.
    """
    if price_artifact is not None:
        try:
            price_model = price_artifact["model"]
            feature_cols = price_artifact["feature_cols"]

            input_df = build_model_input(form_data, feature_cols)

            pred_log = price_model.predict(input_df)[0]
            pred_price_ty = np.expm1(pred_log)
            pred_price_vnd = pred_price_ty * 1_000_000_000

            return max(float(pred_price_vnd), 0.0)

        except Exception as e:
            st.warning(f"Lỗi khi dự đoán model: {e}")

    base_vnd = (
        form_data.get("dien_tich", 0) * 70_000_000
        + form_data.get("so_phong_ngu", 0) * 300_000_000
        + form_data.get("tong_so_tang", 0) * 250_000_000
    )
    return max(base_vnd, 500_000_000)

def analyze_anomaly_for_admin(form_data, suggested_price):
    """
    Dùng trực tiếp Anomaly_artifact để kiểm tra bất thường
    """
    if anomaly_artifact is None:
        return {
            "suggested_price": float(suggested_price),
            "score": 0.0,
            "is_anomaly": False,
            "reasons": ["Không load được mô hình anomaly"]
        }

    result = anomaly_artifact.predict(form_data, float(suggested_price))

    return {
        "suggested_price": float(suggested_price),
        "score": float(result["score"]),
        "is_anomaly": bool(result["is_anomaly"]),
        "reasons": result["reasons"]
    }

COLUMN_MAPPING = {
    "tieu_de": "Tiêu đề",
    "dia_chi": "Địa chỉ",
    "so_dien_thoai": "Số điện thoại",
    "quan": "Quận",
    "loai_hinh": "Loại hình",
    "dien_tich": "Diện tích (m²)",
    "dien_tich_dat": "Diện tích đất (m²)",
    "dien_tich_su_dung": "Diện tích sử dụng (m²)",
    "chieu_ngang": "Chiều ngang (m)",
    "chieu_dai": "Chiều dài (m)",
    "so_phong_ngu": "Số phòng ngủ",
    "so_phong_ve_sinh": "Số phòng vệ sinh",
    "tong_so_tang": "Tổng số tầng",
    "mo_ta": "Mô tả",
    "huong_cua_chinh": "Hướng cửa chính",
    "giay_to_phap_ly": "Giấy tờ pháp lý",
    "tinh_trang_noi_that": "Tình trạng nội thất",
    "dac_diem": "Đặc điểm",
    "gia_dang_vnd": "Giá đăng (VND)"
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Menu chính")

admin_mode = False

with st.sidebar:
    show_admin = st.checkbox("Admin mode")
    if show_admin:
        admin_password = st.text_input("Nhập mật khẩu", type="password")
        if admin_password == "123456":
            admin_mode = True
        else:
            st.warning("Sai mật khẩu")

if admin_mode:
    page = st.sidebar.radio(
        "**Menu**",
        ["Giới thiệu", "Đăng tin", "Quản trị tin đăng", "Bất động sản đang bán", "Thông tin mô hình"]
    )
else:
    page = st.sidebar.radio(
        "**Menu**",
        ["Giới thiệu", "Đăng tin", "Bất động sản đang bán"]
    )

# =========================
# HEADER
# =========================
col1, col2 = st.columns([2, 10])
with col1:
    st.image(img_logo, width=150, output_format="PNG")
with col2:
    st.markdown(
        "<h1 style='margin-top:30px;'>Smart Home Pricing</h1>",
        unsafe_allow_html=True
    )
    st.caption("Đăng tin bán nhà, nhận giá đề xuất và hỗ trợ kiểm tra bất thường tại TP.HCM")

st.image(img_banner, use_container_width=True)

# =========================
# PAGE: INTRODUCTION
# =========================
if page == "Giới thiệu":
    st.markdown("#### Bối cảnh")
    st.write("Với tình hình biến động của thị trường BĐS, hiện nay, chúng tôi nhận thấy có nhiều tin đăng nhà bán/ cho thuê với mức giá chưa hợp lý với thị trường, gây khó khăn cho việc ra quyết định của cả người mua và người bán/cho thuê.")

    st.divider()

    st.markdown("#### Mục tiêu của ứng dụng")
    st.write("Từ nhu cầu trên, ứng dụng xác định 2 mục tiêu chính:")
    st.write("1. Xây dựng mô hình dự đoán và gợi ý giá đăng bán hợp lý.")
    st.write("2. Phát hiện bất thường đối với các tin đăng có giá quá thấp hoặc quá cao so với mặt bằng thị trường.")

# =========================
# PAGE: USER
# =========================
elif page == "Đăng tin":
    st.markdown("### Đăng tin bán bất động sản")

    HCM_DISTRICTS = [
        "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8",
        "Quận 10", "Quận 11", "Quận 12",
        "Quận Bình Thạnh", "Quận Gò Vấp", "Quận Phú Nhuận",
        "Quận Tân Bình", "Quận Tân Phú", "Quận Bình Tân",
        "TP Thủ Đức",
        "Huyện Bình Chánh", "Huyện Nhà Bè", "Huyện Hóc Môn", "Huyện Củ Chi", "Huyện Cần Giờ"
    ]

    DISTRICT_MAP = {
        "Quận Bình Thạnh": "Binh Thanh",
        "Quận Gò Vấp": "Go Vap",
        "Quận Phú Nhuận": "Phu Nhuan",
    }

    danh_sach_loai_hinh_df = sorted(list(df["loai_hinh"].dropna().astype(str).str.strip().unique()))

    tab_manual, tab_csv = st.tabs(["📝 ***Đăng tin thủ công***", "📂 ***Upload CSV hàng loạt***"])

    with tab_manual:
        st.markdown("### Loại bất động sản")
        c1, c2, c3 = st.columns(3)
        with c1:
            muc_dich = st.selectbox("**Tôi muốn**", ["Bán", "Cho thuê"])
        with c2:
            loai_hinh = st.selectbox("**Loại hình BĐS**", danh_sach_loai_hinh_df)
        with c3:
            Quan = st.selectbox("**Quận/Huyện (TP.HCM)**", HCM_DISTRICTS)

        st.divider()

        st.markdown("### Vị trí bất động sản")
        ten_du_an = st.text_input("**Tên tòa nhà / khu dân cư / dự án**", max_chars=100)
        dia_chi = st.text_input("**Địa chỉ**", max_chars=100)
        so_dien_thoai = st.text_input("**Số điện thoại liên hệ**", max_chars=12)
        if so_dien_thoai and not so_dien_thoai.isdigit():
            st.warning("⚠️ Số điện thoại không đúng vui lòng kiểm tra lại")
        ma_can = st.text_input("**Mã căn**", max_chars=50)
        ten_phan_khu = st.text_input("**Tên phân khu / lô**", max_chars=100)

        st.divider()

        st.markdown("### Đặc điểm bất động sản")
        col_a, col_b = st.columns(2)
        with col_a:
            dien_tich = st.number_input("**Diện tích đất (m²)**", min_value=0.0, value=50.0, step=0.1, format="%.2f")
            so_phong_ngu = st.number_input("**Số phòng ngủ**", min_value=0, value=2, step=1)
            chieu_ngang = st.number_input("**Chiều ngang (m)**", min_value=0.0, value=4.0, step=0.1, format="%.2f")
            tong_so_tang = st.number_input("**Tổng số tầng**", min_value=0, value=2, step=1)
            huong_cua_chinh = st.selectbox("**Hướng cửa chính**", ["Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc", "Không rõ"])
            tinh_trang_noi_that = st.selectbox("**Tình trạng nội thất**", ["Cơ bản", "Đầy đủ", "Cao cấp", "Không rõ"])

        with col_b:
            dien_tich_su_dung = st.number_input("**Diện tích sử dụng (m²)**", min_value=0.0, value=50.0, step=0.1, format="%.2f")
            so_phong_ve_sinh = st.number_input("**Số phòng vệ sinh**", min_value=0, value=2, step=1)
            chieu_dai = st.number_input("**Chiều dài (m)**", min_value=0.0, value=10.0, step=0.1, format="%.2f")
            giay_to_phap_ly = st.selectbox("**Giấy tờ pháp lý**", ["Sổ hồng", "Hợp đồng mua bán", "Đang chờ sổ", "Không rõ"])
            dac_diem = st.selectbox("**Đặc điểm**", ["Mặt tiền", "Hẻm xe hơi", "Gần trung tâm", "Nhà nở hậu", "Nhà chưa hoàn công", "Hiện trạng khác"])

        st.divider()

        st.markdown("### Nội dung tin đăng")
        uploaded_files = st.file_uploader("**Hình ảnh / Video**", accept_multiple_files=True, type=["png", "jpg", "jpeg", "mp4"])
        tieu_de = st.text_input("**Tiêu đề tin đăng**", max_chars=180)
        gia_dang_vnd = st.number_input("**Giá bán (VND)**", min_value=0, value=0, step=100000000)
        gia_dang_hien_thi = format_vnd(gia_dang_vnd)
        st.markdown(
            f"<p style='color:#d93025; font-style:italic; margin-top:-10px;'>💸 Giá ghi nhận: <b>{gia_dang_hien_thi}</b></p>",
            unsafe_allow_html=True
        )
        mo_ta = st.text_area("**Mô tả**", height=300, max_chars=2000)

        quan_model = DISTRICT_MAP.get(Quan, Quan)

        form_data = {
            "muc_dich": muc_dich,
            "quan": quan_model,
            "quan_hien_thi": Quan,
            "loai_hinh": loai_hinh,
            "ten_du_an": ten_du_an,
            "dia_chi": dia_chi,
            "so_dien_thoai": so_dien_thoai,
            "ma_can": ma_can,
            "ten_phan_khu": ten_phan_khu,
            "dien_tich": dien_tich,
            "dien_tich_dat": dien_tich,
            "dien_tich_su_dung": dien_tich_su_dung,
            "so_phong_ngu": so_phong_ngu,
            "so_phong_ve_sinh": so_phong_ve_sinh,
            "chieu_ngang": chieu_ngang,
            "chieu_dai": chieu_dai,
            "tong_so_tang": tong_so_tang,
            "huong_cua_chinh": huong_cua_chinh,
            "giay_to_phap_ly": giay_to_phap_ly,
            "tinh_trang_noi_that": tinh_trang_noi_that,
            "dac_diem": dac_diem,
            "tieu_de": tieu_de,
            "mo_ta": mo_ta,
            "gia_dang_vnd": gia_dang_vnd,
            "gia_dang": gia_dang_vnd / 1_000_000_000,
            "so_anh_video": len(uploaded_files) if uploaded_files else 0
        }

        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Giá đề xuất", type="primary", use_container_width=True, key="manual_predict"):
                suggested_price = predict_price(form_data)
                st.session_state.last_suggested_price = suggested_price
        with b2:
            if st.button("Đăng tin", type="primary", use_container_width=True, key="manual_submit"):
                suggested_price = st.session_state.last_suggested_price
                if suggested_price is None:
                    suggested_price = predict_price(form_data)

                new_post = {
                    "id": len(st.session_state.submitted_posts) + 1,
                    "status": "Chờ duyệt",
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "form_data": form_data,
                    "suggested_price": suggested_price,
                    "admin_checked": False,
                    "admin_result": None,
                    "approved": False,
                    "rejected": False
                }

                st.session_state.submitted_posts.append(new_post)
                save_posts(st.session_state.submitted_posts)
                st.success("Đã gửi tin đăng sang bộ phận kiểm duyệt.")

        if st.session_state.last_suggested_price is not None:
            st.info(f"Giá đề xuất: {format_vnd(st.session_state.last_suggested_price)}")

    with tab_csv:
        st.markdown("### Tải template CSV mẫu")

        template_df = pd.DataFrame([
            {
                "tieu_de": "Nhà hẻm xe hơi Bình Thạnh",
                "dia_chi": "123 Nguyễn Xí",
                "so_dien_thoai": "0909123456",
                "quan": "Quận Bình Thạnh",
                "loai_hinh": "Nhà ngõ, hẻm",
                "dien_tich": 50,
                "dien_tich_dat": 50,
                "dien_tich_su_dung": 50,
                "chieu_ngang": 4,
                "chieu_dai": 12,
                "so_phong_ngu": 2,
                "so_phong_ve_sinh": 2,
                "tong_so_tang": 2,
                "mo_ta": "Nhà đẹp gần trung tâm, vào ở ngay",
                "huong_cua_chinh": "Đông",
                "giay_to_phap_ly": "Sổ hồng",
                "tinh_trang_noi_that": "Cơ bản",
                "dac_diem": "Hẻm xe hơi",
                "gia_dang_vnd": 3200000000
            },
            {
                "tieu_de": "Nhà mặt tiền kinh doanh tốt",
                "dia_chi": "456 Phan Văn Trị",
                "so_dien_thoai": "0911222333",
                "quan": "Quận Gò Vấp",
                "loai_hinh": "Nhà mặt phố, mặt tiền",
                "dien_tich": 65,
                "dien_tich_dat": 65,
                "dien_tich_su_dung": 80,
                "chieu_ngang": 5,
                "chieu_dai": 13,
                "so_phong_ngu": 3,
                "so_phong_ve_sinh": 3,
                "tong_so_tang": 3,
                "mo_ta": "Phù hợp kinh doanh buôn bán",
                "huong_cua_chinh": "Nam",
                "giay_to_phap_ly": "Sổ hồng",
                "tinh_trang_noi_that": "Đầy đủ",
                "dac_diem": "Mặt tiền",
                "gia_dang_vnd": 5800000000
            }
        ])

        template_df_display = template_df.rename(columns=COLUMN_MAPPING)
        template_csv = template_df_display.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 Tải template CSV",
            data=template_csv,
            file_name="template_bds.csv",
            mime="text/csv"
        )

        st.divider()

        st.markdown("### Upload CSV hàng loạt để dự đoán giá")
        csv_file = st.file_uploader(
            "**Tải file CSV danh sách tin đăng**",
            type=["csv"],
            key="bulk_csv_uploader"
        )

        st.caption("*File CSV nên có các cột như: Quận, Loại hình, Diện tích (m²), Diện tích đất (m²), Diện tích sử dụng (m²), Chiều ngang (m), Chiều dài (m), Số phòng ngủ, Số phòng vệ sinh, Tổng số tầng, Hướng cửa chính, Giấy tờ pháp lý, Tình trạng nội thất, Đặc điểm, Giá đăng (VND)")
        if csv_file is not None:
            try:
                # Dataframe để HIỂN THỊ cho user
                bulk_df_display = pd.read_csv(csv_file)
                bulk_df_display.insert(0, "STT", range(1, len(bulk_df_display) + 1))

                st.markdown("#### Dữ liệu CSV đã tải lên")
                with st.container(border=True):
                    st.dataframe(
                        bulk_df_display.style.set_properties(**{'background-color': '#f5e9c1'}), hide_index=True, use_container_width=True, height=300)

                # Dataframe để CHẠY MODEL
                reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
                bulk_df_model = bulk_df_display.rename(columns=reverse_mapping)

                required_columns = [
                    "quan",
                    "loai_hinh",
                    "dien_tich",
                    "dien_tich_dat",
                    "dien_tich_su_dung",
                    "chieu_ngang",
                    "chieu_dai",
                    "so_phong_ngu",
                    "so_phong_ve_sinh",
                    "tong_so_tang",
                    "huong_cua_chinh",
                    "giay_to_phap_ly",
                    "tinh_trang_noi_that",
                    "dac_diem"
                ]

                missing_cols = [col for col in required_columns if col not in bulk_df_model.columns]
                if missing_cols:
                    st.error(f"❌ File CSV thiếu các cột bắt buộc: {', '.join(missing_cols)}")
                    st.stop()

                if bulk_df_model[required_columns].isnull().any().any():
                    st.error("❌ File CSV có giá trị bị thiếu trong các cột bắt buộc. Vui lòng kiểm tra lại.")
                    st.stop()

                c_bulk1, c_bulk2 = st.columns(2)

                with c_bulk1:
                    if st.button("Dự đoán giá hàng loạt", type="primary", use_container_width=True, key="bulk_predict"):
                        result_rows = []
                        posts_to_submit = []

                        for _, row in bulk_df_model.iterrows():
                            row_quan_hien_thi = str(row.get("quan", "Không rõ")).strip()

                            if row_quan_hien_thi in DISTRICT_MAP:
                                row_quan_model = DISTRICT_MAP[row_quan_hien_thi]
                            else:
                                row_quan_model = row_quan_hien_thi

                            row_form_data = {
                                "muc_dich": row.get("muc_dich", "Bán"),
                                "quan": row_quan_model,
                                "quan_hien_thi": row_quan_hien_thi,
                                "loai_hinh": row.get("loai_hinh", "Không rõ"),
                                "ten_du_an": row.get("ten_du_an", ""),
                                "dia_chi": row.get("dia_chi", ""),
                                "so_dien_thoai": str(row.get("so_dien_thoai", "")),
                                "ma_can": row.get("ma_can", ""),
                                "ten_phan_khu": row.get("ten_phan_khu", ""),
                                "dien_tich": row.get("dien_tich", 0),
                                "dien_tich_dat": row.get("dien_tich_dat", row.get("dien_tich", 0)),
                                "dien_tich_su_dung": row.get("dien_tich_su_dung", row.get("dien_tich", 0)),
                                "so_phong_ngu": row.get("so_phong_ngu", 0),
                                "so_phong_ve_sinh": row.get("so_phong_ve_sinh", 0),
                                "chieu_ngang": row.get("chieu_ngang", 0),
                                "chieu_dai": row.get("chieu_dai", 0),
                                "tong_so_tang": row.get("tong_so_tang", 0),
                                "huong_cua_chinh": row.get("huong_cua_chinh", "Không rõ"),
                                "giay_to_phap_ly": row.get("giay_to_phap_ly", "Không rõ"),
                                "tinh_trang_noi_that": row.get("tinh_trang_noi_that", "Không rõ"),
                                "dac_diem": row.get("dac_diem", "Không rõ"),
                                "tieu_de": row.get("tieu_de", f"Tin CSV {row['STT']}"),
                                "mo_ta": row.get("mo_ta", ""),
                                "gia_dang_vnd": row.get("gia_dang_vnd", 0),
                                "gia_dang": row.get("gia_dang_vnd", 0) / 1_000_000_000 if pd.notna(row.get("gia_dang_vnd", 0)) else 0,
                                "so_anh_video": 0
                            }

                            suggested_price = predict_price(row_form_data)

                            result_rows.append({
                                "STT": row["STT"],
                                "Tiêu đề": row.get("tieu_de", f"Tin CSV {row['STT']}"),
                                "Địa chỉ": row.get("dia_chi", ""),
                                "Số điện thoại": row.get("so_dien_thoai", ""),
                                "Quận": row_quan_hien_thi,
                                "Loại hình": row.get("loai_hinh", "Không rõ"),
                                "Diện tích (m²)": row.get("dien_tich", 0),
                                "Diện tích đất (m²)": row.get("dien_tich_dat", 0),
                                "Diện tích sử dụng (m²)": row.get("dien_tich_su_dung", 0),
                                "Chiều ngang (m)": row.get("chieu_ngang", 0),
                                "Chiều dài (m)": row.get("chieu_dai", 0),
                                "Số phòng ngủ": row.get("so_phong_ngu", 0),
                                "Số phòng vệ sinh": row.get("so_phong_ve_sinh", 0),
                                "Tổng số tầng": row.get("tong_so_tang", 0),
                                "Mô tả": row.get("mo_ta", ""),
                                "Hướng cửa chính": row.get("huong_cua_chinh", "Không rõ"),
                                "Giấy tờ pháp lý": row.get("giay_to_phap_ly", "Không rõ"),
                                "Tình trạng nội thất": row.get("tinh_trang_noi_that", "Không rõ"),
                                "Đặc điểm": row.get("dac_diem", "Không rõ"),
                                "Giá đăng (VND)": format_vnd(row.get("gia_dang_vnd", 0)),
                                "Giá đề xuất (VND)": format_vnd(suggested_price)
                            })

                            posts_to_submit.append({
                                "form_data": row_form_data,
                                "suggested_price": suggested_price
                            })

                        st.session_state.bulk_result_df = pd.DataFrame(result_rows)
                        st.session_state.bulk_posts_to_submit = posts_to_submit
                        st.success(f"Đã dự đoán giá cho {len(posts_to_submit)} tin từ file CSV.")

                with c_bulk2:
                    if st.button("Đăng tin hàng loạt", type="primary", use_container_width=True, key="bulk_submit"):
                        if not st.session_state.bulk_posts_to_submit:
                            st.warning("Vui lòng bấm 'Dự đoán giá hàng loạt' trước khi đăng tin.")
                        else:
                            created_posts = 0

                            for item in st.session_state.bulk_posts_to_submit:
                                new_post = {
                                    "id": len(st.session_state.submitted_posts) + 1,
                                    "status": "Chờ duyệt",
                                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "form_data": item["form_data"],
                                    "suggested_price": item["suggested_price"],
                                    "admin_checked": False,
                                    "admin_result": None,
                                    "approved": False,
                                    "rejected": False
                                }
                                st.session_state.submitted_posts.append(new_post)
                                created_posts += 1

                            save_posts(st.session_state.submitted_posts)
                            st.session_state.bulk_posts_to_submit = []
                            st.success(f"Đã gửi {created_posts} tin sang bộ phận kiểm duyệt.")

                if st.session_state.bulk_result_df is not None:
                    st.markdown("#### Bảng giá đề xuất cho từng tin")
                    with st.container(border=True):
                        st.dataframe(
                            st.session_state.bulk_result_df.style.set_properties(**{'background-color': '#f5e9c1'}), hide_index=True,use_container_width=True, height=300)

            except Exception as e:
                st.error(f"Lỗi khi đọc hoặc xử lý file CSV: {e}")



# =========================
# PAGE: ADMIN REVIEW
# =========================
elif page == "Quản trị tin đăng":
    st.subheader("Quản trị tin đăng")

    pending_posts = [p for p in st.session_state.submitted_posts if p["status"] == "Chờ duyệt"]
    approved_posts = [p for p in st.session_state.submitted_posts if p["status"] == "Đã duyệt"]
    rejected_posts = [p for p in st.session_state.submitted_posts if p["status"] == "Đã từ chối"]

    tab1, tab2, tab3 = st.tabs(["🔶  **Tin chờ duyệt**", "✅ **Tin đã duyệt**", "❌  **Tin đã từ chối**"])

    with tab1:
        if not pending_posts:
            st.info("Chưa có tin đăng nào đang chờ duyệt.")
        else:
            for post in pending_posts:
                idx = next(i for i, x in enumerate(st.session_state.submitted_posts) if x["id"] == post["id"])
                fd = post["form_data"]

                with st.expander(f"Tin #{post['id']} - {fd.get('tieu_de', 'Không có tiêu đề')} - {post['status']}", expanded=False):
                    c1, c2 = st.columns([1.3, 1])

                    with c1:
                        st.markdown("#### Thông tin tin đăng")
                        st.write(f"**Ngày giờ đăng:** {post.get('created_at', '')}")
                        st.write(f"**Địa chỉ:** {fd.get('dia_chi', '')}")
                        st.write(f"**Số điện thoại liên hệ:** {fd.get('so_dien_thoai', '')}")
                        st.write(f"**Quận:** {fd.get('quan_hien_thi', fd.get('quan', ''))}")
                        st.write(f"**Loại hình:** {fd.get('loai_hinh', '')}")
                        st.write(f"**Diện tích:** {fd.get('dien_tich', 0)} m²")
                        st.write(f"**Phòng ngủ:** {fd.get('so_phong_ngu', 0)}")
                        st.write(f"**Phòng vệ sinh:** {fd.get('so_phong_ve_sinh', 0)}")
                        st.write(f"**Tổng số tầng:** {fd.get('tong_so_tang', 0)}")
                        st.write(f"**Giá đăng:** {format_vnd(fd.get('gia_dang_vnd', 0))}")
                        st.write(f"**Giá đề xuất:** {format_vnd(post.get('suggested_price'))}")
                        st.write(f"**Mô tả:** {fd.get('mo_ta', '')}")

                    with c2:
                        st.markdown("#### Thao tác admin")

                        if st.button("Kiểm tra bất thường", type="primary", key=f"check_{idx}", use_container_width=True):
                            result = analyze_anomaly_for_admin(fd, post["suggested_price"])
                            st.session_state.submitted_posts[idx]["admin_checked"] = True
                            st.session_state.submitted_posts[idx]["admin_result"] = result
                            save_posts(st.session_state.submitted_posts)

                        c_btn1, c_btn2 = st.columns(2)
                        with c_btn1:
                            if st.button("Duyệt tin", type="primary", key=f"approve_{idx}", use_container_width=True):
                                st.session_state.submitted_posts[idx]["status"] = "Đã duyệt"
                                st.session_state.submitted_posts[idx]["approved"] = True
                                st.session_state.submitted_posts[idx]["rejected"] = False
                                save_posts(st.session_state.submitted_posts)
                                st.rerun()

                        with c_btn2:
                            if st.button("Từ chối", type="primary", key=f"reject_{idx}", use_container_width=True):
                                st.session_state.submitted_posts[idx]["status"] = "Đã từ chối"
                                st.session_state.submitted_posts[idx]["approved"] = False
                                st.session_state.submitted_posts[idx]["rejected"] = True
                                save_posts(st.session_state.submitted_posts)
                                st.rerun()

                        if st.session_state.submitted_posts[idx].get("admin_checked", False):
                            result = st.session_state.submitted_posts[idx]["admin_result"]

                            st.markdown("#### Kết quả kiểm tra")
                            st.metric("Giá đề xuất", format_vnd(result["suggested_price"]))
                            st.metric("Điểm bất thường", f"{result['score']:.1f}")

                            if result["is_anomaly"]:
                                st.error("⚠️ Tin đăng bất thường")
                                st.markdown("**Lý do:**")
                                for reason in result["reasons"]:
                                    st.write(f"- {reason}")
                            else:
                                st.success("✅ Nhà bình thường")
                                st.markdown("**Đánh giá:**")
                                st.write("- Tin đăng nằm trong vùng giá chấp nhận được")

    with tab2:
        if not approved_posts:
            st.info("Chưa có tin đã duyệt.")
        else:
            for post in approved_posts:
                fd = post["form_data"]
                with st.expander(f"Tin #{post['id']} - {fd.get('tieu_de', 'Không có tiêu đề')} - {post['status']}", expanded=False):
                    st.markdown("#### Thông tin tin đăng")
                    st.write(f"**Ngày giờ đăng:** {post.get('created_at', '')}")
                    st.write(f"**Địa chỉ:** {fd.get('dia_chi', '')}")
                    st.write(f"**Số điện thoại liên hệ:** {fd.get('so_dien_thoai', '')}")
                    st.write(f"**Quận:** {fd.get('quan_hien_thi', fd.get('quan', ''))}")
                    st.write(f"**Loại hình:** {fd.get('loai_hinh', '')}")
                    st.write(f"**Diện tích:** {fd.get('dien_tich', 0)} m²")
                    st.write(f"**Phòng ngủ:** {fd.get('so_phong_ngu', 0)}")
                    st.write(f"**Phòng vệ sinh:** {fd.get('so_phong_ve_sinh', 0)}")
                    st.write(f"**Tổng số tầng:** {fd.get('tong_so_tang', 0)}")
                    st.write(f"**Giá đăng:** {format_vnd(fd.get('gia_dang_vnd', 0))}")
                    st.write(f"**Giá đề xuất:** {format_vnd(post.get('suggested_price'))}")
                    st.write(f"**Mô tả:** {fd.get('mo_ta', '')}")

    with tab3:
        if not rejected_posts:
            st.info("Chưa có tin đã từ chối.")
        else:
            for post in rejected_posts:
                fd = post["form_data"]
                with st.expander(f"Tin #{post['id']} - {fd.get('tieu_de', 'Không có tiêu đề')} - {post['status']}", expanded=False):
                    st.markdown("#### Thông tin tin đăng")
                    st.write(f"**Ngày giờ đăng:** {post.get('created_at', '')}")
                    st.write(f"**Địa chỉ:** {fd.get('dia_chi', '')}")
                    st.write(f"**Số điện thoại liên hệ:** {fd.get('so_dien_thoai', '')}")
                    st.write(f"**Quận:** {fd.get('quan_hien_thi', fd.get('quan', ''))}")
                    st.write(f"**Loại hình:** {fd.get('loai_hinh', '')}")
                    st.write(f"**Diện tích:** {fd.get('dien_tich', 0)} m²")
                    st.write(f"**Phòng ngủ:** {fd.get('so_phong_ngu', 0)}")
                    st.write(f"**Phòng vệ sinh:** {fd.get('so_phong_ve_sinh', 0)}")
                    st.write(f"**Tổng số tầng:** {fd.get('tong_so_tang', 0)}")
                    st.write(f"**Giá đăng:** {format_vnd(fd.get('gia_dang_vnd', 0))}")
                    st.write(f"**Giá đề xuất:** {format_vnd(post.get('suggested_price'))}")
                    st.write(f"**Mô tả:** {fd.get('mo_ta', '')}")

# =========================
# PAGE: DATA
# =========================
elif page == "Bất động sản đang bán":
    st.subheader("Bất động sản đang bán")
    df_original = load_data_original()
    df_original["Giá bán (tỷ)"] = pd.to_numeric(df_original["Giá bán (tỷ)"], errors="coerce")
    df_original["Diện tích (m²)"] = pd.to_numeric(df_original["Diện tích (m²)"], errors="coerce")

    min_area = float(df_original["Diện tích (m²)"].quantile(0.01)) if not df_original["Diện tích (m²)"].empty else 0
    max_area = float(df_original["Diện tích (m²)"].quantile(0.95)) if not df_original["Diện tích (m²)"].empty else 500
    min_price = float(df_original["Giá bán (tỷ)"].min()) if not df_original["Giá bán (tỷ)"].empty else 0
    max_price = float(df_original["Giá bán (tỷ)"].max()) if not df_original["Giá bán (tỷ)"].empty else 500

    with st.form(key='search_form'):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            area_filter = st.slider(
                "**Diện tích (m²)**",
                min_value=min_area,
                max_value=max_area,
                value=(min_area, max_area),
                step=5.0
            )
        with c2:
            price_filter = st.slider(
                "**Giá bán (tỷ)**",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=0.5
            )
        with c3:
            danh_sach_quan = ["Tất cả"] + sorted(list(df_original["Quận"].dropna().astype(str).str.strip().unique()))
            quan_filter = st.selectbox("**Quận**", danh_sach_quan)
        with c4:
            danh_sach_loai_hinh = ["Tất cả"] + sorted(list(df_original["Loại hình"].dropna().astype(str).str.strip().unique()))
            loai_hinh_filter = st.selectbox("**Loại hình BĐS**", danh_sach_loai_hinh)

        submit_search = st.form_submit_button("🔎 Tìm kiếm", type="primary")

    st.divider()
    result_container = st.container()

    df_view = df_original.copy()
    if quan_filter != "Tất cả":
        df_view = df_view[df_view["Quận"].astype(str).str.strip() == quan_filter.strip()]
    if loai_hinh_filter != "Tất cả":
        df_view = df_view[df_view["Loại hình"].astype(str).str.strip() == loai_hinh_filter.strip()]

    df_view = df_view[(df_view["Diện tích (m²)"] >= area_filter[0]) & (df_view["Diện tích (m²)"] <= area_filter[1])]
    df_view = df_view[(df_view["Giá bán (tỷ)"] >= price_filter[0]) & (df_view["Giá bán (tỷ)"] <= price_filter[1])]

    with result_container:
        if df_view.empty:
            st.warning("Không tìm thấy bất động sản nào khớp với tiêu chí tìm kiếm.")
        else:
            if submit_search:
                st.success(f"Tìm thấy {len(df_view)} bất động sản phù hợp.")
            with st.container(border=True):
                st.dataframe(
                    df_view.head(200),
                    hide_index=True,
                    use_container_width=True,
                    height=500
                )

# =========================
# PAGE: MODEL INFORMATION
# =========================
elif page == "Thông tin mô hình":
    st.markdown("##### Project 1: Dự đoán giá nhà và phát hiện giá bất thường cho nhà ở trên Nhà Tốt")
    st.markdown("##### Môn học: Data Science / Machine Learning")
    st.markdown("##### Bộ dữ liệu: 3 file CSV Thông tin nhà ở tại Bình Thạnh, Gò Vấp, Phú Nhuận đăng bán trên Nhà Tốt")

    st.divider()

    st.markdown("### 1. Thông tin dữ liệu gốc")
    st.write("- Gồm 3 file CSV: quận Bình Thạnh, quận Gò Vấp, quận Phú Nhuận")
    st.write("- Tổng số dòng dữ liệu gốc: 8.273")
    st.write("- Tổng số cột dữ liệu gốc: 24")

    st.divider()

    st.markdown("### 2. Sau xử lý EDA (final_master.csv)")
    st.write("- Số dòng sau xử lý: 7.939")
    st.write("- Số cột sau xử lý: 21")
    st.write("- Dữ liệu đã được làm sạch, chuẩn hóa kiểu dữ liệu và tạo thêm feature phục vụ modeling")

    st.divider()

    st.markdown("### 3. Các feature tạo thêm")
    st.write([
        "room_density",
        "bath_density",
        "area_per_floor",
        "frontage_density",
        "desc_length"
    ])

    st.markdown("##### Phân bố số lượng dữ liệu theo quận")
    district_counts = df["quan"].value_counts()
    st.bar_chart(district_counts, sort=False)

    st.markdown("##### Phân bố dữ liệu theo khoảng giá (tỷ)")
    df_temp = df.copy()
    df_temp["price_range"] = pd.cut(
        df_temp["gia_ban"],
        bins=[0, 2, 4, 6, 10, 20, 50],
        labels=["<2", "2-4", "4-6", "6-10", "10-20", "20+"]
    )
    price_dist = df_temp["price_range"].value_counts().sort_index()
    st.bar_chart(price_dist, x_label="Khoảng giá (tỷ)", y_label="số lượng")

    st.divider()

    st.markdown("### 4. Thông số mô hình được chọn để dự đoán giá")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model", "Random Forest")
    with c2:
        st.metric("MAE", "0.1376")
    with c3:
        st.metric("RMSE", "0.1987")
    with c4:
        st.metric("R²", "0.8299")

    st.divider()

    st.markdown("### 5. Feature Importance của Random Forest")
    st.write("Các yếu tố quan trọng ảnh hưởng tới giá")

    fi_df = pd.DataFrame({
        "feature": [
            "dien_tich",
            "so_phong_ngu",
            "so_phong_ve_sinh",
            "loai_hinh_Nhà ngõ, hẻm",
            "quan_Go Vap",
            "loai_hinh_Nhà mặt phố, mặt tiền",
            "room_density",
            "tong_so_tang",
            "desc_length",
            "dien_tich_su_dung",
            "quan_Phu Nhuan",
            "frontage_density",
            "bath_density",
            "area_per_floor",
            "chieu_ngang"
        ],
        "importance": [
            0.679087,
            0.073229,
            0.034336,
            0.028410,
            0.026453,
            0.021175,
            0.017738,
            0.015940,
            0.014120,
            0.014110,
            0.013823,
            0.013639,
            0.011679,
            0.008867,
            0.008136
        ]
    })
    with st.container(border=True):
        st.dataframe(fi_df.style.set_properties(**{'background-color': '#f5e9c1'}), hide_index=True, use_container_width=True)
    fi_df_sorted = fi_df.sort_values("importance", ascending=False)
    st.bar_chart(fi_df_sorted.set_index("feature")["importance"], horizontal=True, sort=False)

    st.divider()

    st.markdown("### 6. Trọng số Composite Score của mô hình Anomaly Detection")
    st.write("Total Score = w1*S_resid + w2*S_minmax + w3*S_percentile + w4*S_ml")
    weight_df = pd.DataFrame({
        "Thành phần": [
            "Residual",
            "Vi phạm Min-Max",
            "Ngoài khoảng tin cậy P10 - P90",
            "Isolation Forest"
        ],
        "Trọng số": [
            0.4,
            0.1,
            0.2,
            0.3
        ]
    })
    with st.container(border=True):
            st.dataframe(weight_df.style.format({"Trọng số": "{:.2f}"}).set_properties(**{'background-color': '#f5e9c1'}), hide_index=True, use_container_width=True)

    st.divider()

    st.markdown("### 7. Một số Kết quả dự đoán từ dữ liệu mẫu")

    try:
        df_anom = pd.read_csv("model/anomaly_results.csv")
        show_cols = [
            "quan",
            "loai_hinh",
            "gia_ban",
            "gia_du_doan",
            "anomaly_score",
            "is_anomaly",
            "ly_do_bat_thuong"
        ]

        show_cols = [c for c in show_cols if c in df_anom.columns]
        with st.container(border=True):
                    st.dataframe(df_anom[show_cols][df_anom["is_anomaly"]==1].style.set_properties(**{'background-color': '#f5e9c1'}), hide_index=True, use_container_width=True, height=500)
            except:
                st.warning("Không load được thông tin")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("*Smart Home Pricing*")
st.caption("**Người thực hiện:**")
st.caption(" - Lê Văn Linh - [levanlinh8583@gmail.com](mailto:levanlinh8583@gmail.com)")
st.caption(" - Nguyễn Trọng Khiêm - [keyzyah@gmail.com](mailto:keyzyah@gmail.com)")

st.caption("**GV hướng dẫn:** Cô Khuất Thùy Phương")
st.caption("**Project:** Dự đoán giá nhà và phát hiện giá bất thường cho nhà ở TP.HCM")
