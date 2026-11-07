import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


WINE = "#722F37"
BURGUNDY = "#4A1025"
ROSE = "#C77D8D"
CREAM = "#FFF8F3"
PLUM = "#8E5572"
GOLD = "#C6A667"

plt.rcParams.update({
    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,
    "axes.edgecolor": ROSE,
    "axes.labelcolor": BURGUNDY,
    "text.color": BURGUNDY,
    "xtick.color": BURGUNDY,
    "ytick.color": BURGUNDY,
    "grid.color": "#E8CCD2",
    "grid.alpha": 0.5,
    "font.size": 11
})

st.set_page_config(
    page_title="Wine Quality",
    page_icon="🍷",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: #FFF8F3;
    color: #4A1025;
}

h1, h2, h3 {
    color: #722F37 !important;
}

[data-testid="stSidebar"] {
    background-color: #F5E6E8;
}

div[data-testid="stMetric"] {
    background-color: #F9EAEC;
    border: 1px solid #C77D8D;
    border-radius: 15px;
    padding: 15px;
}

.stButton > button {
    background-color: #722F37;
    color: white;
    border-radius: 12px;
}

.stButton > button:hover {
    background-color: #4A1025;
    color: white;
}
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")

model = load_model()

with open(APP_DIR / "config.json", encoding="utf-8") as f:
    config = json.load(f)

@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / "test.csv")

df = load_test_data()

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

features = df.drop(columns="target")

default_threshold = float(config["threshold"])
col_title, col_hippo = st.columns([5, 1])
with col_title:
    st.title("🍷 Wine Quality")
    st.write("Приложение для оценки качества вина")

with col_hippo:
    st.image(str(APP_DIR / "hippo_icon.png"), width=100)
st.sidebar.header("Настройки модели")

threshold = st.sidebar.slider(
    "Порог классификации",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.001
)

st.sidebar.write(f"Текущий порог: {threshold:.3f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Оценка одного вина",
    "Модельные метрики",
    "Пакетное предсказание",
    "Что выучила модель?"
])

with tab1:

    st.subheader("Введите характеристики вина")
    values = {}
    for column in features.columns:
        if column == "is_red":
            is_red = st.toggle(
                "Красное вино (выкл. — белое)",
                value=bool(features[column].median())
            )
            values[column] = int(is_red)

        else:
            min_value = float(features[column].min())
            max_value = float(features[column].max())
            median_value = float(features[column].median())

            values[column] = st.number_input(
                column,
                min_value=min_value,
                max_value=max_value,
                value=median_value
            )

    wine = pd.DataFrame([values], columns=features.columns)

    probability = model.predict_proba(wine)[0, 1]

    prediction = int(probability >= threshold)

    st.subheader("Результат оценки")
    col_result, col_mascot = st.columns([3, 1])
    with col_result:
        st.metric(
            "Вероятность хорошего вина",
            f"{probability:.1%}"
        )

        if prediction == 1:
            st.success("🍷 Отличный выбор! Вино рекомендуется!")
        else:
            st.warning("Вино не рекомендуется к подборке.")
    with col_mascot:
        st.image(
            str(APP_DIR / "hippo_wine.png"),
            width=180
        )

    st.write(f"Порог классификации: {threshold:.3f}")


with tab2:
    st.header("Модельные метрики")

    X_test = df.drop(columns="target")
    y_test = df["target"]

    y_proba = test_proba(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    precision = metrics.precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = metrics.recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = metrics.f1_score(
        y_test, y_pred, zero_division=0
    )

    st.subheader("Метрики при выбранном пороге")

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1", f"{f1:.3f}")

    st.subheader("Матрица ошибок")

    cm = metrics.confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )

    wine_cmap = LinearSegmentedColormap.from_list(
    "wine",
    ["#FFF0F2", "#C77D8D", "#722F37", "#4A1025"]
    )

    disp.plot(
        ax=ax_cm,
        cmap=wine_cmap,
        colorbar=False
    )

    ax_cm.set_title(
        "Матрица ошибок",
        color=WINE,
        fontsize=15,
        fontweight="bold"
    )

    st.pyplot(fig_cm)
    plt.close(fig_cm)
    st.subheader("ROC и Precision-Recall кривые")
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    tn, fp, fn, tp = cm.ravel()
    current_fpr = fp / (fp + tn)
    current_tpr = tp / (tp + fn)

    pr_precision, pr_recall, _ = (
        metrics.precision_recall_curve(y_test, y_proba)
    )

    col_roc, col_pr = st.columns(2)

    with col_roc:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

        ax_roc.plot(
            fpr,
            tpr,
            color=WINE,
            linewidth=3,
            label="ROC"
        )
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")

        ax_roc.scatter(
            current_fpr,
            current_tpr,
            color="red",
            s=90,
            label=f"Threshold = {threshold:.3f}",
            zorder=5
        )

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC curve")
        ax_roc.legend()
        ax_roc.grid(True)

        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with col_pr:
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

        ax_pr.plot(
            pr_recall,
            pr_precision,
            color=PLUM,
            linewidth=3,
            label="Precision-Recall"
        )

        ax_pr.axhline(
            y_test.mean(),
            color="gray",
            linestyle="--",
            label="Random"
        )

        ax_pr.scatter(
            recall,
            precision,
            color=GOLD,
            s=90,
            label=f"Threshold = {threshold:.3f}",
            zorder=5
        )

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall curve")
        ax_pr.legend()
        ax_pr.grid(True)

        st.pyplot(fig_pr)
        plt.close(fig_pr)


    st.subheader("Как порог влияет на подборку?")

    st.write(
        "Если увеличить порог, модель будет рекомендовать "
        "меньше вин. Обычно Precision при этом повышается, "
        "а Recall снижается: подборка становится более "
        "избирательной, но часть хороших вин пропускается."
    )

    st.write(
        "Если уменьшить порог, в подборку попадёт больше вин. "
        "Recall обычно увеличивается, но Precision может "
        "снизиться, поскольку возрастает риск рекомендовать "
        "посредственные вина."
    )


with tab3:
    st.header("Пакетное предсказание")

    st.write(
        "Загрузите CSV-файл с характеристиками вин. "
        "Приложение рассчитает вероятность хорошего качества "
        "и решение модели для каждого вина."
    )

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            required_columns = list(features.columns)
            missing_columns = [
                col for col in required_columns
                if col not in uploaded_df.columns
            ]

            if missing_columns:
                st.error(
                    "В файле отсутствуют необходимые столбцы: "
                    + ", ".join(missing_columns)
                )

            else:
                X_uploaded = uploaded_df[required_columns].copy()
                if X_uploaded.empty:
                    st.error("Загруженный CSV-файл не содержит строк.")
                else:
                    X_uploaded = X_uploaded.apply(
                        pd.to_numeric, errors="coerce"
                    )
                    if X_uploaded.isna().any().any():
                        st.error(
                            "В файле есть пропущенные или "
                            "некорректные числовые значения."
                        )
                    else:
                        probabilities = model.predict_proba(
                            X_uploaded
                        )[:, 1]
                        predictions = (
                            probabilities >= threshold
                        ).astype(int)
                        result_df = uploaded_df.copy()
                        result_df["probability"] = probabilities
                        result_df["prediction"] = predictions
                        st.success(
                            f"Обработано вин: {len(result_df)}"
                        )
                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )
                        st.write(
                            "Рекомендовано вин:",
                            int(predictions.sum())
                        )
                        csv = result_df.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            label="Скачать результаты CSV",
                            data=csv,
                            file_name="wine_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(
                f"Не удалось обработать файл: {e}"
            )


with tab4:
    st.header("Что выучила модель?")
    st.write(
        "Коэффициенты логистической регрессии показывают, "
        "как признаки влияют на вероятность того, "
        "что вино относится к классу хороших."
    )
    logreg = model.named_steps["logisticregression"]
    coefficients = logreg.coef_[0]
    feature_names = features.columns
    coef_df = pd.DataFrame({
        "Признак": feature_names,
        "Коэффициент": coefficients
    }).sort_values("Коэффициент")
    colors = [
        WINE if coef > 0 else ROSE
        for coef in coef_df["Коэффициент"]
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        coef_df["Признак"],
        coef_df["Коэффициент"],
        color=colors
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент")
    ax.set_ylabel("Признак")
    ax.set_title("Коэффициенты логистической регрессии")
    st.pyplot(fig)
    plt.close(fig)
    positive_features = coef_df[
        coef_df["Коэффициент"] > 0
    ].sort_values("Коэффициент", ascending=False)
    negative_features = coef_df[
        coef_df["Коэффициент"] < 0
    ].sort_values("Коэффициент")
    st.subheader("Интерпретация")
    if not positive_features.empty:
        st.success(
            "Сильнее всего за хорошее вино: "
            + ", ".join(positive_features["Признак"].head(3))
        )
    if not negative_features.empty:
        st.error(
            "Сильнее всего против хорошего вина: "
            + ", ".join(negative_features["Признак"].head(3))
        )
    st.caption(
        "Положительные коэффициенты увеличивают "
        "предсказанные шансы хорошего качества вина, "
        "отрицательные уменьшают. "
        "Коэффициенты рассчитаны после стандартизации признаков."
    )
