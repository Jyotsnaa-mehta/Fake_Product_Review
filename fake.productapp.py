import streamlit as st
import pandas as pd
import sqlite3
import re
import datetime
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide"
)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("reviews.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review TEXT,
    prediction TEXT,
    date TEXT
)
""")
conn.commit()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------- TITLE ----------------
st.title("🔍 Fake Product Review Detection")
st.markdown("### Detect whether a product review is Genuine or Fake")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Dashboard")

total_reviews = pd.read_sql(
    "SELECT COUNT(*) as count FROM predictions", conn
)["count"][0]

st.sidebar.metric("Total Predictions", total_reviews)

# ---------------- INPUT ----------------
review = st.text_area(
    "Enter Product Review",
    height=150,
    placeholder="Type review here..."
)

col1, col2 = st.columns(2)

# ---------------- PREDICT ----------------
with col1:
    if st.button("🚀 Detect Review", use_container_width=True):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])

            prediction = model.predict(vec)[0]

            if prediction.lower() == "fake":
                st.error("❌ Fake Review Detected")
            else:
                st.success("✅ Genuine Review")

            c.execute(
                "INSERT INTO predictions(review,prediction,date) VALUES(?,?,?)",
                (
                    review,
                    prediction,
                    str(datetime.datetime.now())
                )
            )
            conn.commit()

# ---------------- CLEAR HISTORY ----------------
with col2:
    if st.button("🗑 Clear History", use_container_width=True):
        c.execute("DELETE FROM predictions")
        conn.commit()
        st.success("History Cleared")

# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("📜 Prediction History")

history = pd.read_sql(
    "SELECT * FROM predictions ORDER BY id DESC",
    conn
)

if len(history) > 0:

    for index, row in history.iterrows():

        with st.expander(f"Review #{row['id']}"):

            st.write("**Review:**")
            st.write(row["review"])

            st.write("**Prediction:**", row["prediction"])
            st.write("**Date:**", row["date"])

            if st.button(
                f"Delete Review {row['id']}",
                key=row["id"]
            ):
                c.execute(
                    "DELETE FROM predictions WHERE id=?",
                    (row["id"],)
                )
                conn.commit()
                st.rerun()

else:
    st.info("No predictions yet.")

# ---------------- STATS ----------------
st.markdown("---")
st.subheader("📈 Statistics")

if len(history) > 0:

    stats = history["prediction"].value_counts()

    st.bar_chart(stats)

    st.dataframe(
        history,
        use_container_width=True
    )
