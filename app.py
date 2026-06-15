from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import plotly.express as px

from model import run_clustering

app = Flask(__name__)

UPLOAD_FOLDER = "data"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load data
    df = pd.read_csv(filepath)
    result_df = run_clustering(df)

    # Save output
    output_path = os.path.join(OUTPUT_FOLDER, "segmented_output.csv")
    result_df.to_csv(output_path, index=False)

    # =========================
    # KPI METRICS
    # =========================
    total_customers = len(result_df)
    total_clusters = result_df["Cluster"].nunique()
    avg_income = round(result_df["Income"].mean(), 2)
    avg_spending = round(result_df["SpendingScore"].mean(), 2)

    # =========================
    # CLUSTER SUMMARY TABLE
    # =========================
    cluster_summary = result_df.groupby("Cluster").mean(numeric_only=True).round(2).reset_index()

    # =========================
    # PLOTLY CHART
    # =========================
    fig = px.scatter(
        result_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        size="Income",
        title="AI Customer Segmentation Dashboard",
        template="plotly_dark"
    )

    graph_html = fig.to_html(full_html=False)

    # =========================
    # INSIGHTS
    # =========================
    insights = f"""
    Total Customers: {total_customers}
    Total Clusters: {total_clusters}
    Average Income: {avg_income}
    Average Spending Score: {avg_spending}
    """

    return render_template(
        "dashboard.html",
        graph=graph_html,
        tables=[cluster_summary.to_html(classes="table")],
        download_link="/download",
        kpi={
            "customers": total_customers,
            "clusters": total_clusters,
            "income": avg_income,
            "spending": avg_spending
        },
        insights=insights
    )


@app.route("/download")
def download_file():
    return send_file("output/segmented_output.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)