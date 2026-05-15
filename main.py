import os
import shutil

from etl_preprocessing        import preprocess_data
from exploratory_analysis      import run_exploratory_analysis
from clustering_kmeans         import perform_clustering
from prediction_neural_network import train_model
from intervention_system       import apply_intervention, print_at_risk_report
from visualization_dashboard   import run_dashboard

FOLDERS = {
    "database":     "outputs/1_database",
    "eda":          "outputs/2_eda_charts",
    "clustering":   "outputs/3_clustering",
    "model":        "outputs/4_model",
    "intervention": "outputs/5_intervention",
    "viz":          "outputs/6_viz_charts",
}


def setup_folders():
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    for path in FOLDERS.values():
        os.makedirs(path)


def move(filename, folder_key):
    if os.path.exists(filename):
        shutil.move(filename, os.path.join(FOLDERS[folder_key], filename))


def organise_outputs():
    move("student_dw.db",               "database")
    move("cleaned_student_data.csv",    "database")

    move("eda_grade_distribution.png",  "eda")
    move("eda_all_grades.png",          "eda")
    move("eda_class_distribution.png",  "eda")
    move("eda_correlation_heatmap.png", "eda")
    move("eda_feature_importance.png",  "eda")
    move("eda_absences_vs_grade.png",   "eda")
    move("eda_boxplot_by_class.png",    "eda")

    move("cluster_assignments.csv",     "clustering")
    move("cluster_nn_metrics.csv",      "clustering")
    move("cluster_labels.npy",          "clustering")
    move("wcss.npy",                    "clustering")
    move("X_pca.npy",                   "clustering")
    move("X_scaled.npy",               "clustering")
    move("y_labels.npy",                "clustering")

    move("nn_metrics.csv",              "model")
    move("confusion_matrix.npy",        "model")
    move("feature_names.txt",           "model")

    move("intervention_report.csv",     "intervention")

    move("viz_cluster_chart.png",           "viz")
    move("viz_elbow_curve.png",             "viz")
    move("viz_kmeans_scatter.png",          "viz")
    move("viz_cluster_profile.png",         "viz")
    move("viz_classifier_comparison.png",   "viz")
    move("viz_all_metrics.png",             "viz")
    move("viz_confusion_matrix.png",        "viz")
    move("viz_g3_boxplot.png",              "viz")
    move("viz_intervention_urgency.png",    "viz")

def main():
    setup_folders()

    data = preprocess_data()
    run_exploratory_analysis(data)
    data = perform_clustering(data)
    model = train_model(data)
    data = apply_intervention(data)
    print_at_risk_report(data, top_n=10)

    output_cols = ["student_id", "G1", "G2", "G3", "absences", "failures",
                   "cluster_label", "performance_class", "urgency",
                   "intervention", "risk_factor_count"]
    output_cols = [c for c in output_cols if c in data.columns]
    data[output_cols].to_csv("intervention_report.csv", index=False)

    run_dashboard(data)

    organise_outputs()

    print("\nPipeline complete. Results saved in outputs.")


if __name__ == "__main__":
    main()