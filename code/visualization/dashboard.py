import sys
import os

# Automatically add the project root (two levels up from this file) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the modules from your package structure
from code.preprocessing.privacy_utils import apply_differential_privacy
from code.evaluation.bias_audit import run_bias_audit  # This can be your real function; using dummy for now
from code.evaluation.transparency_report import generate_transparency_report

import json
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static

# -------------------------------
# Dummy bias audit function for demonstration.
# Replace with your actual function if available.
def dummy_bias_audit():
    return {"accuracy": 90.0, "total_tests": 100, "discrepancies": 10}

# Placeholder function for generative feedback.
def generate_feedback(recommendation: str) -> str:
    feedback_map = {
        "Revise tone for better engagement": "Consider adjusting your language and visuals to resonate better.",
        "Content is okay, consider minor adjustments": "Your content performs adequately; a slight tweak could boost engagement.",
        "High engagement predicted: Post immediately": "Your post has great potential. Publishing now might maximize reach!"
    }
    return feedback_map.get(recommendation, "No additional feedback available.")

# -------------------------------
# Dashboard Application using Textual's compose API
class DashboardApp(App):
    def compose(self) -> ComposeResult:
        # Yield a header at the top.
        yield Header()

        # Create a DataTable widget to display recommendations.
        table = DataTable()
        table.add_column("Post ID", width=12)
        table.add_column("Sentiment Class", width=16)
        table.add_column("Noisy Sentiment", width=16)
        table.add_column("Recommendation", width=40)
        table.add_column("Feedback", width=50)

        # Load recommendations from the JSON file generated in the federated beta phase.
        recommendations_path = r"C:\EngineRoom\experiments\recommendations.json"
        try:
            with open(recommendations_path, "r") as f:
                results = json.load(f)
        except Exception as e:
            self.log(f"Error loading recommendations: {e}")
            results = []

        # Set a static privacy budget (epsilon) for demonstration.
        epsilon = 3.0

        # Populate the table with recommendations and apply differential privacy to the sentiment class.
        for row in results:
            original_sentiment = float(row["predicted_sentiment_class"])
            # Apply differential privacy noise.
            noisy_sentiment = apply_differential_privacy(original_sentiment, epsilon)
            noisy_sentiment_display = round(noisy_sentiment, 2)
            feedback = generate_feedback(row["recommendation"])
            table.add_row(
                row["post_id"],
                str(row["predicted_sentiment_class"]),
                str(noisy_sentiment_display),
                row["recommendation"],
                feedback
            )

        # Generate a transparency report using a dummy bias audit and sample model accuracy.
        dummy_audit_report = dummy_bias_audit()
        transparency_json = generate_transparency_report(dummy_audit_report, privacy_setting=epsilon, model_accuracy=85.0)
        transparency_panel = Static(f"Transparency Report:\n{transparency_json}", expand=True)

        # Yield the table and the transparency report panel.
        yield table
        yield transparency_panel

        # Yield a footer at the bottom.
        yield Footer()

if __name__ == "__main__":
    DashboardApp().run()
