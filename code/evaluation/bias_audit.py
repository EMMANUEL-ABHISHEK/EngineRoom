import torch

def run_bias_audit(model, test_data, expected_labels):
    """
    Runs a bias audit on the model using test data and expected labels.
    
    Parameters:
    - model: The trained sentiment analysis model.
    - test_data: A PyG Data object containing adversarial or curated test examples.
    - expected_labels: The expected labels for the test data.
    
    Returns:
    - audit_report: A dictionary with audit results (accuracy, discrepancies, etc.).
    """
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        predictions = torch.argmax(logits, dim=1)
    
    discrepancies = (predictions != expected_labels).sum().item()
    total = expected_labels.size(0)
    accuracy = 100 * (total - discrepancies) / total
    
    audit_report = {
        "accuracy": accuracy,
        "total_tests": total,
        "discrepancies": discrepancies
    }
    return audit_report

# Example usage (assuming you have test_data and expected_labels tensors):
# audit_report = run_bias_audit(model, test_data, expected_labels)
# print("Bias Audit Report:", audit_report)
