Table~\ref{tab:knowledge_preservation} examines whether personality guidance preserves general knowledge performance.
For each model family and benchmark, we evaluate all personality-guided variants and compare their averaged edited scores with the corresponding base model using a one-sided \(t\)-test for performance drop.
The reported \(p_{\mathrm{drop}}\) value tests whether the edited variants show a systematic decrease relative to the base model.
Under the threshold \(\alpha = 0.05\), \(p_{\mathrm{drop}} < 0.05\) indicates significant evidence of degradation, while \(p_{\mathrm{drop}} \geq 0.05\) indicates insufficient evidence for systematic drop.
All benchmark-level and pooled decisions remain negative for systematic degradation across the three model families.
For Qwen2.5-7B, the updated paired rerun gives \(p_{\mathrm{drop}} = 0.227\) on ARC-Easy, \(0.146\) on BoolQ, and \(0.960\) on GSM8K, so none of the individual benchmark tests falls below 0.05.
This suggests that the behavioural shifts analysed below are unlikely to be explained by broad knowledge degradation after personality guidance.
