# Visually-Grounded Safety Process Verification (VG-SPV)

As Multimodal Large Language Models (MLLMs) become more capable, they remain vulnerable to sophisticated multimodal jailbreaks (like VisCRA) that exploit the "modality gap" between textual safety priors and visual evidence. 

**VG-SPV** is a defense framework that forces MLLMs to explicitly ground their safety critiques in verifiable visual evidence. By combining Introspective Reasoning (CoT) with automated Spatial Process Rewards via Grounding DINO, VG-SPV ensures that if a model claims a safety threat exists, it must accurately provide the bounding box coordinates for it. The model is optimized using fine-grained Direct Preference Optimization (fDPO) and Vision-Guided Loss (V-DPO) to effectively neutralize visual context attacks without increasing benign over-refusals.

Built for a research project in CS 8803 (LLMs) at Georgia Tech.