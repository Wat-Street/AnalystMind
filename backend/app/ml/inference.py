# ASSIGNED TO: Aadesh
#
# Task: Implement persona-level and consensus inference against a trained checkpoint.
#
# What to implement:
#
#   class PersonaInference:
#     - load_model(checkpoint_path): load FTTransformerModel weights from .pt file
#
#     - run_persona(ticker, persona_name, db_session) -> PersonaOutput:
#         1. Fetch latest feature snapshot for ticker from DB
#         2. Build feature tensor (must follow FEATURE_ORDER from features.py)
#         3. Call model.forward(x, persona_name=persona_name)
#         4. Convert predicted_return[horizon=3M] → price_target = current_price * (1 + r)
#         5. Return PersonaOutput (see schemas/persona.py)
#
#     - run_consensus(ticker, db_session) -> ConsensusPT:
#         1. Call run_persona() for all 9 personas
#         2. Confidence-weighted mean PT
#         3. Flag outliers > 1.5σ, downweight to 30%, recompute
#         4. conviction_score = 1 - (sigma / consensus_pt), capped [0, 1]
#         5. Return ConsensusPT (see schemas/persona.py)
#
# Also wire run_consensus() into POST /analyze in backend/app/api/routes.py.
#
# Depends on: backbone.py (FTTransformerModel), features.py, schemas/persona.py (Afreed)
