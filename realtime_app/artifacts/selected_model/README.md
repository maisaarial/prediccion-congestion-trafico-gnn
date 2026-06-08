# Selected model artifact slot

This folder contains the temporary real-inference artifacts used by the realtime MVP.

Current temporary model:

- `model.pt`: checkpoint used only for technical integration.
- `graph.pt`, `edges.csv`, `tabla_gnn.csv`: graph/table artifacts from the same configuration.
- `model_config.json`: architecture and window metadata for loading the checkpoint.

The temporary model requires 12 blocks of 15 minutes. If fewer blocks are available,
the backend falls back to `DEMO_HISTORICO_INSUFICIENTE` and uses the mock identity
prediction.

Future replacement:

1. Replace `model.pt` with the final checkpoint.
2. Replace `model_config.json` with the final model configuration.
3. Keep `graph.pt`, `edges.csv`, and `tabla_gnn.csv` from the same case/adjacency as the checkpoint.
4. Do not mix models and graphs from different configurations.
