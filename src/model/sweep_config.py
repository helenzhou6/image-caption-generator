sweep_configuration = {
    "method": "grid",
    "name": "meteor_sweep",
    "metric": {
        "name": "meteor",
        "goal": "minimize"
    },
    "parameters": {
        "EMBEDDING_DIM": {
            "values": [512]
        },
        "NUM_HEADS": {
            "values": [8, 16]
        },
        "NUM_LAYERS": {
            "values": [6, 8, 10]
        },
    }
}