from importlib import import_module

MODELS = {"seasonal_naive": "SeasonalNaive", "moving_average": "MovingAverage",
          "lightgbm_global": "LightGBMGlobal", "lstm_global": "LSTMGlobal",
          "chronos2": "Chronos2", "lag_llama": "LagLlama", "timegpt": "TimeGPT"}


def get_model(name):
    return getattr(import_module(f".{name}", __name__), MODELS[name])
