from importlib import import_module

MODELS = {"seasonal_naive": "SeasonalNaive", "moving_average": "MovingAverage",
          "lightgbm_global": "LightGBMGlobal", "lstm_global": "LSTMGlobal",
          "chronos2": "Chronos2", "lag_llama": "LagLlama", "timegpt": "TimeGPT"}

CELLS = [("seasonal_naive", "statistical"), ("moving_average", "statistical"),
         ("lightgbm_global", "statistical"), ("lstm_global", "statistical"),
         ("chronos2", "zero_shot"), ("chronos2", "fine_tune"),
         ("lag_llama", "zero_shot"), ("lag_llama", "fine_tune"),
         ("timegpt", "zero_shot"), ("timegpt", "fine_tune")]


def get_model(name):
    return getattr(import_module(f".{name}", __name__), MODELS[name])
