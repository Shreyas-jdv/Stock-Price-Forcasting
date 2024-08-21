import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import neptune.new as neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

# Load your dataset (Assuming the CSV file is in the same directory)
stockprices = pd.read_csv("stock_market_data-AAPL.csv", index_col="Date")


test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))

train = stockprices[:train_size][["Close"]]
test = stockprices[train_size:][["Close"]]

def extract_seqX_outcomeY(data, N, offset):
    X, y = [], []
    for i in range(offset, len(data)):
        X.append(data[i - N: i])
        y.append(data[i])
    return np.array(X), np.array(y)


def calculate_perf_metrics(var):
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )
    mape = calculate_mape(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )

    run["RMSE"] = rmse
    run["MAPE (%)"] = mape

    return rmse, mape

def plot_stock_trend(var, cur_title, stockprices=stockprices):
    ax = stockprices[["Close", var, "200day"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

    run["Plot of Stock Predictions"].upload(
        neptune.types.File.as_image(ax.get_figure())
    )

window_size = 50

run = neptune.init_run(
    project="your-neptune-project",
    name="SMA",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Simple", "neptune"],
)

window_var = f"{window_size}day"

stockprices[window_var] = stockprices["Close"].rolling(window_size).mean()
stockprices["200day"] = stockprices["Close"].rolling(200).mean()

plot_stock_trend(var=window_var, cur_title="Simple Moving Averages")
rmse_sma, mape_sma = calculate_perf_metrics(var=window_var)

run.stop()


run = neptune.init_run(
    project="your-neptune-project",
    name="EMA",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Exponential", "neptune"],
)

window_ema_var = f"{window_var}_EMA"

stockprices[window_ema_var] = (
    stockprices["Close"].ewm(span=window_size, adjust=False).mean()
)
stockprices["200day"] = stockprices["Close"].rolling(200).mean()

plot_stock_trend(
    var=window_ema_var, cur_title="Exponential Moving Averages")
rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var)

run.stop()


layer_units = 50
optimizer = "adam"
cur_epochs = 15
cur_batch_size = 20

cur_LSTM_args = {
    "units": layer_units,
    "optimizer": optimizer,
    "batch_size": cur_batch_size,
    "epochs": cur_epochs,
}

run = neptune.init_run(
    project="your-neptune-project",
    name="LSTM",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "LSTM", "neptune"],
)
run["LSTM_args"] = cur_LSTM_args

scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[["Close"]])
scaled_data_train = scaled_data[: train.shape[0]]

X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)


def Run_LSTM(X_train, layer_units=50):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

model = Run_LSTM(X_train, layer_units=layer_units)

history = model.fit(
    X_train,
    y_train,
    epochs=cur_epochs,
    batch_size=cur_batch_size,
    verbose=1,
    validation_split=0.1,
    shuffle=True,
    callbacks=[neptune_callback],
)

def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):
    raw = data["Close"][len(data) - len(test) - window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)

    X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

X_test = preprocess_testdat()

predicted_price_ = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price_)

test["Predictions_lstm"] = predicted_price

rmse_lstm = calculate_rmse(np.array(test["Close"]), np.array(test["Predictions_lstm"]))
mape_lstm = calculate_mape(np.array(test["Close"]), np.array(test["Predictions_lstm"]))

run["RMSE"] = rmse_lstm
run["MAPE (%)"] = mape_lstm

def plot_stock_trend_lstm(train, test):
    fig = plt.figure(figsize = (20,10))
    plt.plot(np.asarray(train.index), np.asarray(train["Close"]), label = "Train Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Close"]), label = "Test Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Predictions_lstm"]), label = "Predicted Closing Price")
    plt.title("LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper left")

    run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(fig))

plot_stock_trend_lstm(train, test)

run.stop()
