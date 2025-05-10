import pandas as pd
import numpy as np
import os
import ccxt
import tensorflow as tf
import psutil
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set TensorFlow to use 5 threads
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

# Directories and paths
CHECKPOINT_DIR = "stock_model_checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stock_lstm.keras")
BATCH_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "batch_model.keras")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.pkl")
BATCH_PROGRESS_PATH = os.path.join(CHECKPOINT_DIR, "batch_progress.pkl")
BACKTEST_PATH = os.path.join(CHECKPOINT_DIR, "backtest_results.pkl")

# Create checkpoint directory
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    logger.info(f"Created checkpoint directory: {CHECKPOINT_DIR}")

# Custom callback for live stats, accuracy check, and batch checkpointing
class LiveStatsCallback(Callback):
    def __init__(self, X_val, y_val, scaler, features, X_train, batch_size):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler
        self.features = features
        self.X_train = X_train
        self.batch_size = batch_size
        self.history = {'loss': [], 'val_loss': [], 'rmse': [], 'r2': []}
        self.batch_losses = []
        self.target_reached = False
        self.current_batch = 0
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.current_batch = 0
        self.batch_losses = []
        logger.info(f"Starting Epoch {self.current_epoch}")

    def on_batch_end(self, batch, logs=None):
        self.current_batch += 1
        self.batch_losses.append(logs.get('loss', 0))
        if batch % 10000 == 0 or batch == 0:
            logger.info(f"Processed batch {batch}, Loss: {logs.get('loss', 0):.4f}")
            logger.info(f"RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1024**3:.2f} GB), CPU: {psutil.cpu_percent()}%")
            # Save batch checkpoint
            logger.info(f"Saving batch checkpoint: {BATCH_MODEL_PATH}, {BATCH_PROGRESS_PATH}")
            self.model.save(BATCH_MODEL_PATH)
            with open(BATCH_PROGRESS_PATH, 'wb') as f:
                pickle.dump({
                    'epoch': self.current_epoch,
                    'batch': self.current_batch,
                    'batch_losses': self.batch_losses,
                    'history': self.history
                }, f)

    def on_epoch_end(self, epoch, logs=None):
        logger.info("Computing validation metrics...")
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_val_rescaled = self.scaler.inverse_transform(
            np.concatenate([self.X_val[:, -1, :-1], self.y_val.reshape(-1, 1)], axis=1))[:, -1]
        y_pred_rescaled = self.scaler.inverse_transform(
            np.concatenate([self.X_val[:, -1, :-1], y_pred], axis=1))[:, -1]
        rmse = np.sqrt(mean_squared_error(y_val_rescaled, y_pred_rescaled))
        r2 = r2_score(y_val_rescaled, y_pred_rescaled)
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        self.history['rmse'].append(rmse)
        self.history['r2'].append(r2)
        logger.info(f"Epoch {epoch + 1}: Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        logger.info(f"Saving epoch checkpoint: {HISTORY_PATH}")
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(self.history, f)
        if r2 >= 0.9 and not self.target_reached:
            self.target_reached = True
            logger.info("Reached 90% accuracy (R² ≥ 0.9). Continue training to 100%? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                self.model.stop_training = True

# Create sequences
def create_sequences(data, seq_length):
    logger.info("Creating sequences...")
    start_time = time.time()
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, -1])  # Predict next Close
    X, y = np.array(X), np.array(y)
    logger.info(f"Sequences created: {len(X)} samples, Time: {time.time() - start_time:.2f}s")
    return X, y

# Process chunk
def process_chunk(chunk, scaler, seq_length, features):
    logger.info(f"Processing chunk with {len(chunk)} rows")
    scaled_data = scaler.transform(chunk)
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=chunk.index)
    X, y = create_sequences(scaled_df.values, seq_length)
    return X, y

# Backtest strategy
def backtest(data, model, scaler, seq_length, features, trade_duration=20):
    logger.info("Running backtest...")
    X, _ = create_sequences(scaler.transform(data[features]), seq_length)
    predictions = model.predict(X, verbose=0)
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate([X[:, -1, :-1], predictions], axis=1))[:, -1]
    actual = data['Close'].values[seq_length:-1]
    
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trades = []
    entry_time = None
    
    for i in range(len(predictions_rescaled) - 1):
        pred = predictions_rescaled[i]
        actual_price = actual[i]
        next_price = actual[i + 1]
        
        if pred > actual_price * 1.001:  # Buy if predicted 0.1% increase
            if position == 0:
                position = balance / actual_price
                balance = 0
                entry_time = data.index[i + seq_length]
                trades.append(('buy', actual_price, entry_time))
        elif pred < actual_price * 0.999 or (
            entry_time and (data.index[i + seq_length] - entry_time).total_seconds() / 60 >= trade_duration
        ):  # Sell if predicted 0.1% decrease or 20 min elapsed
            if position > 0:
                balance = position * actual_price
                position = 0
                trades.append(('sell', actual_price, data.index[i + seq_length]))
    
    if position > 0:
        balance = position * actual[-1]
    profit_pct = (balance - initial_balance) / initial_balance * 100
    logger.info(f"Backtest completed. Profit: {profit_pct:.2f}%")
    return profit_pct, trades

# Live trading
def live_trading(model, scaler, seq_length, features, exchange):
    logger.info("Starting live trading (1-20 min trades)...")
    last_sequence = None
    position = 0
    entry_time = None
    trade_duration = 20 * 60  # 20 min in seconds
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=seq_length + 1)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df.set_index('Timestamp')
            
            scaled_data = scaler.transform(df[features])
            if last_sequence is None:
                last_sequence = scaled_data[:-1]
            else:
                last_sequence = np.vstack([last_sequence[1:], scaled_data[-1]])
            
            X = last_sequence.reshape(1, seq_length, len(features))
            pred = model.predict(X, verbose=0)
            pred_price = scaler.inverse_transform(
                np.concatenate([X[:, -1, :-1], pred], axis=1))[:, -1][0]
            current_price = df['Close'].iloc[-1]
            
            if pred_price > current_price * 1.001 and position == 0:
                logger.info(f"Buy at {current_price:.2f} (Predicted: {pred_price:.2f})")
                position = 1
                entry_time = df.index[-1]
            elif (pred_price < current_price * 0.999 or (
                entry_time and (df.index[-1] - entry_time).total_seconds() >= trade_duration
            )) and position > 0:
                logger.info(f"Sell at {current_price:.2f} (Predicted: {pred_price:.2f})")
                position = 0
                entry_time = None
            
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            time.sleep(60)

# Load and process data
logger.info("Loading data...")
chunk_size = 25000  # Optimized for 6 GB RAM
batch_size = 8  # Reduced for lower RAM usage
features = ['Open', 'High', 'Low', 'Close', 'Volume']
seq_length = 6  # 6 minutes
X_train, y_train, X_test, y_test = [], [], [], []

# Initialize scaler
logger.info("Initializing scaler...")
scaler = MinMaxScaler()
first_chunk = pd.read_csv('d.csv', nrows=1000, dtype={'Timestamp': np.float64, 'Open': np.float32, 'High': np.float32, 'Low': np.float32, 'Close': np.float32, 'Volume': np.float32})
first_chunk['Timestamp'] = pd.to_datetime(first_chunk['Timestamp'], unit='s')
first_chunk = first_chunk.set_index('Timestamp')
scaler.fit(first_chunk[features])
logger.info(f"Saving scaler: {SCALER_PATH}")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# Process chunks
for chunk in pd.read_csv('d.csv', chunksize=chunk_size, dtype={'Timestamp': np.float64, 'Open': np.float32, 'High': np.float32, 'Low': np.float32, 'Close': np.float32, 'Volume': np.float32}):
    chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], unit='s')
    chunk = chunk.set_index('Timestamp')
    chunk = chunk.resample('1T').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    if chunk.empty:
        continue
    train_size = int(0.8 * len(chunk))
    train_chunk = chunk.iloc[:train_size]
    test_chunk = chunk.iloc[train_size:]
    if len(train_chunk) > seq_length:
        X, y = process_chunk(train_chunk[features], scaler, seq_length, features)
        X_train.append(X)
        y_train.append(y)
    if len(test_chunk) > seq_length:
        X, y = process_chunk(test_chunk[features], scaler, seq_length, features)
        X_test.append(X)
        y_test.append(y)

X_train = np.concatenate(X_train) if X_train else np.array([])
y_train = np.concatenate(y_train) if y_train else np.array([])
X_test = np.concatenate(X_test) if X_test else np.array([])
y_test = np.concatenate(y_test) if y_test else np.array([])

logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Load or build model
start_batch = 0
batch_losses = []
if os.path.exists(BATCH_MODEL_PATH) and os.path.exists(BATCH_PROGRESS_PATH):
    logger.info("Loading existing batch checkpoint...")
    model = load_model(BATCH_MODEL_PATH)
    with open(BATCH_PROGRESS_PATH, 'rb') as f:
        progress = pickle.load(f)
        start_batch = progress['batch']
        batch_losses = progress['batch_losses']
        history = progress['history']
        start_epoch = progress['epoch']
else:
    logger.info("Building new model...")
    model = Sequential([
        LSTM(64, input_shape=(seq_length, len(features)), return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = {'loss': [], 'val_loss': [], 'rmse': [], 'r2': []}
    start_epoch = 1

# Train model
logger.info("Starting training...")
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
live_stats = LiveStatsCallback(X_test, y_test, scaler, features, X_train, batch_size)
model.fit(X_train[start_batch * batch_size:], y_train[start_batch * batch_size:], 
         epochs=50, batch_size=batch_size, validation_data=(X_test, y_test),
         callbacks=[checkpoint, live_stats], verbose=1, initial_epoch=start_epoch - 1)

# Backtest if training stopped
if live_stats.target_reached:
    logger.info("Running backtest...")
    data = pd.read_csv('d.csv', dtype={'Timestamp': np.float64, 'Open': np.float32, 'High': np.float32, 'Low': np.float32, 'Close': np.float32, 'Volume': np.float32})
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data = data.set_index('Timestamp')
    data = data.resample('1T').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    profit_pct, trades = backtest(data, model, scaler, seq_length, features)
    logger.info(f"Backtest Profit: {profit_pct:.2f}%")
    logger.info(f"Saving backtest results: {BACKTEST_PATH}")
    with open(BACKTEST_PATH, 'wb') as f:
        pickle.dump({'profit_pct': profit_pct, 'trades': trades}, f)
    
    logger.info("Starting live trading...")
    exchange = ccxt.binance()
    live_trading(model, scaler, seq_length, features, exchange)
else:
    logger.info("Training did not reach 90% accuracy. Final R²: %.2f" % live_stats.history['r2'][-1])

# Plot training history
plt.plot(live_stats.history['rmse'], label='RMSE')
plt.plot(live_stats.history['r2'], label='R²')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_progress.png')
plt.close()

# Save final model
logger.info(f"Saving final model: {MODEL_PATH}")
model.save(MODEL_PATH)
