import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class TradingAlgorithm:
    def __init__(self, symbol="BTCUSD", timeframe=mt5.TIMEFRAME_M30, lot_size=0.1):
        # MT5 Settings
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot_size = lot_size
        self.magic_number = 1234  # As requested
        
        # Capital Settings
        self.initial_capital = 20000
        self.position_size = 13000
        
        # Rule Weights - EXACT from your strategy
        self.RULE_WEIGHTS = [0.914, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 0.690, 0.0, 0.0, 0.0, 1.424]
        
        # Rule Parameters - EXACT from your strategy
        self.R1_PERIOD_1 = 41
        self.R1_PERIOD_2 = 50
        self.R2_PERIOD_1 = 27
        self.R2_PERIOD_2 = 27
        self.R3_PERIOD_1 = 41
        self.R3_PERIOD_2 = 61
        self.R4_PERIOD_1 = 1
        self.R4_PERIOD_2 = 3
        self.R5_PERIOD_1 = 15
        self.R5_PERIOD_2 = 23
        self.R6_PERIOD_1 = 1
        self.R6_PERIOD_2 = 3
        self.R7_STOCH_PERIOD = 11
        self.R7_SMA_PERIOD = 19
        self.R8_VI_POS_PERIOD = 7
        self.R8_VI_NEG_PERIOD = 27
        self.R9_TENKAN = 19
        self.R9_KIJUN = 23
        self.R10_RSI_PERIOD = 3
        self.R10_THRESHOLD = 85
        self.R11_CCI_PERIOD = 3
        self.R11_THRESHOLD = 60
        self.R12_RSI_PERIOD = 3
        self.R12_UPPER_BAND = 90
        self.R12_LOWER_BAND = 85
        self.R13_CCI_PERIOD = 3
        self.R13_UPPER_BAND = 60
        self.R13_LOWER_BAND = 20
        self.R14_LENGTH = 7
        self.R14_MULTIPLIER = 2.0
        self.R15_PERIOD = 5
        self.R16_LENGTH = 27
        self.R16_MULTIPLIER = 2.0
        
        # Strategy Parameters - EXACT from your strategy
        self.threshold = 0.1
        self.confidence_threshold = 0.10
        self.TRAILING_STOP_PCT = 0.7
        self.halt_threshold = -1.0
        self.halt_duration = 10
        self.cooldown_bars = 3
        
        # State Variables
        self.entry_price = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.equity = self.initial_capital
        self.daily_start_equity = None
        self.current_day = None
        self.trading_halted = False
        self.halt_start_bar = None
        self.last_trade_bar = 0
        self.bar_index = 0
        self.last_candle_time = None
        
        # Initialize MT5
        if not mt5.initialize():
            print("? MT5 initialization failed")
            print(f"Error code: {mt5.last_error()}")
            quit()
        
        print(f"? MT5 initialized successfully")
        
        # Enable symbol in Market Watch
        if not mt5.symbol_select(self.symbol, True):
            print(f"? Failed to select symbol {self.symbol}")
            print(f"Error code: {mt5.last_error()}")
            mt5.shutdown()
            quit()
        
        # Validate symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"? Symbol {self.symbol} not found")
            mt5.shutdown()
            quit()
        
        if not symbol_info.visible:
            print(f"Symbol {self.symbol} is not visible, attempting to enable...")
            if not mt5.symbol_select(self.symbol, True):
                print(f"? Failed to enable {self.symbol}")
                mt5.shutdown()
                quit()
        
        # Store symbol info for validation
        self.symbol_info = symbol_info
        self.min_lot = symbol_info.volume_min
        self.max_lot = symbol_info.volume_max
        self.lot_step = symbol_info.volume_step
        self.point = symbol_info.point
        
        # Validate lot size
        if self.lot_size < self.min_lot:
            print(f"? Lot size {self.lot_size} too small. Minimum: {self.min_lot}")
            mt5.shutdown()
            quit()
        
        if self.lot_size > self.max_lot:
            print(f"? Lot size {self.lot_size} too large. Maximum: {self.max_lot}")
            mt5.shutdown()
            quit()
        
        # Round lot size to step
        self.lot_size = round(self.lot_size / self.lot_step) * self.lot_step
        
        print(f"? Symbol {self.symbol} validated")
        print(f"? Lot size: {self.lot_size} (Min: {self.min_lot}, Max: {self.max_lot})")
        print(f"? Point value: {self.point}")
    
    @staticmethod
    def ema(series, period):
        return pd.Series(series).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def sma(series, period):
        return pd.Series(series).rolling(window=period).mean().values
    
    def dema(self, series, period):
        ema1 = self.ema(series, period)
        ema2 = self.ema(ema1, period)
        return 2 * ema1 - ema2
    
    def tema(self, series, period):
        ema1 = self.ema(series, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3
    
    @staticmethod
    def rsi(series, period):
        delta = pd.Series(series).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    def cci(series, period):
        tp = pd.Series(series)
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return ((tp - sma_tp) / (0.015 * mad)).values
    
    @staticmethod
    def stochastic(close, high, low, period):
        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        lowest_low = low_series.rolling(window=period).min()
        highest_high = high_series.rolling(window=period).max()
        return (100 * (close_series - lowest_low) / (highest_high - lowest_low)).values
    
    @staticmethod
    def atr(high, low, close, period):
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift(1))
        tr3 = abs(low_series - close_series.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().values
    
    def check_connection(self):
        """Check MT5 connection health"""
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print("?? MT5 terminal disconnected")
            return False
        
        if not terminal_info.connected:
            print("?? MT5 not connected to trade server")
            return False
        
        return True
    
    def check_market_conditions(self):
        """Check if market conditions are suitable for trading"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            print("?? No tick data available")
            return False
        
        # Check spread
        spread = tick.ask - tick.bid
        max_spread = self.point * 100  # Adjust as needed for your symbol
        
        if spread > max_spread:
            print(f"?? Spread too high: {spread:.5f} (max: {max_spread:.5f})")
            return False
        
        return True
    
    def get_market_data(self):
        """Get market data from MT5"""
        if not self.check_connection():
            return None
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 300)
        if rates is None:
            error = mt5.last_error()
            print(f"?? Failed to get rates. Error: {error}")
            return None
        
        if len(rates) == 0:
            print("?? No rate data returned")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def check_new_candle(self, df):
        """Check if a new candle has opened"""
        if df is None or len(df) == 0:
            return False
            
        current_candle_time = df['time'].iloc[-1]
        
        if self.last_candle_time is None:
            self.last_candle_time = current_candle_time
            return True
        
        if current_candle_time > self.last_candle_time:
            self.last_candle_time = current_candle_time
            return True
        
        return False
    
    def calculate_rules(self, df):
        """Calculate all 16 rules - EXACT logic from your strategy"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if len(close) < 100:
            return None
        
        try:
            # Rule 1: SMA/SMA
            ma1_1 = self.sma(close, self.R1_PERIOD_1)
            ma1_27 = self.sma(close, self.R1_PERIOD_2)
            rule1 = 2 * np.sign(ma1_27[-2] - ma1_1[-2]) + (-1 if ma1_1[-2] == ma1_27[-2] else 0)
            
            # Rule 2: EMA/SMA
            ema2_27 = self.ema(close, self.R2_PERIOD_1)
            ma2_27 = self.sma(close, self.R2_PERIOD_2)
            rule2 = 2 * np.sign(ma2_27[-2] - ema2_27[-2]) + (-1 if ema2_27[-2] == ma2_27[-2] else 0)
            
            # Rule 3: EMA/EMA
            ema3_1 = self.ema(close, self.R3_PERIOD_1)
            ema3_2 = self.ema(close, self.R3_PERIOD_2)
            rule3 = 2 * np.sign(ema3_2[-2] - ema3_1[-2]) + (-1 if ema3_1[-2] == ema3_2[-2] else 0)
            
            # Rule 4: DEMA/SMA
            dema4_1 = self.dema(close, self.R4_PERIOD_1)
            ma4_27 = self.sma(close, self.R4_PERIOD_2)
            rule4 = 2 * np.sign(ma4_27[-2] - dema4_1[-2]) + (-1 if dema4_1[-2] == ma4_27[-2] else 0)
            
            # Rule 5: DEMA/DEMA
            dema5_15 = self.dema(close, self.R5_PERIOD_1)
            dema5_23 = self.dema(close, self.R5_PERIOD_2)
            rule5 = 2 * np.sign(dema5_23[-2] - dema5_15[-2]) + (-1 if dema5_15[-2] == dema5_23[-2] else 0)
            
            # Rule 6: TEMA/SMA
            tema6_1 = self.tema(close, self.R6_PERIOD_1)
            ma6_27 = self.sma(close, self.R6_PERIOD_2)
            rule6 = 2 * np.sign(ma6_27[-2] - tema6_1[-2]) + (-1 if tema6_1[-2] == ma6_27[-2] else 0)
            
            # Rule 7: Stoch/SMA
            stoch7_k = self.stochastic(close, high, low, self.R7_STOCH_PERIOD)
            stoch7_sma = self.sma(stoch7_k, self.R7_SMA_PERIOD)
            rule7 = 2 * np.sign(stoch7_sma[-2] - stoch7_k[-2]) + (-1 if stoch7_k[-2] == stoch7_sma[-2] else 0)
            
            # Rule 8: Vortex
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift(1))
            tr3 = abs(low_series - close_series.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            vm_pos = abs(high_series - low_series.shift(1))
            vm_neg = abs(low_series - high_series.shift(1))
            
            vm_pos_7 = vm_pos.rolling(window=self.R8_VI_POS_PERIOD).sum()
            tr_sum_7 = tr.rolling(window=self.R8_VI_POS_PERIOD).sum()
            vi_pos_7 = (vm_pos_7 / tr_sum_7).values
            
            vm_neg_27 = vm_neg.rolling(window=self.R8_VI_NEG_PERIOD).sum()
            tr_sum_27 = tr.rolling(window=self.R8_VI_NEG_PERIOD).sum()
            vi_neg_27 = (vm_neg_27 / tr_sum_27).values
            
            rule8 = 2 * np.sign(vi_neg_27[-2] - vi_pos_7[-2]) + (-1 if vi_pos_7[-2] == vi_neg_27[-2] else 0)
            
            # Rule 9: Ichimoku
            ich_n1 = self.R9_TENKAN
            ich_n2 = self.R9_KIJUN
            ich_n_mid = int(round((ich_n1 + ich_n2) / 2))
            
            ich_a = (np.max(high[-ich_n1-2:-2]) + np.min(low[-ich_n1-2:-2])) / 2
            ich_b = (np.max(high[-ich_n_mid-2:-2]) + np.min(low[-ich_n_mid-2:-2])) / 2
            close_shift = close[-2]
            
            rule9_cond1 = (close_shift > ich_a) & (close_shift > ich_b)
            rule9_cond2 = (close_shift < ich_b) & (close_shift < ich_a)
            rule9 = -1 if rule9_cond1 else (1 if rule9_cond2 else 0)
            
            # Rule 10: RSI/Threshold
            rsi10 = self.rsi(close, self.R10_RSI_PERIOD)
            rule10 = 2 * np.sign(self.R10_THRESHOLD - rsi10[-2]) + (-1 if rsi10[-2] == self.R10_THRESHOLD else 0)
            
            # Rule 11: CCI/Threshold
            cci11 = self.cci(close, self.R11_CCI_PERIOD)
            rule11 = 2 * np.sign(self.R11_THRESHOLD - cci11[-2]) + (-1 if cci11[-2] == self.R11_THRESHOLD else 0)
            
            # Rule 12: RSI Bands
            rsi12 = self.rsi(close, self.R12_RSI_PERIOD)
            rule12 = -1 if rsi12[-2] > self.R12_UPPER_BAND else (1 if rsi12[-2] < self.R12_LOWER_BAND else 0)
            
            # Rule 13: CCI Bands
            cci13 = self.cci(close, self.R13_CCI_PERIOD)
            rule13 = -1 if cci13[-2] > self.R13_UPPER_BAND else (1 if cci13[-2] < self.R13_LOWER_BAND else 0)
            
            # Rule 14: Keltner
            kelt_basis = self.ema(close, self.R14_LENGTH)
            kelt_range = self.R14_MULTIPLIER * self.atr(high, low, close, self.R14_LENGTH)
            kelt_upper = kelt_basis + kelt_range
            kelt_lower = kelt_basis - kelt_range
            rule14 = -1 if close[-2] > kelt_upper[-2] else (1 if close[-2] < kelt_lower[-2] else 0)
            
            # Rule 15: Donchian
            don_upper = np.max(high[-self.R15_PERIOD-2:-2])
            don_lower = np.min(low[-self.R15_PERIOD-2:-2])
            rule15 = -1 if close[-2] > don_upper else (1 if close[-2] < don_lower else 0)
            
            # Rule 16: Bollinger Bands
            bb_basis = self.sma(close, self.R16_LENGTH)
            bb_dev = self.R16_MULTIPLIER * pd.Series(close).rolling(window=self.R16_LENGTH).std().values
            bb_upper = bb_basis + bb_dev
            bb_lower = bb_basis - bb_dev
            rule16 = -1 if close[-2] > bb_upper[-2] else (1 if close[-2] < bb_lower[-2] else 0)
            
            return [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, 
                    rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16]
            
        except Exception as e:
            print(f"? Error calculating rules: {e}")
            return None
    
    def calculate_signal(self, rules):
        """Calculate weighted signal - EXACT from your strategy"""
        if rules is None:
            return 0.0, 0.0
        
        total_signal = 0.0
        for i in range(16):
            total_signal += rules[i] * self.RULE_WEIGHTS[i]
        
        signal_confidence = abs(total_signal)
        return total_signal, signal_confidence
    
    def check_halt_system(self):
        """Daily P&L and Halt System"""
        current_time = datetime.now()
        current_day_num = current_time.day
        
        if current_day_num != self.current_day:
            self.current_day = current_day_num
            account_info = mt5.account_info()
            self.daily_start_equity = account_info.equity if account_info else self.initial_capital
            self.trading_halted = False
            self.halt_start_bar = None
            print(f"?? New trading day - Start equity: ${self.daily_start_equity:.2f}")
        
        account_info = mt5.account_info()
        current_equity = account_info.equity if account_info else self.initial_capital
        daily_pnl_pct = ((current_equity - self.daily_start_equity) / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0
        
        if daily_pnl_pct <= self.halt_threshold and not self.trading_halted:
            self.trading_halted = True
            self.halt_start_bar = self.bar_index
            print(f"?? TRADING HALTED - Daily P&L: {daily_pnl_pct:.2f}%")
        
        if self.trading_halted and self.halt_start_bar is not None and (self.bar_index - self.halt_start_bar) >= self.halt_duration:
            self.trading_halted = False
            print("? Trading halt lifted")
    
    def get_position(self):
        """Get current position size (only our magic number)"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return 0
        
        total_volume = 0
        for pos in positions:
            if pos.magic == self.magic_number:
                if pos.type == mt5.ORDER_TYPE_BUY:
                    total_volume += pos.volume
                else:
                    total_volume -= pos.volume
        return total_volume
    
    def close_position(self, comment=""):
        """Close current position (only our magic number)"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return False
        
        for pos in positions:
            if pos.magic == self.magic_number:
                # Get current price
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    print("?? No tick data for closing position")
                    return False
                
                if pos.type == mt5.ORDER_TYPE_BUY:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": pos.ticket,
                        "price": tick.bid,
                        "deviation": 20,
                        "magic": self.magic_number,
                        "comment": comment,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                        "type_time": mt5.ORDER_TIME_GTC
                    }
                else:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "position": pos.ticket,
                        "price": tick.ask,
                        "deviation": 20,
                        "magic": self.magic_number,
                        "comment": comment,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                        "type_time": mt5.ORDER_TIME_GTC
                    }
                
                result = mt5.order_send(request)
                
                if result is None:
                    print(f"? Close failed: No response from server")
                    return False
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"? Close failed: {result.retcode} - {result.comment}")
                    return False
                
                print(f"? Position closed: {comment}")
        
        self.entry_price = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        return True
    
    def open_position(self, direction, comment=""):
        """Open new position"""
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            print("?? No tick data for opening position")
            return False
        
        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            print(f"? Order failed: No response from server")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"? Order failed: {result.retcode} - {result.comment}")
            return False
        
        print(f"? {comment} executed: Order #{result.order} @ {result.price:.5f}")
        return True
    
    def update_trailing_stop(self, current_price):
        """Update trailing stop loss"""
        position = self.get_position()
        if position == 0 or self.entry_price is None:
            return False
        
        # Update highest/lowest prices
        if self.highest_price_since_entry is None:
            self.highest_price_since_entry = current_price
        else:
            self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
        
        if self.lowest_price_since_entry is None:
            self.lowest_price_since_entry = current_price
        else:
            self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)
        
        # Check trailing stop
        if position > 0:  # Long position
            stop_price = self.highest_price_since_entry * (1 - self.TRAILING_STOP_PCT / 100)
            if current_price <= stop_price:
                print(f"?? LONG TRAILING STOP HIT at {current_price:.5f} (Stop: {stop_price:.5f})")
                self.close_position("Trailing Stop")
                return True
        else:  # Short position
            stop_price = self.lowest_price_since_entry * (1 + self.TRAILING_STOP_PCT / 100)
            if current_price >= stop_price:
                print(f"?? SHORT TRAILING STOP HIT at {current_price:.5f} (Stop: {stop_price:.5f})")
                self.close_position("Trailing Stop")
                return True
        
        return False
    
    def execute_trades(self, total_signal, signal_confidence, new_candle):
        """Execute trades based on signals"""
        long_condition = (total_signal > self.threshold and 
                         signal_confidence > self.confidence_threshold and 
                         not self.trading_halted)
        
        short_condition = (total_signal < -self.threshold and 
                          signal_confidence > self.confidence_threshold and 
                          not self.trading_halted)
        
        can_trade = (self.bar_index - self.last_trade_bar) >= self.cooldown_bars
        current_position = self.get_position()
        
        # Emergency halt exit
        if self.trading_halted and current_position != 0:
            self.close_position("Halt Exit")
            return
        
        # Execute trades only on new candles
        if can_trade and not self.trading_halted and new_candle:
            if long_condition:
                if current_position <= 0:  # Not long or short
                    if current_position < 0:
                        self.close_position("Flip to Long")
                        time.sleep(1)
                    
                    if self.open_position("long", "Long Entry"):
                        self.last_trade_bar = self.bar_index
                        tick = mt5.symbol_info_tick(self.symbol)
                        self.entry_price = tick.ask if tick else None
                        self.highest_price_since_entry = self.entry_price
                        self.lowest_price_since_entry = self.entry_price
                        print(f"?? LONG ENTRY at {self.entry_price:.5f}")
            
            elif short_condition:
                if current_position >= 0:  # Not short or long
                    if current_position > 0:
                        self.close_position("Flip to Short")
                        time.sleep(1)
                    
                    if self.open_position("short", "Short Entry"):
                        self.last_trade_bar = self.bar_index
                        tick = mt5.symbol_info_tick(self.symbol)
                        self.entry_price = tick.bid if tick else None
                        self.highest_price_since_entry = self.entry_price
                        self.lowest_price_since_entry = self.entry_price
                        print(f"?? SHORT ENTRY at {self.entry_price:.5f}")
    
    def run(self):
        """Main strategy loop"""
        print("=" * 60)
        print(f"?? Starting Trading Algorithm")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Lot Size: {self.lot_size}")
        print(f"Magic Number: {self.magic_number}")
        print(f"Trailing Stop: {self.TRAILING_STOP_PCT}%")
        print(f"Daily Halt: {self.halt_threshold}%")
        print("=" * 60)
        
        while True:
            try:
                # Check connection
                if not self.check_connection():
                    print("?? Connection issue, retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                
                # Get market data
                df = self.get_market_data()
                if df is None or len(df) < 100:
                    time.sleep(5)
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Check for new candle
                new_candle = self.check_new_candle(df)
                
                if new_candle:
                    self.bar_index += 1
                    self.check_halt_system()
                    
                    # Check market conditions
                    # if not self.check_market_conditions():
                        # print("?? Market conditions not suitable, skipping...")
                        # time.sleep(2)
                        # continue
                    
                    # Calculate rules and signals
                    rules = self.calculate_rules(df)
                    if rules is not None:
                        total_signal, signal_confidence = self.calculate_signal(rules)
                        
                        # Execute trades
                        self.execute_trades(total_signal, signal_confidence, new_candle)
                        
                        # Print status
                        position = self.get_position()
                        account_info = mt5.account_info()
                        equity = account_info.equity if account_info else 0
                        
                        status = "?? LONG" if position > 0 else ("?? SHORT" if position < 0 else "? FLAT")
                        halt_status = " ?? HALTED" if self.trading_halted else ""
                        
                        print(f"Bar {self.bar_index} | {status}{halt_status} | Signal: {total_signal:.4f} | Conf: {signal_confidence:.4f} | Price: {current_price:.5f} | Equity: ${equity:.2f}")
                
                # Update trailing stop on every tick
                self.update_trailing_stop(current_price)
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n" + "=" * 60)
                print("?? Stopping algorithm...")
                print("=" * 60)
                break
            except Exception as e:
                print(f"? Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
        
        mt5.shutdown()
        print("? MT5 connection closed")

if __name__ == "__main__":
    # Initialize with your settings
    algo = TradingAlgorithm(
        symbol="BTCUSD",
        timeframe=mt5.TIMEFRAME_M30,
        lot_size=0.1
    )
    algo.run()



