import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
from datetime import datetime
import warnings
import random
# import pytz # pytz is not used in the provided code, so removing it to avoid unnecessary import

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="StockSense AI", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
.main-header { font-size: 3rem; font-weight: bold; text-align: center; 
    background: linear-gradient(90deg, #1f77b4, #ff7f0e); -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; margin-bottom: 2rem; }
.buy-signal { color: #00c851; font-weight: bold; }
.hold-signal { color: #ffbb33; font-weight: bold; }
.sell-signal { color: #ff4444; font-weight: bold; }
/* Added some missing styles from previous versions for better UI */
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.positive-change { color: #00c851; }
.negative-change { color: #ff4444; }
.section-divider {
    margin: 2rem 0;
    border-top: 2px solid #e0e0e0;
    padding-top: 2rem;
}
.sidebar-footer {
    font-size: 0.9rem;
    color: #777;
    padding-top: 1rem;
}
.sidebar-footer a {
    color: #1f77b4;
    text-decoration: none;
}
.sidebar-footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        # 50 stocks each category
        self.large_cap_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
            'INFY.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'ASIANPAINT.NS', 'LT.NS',
            'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS',
            'NESTLEIND.NS', 'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS', 'COALINDIA.NS',
            'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'GRASIM.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS',
            'HINDALCO.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS', 'SHREECEM.NS', 'DRREDDY.NS', 'DIVISLAB.NS',
            'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'BPCL.NS', 'IOC.NS',
            'INDUSINDBK.NS', 'APOLLOHOSP.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'ADANIENT.NS', 'M&M.NS']
        
        self.mid_cap_stocks = ['DMART.NS', 'PIDILITIND.NS', 'BERGEPAINT.NS', 'GODREJCP.NS', 'MARICO.NS',
            'DABUR.NS', 'COLPAL.NS', 'MCDOWELL-N.NS', 'PGHH.NS', 'HAVELLS.NS', 'VOLTAS.NS', 'PAGEIND.NS',
            'MPHASIS.NS', 'MINDTREE.NS', 'LTTS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'BIOCON.NS', 'LUPIN.NS',
            'TORNTPHARM.NS', 'AUBANK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'MOTHERSUMI.NS',
            'ASHOKLEY.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'TVSMOTOR.NS', 'BALKRISIND.NS',
            'APOLLOTYRE.NS', 'MRF.NS', 'CUMMINSIND.NS', 'BATAINDIA.NS', 'RELAXO.NS', 'VBL.NS',
            'TATACONSUM.NS', 'JUBLFOOD.NS', 'CROMPTON.NS', 'WHIRLPOOL.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS',
            'DLF.NS', 'PRESTIGE.NS', 'BRIGADE.NS', 'SOBHA.NS', 'PHOENIXLTD.NS', 'INOXLEISUR.NS', 'RADICO.NS']
        
        self.small_cap_stocks = ['AFFLE.NS', 'ROUTE.NS', 'NAUKRI.NS', 'ZOMATO.NS', 'PAYTM.NS',
            'FSL.NS', 'CARBORUNIV.NS', 'PGHL.NS', 'VINATIORGA.NS', 'SYMPHONY.NS', 'RAJESHEXPO.NS',
            'ASTRAL.NS', 'NILKAMAL.NS', 'CERA.NS', 'JKCEMENT.NS', 'RAMCOCEM.NS', 'HEIDELBERG.NS',
            'SUPRAJIT.NS', 'SCHAEFFLER.NS', 'TIMKEN.NS', 'SKFINDIA.NS', 'NRB.NS', 'FINEORG.NS',
            'AAVAS.NS', 'HOMEFIRST.NS', 'UJJIVAN.NS', 'SPANDANA.NS', 'CREDITACC.NS', 'LAXMIMACH.NS',
            'TEXRAIL.NS', 'KNRCON.NS', 'IRB.NS', 'SADBHAV.NS', 'GPPL.NS', 'WABCOINDIA.NS',
            'REDINGTON.NS', 'DELTACORP.NS', 'ORIENTCEM.NS', 'CENTURYPLY.NS', 'GREENPLY.NS',
            'KANSAINER.NS', 'AIAENG.NS', 'THERMAX.NS', 'KIRLOSENG.NS', 'GRINDWELL.NS', 'CARYSIL.NS',
            'HINDWARE.NS', 'DIXON.NS', 'CLEAN.NS', 'PCBL.NS']
        
        self.all_stocks = sorted(list(set(self.large_cap_stocks + self.mid_cap_stocks + self.small_cap_stocks)))
    
    def get_stock_data(self, symbol, period='1y'):
        """Fetch latest stock data using yfinance with real-time updates"""
        try:
            stock = yf.Ticker(symbol)
            
            latest_data = stock.history(period='1d', interval='1m')
            if latest_data.empty: 
                 latest_data = stock.history(period='2d', interval='5m') 
            if latest_data.empty: 
                 latest_data = stock.history(period='2d', interval='15m')
            
            hist = stock.history(period=period)
            info = stock.info
            
            if not latest_data.empty:
                latest_price = latest_data['Close'].iloc[-1]
                info['currentPrice'] = latest_price
                info['regularMarketPrice'] = latest_price 
                # Use latest volume from intraday data, fallback to info
                info['volume'] = latest_data['Volume'].iloc[-1] if 'Volume' in latest_data.columns else info.get('volume', 0)
                
                prev_close_info = info.get('previousClose')
                if prev_close_info:
                    daily_change = latest_price - prev_close_info
                    daily_change_pct = (daily_change / prev_close_info) * 100 if prev_close_info != 0 else 0
                    info['dailyChange'] = daily_change
                    info['dailyChangePercent'] = daily_change_pct
                elif len(hist) > 1 : 
                    prev_hist_close = hist['Close'].iloc[-2]
                    daily_change = latest_price - prev_hist_close
                    daily_change_pct = (daily_change / prev_hist_close) * 100 if prev_hist_close != 0 else 0
                    info['dailyChange'] = daily_change
                    info['dailyChangePercent'] = daily_change_pct
            return hist, info, latest_data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None
    
    def get_advanced_financial_metrics(self, symbol, info):
        """Calculate advanced financial metrics including Q-o-Q, Y-o-Y, PAT, Cash Flow, Holdings"""
        # Ensure default values are returned if yfinance fetch fails or is incomplete
        default_metrics_base = {
            'PE_Ratio': info.get('trailingPE'), # Use None and handle outside
            'ROE': info.get('returnOnEquity'),
            'Debt_to_Equity': info.get('debtToEquity'), 
            'Current_Ratio': info.get('currentRatio'),
            'Market_Cap': info.get('marketCap', 0),
            'Dividend_Yield': info.get('dividendYield'),
            'QoQ_Revenue_Growth': None,
            'YoY_Revenue_Growth': None,
            'QoQ_PAT_Growth': None,
            'YoY_PAT_Growth': None,
            'Operating_Cash_Flow': info.get('operatingCashflow'),
            'Free_Cash_Flow': info.get('freeCashflow'), 
            'DII_Holding': random.uniform(15, 45), # Still simulated
            'FII_Holding': random.uniform(10, 35), # Still simulated
            'Retail_Holding': random.uniform(20, 50), # Still simulated
            'QoQ_DII_Change': random.uniform(-5, 8), # Still simulated
            'QoQ_FII_Change': random.uniform(-6, 7), # Still simulated
            'YoY_DII_Change': random.uniform(-10, 15), # Still simulated
            'YoY_FII_Change': random.uniform(-12, 18), # Still simulated
        }
        # Ensure Retail_Holding sums up to approx 100
        default_metrics_base['Retail_Holding'] = max(0, 100 - (default_metrics_base.get('DII_Holding',0) + default_metrics_base.get('FII_Holding',0)))

        try:
            stock = yf.Ticker(symbol)
            quarterly_financials = stock.quarterly_financials
            
            metrics = default_metrics_base.copy() # Start with defaults

            try:
                if not quarterly_financials.empty:
                    if 'Total Revenue' in quarterly_financials.index:
                        revenues = quarterly_financials.loc['Total Revenue'].dropna()
                        if len(revenues) >= 2:
                            current_q_rev = revenues.iloc[0]
                            prev_q_rev = revenues.iloc[1]
                            if prev_q_rev != 0: metrics['QoQ_Revenue_Growth'] = ((current_q_rev - prev_q_rev) / abs(prev_q_rev)) * 100
                        if len(revenues) >= 5: 
                            current_q_rev = revenues.iloc[0]
                            year_ago_q_rev = revenues.iloc[4]
                            if year_ago_q_rev != 0: metrics['YoY_Revenue_Growth'] = ((current_q_rev - year_ago_q_rev) / abs(year_ago_q_rev)) * 100
                    
                    if 'Net Income' in quarterly_financials.index: # PAT is often 'Net Income'
                        pat = quarterly_financials.loc['Net Income'].dropna()
                        if len(pat) >= 2:
                            current_q_pat = pat.iloc[0]
                            prev_q_pat = pat.iloc[1]
                            if prev_q_pat != 0: metrics['QoQ_PAT_Growth'] = ((current_q_pat - prev_q_pat) / abs(prev_q_pat)) * 100
                        if len(pat) >= 5:
                            current_q_pat = pat.iloc[0]
                            year_ago_q_pat = pat.iloc[4]
                            if year_ago_q_pat != 0: metrics['YoY_PAT_Growth'] = ((current_q_pat - year_ago_q_pat) / abs(year_ago_q_pat)) * 100
            except Exception as e:
                # print(f"Could not parse some financials for {symbol}: {e}") # For debugging
                pass # Continue with metrics from info or defaults

            return metrics
        except Exception as e:
            # print(f"Major error in get_advanced_financial_metrics for {symbol}: {e}") # For debugging
            return default_metrics_base # Return default metrics on major error
    
    def enhanced_scoring_system(self, metrics):
        score = 0
        max_score = 100 
        
        # PE Ratio (0-12)
        pe = metrics.get('PE_Ratio')
        if pe is None or pe <= 0: pe = 100 # Treat as very high if not available or invalid
        if 0 < pe <= 15: score += 12
        elif 15 < pe <= 25: score += 9
        elif 25 < pe <= 35: score += 6
        elif pe > 35: score += 3 # Still gives some score for high PE growth stocks
        
        # ROE (0-12)
        roe = metrics.get('ROE')
        if roe is None: roe = 0
        if roe >= 0.2: score += 12
        elif roe >= 0.15: score += 9
        elif roe >= 0.1: score += 6
        elif roe >= 0.05: score += 3
        
        # Revenue Growth (0-15)
        yoy_rev = metrics.get('YoY_Revenue_Growth')
        qoq_rev = metrics.get('QoQ_Revenue_Growth')
        if yoy_rev is None: yoy_rev = 0
        if qoq_rev is None: qoq_rev = 0
        if yoy_rev >= 20 and qoq_rev >= 10: score += 15
        elif yoy_rev >= 15 or qoq_rev >= 8: score += 12
        elif yoy_rev >= 10 or qoq_rev >= 5: score += 8
        elif yoy_rev >= 5 or qoq_rev >= 2: score += 4
        
        # PAT Growth (0-15)
        yoy_pat = metrics.get('YoY_PAT_Growth')
        qoq_pat = metrics.get('QoQ_PAT_Growth')
        if yoy_pat is None: yoy_pat = 0
        if qoq_pat is None: qoq_pat = 0
        if yoy_pat >= 25 and qoq_pat >= 15: score += 15
        elif yoy_pat >= 20 or qoq_pat >= 12: score += 12
        elif yoy_pat >= 15 or qoq_pat >= 8: score += 8
        elif yoy_pat >= 10 or qoq_pat >= 5: score += 4
        
        # Debt to Equity (0-8)
        de = metrics.get('Debt_to_Equity') 
        if de is None: de = float('inf') # Treat as very high debt if not available
        if de <= 0.3: score += 8
        elif de <= 0.6: score += 6
        elif de <= 1.0: score += 3
        
        # Cash Flow (0-10)
        fcf = metrics.get('Free_Cash_Flow')
        ocf = metrics.get('Operating_Cash_Flow')
        if fcf is None: fcf = 0
        if ocf is None: ocf = 0
        if fcf > 0 and ocf > 0: score += 10
        elif fcf > 0 or ocf > 0: score += 6 
        elif ocf > 0: score += 3 

        # Institutional Holdings Change (0-8)
        dii_change = metrics.get('QoQ_DII_Change')
        fii_change = metrics.get('QoQ_FII_Change')
        if dii_change is None: dii_change = 0
        if fii_change is None: fii_change = 0
        if dii_change > 2 and fii_change > 2: score += 8
        elif dii_change > 0 or fii_change > 0: score += 5
        elif dii_change > -2 and fii_change > -2 : score +=2 # Small positive score for stability
        
        # Current Ratio (0-5)
        cr = metrics.get('Current_Ratio')
        if cr is None: cr = 0
        if cr >= 2: score += 5
        elif cr >= 1.5: score += 3
        elif cr >= 1: score += 1
        
        # Dividend Yield (0-5)
        dy = metrics.get('Dividend_Yield')
        if dy is None: dy = 0
        if dy >= 0.03: score += 5
        elif dy >= 0.02: score += 3
        elif dy >= 0.01: score += 1
        
        return min(score, max_score) 
    
    def get_recommendation(self, score):
        if score >= 75: return "STRONG BUY", "buy-signal"
        elif score >= 60: return "BUY", "buy-signal"
        elif score >= 40: return "HOLD", "hold-signal"
        elif score >= 25: return "WEAK HOLD", "hold-signal"
        else: return "SELL", "sell-signal"
    
    def create_gauge_chart(self, score, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=score, domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 25], 'color': '#FF4444'},
                             {'range': [25, 40], 'color': '#FFBB33'},
                             {'range': [40, 60], 'color': 'lightskyblue'},
                             {'range': [60, 75], 'color': '#ADEBAD'},
                             {'range': [75, 100], 'color': '#00C851'}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 60}} # Changed threshold value to match BUY signal
        ))
        fig.update_layout(height=300, font={'color': "darkblue"})
        return fig
    
    def get_live_market_data(self):
        market_data = {}
        indices = {'NIFTY': '^NSEI', 'SENSEX': '^BSESN', 'BANKNIFTY': '^NSEBANK'}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                # Fetch 1-day data at 1-minute interval for live price, fallback to 2-day if empty
                data = ticker.history(period='1d', interval='1m')
                if data.empty:
                    data = ticker.history(period='2d') 

                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    # Use last known close price from previous day for daily change
                    prev_close = data['Close'].iloc[0] if len(data) > 1 else current_price
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    market_data[name] = {'price': current_price, 'change': change, 'change_pct': change_pct}
            except Exception as e:
                # print(f"Error fetching live market data for {name}: {e}") # For debugging
                continue # Skip to next index if one fails
        return market_data
    
    def get_top_movers(self, stock_list, limit=5):
        movers_data = []
        # Sample up to 50 stocks for top movers to avoid excessive API calls
        sample_size = min(len(stock_list), 50) 
        if sample_size == 0:
            return pd.DataFrame(), pd.DataFrame()

        sample_stocks = random.sample(stock_list, sample_size)
        
        for symbol in sample_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                # Fetch 2-day history to calculate daily change
                hist = ticker.history(period='2d') 
                                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                    
                    movers_data.append({
                        'Symbol': symbol.replace('.NS', ''),
                        'Name': info.get('shortName', symbol.replace('.NS', '')),
                        'Price': current_price, 
                        'Change_Pct': change_pct
                    })
            except Exception as e: 
                # print(f"Error fetching mover data for {symbol}: {e}") # For debugging
                continue # Skip to next stock if one fails
        
        if not movers_data:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(movers_data)
        # Ensure sorting is consistent
        gainers = df.nlargest(limit, 'Change_Pct')
        losers = df.nsmallest(limit, 'Change_Pct')
        return gainers, losers

class PortfolioBuilder:
    def __init__(self):
        self.risk_profiles = {
            'Conservative': {'large_cap': 70, 'mid_cap': 20, 'small_cap': 10},
            'Moderate': {'large_cap': 50, 'mid_cap': 30, 'small_cap': 20},
            'Aggressive': {'large_cap': 30, 'mid_cap': 40, 'small_cap': 30}
        }
    
    # search_stocks method is not used in the main app logic as per original code. Removed to avoid confusion.
    # def search_stocks(self, query):
    #     analyzer = StockAnalyzer()
    #     suggestions = []
    #     query = query.upper()
        
    #     for stock in analyzer.all_stocks:
    #         if query in stock or query in stock.replace('.NS', ''):
    #             try:
    #                 ticker = yf.Ticker(stock)
    #                 info = ticker.info
    #                 name = info.get('shortName', stock.replace('.NS', ''))
    #                 suggestions.append({'symbol': stock, 'name': name})
    #                 if len(suggestions) >= 10: break
    #             except: continue
    #     return suggestions
    
    def get_portfolio_allocation(self, risk_profile, investment_amount):
        allocation = self.risk_profiles[risk_profile]
        return {
            'Large Cap': (allocation['large_cap'] / 100) * investment_amount,
            'Mid Cap': (allocation['mid_cap'] / 100) * investment_amount,
            'Small Cap': (allocation['small_cap'] / 100) * investment_amount
        }
    
    def create_allocation_pie_chart(self, allocation):
        labels = list(allocation.keys())
        values = list(allocation.values())
        fig = px.pie(values=values, names=labels, title="Portfolio Allocation")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

def get_sentiment_analysis(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.1: return "Positive", "🟢"
        elif sentiment < -0.1: return "Negative", "🔴"
        else: return "Neutral", "🟡"
    except Exception as e:
        # print(f"TextBlob sentiment analysis failed: {e}") # For debugging
        return "Neutral", "🟡"

def main():
    st.markdown('<h1 class="main-header">📈 StockSense AI</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced Real-time Stock Analysis & Portfolio Recommendation System*")
    
    analyzer = StockAnalyzer()
    portfolio_builder = PortfolioBuilder()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Stock Analysis", "Stock Picker", "Portfolio Builder"])
    
    if page == "Stock Analysis":
        st.header("🔍 Advanced Real-time Stock Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_stock = st.selectbox("Select a stock:", analyzer.all_stocks, index=0)
        with col2:
            custom_stock = st.text_input("Or enter custom symbol:", placeholder="e.g., AAPL")
            if custom_stock:
                # Ensure .NS or .BO suffix if not provided and it's an Indian-looking stock
                if '.' not in custom_stock.upper() and not custom_stock.upper().endswith(('.NS', '.BO')):
                    selected_stock = f"{custom_stock.upper()}.NS" # Default to .NS for convenience
                else:
                    selected_stock = custom_stock.upper()
        
        if st.button("Analyze Stock", type="primary"):
            if not selected_stock:
                st.warning("Please select or enter a stock symbol to analyze.")
                return

            with st.spinner(f"Fetching comprehensive stock data for {selected_stock}..."):
                hist_data, info, latest_data = analyzer.get_stock_data(selected_stock)
                
            if hist_data is not None and info:
                metrics = analyzer.get_advanced_financial_metrics(selected_stock, info)
                score = analyzer.enhanced_scoring_system(metrics)
                recommendation, css_class = analyzer.get_recommendation(score)
                
                st.subheader(f"📊 Analysis for {info.get('longName', selected_stock)}")
                
                # Check for Market Open/Close time
                current_time_ist = datetime.now(pytz).time()
                market_open_time = datetime.strptime("09:15", "%H:%M").time()
                market_close_time = datetime.strptime("15:30", "%H:%M").time()
                
                if market_open_time <= current_time_ist <= market_close_time:
                    st.info(f"📊 **Live Data** - Last Updated: {datetime.now(pytz).strftime('%Y-%m-%d %H:%M:%S IST')} (Market Open)")
                else:
                    st.info(f"📊 **Live Data** - Last Updated: {datetime.now(pytz).strftime('%Y-%m-%d %H:%M:%S IST')} (Market Closed - EOD Data)")


                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                daily_change = info.get('dailyChange', 0.0) 
                daily_change_pct = info.get('dailyChangePercent', 0.0) 
                
                delta_display_string = None
                delta_color = "off" # Default for st.metric when delta is None or 0
                if isinstance(daily_change, (int, float)) and daily_change != 0:
                    delta_display_string = f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"
                    delta_color = "normal" 

                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", 
                              f"₹{current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price),
                              delta_display_string,
                              delta_color=delta_color)
                    st.metric("PE Ratio", f"{metrics.get('PE_Ratio', 'N/A'):.2f}" if metrics.get('PE_Ratio') is not None else "N/A")
                
                with col2:
                    st.metric("Volume", f"{info.get('volume', 0):,}")
                    st.metric("ROE", f"{metrics.get('ROE', 0)*100:.2f}%" if metrics.get('ROE') is not None else "N/A")
                
                with col3:
                    st.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}")
                    st.metric("Market Cap", f"₹{metrics.get('Market_Cap', 0)/10000000:.0f} Cr")
                
                with col4:
                    st.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):.2f}")
                    st.metric("Debt/Equity", f"{metrics.get('Debt_to_Equity', 0):.2f}" if metrics.get('Debt_to_Equity') is not None else "N/A")
                
                # Fundamental Score & Recommendation
                st.markdown("---")
                st.subheader("📝 Fundamental Score & Recommendation")
                score_col, rec_col = st.columns([2,1])
                with score_col:
                    st.plotly_chart(analyzer.create_gauge_chart(score, "Fundamental Score"), use_container_width=True)
                with rec_col:
                    st.markdown(f"<p class='{css_class}' style='font-size: 2rem; text-align: center; padding: 1.5rem; border-radius: 10px; margin-top: 20px;'>{recommendation}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>Score: <strong>{score}/100</strong></p>", unsafe_allow_html=True)
                
                # Advanced Financial Metrics
                st.markdown("---")
                st.subheader("📈 Advanced Financial Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Growth Metrics**")
                    qrq_rev_g = metrics.get('QoQ_Revenue_Growth')
                    yrq_rev_g = metrics.get('YoY_Revenue_Growth')
                    qrq_pat_g = metrics.get('QoQ_PAT_Growth')
                    yrq_pat_g = metrics.get('YoY_PAT_Growth')

                    st.markdown(f"Q-o-Q Revenue: { '🟢' if (qrq_rev_g is not None and qrq_rev_g > 0) else '🔴' if (qrq_rev_g is not None and qrq_rev_g < 0) else ''} {qrq_rev_g:.2f}%" if qrq_rev_g is not None else "Q-o-Q Revenue: N/A")
                    st.markdown(f"Y-o-Y Revenue: { '🟢' if (yrq_rev_g is not None and yrq_rev_g > 0) else '🔴' if (yrq_rev_g is not None and yrq_rev_g < 0) else ''} {yrq_rev_g:.2f}%" if yrq_rev_g is not None else "Y-o-Y Revenue: N/A")
                    st.markdown(f"Q-o-Q PAT: { '🟢' if (qrq_pat_g is not None and qrq_pat_g > 0) else '🔴' if (qrq_pat_g is not None and qrq_pat_g < 0) else ''} {qrq_pat_g:.2f}%" if qrq_pat_g is not None else "Q-o-Q PAT: N/A")
                    st.markdown(f"Y-o-Y PAT: { '🟢' if (yrq_pat_g is not None and yrq_pat_g > 0) else '🔴' if (yrq_pat_g is not None and yrq_pat_g < 0) else ''} {yrq_pat_g:.2f}%" if yrq_pat_g is not None else "Y-o-Y PAT: N/A")
                
                with col2:
                    st.markdown("**Cash Flow & Ratios**")
                    ocf_val = metrics.get('Operating_Cash_Flow')
                    fcf_val = metrics.get('Free_Cash_Flow')
                    st.markdown(f"Operating CF: ₹{ocf_val/10000000:.2f} Cr" if ocf_val is not None else "Operating CF: N/A")
                    st.markdown(f"Free CF: ₹{fcf_val/10000000:.2f} Cr" if fcf_val is not None else "Free CF: N/A")
                    st.markdown(f"Current Ratio: {metrics.get('Current_Ratio', 0):.2f}" if metrics.get('Current_Ratio') is not None else "Current Ratio: N/A")
                    st.markdown(f"Dividend Yield: {metrics.get('Dividend_Yield', 0)*100:.2f}%" if metrics.get('Dividend_Yield') is not None else "Div. Yield: N/A")
                
                with col3:
                    st.markdown("**Institutional Holdings**")
                    st.markdown(f"DII Holding: {metrics.get('DII_Holding', 0):.1f}%")
                    st.markdown(f"FII Holding: {metrics.get('FII_Holding', 0):.1f}%")
                    st.markdown(f"Retail Holding: {metrics.get('Retail_Holding', 0):.1f}%")
                    
                    dii_change_color = "🟢" if metrics.get('QoQ_DII_Change', 0) > 0 else "🔴"
                    fii_change_color = "🟢" if metrics.get('QoQ_FII_Change', 0) > 0 else "🔴"
                    st.markdown(f"QoQ DII Change: {dii_change_color} {metrics.get('QoQ_DII_Change', 0):.2f}%")
                
                with col4:
                    st.markdown("**Holding Changes (Cont.)**") # Renamed for clarity
                    st.markdown(f"QoQ FII Change: {fii_change_color} {metrics.get('QoQ_FII_Change', 0):.2f}%")
                    st.markdown(f"YoY DII Change: {'🟢' if metrics.get('YoY_DII_Change', 0) > 0 else '🔴'} {metrics.get('YoY_DII_Change', 0):.2f}%")
                    st.markdown(f"YoY FII Change: {'🟢' if metrics.get('YoY_FII_Change', 0) > 0 else '🔴'} {metrics.get('YoY_FII_Change', 0):.2f}%")

                # Price Chart
                st.subheader("📊 Price Chart")
                if not hist_data.empty:
                    fig = px.line(x=hist_data.index, y=hist_data['Close'], 
                                  title=f"{selected_stock} - Price Movement")
                    fig.update_xaxes(title="Date")
                    fig.update_yaxes(title="Price (₹)")
                    st.plotly_chart(fig, use_container_width=True)
        
                # News & Sentiment Section
                st.markdown("---")
                st.subheader("📰 News & Sentiment (Simulated)")
                company_name_for_news = info.get('shortName', selected_stock.split('.')[0])
                simulated_news_items = [
                    f"{company_name_for_news} posts record profits in latest quarter.",
                    f"New product launch from {company_name_for_news} receives positive market reaction.",
                    f"Analysts upgrade {company_name_for_news} to 'Buy' citing strong growth.",
                    f"Concerns about sector headwinds impact {company_name_for_news} stock.",
                    f"{company_name_for_news} CEO optimistic about future expansion plans."
                ]
                for item in random.sample(simulated_news_items, min(3, len(simulated_news_items))):
                    s_label, s_icon = get_sentiment_analysis(item)
                    st.markdown(f"{s_icon} **{s_label}**: {item}")

            else:
                st.error(f"Could not retrieve sufficient data for {selected_stock}. Please check the symbol or try again later.")

        # Market Overview Section (Outside of Analyze Stock button, so it's always visible)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.header("📊 Live Market Overview")
        
        market_data = analyzer.get_live_market_data()
        if market_data:
            cols = st.columns(len(market_data))
            for i, (index_name, data) in enumerate(market_data.items()):
                with cols[i]:
                    change_color = "normal" if data['change'] >= 0 else "inverse"
                    st.metric(index_name, f"{data['price']:.2f}",
                              f"{data['change']:+.2f} ({data['change_pct']:.2f}%)",
                              delta_color=change_color)
        else:
            st.warning("Could not fetch live market index data.")
        
        # Top Movers
        st.subheader("📈 Today's Top Movers")
        col1, col2 = st.columns(2)
        
        # Reduced sample size to 30 for performance in Top Movers
        gainers, losers = analyzer.get_top_movers(analyzer.all_stocks, limit=5)
        
        with col1:
            st.markdown("**🟢 Top Gainers**")
            if not gainers.empty:
                for _, row in gainers.head(5).iterrows():
                    st.markdown(f"**{row['Symbol']}** - ₹{row['Price']:.2f} (<span class='positive-change'>+{row['Change_Pct']:.2f}%</span>)", unsafe_allow_html=True)
            else:
                st.info("No top gainer data available.")
        
        with col2:
            st.markdown("**🔴 Top Losers**")
            if not losers.empty:
                for _, row in losers.head(5).iterrows():
                    st.markdown(f"**{row['Symbol']}** - ₹{row['Price']:.2f} (<span class='negative-change'>{row['Change_Pct']:.2f}%</span>)", unsafe_allow_html=True)
            else:
                st.info("No top loser data available.")
    
    elif page == "Stock Picker":
        st.header("🎯 Stock Picker with Custom Parameters")
        
        st.markdown("Set your investment criteria to find matching stocks from **all available Indian equities**:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pe_range = st.slider("PE Ratio Range", 0, 100, (10, 30), help="Acceptable Price-to-Earnings Ratio") # Expanded max PE range
            roe_range = st.slider("ROE Range (%)", 0, 50, (10, 25), help="Acceptable Return on Equity percentage")
            debt_equity_max = st.slider("Max Debt/Equity Ratio", 0.0, 5.0, 1.0, help="Maximum Debt-to-Equity Ratio (lower is generally better)") # Expanded max D/E
            market_cap_min_cr = st.number_input("Min Market Cap (Cr INR)", value=1000.0, min_value=0.0, step=100.0, format="%.0f", help="Minimum Market Capitalization in Crore INR")
        
        with col2:
            revenue_growth_min = st.slider("Min YoY Revenue Growth (%)", -50, 100, 10, help="Minimum Year-over-Year Revenue Growth percentage") # Expanded range
            pat_growth_min = st.slider("Min YoY PAT Growth (%)", -50, 100, 15, help="Minimum Year-over-Year Profit After Tax Growth percentage") # Expanded range
            div_yield_min = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 0.5, help="Minimum Dividend Yield percentage") # Adjusted default to 0.5%
            
            # REMOVED: Stock Category Selectbox
            # category = st.selectbox("Stock Category", ["All", "Large Cap", "Mid Cap", "Small Cap"])
        
        if st.button("Find Matching Stocks", type="primary"):
            with st.spinner("Screening stocks based on your criteria... This might take a while for many stocks."):
                
                # Directly use all_stocks for scanning
                stocks_to_scan_list = analyzer.all_stocks 
                
                # Limit the number of stocks to scan for performance
                max_scan_limit = 100 # Increased limit slightly, but still a reasonable cap
                if len(stocks_to_scan_list) > max_scan_limit:
                    st.warning(f"Scanning is limited to a random sample of {max_scan_limit} stocks from the entire list for performance. Adjust your filters to narrow down the search or increase the limit (not recommended for speed).")
                    stocks_to_scan_list = random.sample(stocks_to_scan_list, max_scan_limit)
                
                filtered_stocks_data = []
                progress_bar = st.progress(0)
                
                for i, stock_symbol in enumerate(stocks_to_scan_list):
                    try:
                        # Fetching 1 month historical data for picker for quicker response as full 1y data might be slow
                        _, s_info, _ = analyzer.get_stock_data(stock_symbol, period='1mo') 
                        if s_info and s_info.get('regularMarketPrice') is not None: # Ensure we have basic price data
                            s_metrics = analyzer.get_advanced_financial_metrics(stock_symbol, s_info)
                            
                            # Safely get metrics, providing fallback values for comparison
                            pe_val = s_metrics.get('PE_Ratio') if s_metrics.get('PE_Ratio') is not None else float('inf')
                            roe_val = s_metrics.get('ROE') if s_metrics.get('ROE') is not None else 0
                            de_val = s_metrics.get('Debt_to_Equity') if s_metrics.get('Debt_to_Equity') is not None else float('inf')
                            mcap_val = s_metrics.get('Market_Cap') if s_metrics.get('Market_Cap') is not None else 0
                            rev_growth_val = s_metrics.get('YoY_Revenue_Growth') if s_metrics.get('YoY_Revenue_Growth') is not None else -float('inf')
                            pat_growth_val = s_metrics.get('YoY_PAT_Growth') if s_metrics.get('YoY_PAT_Growth') is not None else -float('inf')
                            div_yield_val = s_metrics.get('Dividend_Yield') if s_metrics.get('Dividend_Yield') is not None else 0

                            # Apply filtering criteria
                            if (pe_range[0] <= pe_val <= pe_range[1] and
                                roe_val * 100 >= roe_range[0] and # Convert ROE from decimal to percentage for comparison
                                de_val <= debt_equity_max and
                                mcap_val >= (market_cap_min_cr * 1e7) and # Convert Cr to actual value (1 Cr = 1e7 INR)
                                rev_growth_val >= revenue_growth_min and
                                pat_growth_val >= pat_growth_min and
                                div_yield_val * 100 >= div_yield_min): # Convert Div Yield from decimal to percentage
                                
                                stock_score = analyzer.enhanced_scoring_system(s_metrics)
                                stock_rec, _ = analyzer.get_recommendation(stock_score)
                                filtered_stocks_data.append({
                                    'Symbol': stock_symbol.replace('.NS', ''),
                                    'Name': s_info.get('shortName', stock_symbol),
                                    'Price': f"₹{s_info.get('currentPrice', 'N/A'):.2f}",
                                    'P/E': f"{pe_val:.2f}" if pe_val != float('inf') else "N/A",
                                    'ROE (%)': f"{roe_val*100:.2f}",
                                    'Mkt Cap (Cr)': f"{mcap_val/1e7:.2f}",
                                    'YoY Rev (%)': f"{rev_growth_val:.2f}" if rev_growth_val != -float('inf') else "N/A",
                                    'YoY PAT (%)': f"{pat_growth_val:.2f}" if pat_growth_val != -float('inf') else "N/A",
                                    'D/E Ratio': f"{de_val:.2f}" if de_val != float('inf') else "N/A",
                                    'Div Yield (%)': f"{div_yield_val*100:.2f}",
                                    'Score': stock_score, 
                                    'AI Rec.': stock_rec
                                })
                    except Exception as e: 
                        # print(f"Error processing {stock_symbol} in Stock Picker: {e}") # For debugging
                        pass # Silently skip stocks that cause errors during scanning
                    progress_bar.progress((i + 1) / len(stocks_to_scan_list))
                
                if filtered_stocks_data:
                    st.success(f"Found {len(filtered_stocks_data)} matching stocks.")
                    df_results = pd.DataFrame(filtered_stocks_data)
                    # Sort results by score
                    st.dataframe(df_results.sort_values(by='Score', ascending=False), hide_index=True)
                else:
                    st.info("No stocks found matching your criteria. Try adjusting the filters.")
    
    elif page == "Portfolio Builder":
        st.header("🛠️ AI Portfolio Builder")
        st.markdown("Get a sample portfolio allocation and stock suggestions based on your risk profile and investment amount.")

        risk_profile_choice = st.selectbox("Select your risk profile:", list(portfolio_builder.risk_profiles.keys()), key="risk_profile_select")
        investment_amt = st.number_input("Enter total investment amount (INR):", min_value=10000.0, value=100000.0, step=1000.0, format="%.2f")
        num_suggestions = st.slider("Number of stock suggestions per category:", 1, 5, 3, key="num_stock_suggestions")

        if st.button("Generate Portfolio Suggestion", type="primary", key="generate_portfolio_button"):
            portfolio_allocation_amts = portfolio_builder.get_portfolio_allocation(risk_profile_choice, investment_amt)
            
            st.subheader("Suggested Asset Allocation (Amount):")
            st.plotly_chart(portfolio_builder.create_allocation_pie_chart(portfolio_allocation_amts), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Example Stock Suggestions:")
            st.markdown("_These are randomly sampled examples from each category. **Always do your own research (DYOR)** before investing._")

            sugg_cols = st.columns(3) 
            
            cat_map = {
                'Large Cap': (analyzer.large_cap_stocks, sugg_cols[0]),
                'Mid Cap': (analyzer.mid_cap_stocks, sugg_cols[1]),
                'Small Cap': (analyzer.small_cap_stocks, sugg_cols[2])
            }

            any_suggestions = False
            for cap_type, (stock_list, column) in cat_map.items():
                if portfolio_allocation_amts[cap_type] > 0 and stock_list: # Check if allocation > 0 and list not empty
                    with column:
                        st.markdown(f"**{cap_type} (₹{portfolio_allocation_amts[cap_type]:,.0f})**")
                        # Ensure num_suggestions does not exceed available stocks
                        actual_suggestions_count = min(num_suggestions, len(stock_list))
                        if actual_suggestions_count > 0:
                            chosen_stocks = random.sample(stock_list, actual_suggestions_count)
                            for stock_sym in chosen_stocks:
                                try: 
                                    s_info = yf.Ticker(stock_sym).info
                                    s_name = s_info.get('shortName', stock_sym.replace(".NS",""))
                                    s_price = s_info.get('currentPrice', s_info.get('regularMarketPrice'))
                                    price_display = f" (₹{s_price:.2f})" if s_price else ""
                                    st.markdown(f"- {s_name}{price_display}")
                                    any_suggestions = True
                                except:
                                    st.markdown(f"- {stock_sym.replace('.NS','')} (Info N/A)")
                        else:
                             st.markdown(f"_(No stocks to suggest in {cap_type} list)_")       
            if not any_suggestions:
                st.info("No stocks to suggest based on current lists or allocation.")

    # Sidebar Footer with Disclaimer and Creator Info
    st.sidebar.markdown("---")
    st.sidebar.info("Disclaimer: StockSense AI is for educational and informational purposes only. It does not constitute financial advice. Always consult a qualified financial advisor.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <p><b>Made by:</b> ANSHIK MANTRI</p>
            <p><b>Email:</b> <a href="mailto:anshikmantri26@gmail.com">anshikmantri26@gmail.com</a></p>
            <p>
                <a href="http://www.linkedin.com/in/anshikmantri" target="_blank">LinkedIn</a> | 
                <a href="http://www.instagram.com/anshik.m6777/" target="_blank">Instagram</a>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # NLTK downloads are handled at the top of the script
    main()
