# #########################
# Required libraries
# #########################

import yfinance as yf
import QuantLib as ql
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
# Charting library
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# UI widgets
import ipywidgets as widgets
import panel as pn
from IPython.display import display, clear_output
from bokeh.models.widgets.tables import NumberFormatter
from functools import partial

# Needed to display Plotly charts inside widgets in Google Colab
from google.colab import output
output.enable_custom_widget_manager()
pn.extension("tabulator")


def bsm(position, spotPrice = None, atTime = None, ir = 0, iv = None, multiplier = 100, optionPrice = None, resultType = "pnl"):
   """
   Computes the P&L value or the Implied Volatilty of a set of option contracts.

   Parameters:
   - position: DataFrame with option details (optionType, strikePrice, expiryDttm, qty, tradePrice).
   - spotPrice: Current price of the underlying asset. Can be an array or a single value.
   - atTime: Time at which the contract is evaluated.
   - ir: Interest rate.
   - iv: Implied volatility for each option leg.
   - multiplier: Contract multiplier.
   - resultType: the type of output bbeing calculated. Valid values: 'pnl' or 'ImpliedVol' (default: 'pnl')
   - optionPrice: Array/List containing the trade price for each option. Required if resultType = 'ImpliedVol' 

   Returns:
   - The P&L value of the entire position evaluated for each value of spotPrice. (If resultType = 'pnl')
   - The implied volatility for each contract in the position dataframe. (If resultType = 'ImpliedVol')
   """

   # Convert spotPrice to a numpy array for vectorized operations:
   if not isinstance(spotPrice, (list, tuple, np.ndarray, pd.Series)):
      # spotPrice is a scalar value, convert to numpy array and add 0.0 to cast the values to np.float64
      spotPrice = np.array([spotPrice]) + 0.0

   if not isinstance(spotPrice, np.ndarray):
      # spotPrice is an array/list, convert to numpy array and add 0.0 to cast the values to np.float64
      spotPrice = np.array(spotPrice) + 0.0

   # Setup the interest rate term structure
   calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
   rate = ql.SimpleQuote(ir)
   rate_handle = ql.QuoteHandle(rate)
   day_count = ql.Actual365Fixed()

   # Initialize result array with zeros
   if resultType == "ImpliedVol":
      result = np.zeros(len(position))
   else:
      result = np.zeros_like(spotPrice)

   # Iterate over position rows to build each option
   #position = position.reset_index()
   for idx, row in position.iterrows():
      # Determine the option type
      option_type = ql.Option.Put if row['optionType'].lower() == 'put' else ql.Option.Call
      # Get the strike price
      strike = row['strikePrice']
      # Set the Expiry Date
      expiry_date = ql.Date(row['expiryDttm'].day, row['expiryDttm'].month, row['expiryDttm'].year)


      # Setting up the evaluation date
      if atTime is None:
         today = ql.Date().todaysDate()
      else:
         if resultType == "ImpliedVol":
            today = ql.Date(atTime[idx].day, atTime[idx].month, atTime[idx].year)
         else:
            today = ql.Date(atTime.day, atTime.month, atTime.year)
      ql.Settings.instance().evaluationDate = today

      #ttm = day_count.yearFraction(today, expiry_date)  # Now includes time portion
      dte = day_count.dayCount(today, expiry_date)

      # Setup Yield Term Structure
      flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate_handle, day_count))
      dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today, 0, day_count))

      # Use provided implied volatility or fallback to a default (20%)
      try:
         volatility = iv[idx]
      except:
         #print("Exception!")
         volatility = 0.2

      #print(f"idx: {idx}  IV:{volatility}")

      vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, ql.QuoteHandle(ql.SimpleQuote(volatility)), day_count))

      # Setup Black-Scholes-Merton process
      spot = ql.SimpleQuote(spotPrice[idx if resultType == "ImpliedVol" else 0])
      bsm_process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), dividend_yield, flat_ts, vol_handle)

      # Setup the European option object
      payoff = ql.PlainVanillaPayoff(option_type, strike)
      exercise = ql.EuropeanExercise(expiry_date)
      european_option = ql.VanillaOption(payoff, exercise)
      european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

      # Define functions used for vectorized operations
      def get_price(spotPrice):
         spot.setValue(spotPrice)
         return european_option.NPV()

      def get_theta(spotPrice):
         spot.setValue(spotPrice)
         return european_option.thetaPerDay()

      def get_delta(spotPrice):
         spot.setValue(spotPrice)
         return european_option.delta()

      def get_gamma(spotPrice):
         spot.setValue(spotPrice)
         return european_option.gamma()

      def get_vega(spotPrice):
         spot.setValue(spotPrice)
         # Return the sensitivity to 1% change in IV
         return 0.01 * european_option.vega()

      def get_wvega(spotPrice, dte):
         return get_vega(spotPrice) * np.sqrt(30/dte)

      def get_rho(spotPrice):
         spot.setValue(spotPrice)
         return 0.01 * european_option.rho()

      values = np.zeros_like(spotPrice)
      # Special case for the expiration day
      if today == expiry_date:
         # Option has expired. Either zero or intrinsic value.
         option_prices = np.where(option_type == ql.Option.Put, np.maximum(strike - spotPrice, 0), np.maximum(spotPrice - strike, 0))
         if resultType == "pnl":
            values = option_prices - row['tradePrice']
         # elif resultType == "delta":
         #     # Option has expired. Either zero or +/- 1.
         #     values = np.sign(2*int(option_type == ql.Option.Call)-1) * np.sign(option_prices)
         else:
            # All other greeks are zero
            values = np.zeros_like(spotPrice)
      elif today < expiry_date:
         if resultType == "pnl":
            option_prices = np.array([get_price(s) for s in spotPrice])
            #print(f"option_prices: {option_prices}")
            values = option_prices - row['tradePrice']
         elif resultType == "theta":
            values = np.array([get_theta(s) for s in spotPrice])
         elif resultType == "delta":
            values = np.array([get_delta(s) for s in spotPrice])
         elif resultType == "gamma":
            values = np.array([get_gamma(s) for s in spotPrice])
         elif resultType == "vega":
            values = np.array([get_vega(s) for s in spotPrice])
         elif resultType == "wvega":
            values = np.array([get_wvega(s, dte) for s in spotPrice])
         elif resultType == "rho":
            values = np.array([get_rho(s) for s in spotPrice])

      if resultType == "ImpliedVol":
         try:
            result[idx] = european_option.impliedVolatility(optionPrice[idx], bsm_process)
         except:
            result[idx] = np.NaN
      else:
         # Adjust result based on quantity and multiplier
         result += values * row['qty'] * multiplier

   return result


def update_plot(position = None
                , eval_time = None
                , expiry_dttm = None
                , iv_reference = None
                , iv_adj = 0
                , ir = 0
                , N_spot = 301
                , spotPrice = None
                , price_range = [-0.2, 0.2]
                , skew_dict = dict()
                , wVega = True
                , showLegend = True
                , showPNLOpen = True
                , outputWidget = widgets.Output()
                ):
   with outputWidget:
      outputWidget.clear_output(wait = True)

      if spotPrice is None:
         spotPrice = position["spotPrice"].min()

      # Spot price range
      spotRange = np.linspace(start = spotPrice * (1+price_range[0]), stop = spotPrice * (1+price_range[1]), num = N_spot)

      if iv_reference is None:
         iv_reference = position["IV.open"]

      openDTE = position["openDTE"].min().days
      if expiry_dttm is None:
         T_expiry = position["expiryDttm"].min() #- timedelta(minutes = 1)
      else:
         T_expiry = expiry_dttm
      T_open = position["openDttm"].min()
      T_0 = min(datetime.combine(date.today(), T_expiry.time()) , T_expiry)
      T_n = min(eval_time, T_expiry)



      start = datetime.now()
      data = pd.DataFrame(index = range(0, N_spot))
      data["spot"] = spotRange
      # Compute the P&L
      data["P&L (Open)"]     = bsm(position, spotPrice = spotRange, atTime = T_open  , ir = ir, iv = position["IV.open"]  , resultType = "pnl")
      data["P&L (Expiry)"]   = bsm(position, spotPrice = spotRange, atTime = T_expiry, ir = ir, iv = iv_reference + iv_adj, resultType = "pnl")
      data["P&L (@ Now)"]    = bsm(position, spotPrice = spotRange, atTime = T_0     , ir = ir, iv = iv_reference + iv_adj, resultType = "pnl")
      data[f"P&L (@ {T_n})"] = bsm(position, spotPrice = spotRange, atTime = T_n     , ir = ir, iv = iv_reference + iv_adj, resultType = "pnl")
      # Compute the greeks
      data["theta"] = bsm(position, spotPrice = spotRange, atTime = T_n, ir = ir, iv = iv_reference + iv_adj, resultType = "theta")
      data["delta"] = bsm(position, spotPrice = spotRange, atTime = T_n, ir = ir, iv = iv_reference + iv_adj, resultType = "delta")
      data["gamma"] = bsm(position, spotPrice = spotRange, atTime = T_n, ir = ir, iv = iv_reference + iv_adj, resultType = "gamma")
      data["vega"]  = bsm(position, spotPrice = spotRange, atTime = T_n, ir = ir, iv = iv_reference + iv_adj, resultType = "wvega" if wVega else "vega")
      data["rho"]   = bsm(position, spotPrice = spotRange, atTime = T_n, ir = ir, iv = iv_reference + iv_adj, resultType = "rho")
      end = datetime.now()
      elapsed = end-start

      pnl_span = [np.min(data.filter(like = 'P&L', axis=1).min().values)-1, np.max(data.filter(like = 'P&L', axis=1).max().values)+1]

      # Create a chart with multiple subplots: 6 rows and 1 column
      # Set the row_heights to give more space to the P&L chart
      fig = make_subplots(rows = 7, cols = 1, row_heights = [1, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3]
                           , subplot_titles = (f"P&L - IR: {ir:.1%} - IV Adj.: {iv_adj:.1%}", "Skew", "Theta", "Delta", "Gamma", "Weighted Vega" if wVega else "Vega", "Rho")
                           , shared_xaxes = True
                           , vertical_spacing = 0.03
                           )

      if showPNLOpen:
         # Show the P&L @ Open
         fig.add_trace(go.Scatter(x = spotRange, y = data["P&L (Open)"], mode = "lines", name = f"P&L @ Open ({openDTE} DTE)", line = {"color": "darkorange"}), row = 1, col = 1)

      fig.add_traces(data = [
                        go.Scatter(x = spotRange, y = data["P&L (Expiry)"]  , mode = "lines", name = f"P&L @ {T_expiry.strftime('%Y-%m-%d')} (0 DTE): T+{(T_expiry-T_0).days}"      , line = {"color": "darkblue"})
                     , go.Scatter(x = spotRange, y = data[f"P&L (@ {T_n})"], mode = "lines", name = f"P&L @ {T_n.strftime('%Y-%m-%d')} ({(T_expiry-T_n).days} DTE): T+{(T_n-T_0).days}", line = {"color": "darkgrey"})
                     , go.Scatter(x = spotRange, y = data["P&L (@ Now)"]   , mode = "lines", name = f"P&L @ {T_0.strftime('%Y-%m-%d')} ({(T_expiry-T_0).days} DTE): T+0"           , line = {"color": "darkgreen"})
                     , go.Scatter(x = [spotPrice, spotPrice]
                                    , y = pnl_span
                                    , name = "Spot Price"
                                    , mode = "lines"
                                    , line = dict(color = "darkgreen", dash = "dot", width = 0.8)
                                    , showlegend = False
                                    , hoverinfo = "skip"
                                 )
                     ]
                     , rows = 1, cols = 1
                     )


      expiry_dt = T_expiry.strftime('%Y-%m-%d')
      if skew_dict is not None and expiry_dt in skew_dict.keys():
         put_skew = skew_dict[expiry_dt]["puts"].query(f"{spotRange.min()} <= strike <= {np.max([position['strikePrice'].astype(np.float64).max(), spotPrice])+10.0}")
         call_skew = skew_dict[expiry_dt]["calls"].query(f"{np.min([position['strikePrice'].astype(np.float64).min(), spotPrice])-10.0} <= strike <= {spotRange.max()}")
         #skew_span = [np.min([put_skew["impliedVolatility"].min(), call_skew["impliedVolatility"].min(), position["IV.open"].min()])-0.01, np.max([put_skew["impliedVolatility"].max(), call_skew["impliedVolatility"].max(), position["IV.open"].max()])+0.01]
         skew_span = [np.min([put_skew["IV.last"].min(), call_skew["IV.last"].min(), position["IV.open"].min()])-0.01, np.max([put_skew["IV.last"].max(), call_skew["IV.last"].max(), position["IV.open"].max()])+0.01]
         position_ivs = position[["strikePrice", "IV.open"]].sort_values(by = "strikePrice")
         fig.add_traces(data = [
                           #go.Scatter(x = put_skew["strike"] , y = put_skew["impliedVolatility"] , mode = "lines", name = f"Put Skew @ {T_expiry.strftime('%Y-%m-%d')}", line = {"color": "darkred"}, showlegend = False)
                           #, go.Scatter(x = call_skew["strike"], y = call_skew["impliedVolatility"], mode = "lines", name = f"Call Skew @ {T_expiry.strftime('%Y-%m-%d')}", line = {"color": "darkblue"}, showlegend = False)
                           go.Scatter(x = put_skew["strike"] , y = put_skew["IV.last"] , mode = "lines", name = f"Put Skew @ {T_expiry.strftime('%Y-%m-%d')}", line = {"color": "darkred"}, showlegend = False)
                           , go.Scatter(x = call_skew["strike"], y = call_skew["IV.last"], mode = "lines", name = f"Call Skew @ {T_expiry.strftime('%Y-%m-%d')}", line = {"color": "darkblue"}, showlegend = False)
                           , go.Scatter(x = position_ivs["strikePrice"], y = position_ivs["IV.open"], mode = "lines+markers", name = f"Skew @ Open", line = {"color": "darkorange"}, showlegend = False)
                           , go.Scatter(x = [spotPrice, spotPrice]
                                 , y = skew_span
                                 , name = "Spot Price"
                                 , mode = "lines"
                                 , line = dict(color = "darkgreen", dash = "dot", width = 0.8)
                                 , showlegend = False
                                 , hoverinfo = "skip"
                                 )
                        ]
                        , rows = 2, cols = 1
                        )


      for greek, row in zip(["theta", "delta", "gamma", "vega", "rho"], np.linspace(3,7,5, dtype = int)):
         y = data[greek]
         norm_y = (np.array(y) - min(y)) / ((max(y) - min(y)) if (max(y) - min(y)) != 0 else 1)
         # Add the Bar Chart for the Greek
         fig.add_trace(go.Bar(x = spotRange
                              , y = y
                              , name = greek.capitalize()
                              , showlegend = False
                              , marker = {'color': norm_y
                                          , 'colorscale': "RdBu"
                                          , 'line': {'color': 'white', 'width': 1}
                                          }
                              )
                        , row = row, col = 1
                        )
         # Add a vertical line to mark the Spot price
         fig.add_trace(go.Scatter(x = [spotPrice, spotPrice]
                                 , y = [min(y)-1, max(y)+1]
                                 , name = "Spot Price"
                                 , mode = "lines"
                                 , line = dict(color = "darkgreen", dash = "dot", width = 0.8)
                                 , showlegend = False
                                 , hoverinfo = "skip"
                                 )
                     , row = row, col = 1
                     )


   fig.update_layout( title_x = 0.5
                     , autosize = False
                     , width = 1400
                     , height = 900
                     , hovermode = 'x unified'
                     , hoverlabel = dict(bgcolor = "rgba(240,240,250,0.5)")
                     , showlegend = showLegend
                     , legend = dict(x = 0.01, y = 0.85 #y = 0.99 # Positions for the legend
                                    , traceorder = "normal"
                                    , bgcolor = "rgba(240,240,250,0.5)"
                                    , bordercolor = "darkblue"
                                    , borderwidth = 1
                                    , font = dict(size = 10)
                                    #, itemsizing = 'constant'
                                    )
                     #, shapes = [spot_line]
                     )



   annotation_text = "<span style='text-align: left'><b>Position:                                </b></span>"
   for idx, row in position.iterrows():
      delta_t0 = bsm(position.loc[[idx]], spotPrice = spotPrice, atTime = T_0, ir = ir, iv = iv_reference, resultType = "delta")[0]/row['qty']
      annotation_text += f"<br>&nbsp;<span style = 'color: {'darkblue' if row['qty'] > 0 else 'darkred'}'>{row['qty']:+} {row['expiryDttm']:%d%b} {row['strikePrice']} {row['optionType']} Δ {delta_t0:.2f}</span>"

   fig.add_annotation(
         go.layout.Annotation(
            text = annotation_text
            , xref = "paper", yref = "paper"
            , xanchor = "left"
            , x = 0.01, y = 0.99
            , showarrow = False
            , bgcolor = "rgba(240,240,250,0.5)"
            , bordercolor = "darkblue"
            , borderwidth = 1
            , font = dict(size = 10)
         )
      )

   fig.update_xaxes(title_text = "Spot Price", row = 7, col = 1)  # Adding title to the bottom subplot's x-axis
   fig.update_yaxes(tickformat = "$,.1f")
   fig.update_yaxes(tickformat = ",.1%", row = 2, col = 1)
   fig.show()

def app_init(portfolio = None, N_spot = 300, ir = 0.055, marketOpenTime = timedelta(hours = 9, minutes = 30), marketCloseTime = timedelta(hours = 16)):
   tickers_list = sorted(portfolio["ticker"].unique())

   # T+n selector
   time_slider = widgets.Dropdown(options = [], description = 'Eval Time:', value = None, layout={'width': '25%'})
   # IV Adjustment Slider: Up to (+/-)10% adjustment with 0.5% increments
   iv_slider = widgets.FloatSlider(value = 0, min = -0.1, max = 0.1, step = 0.005, continuous_update = True, readout_format='.1%', description = "IV Adjust.", layout={'width': '25%'})
   # Interest Rate: From 0% to 10% with 0.25% increments
   ir_slider = widgets.FloatSlider(value = ir, min = 0, max = 0.10, step = 0.0025, continuous_update = True, readout_format='.1%', description = "Interest Rate", layout={'width': '25%'})
   # Controls the x-axis range of the charts, relative to the current spot price: [spot - x%, spot + y%] where x and y are the values of the range slider
   xrange_slider = widgets.FloatRangeSlider(value = [-0.1, 0.1], min = -0.5, max = 0.5, step = 0.05, continuous_update = True, readout_format = '.1%', readout = True, description = "Price Range", layout={'width': '23%'})
   # Button used to trigger a refresh of the quote data from Yahoo! Finance
   refresh_button = widgets.Button(description = 'Refresh Quote Data', disabled = False, button_style = 'success', tooltip = 'Refresh Quote Data', icon='refresh', layout={'width': '170px'})
   # Weighted Vega Checkbox
   wvega_checkbox = widgets.Checkbox(value = False, description='Weighted Vega', disabled = False, layout = {'width': '15%'})
   # Weighted Vega Checkbox
   legend_checkbox = widgets.Checkbox(value = True, description='Show Legend', disabled = False, layout = {'width': '15%'})
   # Show P&L @ Open Checkbox
   show_pnl_open_checkbox = widgets.Checkbox(value = False, description='Show P&L @ Open', disabled = False, layout = {'width': '15%'})
   # Chart resolution
   resolution_slider = widgets.IntSlider(value = N_spot, min = 100, max = 500, step = 10, continuous_update = True, readout = True, description = "Resolution", layout={'width': '25%'})
   # Warning Label
   warning_label = widgets.HTML()
   # Position details
   pos_details_table = pn.widgets.Tabulator(value = None, show_index = False, disabled = True)

   # Ticker selector
   ticker_dropdown = widgets.Dropdown(options = tickers_list, description = 'Ticker:', value = tickers_list[0], layout = {'width': '15%'})
   # TradeId selector
   #trade_selector = widgets.Dropdown(options = [], description = 'Trade Id:', value = None, layout = {'width': '25%'})
   trade_selector = widgets.SelectMultiple(options = [], description = 'Trade Id:', value = (), layout = {'width': '20%'})

   # Expiration selector
   expiration_selector = widgets.Dropdown(options = [], description = 'Expiration:', value = None) #, layout = {'width': '25%'})
   # Trade Adjustment time
   adjustment_selector = widgets.Dropdown(options = [], description = 'Adjustment Date: ', value = None) #, layout = {'width': '25%'})

   last_box = widgets.HTML(value = "Last Price: {lastPrice}")#, layout = {'width': '15%'})


   # Initialize all widgets and parameters
   params = dict(
         # Ticker selection
         ticker = None
         , marketOpenTime = marketOpenTime
         , marketCloseTime = marketCloseTime
         # IV Skew dictionary
         , skew_dict = None
         , lastPrice = None
         # Portfolio of all positions
         , portfolio = portfolio
         # Position to be analyzed
         , trade_position = portfolio
         , IV_reference = portfolio["IV.open"]
         , ir = ir

         # List of tickers to select from
         , tickers_list = tickers_list

         # T+n selector
         , time_slider = time_slider
         # IV Adjustment Slider: Up to (+/-)10% adjustment with 0.5% increments
         , iv_slider = iv_slider
         # Interest Rate: From 0% to 10% with 0.25% increments
         , ir_slider = ir_slider
         # Controls the x-axis range of the charts, relative to the current spot price: [spot - x%, spot + y%] where x and y are the values of the range slider
         , xrange_slider = xrange_slider
         # Button used to trigger a refresh of the quote data from Yahoo! Finance
         , refresh_button = refresh_button
         # Weighted Vega Checkbox
         , wvega_checkbox = wvega_checkbox
         # Weighted Vega Checkbox
         , legend_checkbox = legend_checkbox
         # Show P&L @ Open Checkbox
         , show_pnl_open_checkbox = show_pnl_open_checkbox
         # Chart resolution
         , resolution_slider = resolution_slider
         # Warning Label
         , warning_label = warning_label
         # Position details
         , pos_details_table = pos_details_table

         # Ticker selector
         , ticker_dropdown = ticker_dropdown
         # TradeId selector
         #trade_selector = widgets.Dropdown(options = [], description = 'Trade Id:', value = None, layout = {'width': '25%'})
         , trade_selector = trade_selector

         # Expiration selector
         , expiration_selector = expiration_selector
         # Trade Adjustment time
         , adjustment_selector = adjustment_selector

         , last_box = last_box
         , outputWidget = widgets.Output()
      )
   
   # Configure observers
   refresh_button.on_click(partial(refresh_quote_data, params = params))
   trade_selector.observe(partial(refresh_quote_data, params = params), names = 'value')
   ticker_dropdown.observe(partial(refresh_quote_data, params = params), names = 'value')
   adjustment_selector.observe(partial(refresh_quote_data, params = params), names = 'value')
   time_slider.observe(partial(sliders_change, params = params), names = 'value')
   iv_slider.observe(partial(sliders_change, params = params), names = 'value')
   ir_slider.observe(partial(sliders_change, params = params), names = 'value')
   xrange_slider.observe(partial(sliders_change, params = params), names = 'value')
   wvega_checkbox.observe(partial(sliders_change, params = params), names = 'value')
   legend_checkbox.observe(partial(sliders_change, params = params), names = 'value')
   show_pnl_open_checkbox.observe(partial(sliders_change, params = params), names = 'value')
   resolution_slider.observe(partial(sliders_change, params = params), names = 'value')
   expiration_selector.observe(partial(update_time_selector, params = params), names = 'value')

   return params

def refresh_quote_data(event, params = dict()):

   # Extract parameters
   portfolio = params.get("portfolio", None)
   adjustment_selector = params.get("adjustment_selector", None)
   ticker_dropdown = params.get("ticker_dropdown", None)
   last_box = params.get("last_box", None)
   trade_selector = params.get("trade_selector", None)

   # Create the ticker object
   ticker = yf.Ticker(ticker_dropdown.value)

   # Get the current price
   price_history = ticker.history(period = "1d")
   openPrice = price_history["Open"][0]
   lastPrice = price_history["Close"][0]
   priceChange = lastPrice/openPrice - 1
   price_color = 'darkgreen' if lastPrice >= openPrice else 'darkred'
   # Update the Price Label
   last_box.value = f"<span style='margin-left: 20px;'>Last Price: <span style = 'color: {price_color}'>{lastPrice: .2f} ({priceChange:+.2%})</span> </span>"

   # Update the Params dictionary
   params["ticker"] = ticker
   params["lastPrice"] = lastPrice


   # Update the list of trades and make sure there is at least one trade selected
   update_tradeid_list(event, params)

   # Filter the position based on the selected trade
   query = f"ticker == '{ticker_dropdown.value}' and tradeId in {trade_selector.value}"
   # Update the adjustment selector
   adjustment_selector.options = [(pd.Timestamp(dt), pd.Timestamp(dt)) for dt in sorted(portfolio.query(query)["openDttm"].unique(), reverse = True)]

   if adjustment_selector.value is not None:
      query += f" and openDttm <= '{pd.Timestamp(adjustment_selector.value):%Y-%m-%d %H:%M:%S}'"

   # Update the list of positions to analyze
   trade_position = portfolio.query(query).copy().reset_index()
   params["trade_position"] = trade_position

   #IV_reference = trade_position["IV.open"]

   update_expiration_selector(event, params)


def update_tradeid_list(event, params = dict()):

   portfolio = params.get("portfolio", None)
   trade_selector = params.get("trade_selector", None)
   ticker_dropdown = params.get("ticker_dropdown", None)

   # Update the list of Trade Id values to choose from
   trade_selector.options = sorted(portfolio.query(f"ticker == '{ticker_dropdown.value}'")["tradeId"].unique())

   # Select a trade if none is selected
   if trade_selector.value is None or len(trade_selector.value) == 0:
      trade_selector.value = (trade_selector.options[0],)


def update_expiration_selector(event, params = dict()):

   # Retrieve the prompt parameters
   ticker = params.get("ticker", None)
   lastPrice = params.get("lastPrice", None)
   trade_position = params.get("trade_position", None)
   expiration_selector = params.get("expiration_selector", None)
   warning_label = params.get("warning_label", None)
   pos_details_table = params.get("pos_details_table", None)
   ir = params.get("ir", None)

   expiration_dates = [pd.Timestamp(dt) for dt in sorted(trade_position["expiryDttm"].unique())]
   expiration_selector.options = expiration_dates #[(dt, dt) for dt in expiration_dates]

   option_chains = dict()
   skew_dict = dict()
   for dt in expiration_dates:
      expiry_dt = dt.strftime("%Y-%m-%d")
      # Get the chain data
      chain = ticker.option_chain(expiry_dt)
      # Set variables needed to compute IV
      chain.puts["atTime"] = datetime.now()
      chain.puts["expiryDttm"] = dt
      chain.puts["optionType"] = "Put"
      chain.puts["strikePrice"] = chain.puts["strike"]
      chain.puts["spotPrice"] = lastPrice
      chain.puts["midPrice"] = chain.puts[["bid", "ask"]].mean(axis = 1)
      chain.calls["atTime"] = datetime.now()
      chain.calls["expiryDttm"] = dt
      chain.calls["optionType"] = "Call"
      chain.calls["strikePrice"] = chain.calls["strike"]
      chain.calls["spotPrice"] = lastPrice
      chain.calls["midPrice"] = chain.calls[["bid", "ask"]].mean(axis = 1)
      # Compute IV
      chain.puts["IV.last"] = bsm(chain.puts, spotPrice = chain.puts["spotPrice"], atTime = chain.puts["atTime"], ir = ir, optionPrice = chain.puts["midPrice"], resultType = "ImpliedVol")
      chain.calls["IV.last"] = bsm(chain.calls, spotPrice = chain.calls["spotPrice"], atTime = chain.calls["atTime"], ir = ir, optionPrice = chain.calls["midPrice"], resultType = "ImpliedVol")
      option_chains[expiry_dt] = chain._asdict()
      skew_dict[expiry_dt] = dict(puts = chain.puts[["strike", "impliedVolatility", "IV.last"]], calls = chain.calls[["strike", "impliedVolatility", "IV.last"]])


   option_prices = []
   option_IVs = []
   for idx, row in trade_position.iterrows():
      # Get the chains
      chains = option_chains[row["expiryDate"]]
      # Get the Calls or Puts chain
      df_chain = chains.get(f"{row['optionType'].lower()}s")
      # Find the specific strike
      option = df_chain.query(f"strike == {row['strikePrice']}").copy()
      #print(f"strike == {row['strikePrice']} -> {option}")
      ###option["midPrice"] = option[["bid", "ask"]].mean(axis = 1)
      # Find the Strike and get the lastPrice
      optionMidPrice = option.iloc[0]["midPrice"]
      optionIV = option.iloc[0]["impliedVolatility"]
      # Append the price to the list
      option_prices.append(optionMidPrice)
      option_IVs.append(optionIV)

   trade_position["atTime"] = datetime.now()
   trade_position["spotPrice"] = lastPrice
   trade_position["optionPrice"] = option_prices
   try:
      trade_position["IV.last"] = bsm(trade_position, spotPrice = trade_position["spotPrice"], atTime = trade_position["atTime"], ir = ir, optionPrice = option_prices, resultType = "ImpliedVol")
      warning_label.value = ""
   except:
      trade_position["IV.last"] = optionIV
      warning_label.value = "<span style = 'color: darkorange'><b>WARNING: Could not compute IV. Using IV quote from Yahoo! Finance</b></span>"

   trade_position["Implied Vol (@Open)"] = trade_position["IV.open"].apply(lambda x: '{:.2%}'.format(x))
   trade_position["Implied Vol (@Now)"] = trade_position["IV.last"].apply(lambda x: '{:.2%}'.format(x))
   trade_position["IV Change"] = (trade_position["IV.last"] - trade_position["IV.open"]).apply(lambda x: '{:+.2%}'.format(x))
   #print(trade_position[['optionType',"expiryDate", "strikePrice", "tradePrice", "optionPrice", "IV.last"]])
   pos_details_table.value = trade_position[["tradeId","openDttm", "optionType","expiryDate", "strikePrice", "tradePrice", "optionPrice", "Implied Vol (@Open)", "Implied Vol (@Now)", "IV Change"]]

   IV_reference = trade_position["IV.last"]

   # Update parameters dictionary
   params["IV_reference"] = IV_reference
   params["skew_dict"] = skew_dict


   #IV_reference = np.array(option_IVs)
   #expiration_selector.value = expiration_selector.options[0]
   update_time_selector(event, params)

def update_time_selector(event, params = dict()):

   marketCloseTime = params.get("marketCloseTime", None)
   expiration_selector = params.get("expiration_selector", None)
   time_slider = params.get("time_slider", None)

   # Current date (@ Market Close)
   time_reference = datetime.combine(date.today(), datetime.min.time()) + marketCloseTime
   # Number of days to the earliest expiration
   #DTE = (trade_position["expiryDttm"].max() - time_reference).days
   DTE = max(0, (expiration_selector.value - time_reference).days)

   # Build the timeline to expiration
   eval_times = np.append(
      [time_reference + timedelta(days = int(i)) for i in np.linspace(start = 0, stop = DTE+1, num = DTE+1, dtype = int, endpoint = False)]
      , [] #[datetime(time_reference.year, time_reference.month, time_reference.day+DTE) + marketOpenTime + timedelta(hours = i*1) for i in range(0, 7)]
   )
   # Build the list of options for the dropdown: (Label, Value) pairs
   eval_times_opts = [
      (f"{t.strftime('%Y-%m-%d')} T+{(t-eval_times[0]+marketCloseTime).days} @ {t.time()}", t) for t in eval_times
   ]
   # Update the Time selector
   time_slider.options = eval_times_opts
   # Set the time (this will trigger a call to slider_change function)
   #time_slider.value = eval_times[0]
   sliders_change(event, params)
   #eval_times_opts
   #T_1 = pd.DataFrame(eval_times, columns = ["eval_time"]).query(f"eval_time > '{datetime.now()}'").head(1)["eval_time"].iloc[0]


def sliders_change(event, params = dict()):

   trade_selector = params.get("trade_selector", None)
   time_slider = params.get("time_slider", None)
   trade_position = params.get("trade_position", None)
   expiration_selector = params.get("expiration_selector", None)
   iv_slider = params.get("iv_slider", None)
   IV_reference = params.get("IV_reference", None)
   resolution_slider = params.get("resolution_slider", None)
   lastPrice = params.get("lastPrice", None)
   xrange_slider = params.get("xrange_slider", None)
   wvega_checkbox = params.get("wvega_checkbox", None)
   legend_checkbox = params.get("legend_checkbox", None)
   show_pnl_open_checkbox = params.get("show_pnl_open_checkbox", None)
   outputWidget = params.get("outputWidget", None)

   ir_slider = params.get("ir_slider", None)
   if trade_selector.value is None:
         return widgets.Label(value = "Please select a Trade Id")
   if time_slider.value is None:
         return widgets.Label(value = "Please select an Eval Time")

   update_plot(trade_position
               , eval_time = time_slider.value
               , expiry_dttm = expiration_selector.value
               , ir = ir_slider.value
               , iv_adj = iv_slider.value
               , iv_reference = IV_reference
               , N_spot = resolution_slider.value
               , spotPrice = lastPrice
               , price_range = xrange_slider.value
               , wVega = wvega_checkbox.value
               , showLegend = legend_checkbox.value
               , showPNLOpen = show_pnl_open_checkbox.value
               , outputWidget = outputWidget
               )

def app_run(portfolio = None, N_spot = 300, ir = 0.055, marketOpenTime = timedelta(hours = 9, minutes = 30), marketCloseTime = timedelta(hours = 16)):
   # Initialize the app
   params = app_init(portfolio = portfolio, N_spot = N_spot, ir = ir, marketOpenTime = marketOpenTime, marketCloseTime = marketCloseTime)

   ticker_dropdown = params.get("ticker_dropdown", None)
   trade_selector = params.get("trade_selector", None)
   expiration_selector = params.get("expiration_selector", None)
   adjustment_selector = params.get("adjustment_selector", None)
   last_box = params.get("last_box", None)
   refresh_button = params.get("refresh_button", None)
   resolution_slider = params.get("resolution_slider", None)
   xrange_slider = params.get("xrange_slider", None)
   legend_checkbox = params.get("legend_checkbox", None)
   wvega_checkbox = params.get("wvega_checkbox", None)
   show_pnl_open_checkbox = params.get("show_pnl_open_checkbox", None)
   time_slider = params.get("time_slider", None)
   iv_slider = params.get("iv_slider", None)
   ir_slider = params.get("ir_slider", None)
   pos_details_table = params.get("pos_details_table", None)
   warning_label = params.get("warning_label", None)
   outputWidget = params.get("outputWidget", None)

   # Arrange sliders horizontally and display with plot
   header1 = widgets.HBox([ticker_dropdown, trade_selector, widgets.VBox([expiration_selector, adjustment_selector]), widgets.VBox([last_box, refresh_button])])
   header2 = widgets.HBox([resolution_slider, xrange_slider, legend_checkbox, wvega_checkbox, show_pnl_open_checkbox])
   sliders = widgets.HBox([time_slider, iv_slider, ir_slider])
   accordion = pn.Accordion(("Position Details", pos_details_table))

   display(header1, header2, sliders, warning_label, accordion, outputWidget)

   # Initial Rendering
   refresh_quote_data(event = 1, params = params)
