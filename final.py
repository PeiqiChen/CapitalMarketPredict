# Import required libraries
import datetime
import time
from polygon.rest import RESTClient 
from sqlalchemy import create_engine 
from sqlalchemy import text
import pandas as pd
from math import sqrt
from math import isnan
from numpy import mean
from numpy import std
from math import floor
import sqlite3
from pycaret.regression import *
'''
THIS IS MAX CODE
'''
# We can buy, sell, or do nothing each time we make a decision.
# This class defies a nobject for keeping track of our current investments/profits for each currency pair
class portfolio(object):
    def __init__(self,from_,to):
        # Initialize the 'From' currency amont to 1
        self.amount = 1
        self.curr2 = 0
        self.from_ = from_
        self.to = to
        # We want to keep track of state, to see what our next trade should be
        self.Prev_Action_was_Buy = False
    
    # This defines a function to buy the 'To' currency. It will always buy the max amount, in whole number
    # increments
    def buy_curr(self, price):
        if self.amount >= 1:
            num_to_buy = floor(self.amount)
            self.amount -= num_to_buy
            self.Prev_Action_was_Buy = True
            self.curr2 += num_to_buy*price
            print("Bought %d worth of the target currency (%s). Our current profits and losses in the original currency (%s) are: %f." % (num_to_buy,self.to,self.from_,(self.amount-1)))
        else:
            print("There was not enough of the original currency (%s) to make another buy." % self.from_)
    # This defines a function to sell the 'To' currency. It will always sell the max amount, in a whole number
    # increments
    def sell_curr(self, price):
        if self.curr2 >= 1:
            num_to_sell = floor(self.curr2)
            self.amount += num_to_sell * (1/price)
            self.Prev_Action_was_Buy = False
            self.curr2 -= num_to_sell
            print("Sold %d worth of the target currency (%s). Our current profits and losses in the original currency (%s) are: %f." % (num_to_sell,self.to,self.from_,(self.amount-1)))
        else:
            print("There was not enough of the target currency (%s) to make another sell." % self.to)   
#---------------------------------- HW2 CODE CHANGES --------------------------------------------
'''
Function to calculate 100 Keltner Upper Bands
'''
def calculateKeltnerChannelUB(mean, VOL):
  UB100values = []
  for n in range(1,101):
    UB100values.append((mean+ n*0.025*VOL))
  return UB100values

#function to calculate 100 Keltner Lower Bands
def calculateKeltnerChannelLB(mean, VOL):
  LB100values = []
  for n in range(1,101):
    LB100values.append((mean - n*0.025*VOL))
  return LB100values

'''
Function that counts N i.e the number of times a price crosses a Keltner Channel from lastPrice to currentPrice

Arguments we pass are -
# lastPrice: previous currency price
# currentPrice: current currency price 
# UB (Upper Band): 100 Keltner Upper Bands
# LB (Lower Band): 100 Keltner Lower Bands
'''
def countBandsCrossing(lastPrice, currentPrice, UB, LB):
  '''
  Step 1: We find the startPrice and endPrice so as to cound all the bands lying in between these two prices
          start price is the price which is min of lastPrice and currentPrice and end price is the larger price
  Step 2: We calculate totalBands i.e 200 bands
  Step 3: We calculate filteredTotalBands which are the bands that the price changes crosses
  Step 4: We return the length of filteredTotalBands which is the count of bands crossed from the lastPrice to currentPrice
  
  '''
  # Finding startPrice and endPrice to find all the bands lying in between the given price change
  startPrice = min(lastPrice, currentPrice)
  endPrice = max(lastPrice, currentPrice)

  # totalBands are the 200 (LB + UB) in sorted order
  totalBands = LB[::-1] + UB

  # filteredTotalBands are the bands lying between the startPrice and endPrice
  filteredTotalBands = filter(lambda x: endPrice>x> startPrice, totalBands)

  # In order to return the count of bands crossed we simply find the len(filteredTotalBands)
  return len(list(filteredTotalBands))

#function to calculate FD (Fractal Dimension) using formula FD = N/VOL
#function will return 0 if the VOL is 0 (In order to avoid divide by 0 error) 
def calculateFD(N, VOL):
  if VOL == 0:
    return 0
  else:
    return N/VOL

#---------------------------------- HW2 CODE CHANGES --------------------------------------------

#---------------------------------- HW3 CODE CHANGES --------------------------------------------

'''
Calculates the return according to the formula 𝑟𝑖 = (𝑃𝑖 − 𝑃𝑖−1)⁄(𝑃𝑖−1)

Arguments:
  - currentPrice: Current mean price of currency pair
  - lastPrice: Last mean price of currency pair

Returns:
  - return 𝑟𝑖
'''
def calculateReturn(currentPrice, lastPrice):
  return (currentPrice-lastPrice)/lastPrice


'''
Returns sum of last 10 intervals returnVal

Arguments:
  - rows: sql rows object which has last 10 rows from corresponding aggregated table

Returns:
  - sum of 𝑟𝑖
'''
def getReturnOfLast10(rows):
  returnVal = 0
  for row in rows:
    if row.returnVal:
      returnVal += row.returnVal
  return returnVal

#---------------------------------- HW3 CODE CHANGES --------------------------------------------
from ast import Pass
# Function slightly modified from polygon sample code to format the date string 
def ts_to_datetime(ts) -> str:
    return datetime.datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

# Function which clears the raw data tables once we have aggregated the data in a 6 minute interval
def reset_raw_data_tables(engine,currency_pairs):
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_raw;"))
            conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_raw(ticktime text, fxrate  numeric, inserttime text);"))

# This creates a table for storing the raw, unaggregated price data for each currency pair in the SQLite database
def initialize_raw_data_tables(engine,currency_pairs):
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_raw(ticktime text, fxrate  numeric, inserttime text);"))

            
# This creates a table for storing the (6 min interval) aggregated price data for each currency pair in the SQLite database            
def initialize_aggregated_tables(engine,currency_pairs):
  with engine.begin() as conn:
    for curr in currency_pairs:
      # Initializes aggregate table for all currency pairs with fields inserttime, period, max, min, mean, vol and fd
      conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_agg (inserttime text, period numeric, max numeric, min numeric, mean numeric, vol numeric, fd numeric, returnVal numeric);"))


def aggregate_raw_data_tables(engine, currency_pairs, period, N, currencyCheck):
  # Initialize UB and LB dictionary which has currency pairs as key and stores bands based on mean and vol
  UB = {}
  LB = {}
  returnVals = {}
  stopSigns = {}
  with engine.begin() as conn:
    for curr in currency_pairs:
      if True:#currencyCheck[curr[0]+curr[1]] == True:
        stopSigns[curr[0]+curr[1]] = currencyCheck[curr[0]+curr[1]]
        with engine.begin() as conn:
          # Calculates mean, max, and min from raw tables which has price data of last 6 mins period
          result = conn.execute(text("SELECT AVG(fxrate) as mean, MIN(fxrate) as min, MAX(fxrate) as max FROM "+curr[0]+curr[1]+"_raw;"))
          for row in result:
            mean = row.mean
            min = row.min
            max = row.max
            # print()
            vol = (max-min)/mean

          # Calculate UB and LB for the currency pair and storing it into dict
          UB[curr[0]+curr[1]] = calculateKeltnerChannelUB(mean, vol)
          LB[curr[0]+curr[1]] = calculateKeltnerChannelLB(mean, vol)

          date_res = conn.execute(text("SELECT MAX(ticktime) as last_date FROM " +curr[0]+curr[1]+"_raw;"))
          for row in date_res:
            last_date = row.last_date

          # If it is the 1st period, we skip adding FD value in table 
          if period == 1:
            returnVals[curr[0]+curr[1]] = 0
            conn.execute(text("INSERT INTO " +curr[0]+curr[1]+"_agg (inserttime, period, max, min, mean, vol) VALUES (:inserttime, :period, :max, :min, :mean, :vol);"),[{"inserttime":last_date, "period": period, "max": max, "min": min, "mean": mean, "vol": vol}])
          else:
            fd = calculateFD(N[curr[0]+curr[1]],vol)
            result = conn.execute(text("SELECT * FROM " +curr[0]+curr[1]+"_agg ORDER BY rowid DESC LIMIT 1;"))
            
            for row in result:
              lastMean = row.mean
            returnVal = calculateReturn(mean, lastMean)
            returnVals[curr[0]+curr[1]] = returnVal
            conn.execute(text("INSERT INTO " +curr[0]+curr[1]+"_agg (inserttime, period, max, min, mean, vol, fd, returnVal) VALUES (:inserttime, :period, :max, :min, :mean, :vol, :fd, :returnVal );"),[{"inserttime":last_date, "period": period, "max": max, "min": min, "mean": mean, "vol": vol, "fd": fd, "returnVal":returnVal}])

  # Returning Keltner Channel Upper bound and Lower bound to the main function
  return UB, LB, returnVals, stopSigns
#---------------------------------- HW3 CODE CHANGES --------------------------------------------
# This creates an output table for storing the data after 60 min interval for each currency pair in the SQLite database     
# The table has window, balance, returnVal and position fields 
def initialize_output_tables(engine,currency_pairs):
  with engine.begin() as conn:
    for curr in currency_pairs:
      # Initializes aggregate table for all currency pairs with fields inserttime, period, max, min, mean, vol and fd
      conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_output (window numeric, balance numeric, returnVal numeric, position text);"))

#---------------------------------- HW3 CODE CHANGES --------------------------------------------
#---------------------------------- HW3 CODE CHANGES --------------------------------------------

def fill_output_data_tables(engine, currency_pairs, window, currencyCheck):
  '''
    Function called every 60 minutes to make a decision on current open positions and fill the output data tables
    
    Arguments:
      - engine: Engine object from sqlalchemy
      - currency_pairs: Nested list of each currency pair
      - window: current window count
      - currencyCheck: It is a dictionary which has keys as currencyPair and values as True or False
    Returns:
      - currencyCheck: Updated currencyCheck dict
  '''

  #longCurrency List (BUY)
  longCurrency = ["EURUSD", "GBPEUR", "USDJPY", "USDCZK", "USDINR"]
  #shortCurrency List (SELL)
  shortCurrency = [ "AUDUSD", "USDCAD", "USDCNY", "USDPLN", "USDMXN"]


  for curr in currency_pairs:
    # We will check if currency pair is set to True in the currencyCheck dict which will let us know weather to close the position or continue
    if True:#urrencyCheck[curr[0]+curr[1]] == True:

      # setting position based on longCurrency list
      if curr[0]+curr[1] in longCurrency:
        position = 'LONG'
      else:
        position = 'SHORT'

      with engine.begin() as conn:
        # Fetching last 10 rows from aggregate table to get the return value
        result = conn.execute(text("SELECT *  FROM "+curr[0]+curr[1]+"_agg ORDER BY rowid DESC LIMIT 10;"))
        # R10 stores the sum of last 10 returnVal 
        R10 = getReturnOfLast10(result)

        # At T10, cutoff value to use is 0.250%
        if window == 1:
          balance = 100
          if position == 'LONG':
            # Long condition -> , a profitable trade has a positive return but we have a tolarence of 0.250%
            # 0.250% = 0.0025
            if R10 >= -0.0025:
              balance = balance + 100 + R10
            # If not profitable, we will close the position and set the currencyCheck flag to false
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
          else:
            # Short condition -> a profitable trade has a negative return. 
            if R10 <= 0.0025:
              balance = balance + 100 + R10
            # If not profitable, we will close the position and set the currencyCheck flag to false
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
        
        # At T20, cutoff value to use is 0.150%
        elif window == 2:
          # The below line fetches the last balance of the currency pair of last 60 min period
          result = conn.execute(text("SELECT * FROM " +curr[0]+curr[1]+"_output WHERE window = " + str(window-1)+";"))
          for row in result:
            balance = row.balance

          if position == 'LONG':
            if R10 >= -0.0015:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
          else:
            # Short condition
            if R10 <= 0.0015:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False


        # At T30, value to use is 0.100%
        elif window == 3:
          result = conn.execute(text("SELECT * FROM " +curr[0]+curr[1]+"_output WHERE window = " + str(window-1)+";"))
          for row in result:
            balance = row.balance

          if position == 'LONG':
            if R10 >= -0.001:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
          else:
            if R10 <= 0.001:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False


        # At T40, value to use is 0.050%
        elif window == 4:
          result = conn.execute(text("SELECT * FROM " +curr[0]+curr[1]+"_output WHERE window = " + str(window-1)+";"))
          for row in result:
            balance = row.balance

          if position == 'LONG':
            if R10 >= -0.0005:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
          else:
            if R10 <= 0.0005:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False


        # After T40, value to use is 0.050%
        elif window > 4:
          result = conn.execute(text("SELECT * FROM " +curr[0]+curr[1]+"_output WHERE window = " + str(window-1)+";"))
          for row in result:
            balance = row.balance

          if position == 'LONG':
            if R10 >= -0.0005:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
          else:
            if R10 <= 0.0005:
              balance = balance + 100 + R10
            else:
              balance = balance + R10
              currencyCheck[curr[0]+curr[1]] = False
          
        conn.execute(text("INSERT INTO " +curr[0]+curr[1]+"_output (window, balance, returnVal, position) VALUES (:window, :balance, :returnVal, :position);"),[{"window":window, "balance": balance, "returnVal": R10, "position": position}])

  return currencyCheck

#---------------------------------- HW3 CODE CHANGES --------------------------------------------
# This main function repeatedly calls the polygon api every 1 seconds for 24 hours 
# and stores the results.
# A dictionary defining the set of currency pairs we will be pulling data for
currency_pairs = [
    ["EUR","USD",[],portfolio("EUR","USD")],
    ["GBP","USD",[],portfolio("GBP","USD")],    
    ["USD","CHF",[],portfolio("USD","CHF")],
    ["USD","CAD",[],portfolio("USD","CAD")],
    ["USD","HKD",[],portfolio("USD","HKD")],
    ["USD","AUD",[],portfolio("USD","AUD")],
    ["USD","NZD",[],portfolio("USD","NZD")],
    ["USD","SGD",[],portfolio("USD","SGD")],
    ]

import os
import csv
# import the necessary packages
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.utils.data import random_split
from torch.optim import Adam
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)

class LeNet(Module):
  def __init__(self, numChannels, classes):
		# call the parent constructor
    super(LeNet, self).__init__()
    self.fc1 = Linear(in_features=numChannels, out_features=20)
    self.relu = ReLU()
    # initialize our softmax classifier
    self.fc2 = Linear(in_features=20, out_features=classes)
    self.logSoftmax = LogSoftmax(dim=1)

  def forward(self, x):
    # print('fwd ',x)

    x = self.fc1(x)
    # print('after fc1',x)
    x = self.relu(x)
    # print('after relu',x)

    x = self.fc2(x)
    # print('after fc2',x)

    output = self.logSoftmax(x)
    # print('op ',output)

    # return the output predictions
    return output


def read_data(root_dir, currency_pairs) :
  csv_file_dir2 = root_dir+os.sep+'de2'+os.sep
  csv_file_dir = root_dir+os.sep+'de'+os.sep
  cvs_data = {}
  for currency in currency_pairs:
        base = currency[0]+currency[1]
        cvs_data[base] = []

  for currency in currency_pairs:
        base = currency[0]+currency[1]
        file = csv_file_dir+'raw_'+base+'.csv'
        # opening the CSV file
        with open(file, mode ='r')as file:
   
          # reading the CSV file
          csvFile = csv.reader(file)
      
          # displaying the contents of the CSV file
          for lines in csvFile:
            if len(cvs_data[base]) > 60*60*10: # 10 hour
              break
            cvs_data[base].append(float(lines[1]))
        
        # file = csv_file_dir2+'raw_'+base+'.csv'
        # file2UsedCount = 1
        # with open(file, mode ='r')as file:
        #   csvFile = csv.reader(file)
        #   for lines in csvFile:
        #     if len(cvs_data[base]) > 60*60*10: # 10 hour

        #       with open("used_data.txt","a") as file:
        #             file.write(str(file2UsedCount)+"\n")
        #       break
        #     cvs_data[base].append(float(lines[1]))
        #     file2UsedCount+=1
  for currency in currency_pairs:
        base = currency[0]+currency[1]
        print(base, len(cvs_data[base]))
  return cvs_data

def toCSV(dataframe,csv_name, class_1_size, class_2_size):
  count = 0
  data = {}
  for index, row in dataframe.iterrows():
    insert_time = row['period']
    mean_value = row['mean']
    return_value = row['returnVal'] # target
    if not isnan(return_value):
      return_value *= 100000
    else: 
      return_value = 0
    if count < class_1_size:
      with open(csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([insert_time, 1, 1, mean_value, return_value])
      data[insert_time] = [1, 1, mean_value, return_value]
    elif count < class_2_size:
      if count == class_1_size:
        with open('break.txt', 'a') as f:
          writer = csv.writer(f)
          writer.writerow([csv_name, 'bound1', row['vol'],row['fd']])
      
      with open(csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([insert_time, 2, 2, mean_value, return_value])
      data[insert_time] = [2, 2, mean_value, return_value]
    else:
      if count == class_2_size:
        with open('break.txt', 'a') as f:
          writer = csv.writer(f)
          writer.writerow([csv_name, 'bound2', row['vol'],row['fd']])
      with open(csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([insert_time, 3, 3, mean_value, return_value])
      data[insert_time] = [3, 3, mean_value, return_value]
    count += 1
  return data

def gen_train_csv(df, key):
  class_1_size = len(df)//3
  class_3_size = class_1_size
  class_2_size = len(df) - 2*class_3_size
  train_dir = 'train' + os.sep

  labeled_data = {}
  df1 = df.sort_values(by=['vol','fd'], ascending=True)
  csv_name = train_dir+key+'_vol_only.csv'
  data = toCSV(df1, csv_name, class_1_size, class_2_size)
  labeled_data['_vol_only'] = data
 
  df2 = df.sort_values(by='vol', ascending=True)
  csv_name = train_dir+key+'_vol_1st_fd_2nd.csv'
  data = toCSV(df2, csv_name, class_1_size, class_2_size)
  labeled_data['_vol_1st_fd_2nd'] = data

  df3 = df.sort_values(by=['fd','vol'], ascending=True)
  csv_name = train_dir+key+'_fd_1st_vol_2nd.csv'
  data = toCSV(df3, csv_name, class_1_size, class_2_size)
  labeled_data['_fd_1st_vol_2nd'] = data

  return labeled_data





class CustomDataset(Dataset):
    def __init__(self, text, labels):
        print('text ', len(text))
        print('label ', len(labels))
        print(type(text))
        print(text)

        self.text = text
        self.labels = labels
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]

def sort_add_label(data):

  count = 0
  agg_count = 0
  decision_count = 0
  currencyCheck = {}
  engine = create_engine("sqlite:///final.db", echo=False, future=True)
  with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_raw;"))
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_agg;"))
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_output;"))
  # Create the needed tables in the database
  initialize_raw_data_tables(engine,currency_pairs)
  initialize_aggregated_tables(engine,currency_pairs)
  # Create the output tables for all currency pairs
  initialize_output_tables(engine,currency_pairs)

  TEN_HOUR = 36000
  SIX_MINUTE = 360
  ONE_HOUR = 3600
  period_count = 0 
  window_count = 0
  N = {}
  UB = {}
  LB = {}
  lastPrice = {}
  while count < TEN_HOUR:
      if agg_count == SIX_MINUTE:
        period_count+=1
        UB, LB, returnVals, noStops = aggregate_raw_data_tables(engine, currency_pairs, period_count, N, currencyCheck)
        reset_raw_data_tables(engine,currency_pairs)
        agg_count = 0
        N = {}
        lastPrice = {}
      if decision_count == ONE_HOUR:
        window_count +=1
        currencyCheck = fill_output_data_tables(engine, currency_pairs, window_count, currencyCheck)
        decision_count = 0

      # Loop through each currency pair
      for currency in currency_pairs:
        # Set the input variables to the API
        from_ = currency[0]
        to = currency[1]

        # initializing currencyCheck for all currency pair with True
        if from_+to not in currencyCheck:
          currencyCheck[from_+to] = True

        # Run the below logic only when currencyCheck of the corresponding currency pair is set to True 
        if True:#currencyCheck[from_+to] == True:
          avg_price = data[from_+to][count]
          # Format the timestamp from the result
          dt = count
          # Get the current time and format it
          insert_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          # In the 1st period, UB and LB will be an empty dictionary so this condition will be false for 1st period
          if UB!={} and LB!={}:
            # From 2nd period to 100th period
            # if currency[0]+currency[1] key is in lastPrice, it means we have a last price to send in the countBandsCrossing() function and we update lastPrice with the current price which is the avg_price
            if currency[0]+currency[1] in lastPrice:
              N[currency[0]+currency[1]] += countBandsCrossing(lastPrice[currency[0]+currency[1]], avg_price, UB[currency[0]+currency[1]], LB[currency[0]+currency[1]])    
              lastPrice[currency[0]+currency[1]] = avg_price
            
            # if currency[0]+currency[1] key not in lastPrice and N, we initialize here
            else:
              lastPrice[currency[0]+currency[1]] = avg_price
              N[currency[0]+currency[1]] = 0

          # Write the data to the SQLite database, raw data tables
          with engine.begin() as conn:
              conn.execute(text("INSERT INTO "+from_+to+"_raw(ticktime, fxrate, inserttime) VALUES (:ticktime, :fxrate, :inserttime)"),[{"ticktime": dt, "fxrate": avg_price, "inserttime": insert_time}])
      
      # Increment the counters
      count += 1
      agg_count +=1
      decision_count +=1

  conn = sqlite3.connect('final.db', isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)

  print("Done computing vol and fd, begin sort")
  sort_type = ['_vol_only','_vol_1st_fd_2nd','_fd_1st_vol_2nd']

  labeled_data = {}
  for key in data:
    sql = "SELECT * FROM "+key+"_agg;"
    db_df = pd.read_sql_query(sql, conn)
    labeled_data[key] = gen_train_csv(db_df, key)
    weighted_labeled_data = {}
  for key in labeled_data:
    weights = [10]*3+[20]*3+[30]*3+[40]*3+[50]*3+[60]*3+[70]*3+[80]*3+[90]*3+[100]*3
    labeled_cur = labeled_data[key]
    weighted_labeled_data[key] = {}
    for tp in sort_type:
      unweighted_data = []
      weighted_labeled_data[key][tp] = []
      for k, v in labeled_cur[tp].items():
        unweighted_data = v[:-1]
        for currency in currency_pairs:
          other_coin_key = currency[0] + currency[1]
          if other_coin_key != key:
            # print(k, k in labeled_data[other_coin_key][tp].keys(),labeled_data[other_coin_key].keys())
            if k in labeled_data[other_coin_key][tp].keys():
              unweighted_data += labeled_data[other_coin_key][tp][k][:-1]
            else:
              unweighted_data += [0,0,0]
        weighted_data = [sum(v) for v in zip(unweighted_data, weights[0: len(unweighted_data)])]
        # print(len(unweighted_data), unweighted_data, weighted_data)
        label = 0
        if v[-1] > 0:
          label = 10
        elif v[-1] < 0:
          label = -10
        weighted_labeled_data[key][tp].append(weighted_data+[label])
      data_df = pd.DataFrame(data = weighted_labeled_data[key][tp], 
                        columns = [
                          "v11", "v12", "v13",
                          "v21", "v22", "v23",
                          "v31", "v32", "v33",
                          "v41", "v42", "v43",
                          "v51", "v52", "v53",
                          "v61", "v62", "v63",
                          "v71", "v72", "v73",
                          "v81", "v82", "v83",
                          # "v91", "v92", "v93",
                          # "v101", "v102", "v103",
                          "target"
                      ])
      
      from sklearn.model_selection import train_test_split

      train_val, test = train_test_split(data_df, test_size=0.1)
      train, val = train_test_split(train_val, test_size=0.1)


    
      BATCH_SIZE = 16
      TRAIN_SPLIT = 0.9
      INIT_LR = 1e-3
      EPOCHS = 10
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      
      dataset_train = CustomDataset(torch.tensor(train.drop('target', axis = 1).values.astype(np.float32)) , 
      torch.tensor(train['target'].values.astype(np.float32)))
      trainDataLoader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

      # dataset_val = CustomDataset(torch.tensor(list(pd.DataFrame(val, columns = ["v11", "v12", "v13",
      #                     "v21", "v22", "v23",
      #                     "v31", "v32", "v33",
      #                     "v41", "v42", "v43",
      #                     "v51", "v52", "v53",
      #                     "v61", "v62", "v63",
      #                     "v71", "v72", "v73",
      #                     "v81", "v82", "v83"]).values)), torch.tensor(list(pd.DataFrame(val, columns = ['target']))))
      dataset_val = CustomDataset(torch.tensor(val.drop('target', axis = 1).values.astype(np.float32)) , 
      torch.tensor(val['target'].values.astype(np.float32)))
      valDataLoader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)


      # dataset_test = CustomDataset(torch.tensor(list(pd.DataFrame(test, columns = ["v11", "v12", "v13",
      #                     "v21", "v22", "v23",
      #                     "v31", "v32", "v33",
      #                     "v41", "v42", "v43",
      #                     "v51", "v52", "v53",
      #                     "v61", "v62", "v63",
      #                     "v71", "v72", "v73",
      #                     "v81", "v82", "v83"]).values)), torch.tensor(list(pd.DataFrame(test, columns = ['target']))))
      dataset_test = CustomDataset(torch.tensor(test.drop('target', axis = 1).values.astype(np.float32)) , 
      torch.tensor(test['target'].values.astype(np.float32)))
      testDataLoader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
      
      # trainDataLoader = DataLoader(CustomDataset(train), batch_size=BATCH_SIZE, shuffle=True)          
      # valDataLoader = DataLoader(CustomDataset(val), batch_size=BATCH_SIZE)
      # testDataLoader = DataLoader(CustomDataset(test), batch_size=BATCH_SIZE, shuffle=True)  
    

      print(trainDataLoader)
      print(trainDataLoader.dataset)
      print(len(trainDataLoader.dataset))
      # print(trainDataLoader.dataset[0])
      # calculate steps per epoch for training and validation set
      trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
      valSteps = len(valDataLoader.dataset) // BATCH_SIZE
      # initialize the LeNet model
      print("[INFO] initializing the LeNet model...")
      model = LeNet(
      	numChannels=8*3,
      	classes=3).to(device) # loss -10 0 10
      # initialize our optimizer and loss function
      opt = Adam(model.parameters(), lr=INIT_LR)
      lossFn = nn.NLLLoss()
      # initialize a dictionary to store training history
      H = {
      	"train_loss": [],
      	"train_acc": [],
      	"val_loss": [],
      	"val_acc": []
      }
      # measure how long training is going to take
      print("[INFO] training the network...")
      
      # startTime = time.time()
      # loop over our epochs
      for e in range(0, EPOCHS):
        print('inside ', e)
        # set the model in training mode
        model.train()
        print('train done ')
        
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        print('into train loader')

        for i, data in enumerate(trainDataLoader, 0):
          print(i, data)
          (x,y) = data
          # send the input to the device
          (x, y) = (x.to(device), y.to(device))
          # perform a forward pass and calculate the training loss
          pred = model(x)
          loss = lossFn(pred, y)
          # zero out the gradients, perform the backpropagation step,
          # and update the weights
          opt.zero_grad()
          loss.backward()
          opt.step()
          # add the loss to the total training loss so far and
          # calculate the number of correct predictions
          totalTrainLoss += loss
          trainCorrect += (pred.argmax(1) == y).type(
          	torch.float).sum().item()
          # switch off autograd for evaluation
          with torch.no_grad():
          	# set the model in evaluation mode
            model.eval()
          	# loop over the validation set
            for (x, y) in valDataLoader:
            	# send the input to the device
              (x, y) = (x.to(device), y.to(device))
            	# make the predictions and calculate the validation loss
              pred = model(x)
              totalValLoss += lossFn(pred, y)
              # calculate the number of correct predictions
              valCorrect += (pred.argmax(1) == y).type(
              	torch.float).sum().item()
          	# calculate the average training and validation loss
          avgTrainLoss = totalTrainLoss / trainSteps
          avgValLoss = totalValLoss / valSteps
          # calculate the training and validation accuracy
          trainCorrect = trainCorrect / len(trainDataLoader.dataset)
          valCorrect = valCorrect / len(valDataLoader.dataset)
          # update our training history
          H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
          H["train_acc"].append(trainCorrect)
          H["val_loss"].append(avgValLoss.cpu().detach().numpy())
          H["val_acc"].append(valCorrect)
          # print the model training and validation information
          print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
          print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
          	avgTrainLoss, trainCorrect))
          print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))
  # finish measuring how long training took
      # endTime = time.time()
      # print("[INFO] total time taken to train the model: {:.2f}s".format(
      # 	endTime - startTime))
      # we can now evaluate the network on the test set
      print("[INFO] evaluating network...")
      # turn off autograd for testing evaluation
      with torch.no_grad():
      	# set the model in evaluation mode
        model.eval()

      	# initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
        	# send the input to the device
          x = x.to(device)
        	# make the predictions and add them to the list
          pred = model(x)
          preds.extend(pred.argmax(axis=1).cpu().numpy())
      # generate a classification report
      print(classification_report(testData.targets.cpu().numpy(),
	np.array(preds), target_names=testData.classes))


  return weighted_labeled_data

def get_prediction(currency_pairs, data, data_classification_bound, model_type):
    # Number of list iterations - each one should last about 1 second
    count = 0
    agg_count = 0
    # Counter to keep track of 10 intervals which is after every 60 minutes
    decision_count = 0

    # currencyCheck dictionary has keys as currencyPair and values as True or False
    # By default all are set to True but if we decide not to invest more and stop the position, we set this to False
    currencyCheck = {}
    
    # Create an engine to connect to the database; setting echo to false should stop it from logging in std.out
    engine = create_engine("sqlite:///final.db", echo=False, future=True)
    
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_raw;"))
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_agg;"))
            conn.execute(text("DROP TABLE "+curr[0]+curr[1]+"_output;"))
  
    # Create the needed tables in the database
    initialize_raw_data_tables(engine,currency_pairs)
    initialize_aggregated_tables(engine,currency_pairs)
    # Create the output tables for all currency pairs
    initialize_output_tables(engine,currency_pairs)
    
   
    # counter for period 
    period_count = 0 
    # counter for window 
    window_count = 0
    # dictionary to store N values for all currency pairs
    N = {}
    UB = {}
    LB = {}
    

    # stores last price of currencypairs
    lastPrice = {}

    # Loop that runs until the total duration of the program hits 10 hours. 
    # 36000 seconds = 10 hours_
    while count < 36000:
      
      # Make a check to see if 6 minutes has been reached or not
      if agg_count == 360:
        # Aggregate the data and clear the raw data tables
        period_count+=1
        UB, LB, returnVals, noStops = aggregate_raw_data_tables(engine, currency_pairs, period_count, N, currencyCheck)
        # print(UB, returnVals)
        with engine.begin() as conn:
          for curr in currency_pairs:
            # if curr[0]+curr[1] == 'USDCAD':
            #   continue
            
            sort_tp = model_type[curr[0]+curr[1]]
            model_path = './models/'+curr[0]+curr[1]+'_'+sort_tp
              # continue
            result = conn.execute(text("SELECT AVG(fxrate) as mean, MIN(fxrate) as min, MAX(fxrate) as max FROM "+curr[0]+curr[1]+"_raw;"))
            for row in result:
              mean = row.mean
              min = row.min
              max = row.max
              vol = (max-min)/mean
            if period_count == 1:
              fd = 0
            else:
              print(period_count )
              fd = calculateFD(N[curr[0]+curr[1]],vol)
              
            returnVal = returnVals[curr[0]+curr[1]] * 100000
            model_band = data_classification_bound[curr[0]+curr[1]]
            
            print(curr[0]+curr[1],count,sort_tp)
            if(sort_tp == '_vol_only' or sort_tp == '_vol_1st_fd_2nd' ):
              if (vol < model_band[0]): 
                vol = 1
                fd = 1
              elif(vol < model_band[2]):
                vol = 2
                fd = 2
              else:
                vol = 3
                fd = 3
            else:  # fd_1st
              if (fd < model_band[1]): 
                vol = 1
                fd = 1
              elif(fd < model_band[3]):
                vol = 2
                fd = 2
              else:
                vol = 3
                fd = 3
            
            pred_data = [vol, fd, mean]
            

            weights = [10]*3+[20]*3+[30]*3+[40]*3+[50]*3+[60]*3+[70]*3+[80]*3+[90]*3+[100]*3
            for o in currency_pairs:
              if(curr[0]+curr[1] != o[0]+o[1]):
                o_result = conn.execute(text("SELECT AVG(fxrate) as mean, MIN(fxrate) as min, MAX(fxrate) as max FROM "+o[0]+o[1]+"_raw;"))
                for o_row in o_result:
                  o_mean = o_row.mean
                  o_min = o_row.min
                  o_max = o_row.max
                  print(o_row,o[0]+o[1])
                  if o_min is None:
                    o_vol = 0
                  else:
                    o_vol = (o_max-o_min)/o_mean
                if period_count == 1:
                  o_fd = 0
                else:
                  print(o[0]+o[1],o_vol)
                  print(N)
                  o_fd = calculateFD(N[o[0]+o[1]],o_vol)
                o_model_band = data_classification_bound[o[0]+o[1]]
                o_sort_tp = model_type[o[0]+o[1]]
                if(o_sort_tp == '_vol_only' or o_sort_tp == '_vol_1st_fd_2nd' ):
                  if (o_vol < o_model_band[0]): 
                    o_vol = 1
                    o_fd = 1
                  elif(o_vol < o_model_band[2]):
                    o_vol = 2
                    o_fd = 2
                  else:
                    o_vol = 3
                    o_fd = 3
                else:  # fd_1st
                  if (o_fd < o_model_band[1]): 
                    o_vol = 1
                    o_fd = 1
                  elif(o_fd < o_model_band[3]):
                    o_vol = 2
                    o_fd = 2
                  else:
                    o_vol = 3
                    o_fd = 3
                pred_data += [o_vol, o_fd, o_mean]
            
            weighted_data = [sum(v) for v in zip(pred_data, weights[0: len(pred_data)])]
            weighted_data += [returnVal]
            data_df = pd.DataFrame([weighted_data])
            data_df.columns =[
                          "v11", "v12", "v13",
                          "v21", "v22", "v23",
                          "v31", "v32", "v33",
                          "v41", "v42", "v43",
                          "v51", "v52", "v53",
                          "v61", "v62", "v63",
                          "v71", "v72", "v73",
                          "v81", "v82", "v83",
                          # "v91", "v92", "v93",
                          # "v101", "v102", "v103",
                          "target"
                      ]
            # print(model_path)
            model = load_model(model_path) 
            # print(model_path, model)
            prediction = predict_model(model, data = data_df)
            

            prediction['loss'] = prediction['target'] - prediction['prediction_label']
            pred_value = list(prediction['prediction_label'])[0]
            target_value = list(prediction['target'])[0]
            signalAligned = (target_value * pred_value >= 0)
            if signalAligned:
              if noStops[curr[0]+curr[1]]:
                prediction['stop'] =  'REINVEST'
              else:
                prediction['stop'] ='NoAction'
            else:
              prediction['stop'] = 'STOP'
            # to_file_pred = pd.DataFrame(prediction, columns=['v13','loss','stop','target','prediction_label'])
            if period_count == 1:
              prediction.to_csv(curr[0]+curr[1]+'PredResult.csv', mode='a', index=False, header=True)

            else:
              prediction.to_csv(curr[0]+curr[1]+'PredResult.csv', mode='a', index=False, header=False)
            print(prediction)

        reset_raw_data_tables(engine,currency_pairs)

        agg_count = 0
        N = {}
        lastPrice = {}
        
      # Only call the api every 1 second, so wait here for 0.75 seconds, because the 
      # code takes about .15 seconds to run
      # time.sleep(0.75)

      # Only called when 10 periods of 6 mins are done i.e after every 60 mins
      if decision_count == 3600:
        window_count +=1
        currencyCheck = fill_output_data_tables(engine, currency_pairs, window_count, currencyCheck)
        decision_count = 0
      
      # Increment the counters
      count += 1
      agg_count +=1
      decision_count +=1

      # Loop through each currency pair
      for currency in currency_pairs:
        # Set the input variables to the API
        from_ = currency[0]
        to = currency[1]

        # initializing currencyCheck for all currency pair with True
        if from_+to not in currencyCheck:
          currencyCheck[from_+to] = True

        # Run the below logic only when currencyCheck of the corresponding currency pair is set to True 
        if True:#currencyCheck[from_+to] == True:
          # Call the API with the required parameters
          avg_price = data[from_+to][count-1]
          # print(count-1, len(data[from_+to]))
          # if from_+to == 'EURUSD':
            # print(avg_price, count)

          dt = count-1

          # Get the current time and format it
          insert_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          
          
          # In the 1st period, UB and LB will be an empty dictionary so this condition will be false for 1st period
          if UB!={} and LB!={}:

            # From 2nd period to 100th period
            # if currency[0]+currency[1] key is in lastPrice, it means we have a last price to send in the countBandsCrossing() function and we update lastPrice with the current price which is the avg_price
            if currency[0]+currency[1] in lastPrice:
              # print(UB)
              # print(LB)
              if currency[0]+currency[1] not in UB.keys():
                N[currency[0]+currency[1]] = N[currency[0]+currency[1]]
              else:
                N[currency[0]+currency[1]] += countBandsCrossing(lastPrice[currency[0]+currency[1]], avg_price, UB[currency[0]+currency[1]], LB[currency[0]+currency[1]])    
              lastPrice[currency[0]+currency[1]] = avg_price
              # print('bp1',N)
            
            # if currency[0]+currency[1] key not in lastPrice and N, we initialize here
            else:
              lastPrice[currency[0]+currency[1]] = avg_price
              N[currency[0]+currency[1]] = 0
              # print(N)
              # print('bp2',N)

          # Write the data to the SQLite database, raw data tables
          with engine.begin() as conn:
              conn.execute(text("INSERT INTO "+from_+to+"_raw(ticktime, fxrate, inserttime) VALUES (:ticktime, :fxrate, :inserttime)"),[{"ticktime": dt, "fxrate": avg_price, "inserttime": insert_time}])


data = read_data('./', currency_pairs)
labeled_data = sort_add_label(data)


# cvs_data = {}


# for currency in currency_pairs:
#   base = currency[0]+currency[1]
#   cvs_data[base] = []

# for currency in currency_pairs:
#   base = currency[0]+currency[1]
#   file = './de2/'+'raw_'+base+'.csv'
#   count = 0
#   with open(file, mode ='r')as file:
#     csvFile = csv.reader(file)
#     for lines in csvFile:
#       if len(cvs_data[base]) < 60*60*10: # 10 hour  
#         cvs_data[base].append(float(lines[1]))
#       else:
#         break
# print("begin predicting")


# get_prediction(currency_pairs, cvs_data, data_classification_bound, model_type)

