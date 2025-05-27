# algorithm-neuralnetwork-ml-
this is my attempt on the idea that i proposed
# Imports
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings

warnings.filterwarnings('ignore')
truedatabase = smp500results

inputs = smp500
newinputs = tradinginput
def function volatility(x)
x =((np.log(df['high'] / df['low']) ** 2) / 2 -
    (2 * np.log(2) - 1) * (np.log(df['adj close'] / df['open']) ** 2)
return(x)
def function rsi(xs)
xs = df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
return(xs)
def function boling(bb)
bb = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=x, length=20))
df['bb_upper'] = bb['BBU_20_2.0']
df['bb_lower'] = bb['BBL_20_2.0']
return(bb)
def function atr(art)
art =df['atr'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.atr(high=df['high'], low=df['low'], close=x, length=14))
return(art)
input1 = volatility(inputs)
input2 = rsi(inputs)
input3 = boling(inputs)
input4 = art(inputs)
if input1 = low:
	input1 = 1
elif input1 = high:
	input1 = 0
if input2 > 70:
	input2 = 0
elif input2 <30:
	input2 = 0
else input2 = 1
if input3 = rising:
 input3 = 0:
if input3 = low:
 input3 = 1
if input4 = bullish:
	input4 = 1
elif input 4 = bearish:
	input4 = 0
elif input4 > 0
	input4 = 1
elif input 4 <0:
	input 4 = 0
epoch = 10000
for I in range(epoch)
weigths = np.random(-1,1).uniform(4,4)
weigths2 = np.random(-1,1).uniform(4,4)
weigths3 = np.random(-1,1).uniform(4,4)
input = [input1,input2,input3,input4]
input = np.reshape(input(1,4):
def function sigmoid(x)
x = 1/1-x**-x
return(x)
def function sigmoidderivative(x)
z1 = np.dot(weights,input)+3
layer1 = sigmoid(z1)
z2 = np.dot(layer1,weights2)+3
layer2 = sigmoid(z2)
z3 = np.dot(layer2,weights3)+3
output = sigmoid(z3)
cost = (output - truedatabase)**2
costderivative = 2(output-truedatabase)
l1derive = sigmoid(z1)
l2derive = sigmoid(z2)
l3derive = sigmoid(z3)
derivative = input
backpropagation1 = costderivative*l1derive*input
backpropagation2 = costderivative*l2derive*input
lrate = 0.1
weight = weight-backpropogation*lrate
weight2 = weight2-backpropogation2*lrate
weight3 = weight3-backpropogation3*lrate
if epoch = 200:
	print(cost,output)
	data = input("do you like this data")
	if data = yes:
	input = newinputs
else:
	print(ok)
if epoch = 1000:
	print(cost,output)
	data = input("do you like this data")
	if data = yes:
	input = newinputs
else:
	print("ok")
if epoch = 5000:
	print(cost,output)
	data = input("do you like this data")
	if data = yes:
	input = newinputs
else:
	print("ok")
if epoch = 10000:
	print(cost,output)
	data = input("do you like this data")
	if data = yes:
	input = newinputs
else:
	print("ok")
