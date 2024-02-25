from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import random
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import csv

def PSK8(): #definicja punktów konstelacji
    PSK8_points=[math.sqrt(2+math.sqrt(2))/2+1j*math.sqrt(2-math.sqrt(2))/2,math.sqrt(2-math.sqrt(2))/2+1j*math.sqrt(2+math.sqrt(2))/2,-math.sqrt(2-math.sqrt(2))/2+1j*math.sqrt(2+math.sqrt(2))/2,-math.sqrt(2+math.sqrt(2))/2+1j*math.sqrt(2-math.sqrt(2))/2,-math.sqrt(2+math.sqrt(2))/2-1j*math.sqrt(2-math.sqrt(2))/2,-math.sqrt(2-math.sqrt(2))/2-1j*math.sqrt(2+math.sqrt(2))/2,math.sqrt(2-math.sqrt(2))/2-1j*math.sqrt(2+math.sqrt(2))/2,math.sqrt(2+math.sqrt(2))/2-1j*math.sqrt(2-math.sqrt(2))/2] #przesunięte 8psk
    #PSK8_points=[1, 0.7071 + 1j * 0.7071, 1j, -0.7071 + 1j * 0.7071, -1, -0.7071 - 1j * 0.7071, -1j, 0.7071 - 1j * 0.7071] #typowe 8psk
    return PSK8_points

def QAM8(): #definicja punktów konstelacji
    QAM8_points=[3+1j, 1+1j, -1+1j, -3+1j, -3-1j, -1-1j, 1-1j, 3-1j]
    return QAM8_points

def plot_points(points,konstelacja,koder): #wykres konstelacji
    x=[ele.real for ele in points] 
    y=[ele.imag for ele in points]
    plt.plot(x,y, 'ro',label=(konstelacja+" "+koder))
    if konstelacja=="QAM8":
        QAM8_points=QAM8()
        x_QAM8=[ele.real for ele in QAM8_points] 
        y_QAM8=[ele.imag for ele in QAM8_points]
        plt.plot(x_QAM8,y_QAM8, 'bx')
    elif konstelacja=="PSK8":
        PSK8_points=PSK8()
        x_PSK8=[ele.real for ele in PSK8_points] 
        y_PSK8=[ele.imag for ele in PSK8_points]
        plt.plot(x_PSK8,y_PSK8, 'bx')

    plt.legend()
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid()
    plt.show()
    return

def data_to_plot8(data,PSK8_points,elements): #mapowanie punktów
    data2plot=np.zeros(elements,dtype=np.complex_)
    for i in range(len(data)):
        if data[i]==0b000:
            data2plot[i]=PSK8_points[0]
        if data[i]==0b001:
            data2plot[i]=PSK8_points[1]
        if data[i]==0b010:
            data2plot[i]=PSK8_points[2]
        if data[i]==0b011:
            data2plot[i]=PSK8_points[3]
        if data[i]==0b100:
            data2plot[i]=PSK8_points[4]
        if data[i]==0b101:
            data2plot[i]=PSK8_points[5]
        if data[i]==0b110:
            data2plot[i]=PSK8_points[6]
        if data[i]==0b111:
            data2plot[i]=PSK8_points[7]
    return data2plot

def ber_plot(x,y,konstelacja,koder): #wykres BER
    plt.plot(x,y,'r',label=(konstelacja+" "+koder))
    plt.xlabel('SNR[dB]')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()
    return

def ber_plot_file(): #porównanie BER
    with open('csv\PSK8_Naiwny.csv') as dane:
        PSK8_silly_y=[]
        PSK8_silly_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])-1):
            PSK8_silly_x.append(temp2[0][i])
        for i in range(len(temp)-1):
            PSK8_silly_y.append(float(temp[i]))
            
    with open('csv\QAM8_Naiwny.csv') as dane:
        QAM8_silly_y=[]
        QAM8_silly_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            QAM8_silly_x.append(temp2[0][i])
        for i in range(len(temp)):
            QAM8_silly_y.append(float(temp[i]))
    
    with open('csv\PSK8_25.csv') as dane:
        PSK8_25_y=[]
        PSK8_25_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            PSK8_25_x.append(temp2[0][i])
        for i in range(len(temp)):
            PSK8_25_y.append(float(temp[i]))
    with open('csv\QAM8_25.csv') as dane:
        QAM8_25_y=[]
        QAM8_25_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            QAM8_25_x.append(temp2[0][i])
        for i in range(len(temp)):
            QAM8_25_y.append(float(temp[i]))
    with open('csv\QAM8_43.csv') as dane:
        QAM8_43_y=[]
        QAM8_43_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            QAM8_43_x.append(temp2[0][i])
        for i in range(len(temp)):
            QAM8_43_y.append(float(temp[i]))
    with open('csv\PSK8_43.csv') as dane:
        PSK8_43_y=[]
        PSK8_43_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            PSK8_43_x.append(temp2[0][i])
        for i in range(len(temp)):
            PSK8_43_y.append(float(temp[i]))
    with open('csv\PSK8_57.csv') as dane:
        PSK8_57_y=[]
        PSK8_57_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            PSK8_57_x.append(temp2[0][i])
        for i in range(len(temp)):
            PSK8_57_y.append(float(temp[i]))
    with open('csv\QAM8_57.csv') as dane:
        QAM8_57_y=[]
        QAM8_57_x=[]
        csvreader=csv.reader(dane, delimiter=',')
        temp=next(csvreader)
        temp2=[row for row in csvreader]
        for i in range(len(temp2[0])):
            QAM8_57_x.append(temp2[0][i])
        for i in range(len(temp)):
            QAM8_57_y.append(float(temp[i]))
    plt.plot(PSK8_silly_x,PSK8_silly_y, label='PSK8_Naiwny')
    plt.plot(QAM8_silly_x,QAM8_silly_y, label='QAM8_Naiwny')
    plt.plot(PSK8_25_x,PSK8_25_y, label='PSK8_25')
    plt.plot(QAM8_25_x,QAM8_25_y, label='QAM8_25')
    plt.plot(QAM8_43_x,QAM8_43_y, label='QAM8_43')
    plt.plot(PSK8_43_x,PSK8_43_y, label='PSK8_43')
    plt.plot(QAM8_57_x,QAM8_57_y, label='QAM8_57')
    plt.plot(PSK8_57_x,PSK8_57_y, label='PSK8_57')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.show()
    
    return

def data_gen(elements): #generowanie ciągu wejściowego
    bit_list=[0b00,0b01,0b10,0b11]
    input_data=[]
    for i in range(elements):
        input_data.append(random.choice(bit_list))
    return input_data

def coder_silly(input_data): #koder naiwny
    code_data=[]
    memory=0
    for i in range(len(input_data)):
        if i>0:
            memory=input_data[i-1]&0b01
        code_data.append((input_data[i]<<1)+memory)
    return code_data

def coder_5_7(input_data): #koder 57
    code_data=[]
    memo1=0
    memo2=0
    for i in range(len(input_data)):
      memory1=((input_data[i]&0b01)^memo2)<<1
      memory2=(input_data[i]&0b01)^memo2^memo1
      pierwszy_bit=(input_data[i]&0b10)<<1
      if i>0:
        memo2=memo1
        memo1=input_data[i-1]&0b01
        memory1=((input_data[i]&0b01)^memo2)<<1
        memory2=(input_data[i]&0b01)^memo2^memo1

      code_data.append(pierwszy_bit + memory1 + memory2)

    return code_data

def coder_4_3(input_data): #koder 43
    code_data=[]
    memo1=0
    memo2=0
    memory=memo2^memo1
    for i in range(len(input_data)):
        if i>0:
            memo2=memo1
            memo1=input_data[i-1]&0b01
            memory=memo2^memo1
        code_data.append((input_data[i]<<1)+memory)
    return code_data

def coder_2_5(input_data): #koder 25
    code_data=[]
    memo1=0
    memo2=0
    for i in range(len(input_data)):
      memory1=memo1<<1
      memory2=(input_data[i]&0b01)^memo2
      pierwszy_bit=(input_data[i]&0b10)<<1
      if i>0:
        memo2=memo1
        memo1=input_data[i-1]&0b01
        memory1=memo1<<1
        memory2=memo2^(input_data[i]&0b01)
      code_data.append(pierwszy_bit + memory1 + memory2)
    return code_data

def add_noise(code_data,SNR): #generowanie i dodawanie szumu
    noise_level=(math.sqrt(2)/2)*10**(-SNR/20)
    code_data_noised=[]
    # noise=np.random.randn(lenght)*noise_level + 1j*np.random.randn(lenght)*noise_level #inny typ szumu
    for i in range(len(code_data)):
        noise_real=np.random.normal(0,1)*noise_level
        noise_imag=1j*np.random.normal(0,1)*noise_level
        code_data_noised.append(code_data[i]+noise_real+noise_imag)
    return code_data_noised

def decoder_silly(code_data_noised,elements,points8): #dekoder naiwny
    uncoded=[]
    decoded=[]
    decoded_string=[]
    cost=np.zeros((2,elements))
    step=np.zeros((2,elements))
    for i in range(len(code_data_noised)):
        if code_data_noised[i].imag>=0:
            uncoded.append(0)
            if i==0:
                cost[0][i]=math.sqrt(((points8[0].real-code_data_noised[i].real)**2)+((points8[0].imag-code_data_noised[i].imag)**2))
                cost[1][i]=math.sqrt(((points8[2].real-code_data_noised[i].real)**2)+((points8[2].imag-code_data_noised[i].imag)**2))
                step[0][i]=0
                step[1][i]=0    
            else:
                cost_0_to_0=(cost[0][i-1]+math.sqrt(((points8[0].real-code_data_noised[i].real)**2)+((points8[0].imag-code_data_noised[i].imag)**2)))
                cost_1_to_0=(cost[1][i-1]+math.sqrt(((points8[1].real-code_data_noised[i].real)**2)+((points8[1].imag-code_data_noised[i].imag)**2)))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_1_to_1=(cost[1][i-1]+math.sqrt(((points8[3].real-code_data_noised[i].real)**2)+((points8[3].imag-code_data_noised[i].imag)**2)))
                cost_0_to_1=(cost[0][i-1]+math.sqrt(((points8[2].real-code_data_noised[i].real)**2)+((points8[2].imag-code_data_noised[i].imag)**2)))
                if cost_1_to_1>=cost_0_to_1:
                    cost[1][i]=cost_0_to_1
                    step[1][i]=0
                else:
                    cost[1][i]=cost_1_to_1
                    step[1][i]=1
        else:
            uncoded.append(1)
            if i==0:
                cost[0][i]=math.sqrt(((points8[4].real-code_data_noised[i].real)**2)+((points8[4].imag-code_data_noised[i].imag)**2))
                cost[1][i]=math.sqrt(((points8[6].real-code_data_noised[i].real)**2)+((points8[6].imag-code_data_noised[i].imag)**2))
                step[0][i]=0
                step[1][i]=0
            else:
                cost_0_to_0=(cost[0][i-1]+math.sqrt(((points8[4].real-code_data_noised[i].real)**2)+((points8[4].imag-code_data_noised[i].imag)**2)))
                cost_1_to_0=(cost[1][i-1]+math.sqrt(((points8[5].real-code_data_noised[i].real)**2)+((points8[5].imag-code_data_noised[i].imag)**2)))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_1_to_1=(cost[1][i-1]+math.sqrt(((points8[7].real-code_data_noised[i].real)**2)+((points8[7].imag-code_data_noised[i].imag)**2)))
                cost_0_to_1=(cost[0][i-1]+math.sqrt(((points8[6].real-code_data_noised[i].real)**2)+((points8[6].imag-code_data_noised[i].imag)**2)))
                if cost_1_to_1>=cost_0_to_1:
                    cost[1][i]=cost_0_to_1
                    step[1][i]=0
                else:
                    cost[1][i]=cost_1_to_1
                    step[1][i]=1
         
    if cost[0][len(code_data_noised)-1]<cost[1][len(code_data_noised)-1]:
        row=0
    else:
        row=1

    for i in range(step.shape[1]-1,-1,-1):
        temp=step[row][i]
        if row==0 and temp==0:
            decoded.append(0)
            row=0
        elif row==0 and temp==1:
            decoded.append(0)
            row=1
        elif row==1 and temp==0:
            decoded.append(1)
            row=0
        elif row==1 and temp==1:
            decoded.append(1)
            row=1

    decoded.reverse()

    decoded_string=bits_to_decoded(uncoded,decoded,elements)

    return decoded_string

def decoder_5_7(code_data_noised,elements,points8): #dekoder 57
    uncoded=[]
    decoded=[]
    decoded_string=[]
    cost = np.zeros((4,elements))
    step = np.zeros((4, elements))
    for i in range (len(code_data_noised)):
        if code_data_noised[i].imag>=0:
            uncoded.append(0)
            if i==0:
                for j in range(1, 4, 1):
                    if j==2:
                        cost[0][i]=math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        cost[2][i]=math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag])   
                        step[0][i]=0
                        step[2][i]=0
                    else:
                        cost[j][i]=10000000
                        step[j][i]=5
            elif i==1:
                    cost[0][i]=(cost[0][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[1][i]=(cost[2][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[2][i]=(cost[0][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag]))                    
                    cost[3][i]=(cost[2][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    step[0][i]=0
                    step[1][i]=2
                    step[2][i]=0
                    step[3][i]=2
            else:
                cost_0_to_0=(cost[0][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_0=(cost[1][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_2_to_1=(cost[2][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_1=(cost[3][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag])) 
                if cost_2_to_1>=cost_3_to_1:
                    cost[1][i]=cost_3_to_1
                    step[1][i]=3
                else:
                    cost[1][i]=cost_2_to_1
                    step[1][i]=2
                cost_0_to_2=(cost[0][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_2=(cost[1][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_2>=cost_1_to_2:
                    cost[2][i]=cost_1_to_2
                    step[2][i]=1
                else:
                    cost[2][i]=cost_0_to_2
                    step[2][i]=0
                cost_2_to_3=(cost[2][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_3=(cost[3][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_3>=cost_3_to_3:
                    cost[3][i]=cost_3_to_3
                    step[3][i]=3
                else:
                    cost[3][i]=cost_2_to_3
                    step[3][i]=2
        else:
            uncoded.append(1)
            if i==0:
                for j in range(1, 4, 1):
                    if j==2:
                        cost[0][i]=math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        cost[2][i]=math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]) 
                        step[0][i]=0
                        step[2][i]=0
                    else:
                        cost[j][i]=10000000
                        step[j][i]=5
            elif i==1:
                    cost[0][i]=(cost[0][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[1][i]=(cost[2][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[2][i]=(cost[0][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[3][i]=(cost[2][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    step[0][i]=0
                    step[1][i]=2
                    step[2][i]=0
                    step[3][i]=2
            else:
                cost_0_to_0=(cost[0][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_0=(cost[1][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_2_to_1=(cost[2][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_1=(cost[3][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_1>=cost_3_to_1:
                    cost[1][i]=cost_3_to_1
                    step[1][i]=3
                else:
                    cost[1][i]=cost_2_to_1
                    step[1][i]=2
                cost_0_to_2=(cost[0][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_2=(cost[1][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_2>=cost_1_to_2:
                    cost[2][i]=cost_1_to_2
                    step[2][i]=1
                else:
                    cost[2][i]=cost_0_to_2
                    step[2][i]=0
                cost_2_to_3=(cost[2][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_3=(cost[3][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_3>=cost_3_to_3:
                    cost[3][i]=cost_3_to_3
                    step[3][i]=3
                else:
                    cost[3][i]=cost_2_to_3
                    step[3][i]=2

    lowest=cost[0][len(code_data_noised)-1]
    current_state=0
    for i in range (1,4,1):
        if lowest>cost[i][len(code_data_noised)-1]:
            lowest=cost[i][len(code_data_noised)-1]
            current_state=i

    decoded=step_to_bits(step,current_state)

    decoded.reverse()

    decoded_string=bits_to_decoded(uncoded,decoded,elements)
    # print(decoded_string)

    return (decoded_string)

def decoder_4_3(code_data_noised,elements,points8): #dekoder 43
    uncoded=[]
    decoded=[]
    decoded_string=[]
    cost = np.zeros((4,elements))
    step = np.zeros((4, elements))
    for i in range (len(code_data_noised)):
        if code_data_noised[i].imag>=0:
            uncoded.append(0)
            if i==0:
                for j in range(1, 4, 1):
                    if j==2:
                        cost[0][i]=math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        cost[2][i]=math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag])   
                        step[0][i]=0
                        step[2][i]=0
                    else:
                        cost[j][i]=10000000
                        step[j][i]=5
            elif i==1:
                    cost[0][i]=(cost[0][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[1][i]=(cost[2][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[2][i]=(cost[0][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[3][i]=(cost[2][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    step[0][i]=0
                    step[1][i]=2
                    step[2][i]=0
                    step[3][i]=2
            else:
                cost_0_to_0=(cost[0][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_0=(cost[1][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_2_to_1=(cost[2][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_1=(cost[3][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag])) 
                if cost_2_to_1>=cost_3_to_1:
                    cost[1][i]=cost_3_to_1
                    step[1][i]=3
                else:
                    cost[1][i]=cost_2_to_1
                    step[1][i]=2
                cost_0_to_2=(cost[0][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_2=(cost[1][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_2>=cost_1_to_2:
                    cost[2][i]=cost_1_to_2
                    step[2][i]=1
                else:
                    cost[2][i]=cost_0_to_2
                    step[2][i]=0
                cost_2_to_3=(cost[2][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_3=(cost[3][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_3>=cost_3_to_3:
                    cost[3][i]=cost_3_to_3
                    step[3][i]=3
                else:
                    cost[3][i]=cost_2_to_3
                    step[3][i]=2
        else:
            uncoded.append(1)
            if i==0:
                for j in range(1, 4, 1):
                    if j==2:
                        cost[0][i]=math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag])  
                        cost[2][i]=math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        step[0][i]=0
                        step[2][i]=0
                    else:
                        cost[j][i]=10000000
                        step[j][i]=5
            elif i==1:
                    cost[0][i]=(cost[0][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[1][i]=(cost[2][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[2][i]=(cost[0][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[3][i]=(cost[2][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    step[0][i]=0
                    step[1][i]=2
                    step[2][i]=0
                    step[3][i]=2
            else:
                cost_0_to_0=(cost[0][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_0=(cost[1][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_2_to_1=(cost[2][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_1=(cost[3][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_1>=cost_3_to_1:
                    cost[1][i]=cost_3_to_1
                    step[1][i]=3
                else:
                    cost[1][i]=cost_2_to_1
                    step[1][i]=2
                cost_0_to_2=(cost[0][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_2=(cost[1][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_2>=cost_1_to_2:
                    cost[2][i]=cost_1_to_2
                    step[2][i]=1
                else:
                    cost[2][i]=cost_0_to_2
                    step[2][i]=0
                cost_2_to_3=(cost[2][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_3=(cost[3][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_3>=cost_3_to_3:
                    cost[3][i]=cost_3_to_3
                    step[3][i]=3
                else:
                    cost[3][i]=cost_2_to_3
                    step[3][i]=2

    lowest=cost[0][len(code_data_noised)-1]
    current_state=0
    for i in range (1,4,1):
        if lowest>cost[i][len(code_data_noised)-1]:
            lowest=cost[i][len(code_data_noised)-1]
            current_state=i

    decoded=step_to_bits(step,current_state)

    decoded.reverse()

    decoded_string=bits_to_decoded(uncoded,decoded,elements)
    # print(decoded_string)

    return (decoded_string) 

def decoder_2_5(code_data_noised,elements,points8): #dekoder 25
    uncoded=[]
    decoded=[]
    decoded_string=[]
    cost = np.zeros((4,elements))
    step = np.zeros((4,elements))
    for i in range (len(code_data_noised)):
        if code_data_noised[i].imag>=0:
            uncoded.append(0)
            if i==0:
                for j in range(1, 4, 1):
                    if j==2:
                        cost[0][i]=math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        cost[2][i]=math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        step[0][i]=0
                        step[2][i]=0
                    else:
                        cost[j][i]=10000000
                        step[j][i]=5
            elif i==1:
                    cost[0][i]=cost[0][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                    cost[1][i]=cost[2][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                    cost[2][i]=cost[0][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                    cost[3][i]=cost[2][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                    step[0][i]=0
                    step[1][i]=2
                    step[2][i]=0
                    step[3][i]=2
            else:
                cost_0_to_0=(cost[0][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_0=(cost[1][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_2_to_1=(cost[2][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_1=(cost[3][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag])) 
                if cost_2_to_1>=cost_3_to_1:
                    cost[1][i]=cost_3_to_1
                    step[1][i]=3
                else:
                    cost[1][i]=cost_2_to_1
                    step[1][i]=2
                cost_0_to_2=(cost[0][i-1]+math.dist([points8[1].real,points8[1].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_2=(cost[1][i-1]+math.dist([points8[0].real,points8[0].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_2>=cost_1_to_2:
                    cost[2][i]=cost_1_to_2
                    step[2][i]=1
                else:
                    cost[2][i]=cost_0_to_2
                    step[2][i]=0
                cost_2_to_3=(cost[2][i-1]+math.dist([points8[3].real,points8[3].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_3=(cost[3][i-1]+math.dist([points8[2].real,points8[2].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_3>=cost_3_to_3:
                    cost[3][i]=cost_3_to_3
                    step[3][i]=3
                else:
                    cost[3][i]=cost_2_to_3
                    step[3][i]=2
        else:
            uncoded.append(1)
            if i==0:
                for j in range(1, 4, 1):
                    if j==2:
                        cost[0][i]=math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        cost[2][i]=math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag])
                        step[0][i]=0
                        step[2][i]=0
                    else:
                        cost[j][i]=10000000
                        step[j][i]=5
            elif i==1:
                    cost[0][i]=(cost[0][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[1][i]=(cost[2][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[2][i]=(cost[0][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    cost[3][i]=(cost[2][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                    step[0][i]=0
                    step[1][i]=2
                    step[2][i]=0                                         
                    step[3][i]=2
            else:
                cost_0_to_0=(cost[0][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_0=(cost[1][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_0>=cost_1_to_0:
                    cost[0][i]=cost_1_to_0
                    step[0][i]=1
                else:
                    cost[0][i]=cost_0_to_0
                    step[0][i]=0
                cost_2_to_1=(cost[2][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_1=(cost[3][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_1>=cost_3_to_1:
                    cost[1][i]=cost_3_to_1
                    step[1][i]=3
                else:
                    cost[1][i]=cost_2_to_1
                    step[1][i]=2
                cost_0_to_2=(cost[0][i-1]+math.dist([points8[5].real,points8[5].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_1_to_2=(cost[1][i-1]+math.dist([points8[4].real,points8[4].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_0_to_2>=cost_1_to_2:
                    cost[2][i]=cost_1_to_2
                    step[2][i]=1
                else:
                    cost[2][i]=cost_0_to_2
                    step[2][i]=0
                cost_2_to_3=(cost[2][i-1]+math.dist([points8[7].real,points8[7].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                cost_3_to_3=(cost[3][i-1]+math.dist([points8[6].real,points8[6].imag],[code_data_noised[i].real,code_data_noised[i].imag]))
                if cost_2_to_3>=cost_3_to_3:
                    cost[3][i]=cost_3_to_3
                    step[3][i]=3
                else:
                    cost[3][i]=cost_2_to_3
                    step[3][i]=2

    lowest=cost[0][len(code_data_noised)-1]
    current_state=0
    for i in range (1,4,1):
        if lowest>cost[i][len(code_data_noised)-1]:
            lowest=cost[i][len(code_data_noised)-1]
            current_state=i

    decoded=step_to_bits(step,current_state)

    decoded.reverse()

    decoded_string=bits_to_decoded(uncoded,decoded,elements)

    return (decoded_string) 

def error_counter(data_oryginal,data_decoded): #porównanie bitów oryginalnych i zdekodowanych
    counter=0
    for i in range(len(data_decoded)):
        if data_oryginal[i]!=data_decoded[i]:
            counter+=1
    return counter

def step_to_bits(step,current_state):
    decoded=[]
    for i in range(step.shape[1]-1,-1,-1):
        x = step[current_state][i]
        if current_state == 0 and x==0:
            decoded.append(0)
            current_state=0
        elif current_state==0 and x==1:
            decoded.append(0)
            current_state=1
        elif current_state == 1 and x==2:
            decoded.append(0)
            current_state=2
        elif current_state==1 and x==3:
            decoded.append(0)
            current_state=3
        elif current_state==2 and x==0:
            decoded.append(1)
            current_state=0
        elif current_state==2 and x==1:
            decoded.append(1)
            current_state=1
        elif current_state==3 and x==2:
            decoded.append(1)
            current_state=2
        elif current_state==3 and x==3:
            decoded.append(1)
            current_state=3
    return decoded

def bits_to_decoded(uncoded,decoded,elements):
    decoded_string=[]
    for i in range(0,elements,1):
        if (uncoded[i]==0 and decoded[i]==0):
            decoded_string.append(0b00)
        if (uncoded[i]==0 and decoded[i]==1):
            decoded_string.append(0b01)
        if (uncoded[i]==1 and decoded[i]==0):
            decoded_string.append(0b10)
        if (uncoded[i]==1 and decoded[i]==1):
            decoded_string.append(0b11)
    return decoded_string

def run(snr,elem,konstelacja,Name,koder,max):   #kompilacja
    root.quit()
    max=int(max)
    SNR=int(snr)
    elements=int(elem)
    if konstelacja=='PSK8':
        points=PSK8()
    elif konstelacja=='QAM8':
        points=QAM8()
    error_plot=[]
    snr_plot=[]
    T.insert(END,konstelacja+" "+koder+"\n")
    for s in range(SNR,20,1):
        counter=0
        error_count=0
        data_with_noise=[]
        while(error_count<100):
            data_oryginal=data_gen(elements)
            if koder=='Naiwny':
                data=coder_silly(data_oryginal)
            if koder=='25':
                data=coder_2_5(data_oryginal)
            if koder=='43':
                data=coder_4_3(data_oryginal)
            if koder=='57':
                data=coder_5_7(data_oryginal)
            data=data_to_plot8(data,points,elements)
            data=add_noise(data,s)
            for i in range(len(data)):
                data_with_noise.append(data[i])
            if koder=='Naiwny':
                data_decoded=decoder_silly(data,elements,points)
            if koder=='25':
                data_decoded=decoder_2_5(data,elements,points)
            if koder=='43':
                data_decoded=decoder_4_3(data,elements,points)
            if koder=='57':
                data_decoded=decoder_5_7(data,elements,points)
            error_count+=error_counter(data_oryginal,data_decoded)
            counter+=elements
            if counter>max:
                break
        #plot_points(data_with_noise)
        error_plot.append(error_count/counter)
        snr_plot.append(s)
        # print(f"SNR: {s}, Ilość błędów: {error_count}, Ilość bitów: {counter}\n")
        T.insert(END,f"SNR: {s}, Ilość błędów: {error_count}, Ilość bitów: {counter}\n")
        if error_count==0:
            break
    # print(error_plot)
    with open(f"csv\{Name}.csv",'w') as file:
            for i in range(len(error_plot)):
                if i==0:
                    file.write(str(error_plot[i]))
                else:
                    file.write(','+str(error_plot[i]))
            file.write('\n')
            for i in range(len(snr_plot)):
                if i==0:
                    file.write(str(snr_plot[i]))
                else:
                    file.write(','+str(snr_plot[i]))    
    ber_plot(snr_plot,error_plot,koder,konstelacja)
    return

if not os.path.exists('csv'): #tworzenie folderu do danych
    os.makedirs('csv')

root = Tk()
root.geometry("700x500")
root.title("Projekt ZTT")
root.configure(bg="#c6dff6")
style= ttk.Style()
style.theme_use('default')
style.configure("TCombobox",selectbackground="#dddddd",placeholderforeground="#dddddd",insertcolor="#dddddd",selectforeground="black")
#podpisy
text=ttk.Label(root, text="Projekt wykonali: ",font=("Arial",10),background="#c6dff6").place(relx="0.1",rely="0.85")
text=ttk.Label(root, text="Kinga Żywańska ",font=("Arial",10),background="#c6dff6").place(relx="0.1",rely="0.9")
text=ttk.Label(root, text="Paweł Płaczkiewicz ",font=("Arial",10),background="#c6dff6").place(relx="0.25",rely="0.9")
text=ttk.Label(root, text="Patryk Łuczak ",font=("Arial",10),background="#c6dff6").place(relx="0.43",rely="0.9")
text=ttk.Label(root, text="Wybierz konstelacje: ",font=("Arial",17),background="#c6dff6").place(relx="0.05",rely="0.05")
cmb1 = ttk.Combobox(root, width=34, values=("QAM8","PSK8"),state="readonly") #lista wybierana
text=ttk.Label(root, text="Wybierz koder: ",font=("Arial",17),background="#c6dff6").place(relx="0.05",rely="0.25")
cmb2 = ttk.Combobox(root, width=34, values=("43","Naiwny","25","57"),state="readonly") #lista wybierana
window2 = Tk()
window2.geometry("400x300")
S = Scrollbar(window2)
T = Text(window2, height=4, width=50,bg="#c6dff6")
S.pack(side=RIGHT, fill=Y)
T.pack(side=LEFT, fill=Y)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)
window2.title("Wyniki dekodowania")
window2.configure(bg="#c6dff6")

#pierwszy segment - wybor konstelacji
text=ttk.Label(root, text="Zaznacz dolną granice SNR: ",font=("Arial",17),background="#c6dff6").place(relx="0.5",rely="0.05")
slider=Scale(root,from_=0,to=20,length=300,tickinterval=2,background="#dddddd",orient=HORIZONTAL)
#,bg="#dddddd",selectbackground="#dddddd",selectforeground="black"

#drugi segment - pole do wpisania snr
text=ttk.Label(root, text="Długość ramki: ",font=("Arial",17),background="#c6dff6").place(relx="0.05",rely="0.42")
e1=Entry(root,width=36,bg="#dddddd",selectbackground="#dddddd",selectforeground="black")
text=ttk.Label(root, text="Maksymalna ilość bitów : ",font=("Arial",17),background="#c6dff6").place(relx="0.5",rely="0.42")
e2=Entry(root,width=36,bg="#dddddd",selectbackground="#dddddd",selectforeground="black")

#trzeci segment - pole do wpisania ilosci elementow
#funkcja ktora odpala sie po kliknieciu
def checkcmbo():
    try:
        int(e1.get())
    except ValueError:
        messagebox.showinfo("Błąd", "Wpisz liczbe w miejsce ilosci elementow")
    try:
        int(e2.get())
    except ValueError:
        messagebox.showinfo("Błąd", "Wpisz liczbe w miejsce ilosci elementow")
    #sprawdzenie czy wpisana rzecz do pola snr jest intem
    if slider.get()=="":
        messagebox.showinfo("Błąd", "Proszę nie zostawiać pustego miejsca")
    #sprawdzenie czy nie zostały puste miejsca
    else:
        if cmb1.get() == "QAM8" and cmb2.get()=="43":
            choose("QAM8","43",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="Naiwny":
            choose("PSK8","Naiwny",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="43":
            choose("PSK8","43",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "QAM8" and cmb2.get()=="Naiwny":
            choose("QAM8","Naiwny",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="25":
            choose("PSK8","25",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "QAM8" and cmb2.get()=="25":
            choose("QAM8","25",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="57":
            choose("PSK8","57",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "QAM8" and cmb2.get()=="57":
            choose("QAM8","57",slider.get(),e1.get(),e2.get())
    #ogarniecie list wybieranych razem z wartosciami
def ber():
    try:
        int(e1.get())
    except ValueError:
        messagebox.showinfo("Błąd", "Wpisz liczbe w miejsce ilosci elementow")
    try:
        int(e2.get()) or int(e1.get())
    except ValueError:
        messagebox.showinfo("Błąd", "Wpisz liczbe w miejsce ilosci elementow")
    #sprawdzenie czy wpisana rzecz do pola snr jest intem
    if slider.get()=="":
        messagebox.showinfo("Błąd", "Proszę nie zostawiać pustego miejsca")
    #sprawdzenie czy nie zostały puste miejsca
    else:
        choose("QAM8","25",slider.get(),e1.get(),e2.get())
        choose("PSK8","25",slider.get(),e1.get(),e2.get())
        choose("QAM8","43",slider.get(),e1.get(),e2.get())
        choose("PSK8","43",slider.get(),e1.get(),e2.get())
        choose("QAM8","Naiwny",slider.get(),e1.get(),e2.get())
        choose("PSK8","Naiwny",slider.get(),e1.get(),e2.get())
        choose("QAM8","57",slider.get(),e1.get(),e2.get())
        choose("PSK8","57",slider.get(),e1.get(),e2.get())
        ber_plot_file()
    
def stop():
    root.quit()
    window2.quit()
def temp():
    try:
        int(e1.get())
    except ValueError:
        messagebox.showinfo("Błąd", "Wpisz liczbe w miejsce ilosci elementow")
    try:
        int(e2.get()) or int(e1.get())
    except ValueError:
        messagebox.showinfo("Błąd", "Wpisz liczbe w miejsce ilosci elementow")
    #sprawdzenie czy wpisana rzecz do pola snr jest intem
    if slider.get()=="":
        messagebox.showinfo("Błąd", "Proszę nie zostawiać pustego miejsca")
    #sprawdzenie czy nie zostały puste miejsca
    else:
        if cmb1.get() == "QAM8" and cmb2.get()=="43":
            konste("QAM8","43",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="Naiwny":
            konste("PSK8","Naiwny",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="43":
            konste("PSK8","43",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "QAM8" and cmb2.get()=="Naiwny":
            konste("QAM8","Naiwny",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="25":
            konste("PSK8","25",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "QAM8" and cmb2.get()=="25":
            konste("QAM8","25",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "PSK8" and cmb2.get()=="57":
            konste("PSK8","57",slider.get(),e1.get(),e2.get())
        elif cmb1.get() == "QAM8" and cmb2.get()=="57":
            konste("QAM8","57",slider.get(),e1.get(),e2.get())
def konste(konstelacja,koder,snr,elem,max):
    root.quit()
    max=int(max)
    SNR=int(snr)
    elements=int(elem)
    if konstelacja=='PSK8':
        points=PSK8()
    elif konstelacja=='QAM8':
        points=QAM8()
    counter=0
    error_count=0
    data_with_noise=[]
    while(error_count<100):
        data_oryginal=data_gen(elements)
        if koder=='Naiwny':
            data=coder_silly(data_oryginal)
        if koder=='25':
            data=coder_2_5(data_oryginal)
        if koder=='43':
            data=coder_4_3(data_oryginal)
        if koder=='57':
            data=coder_5_7(data_oryginal)
        data=data_to_plot8(data,points,elements)
        data=add_noise(data,SNR)
        for i in range(len(data)):
            data_with_noise.append(data[i])
        if koder=='Naiwny':
            data_decoded=decoder_silly(data,elements,points)
        if koder=='25':
            data_decoded=decoder_2_5(data,elements,points)
        if koder=='43':
            data_decoded=decoder_4_3(data,elements,points)
        if koder=='57':
            data_decoded=decoder_5_7(data,elements,points)
        error_count+=error_counter(data_oryginal,data_decoded)
        counter+=elements
        if counter>max:
            break
    plot_points(data_with_noise,konstelacja,koder)

cmb1.place(relx="0.05",rely="0.14")
cmb2.place(relx="0.05",rely="0.34")
e1.place(relx="0.05",rely="0.5")
e2.place(relx="0.5",rely="0.5")
slider.place(relx="0.5",rely="0.2")

#pozycjonowanie elementow do wpisania
btn1 = Button(root, text="Skompiluj dane",width=40,height=3,command=checkcmbo,bd=0.5,bg="#a1cbf0")
btn1.place(relx="0.06",rely="0.6")
btn2 = Button(root, text="Wykres konstelacji",width=40,height=3,command=temp,bd=0.5,bg="#a1cbf0")
btn2.place(relx="0.48",rely="0.6")
btn3 = Button(root, text="Wykres BER",width=40,height=3,command=ber,bd=0.5,bg="#a1cbf0")
btn3.place(relx="0.06",rely="0.72")
btn4 = Button(root, text="Wyłącz program",fg='#d80000',width=40,height=3,command=stop,bd=0.5,bg="#a1cbf0")
btn4.place(relx="0.48",rely="0.72")

#czesc juz po za gui
def choose(konstelacja,koder,snr,elem,max):
    if konstelacja=="PSK8" and koder=="Naiwny":
        run(snr,elem,konstelacja,'PSK8_Naiwny',koder,max)  
    elif konstelacja=="QAM8" and koder=="Naiwny":
        run(snr,elem,konstelacja,'QAM8_Naiwny',koder,max)

    elif konstelacja=="PSK8" and koder=="25":
        run(snr,elem,konstelacja,'PSK8_25',koder,max)
    elif konstelacja=="QAM8" and koder=="25":
        run(snr,elem,konstelacja,'QAM8_25',koder,max)

    elif konstelacja=="PSK8" and koder=="43":
        run(snr,elem,konstelacja,'PSK8_43',koder,max)
    elif konstelacja=="QAM8" and koder=="43":
        run(snr,elem,konstelacja,'QAM8_43',koder,max)

    elif konstelacja=="PSK8" and koder=="57":
        run(snr,elem,konstelacja,'PSK8_57',koder,max)
    elif konstelacja=="QAM8" and koder=="57":
        run(snr,elem,konstelacja,'QAM8_57',koder,max)

root.mainloop()