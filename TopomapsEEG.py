#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpy as sy
from scipy.integrate import trapz
import math as math
import itertools
import scipy.io
import mne


# In[51]:


data = scipy.io.loadmat('C:\\Users\\luanp\\OneDrive\\Área de Trabalho\\PSB2\\EEGs Pacientes\\V0072601_100.mat') # arquivo .mat com o EEG


# In[52]:


XN = data['xn'] # Matriz com coleta do exame

fa = data['fa']
fa = int(fa) # Vetores de tempo de cada época. São 10 vetores

t = data['t'] # Valores de amplitudes dos 20 eletrodos conforme os tempos das épocas do epochTime

nameChannel = list(itertools.chain.from_iterable(data['nameChannels'])) # Transforma linha alinhada em linha plana
canais = [arr[0].item() for arr in nameChannel] # Mais um processo de desanimanhento de arrays


# In[53]:


# Tempo em segundos
ti = 1139
tf = 1319
n_epocas = 30


# In[54]:


comp_epoca = int(round((tf-ti)*fa/n_epocas, 0))
print(comp_epoca)


# In[55]:


epoca = np.zeros((comp_epoca,20))


# In[56]:


for i in range(i):
    epoca = XN[:, 1*ti*fa+(i*comp_epoca) : 1*ti*fa+((i+1)*comp_epoca)]


# In[57]:


# Receberá valores das potencias das ondas
matriz3d = np.zeros((len(XN), 4, n_epocas))


# In[58]:


for i in range(n_epocas):
    epoca = XN[:, 1*ti*fa+(i*comp_epoca) : 1*ti*fa+((i+1)*comp_epoca)]    
    matriz = np.zeros(4)
    
    for j in range(len(epoca)):
        N = comp_epoca # Comprimento da época
        T = 1/fa # Período

        f = np.fft.fftfreq(N, T) # Eixo de frequencia
        transf = np.fft.fft(epoca[j]) # FFT do sinal
        transf = np.abs(transf) # Valor absoluto da FFT

        # Normalização [Integral]funcao = transf**2
        integral = trapz(transf, f)
        transfNorm = (np.abs(transf)/math.sqrt(np.abs(integral)))/3

        # Monta tabela com frequencias e amplitudes, para auxiar na filtragem das frequencias
        tabela = pd.DataFrame(data = zip(f[f>0], transfNorm[f>0]), columns = ['Frequencia','Amplitude'])

        # Separação de potencias para cada tipo de onda
        delta = tabela[(tabela['Frequencia']>1) & (tabela['Frequencia']<=3)]
        teta = tabela[(tabela['Frequencia']>3) & (tabela['Frequencia']<=7)]
        alfa = tabela[(tabela['Frequencia']>7) & (tabela['Frequencia']<=10)]
        beta = tabela[(tabela['Frequencia']>10) & (tabela['Frequencia']<=30)]

        # Soma das potencias
        s_delta = sum(delta['Amplitude'])
        s_teta = sum(teta['Amplitude'])
        s_alfa = sum(alfa['Amplitude'])
        s_beta = sum(beta['Amplitude'])

        # Count das potencias
        len_delta = len(delta['Amplitude'])
        len_teta = len(teta['Amplitude'])
        len_alfa = len(alfa['Amplitude'])
        len_beta = len(beta['Amplitude'])

        # Adição dos valores da época a matriz final, dividido pela quantidade de itens da respectiva frequencia 
        nova_linha = np.array([s_delta/len_delta, s_teta/len_teta, s_alfa/len_alfa, s_beta/len_beta]).round(2)
        matriz = np.vstack((matriz, nova_linha))
    
    matriz = np.delete(matriz,(0), axis=0) # Remoção da primeira linha zerada
    matriz3d[:,:,i] = matriz


# In[59]:


media_amplitudes = np.mean(matriz3d, axis=2).round(2)


# In[60]:


# Ajuste da matriz de numpy para pandas, para melhor visualização de dados

potencias = pd.DataFrame(data = media_amplitudes, columns = ['Delta','Teta','Alfa','Beta'], index = canais)
print(potencias)


# In[61]:


potencias


# In[62]:


canais = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
info = mne.create_info(ch_names = canais , sfreq = fa, ch_types='eeg')

delta = media_amplitudes[:,0].reshape(-1, 1)
teta = media_amplitudes[:,1].reshape(-1, 1)
alfa = media_amplitudes[:,2].reshape(-1, 1)
beta = media_amplitudes[:,3].reshape(-1, 1)


fig, ax = plt.subplots(1, 4, figsize=(20,4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)

# Exibir Delta
evoked = mne.EvokedArray(delta, info)
montagem = mne.channels.make_standard_montage("standard_1020")
evoked.set_montage(montagem)

im = mne.viz.plot_topomap(evoked.data[:, 0], 
                                    evoked.info, 
                                    vlim=(min(delta),max(delta)), 
                                    names=canais, 
                                    cmap='RdBu_r', 
                                    show=False,
                                    axes=ax[0])

ax[0].set_title("Delta", fontweight = 'bold', fontsize = 15)
cbar = plt.colorbar(im[0], ax=ax[0])
cbar.set_label("Amplitude", fontweight = 'bold', fontsize = 12)


# Exibir Teta
evoked = mne.EvokedArray(teta, info)
montagem = mne.channels.make_standard_montage("standard_1020")
evoked.set_montage(montagem)

im = mne.viz.plot_topomap(evoked.data[:, 0], 
                                    evoked.info, 
                                    vlim=(min(teta),max(teta)), 
                                    names=canais, 
                                    cmap='RdBu_r', 
                                    show=False,
                                    axes=ax[1])

ax[1].set_title("Teta", fontweight = 'bold', fontsize = 15)
cbar = plt.colorbar(im[0], ax=ax[1])
cbar.set_label("Amplitude", fontweight = 'bold', fontsize = 12)

# Exibir Alfa
evoked = mne.EvokedArray(alfa, info)
montagem = mne.channels.make_standard_montage("standard_1020")
evoked.set_montage(montagem)

im = mne.viz.plot_topomap(evoked.data[:, 0], 
                                    evoked.info, 
                                    vlim=(min(alfa),max(alfa)), 
                                    names=canais, 
                                    cmap='RdBu_r', 
                                    show=False,
                                    axes=ax[2])

ax[2].set_title("Alfa", fontweight = 'bold', fontsize = 15)
cbar = plt.colorbar(im[0], ax=ax[2])
cbar.set_label("Amplitude", fontweight = 'bold', fontsize = 12)

# Exibir Beta
evoked = mne.EvokedArray(beta, info)
montagem = mne.channels.make_standard_montage("standard_1020")
evoked.set_montage(montagem)

im = mne.viz.plot_topomap(evoked.data[:, 0], 
                                    evoked.info, 
                                    vlim=(min(beta),max(beta)), 
                                    names=canais, 
                                    cmap='RdBu_r', 
                                    show=False,
                                    axes=ax[3])

ax[3].set_title("Beta", fontweight = 'bold', fontsize = 15)
cbar = plt.colorbar(im[0], ax=ax[3], )
cbar.set_label("Amplitude", fontweight = 'bold', fontsize = 12)


# In[50]:


potencias.to_excel('C:\\Users\\luanp\\OneDrive\\Área de Trabalho\\PSB2\\potencias_3s_finais_501_100.xlsx')
print('DataFrame salvo com sucesso!')


# In[ ]:




