"""
@ Name: Abhishek Dhital
  ID  : 1001548204
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import soundfile as sf
from scipy.signal import spectrogram
import glob


def classifyMusic() :
    testFile, fs= sf.read("testSong.wav")
    namesDatabase= np.array(glob.glob("song-*.wav"))
    database=[["",[]]]*len(namesDatabase)
    
    n=0
    for data in namesDatabase:
        sig=[]
        song, sfreq=sf.read(data)
        f, t, Sxx=spectrogram(song, fs=sfreq, nperseg=sfreq//2)
        Sxx=np.array(Sxx.transpose())
        for time in Sxx:
            sig.append(f[np.where(time==max(time))[0][0]])
        database[n]=data,sig
        n=n+1
    f,t,Sxx=spectrogram(testFile,fs=fs,nperseg=fs//2)
    
    testSignature=[]
    Sxx=np.transpose(Sxx)
    for time in Sxx:
        testSignature.append(f[np.where(time==max(time))[0][0]])
    testSignature=np.array(testSignature)
    
    taxicab=[]
    for data in database:
        array=np.array(data[1])
        taxicab.append(norm((array-testSignature),1))
    taxicab=np.array(taxicab)
    
    output=taxicab.argsort()[:5]
    for x in output:
        print("%d  %s" %(taxicab[x],database[x][0]))
    
    plt.figure()
    plt.specgram(testFile, Fs=fs)
    plt.title("testSong.wav")
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.show()
    
    plt.figure()
    x=sf.read(database[output[0]][0])[0]
    plt.specgram(x, Fs=fs)
    plt.title(database[output[0]][0])
    plt.ylabel("frequency")
    plt.xlabel("time")
    plt.show()
    
    plt.figure()
    x=sf.read(database[output[1]][0])[0]
    plt.specgram(x, Fs=fs)
    plt.title(database[output[1]][0])
    plt.ylabel("frequency")
    plt.xlabel("time")
    plt.show()

    
    
        


###################  main  ###################
if __name__ == "__main__" :
    classifyMusic()
