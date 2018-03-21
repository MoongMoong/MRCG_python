import numpy as np
import numpy.matlib
try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft
from scipy.signal import lfilter
import scipy.io as sio
import time
from scipy import signal

epsc = 0.000001


def mrcg_extract( sig, sampFreq = 16000):
        ######original code######
    beta = 1000 / np.sqrt(sum(map(lambda x:x*x,sig)) / len(sig))
    sig = sig*beta
    sig = sig.reshape(len(sig), 1)
    t0= time.clock()
    g = gammatone(sig, 64, sampFreq)
    t1 = time.clock()
    cochlea1 = np.log10(cochleagram(g, int(sampFreq * 0.025), int(sampFreq * 0.010)))
    t2=time.clock()
    cochlea2 = np.log10(cochleagram(g, int(sampFreq * 0.200), int(sampFreq * 0.010)))
    t3 = time.clock()
    print('gamma total')
    print(t1-t0)
    print('coch1')
    print(t2-t1)
    print('coch2')
    print(t3-t2)
    cochlea1 = cochlea1[:,:]
    cochlea2 = cochlea2[:,:]
    t4 = time.clock()
    cochlea3 = get_avg(cochlea1, 5, 5)
    cochlea4 = get_avg(cochlea1, 11, 11)
    t5 = time.clock()
    print('get avg')
    print(t5-t4)
    all_cochleas = np.concatenate([cochlea1,cochlea2,cochlea3,cochlea4],0)

    del0 = deltas(all_cochleas)
    ddel = deltas(deltas(all_cochleas, 5), 5)

    ouotput = np.concatenate((all_cochleas, del0, ddel), 0)

    return ouotput

def gammatone(insig, numChan=128, fs = 16000):
    fRange = [50, 8000]
    filterOrder = 4
    gL = 2048
    sigLength = len(insig)
    phase = np.zeros([numChan, 1])
    erb_b = hz2erb(fRange)

    ###################
    erb_b_diff = (erb_b[1]-erb_b[0])/(numChan-1)
    erb = np.arange(erb_b[0], erb_b[1]+epsc, erb_b_diff)
    cf = erb2hz(erb)
    b = [1.019 * 24.7 * (4.37 * x / 1000 + 1) for x in cf]
    gt = np.zeros([numChan, gL])
    tmp_t = np.arange(1,gL+1)/fs
    for i in range(numChan):
        gain = 10**((loudness(cf[i])-60)/20)/3*(2 * np.pi * b[i] / fs)**4
        tmp_temp = [gain*(fs**3)*x**(filterOrder - 1)*np.exp(-2 * np.pi * b[i] * x)*np.cos(2 * np.pi * cf[i] * x + phase[i]) for x in tmp_t]
        tmp_temp2 = np.reshape(tmp_temp, [1, gL])

        gt[i, :] = tmp_temp2

    sig = np.reshape(insig,[sigLength,1])
    gt2 = np.transpose(gt)
    resig = np.matlib.repmat(sig,1,numChan)
    t0 = time.clock()
    r = np.transpose(fftfilt(gt2,resig,numChan))
    t1 = time.clock()
    print('fftfilter')
    print(t1-t0)
    return r

def hz2erb(hz):
    erb1 = 0.00437
    # erb2 = [x * erb1 for x in hz]
    # erb3 = [x + 1 for x in erb2]
    erb2 = np.multiply(erb1,hz)
    erb3 = np.subtract(erb2,-1)
    erb4 = np.log10(erb3)
    erb = 21.4 *erb4
    return erb

def erb2hz(erb):
    hz = [(10**(x/21.4)-1)/(0.00437) for x in erb]
    return hz

def loudness(freq):
    dB=60
    # af = [2.3470,2.1900,2.0500,1.8790,1.7240,1.5790,1.5120,1.4660,1.4260,1.3940,1.3720,1.3440,1.3040,1.2560,1.2030,1.1350,1.0620,1.0000,0.9670,0.9430,0.9320,0.9330,0.9370,0.9520,0.9740,1.0270,1.1350,1.2660,1.5010]
    # bf = [0.0056,0.0053,0.0048,0.0040,0.0038,0.0029,0.0026,0.0026,0.0026,0.0026,0.0025,0.0025,0.0023,0.0020,0.0016,0.0011,0.0005,0,-0.0004,-0.0007,-0.0009,-0.0010,-0.0010,-0.0009,-0.0006,0,0.0009,0.0021,0.0049]
    # cf = [74.3000,65.0000,56.3000,48.4000,41.7000,35.5000,29.8000,25.1000,20.7000,16.8000,13.8000,11.2000,8.9000,7.2000,6.0000,5.0000,4.4000,4.2000,3.7000, 2.6000, 1.0000,-1.2000,-3.6000,-3.9000,-1.1000,6.6000,15.3000,16.4000,11.6000]
    # ff = np.multiply([0.0020,0.0025,0.0032,0.0040,0.0050,0.0063,0.0080,0.0100,0.0125,0.0160,0.0200,0.0250,0.0315,0.0400,0.0500,0.0630,0.0800,0.1000,0.1250,0.1600,0.2000,0.2500,0.3150,0.4000,0.5000,0.6300,0.8000,1.0000,1.2500],10000)
    fmat = sio.loadmat('f_af_bf_cf.mat')
    af = fmat['af'][0]
    bf = fmat['bf'][0]
    cf = fmat['cf'][0]
    ff = fmat['ff'][0]
    i = 0
    while ff[i] < freq:
        i = i + 1

    afy = af[i - 1] + (freq - ff[i - 1]) * (af[i] - af[i - 1]) / (ff[i] - ff[i - 1])
    bfy = bf[i - 1] + (freq - ff[i - 1]) * (bf[i] - bf[i - 1]) / (ff[i] - ff[i - 1])
    cfy = cf[i - 1] + (freq - ff[i - 1]) * (cf[i] - cf[i - 1]) / (ff[i] - ff[i - 1])
    loud = 4.2 + afy * (dB - cfy) / (1 + bfy * (dB - cfy))
    return loud

def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""

    return np.ceil(np.log2(abs(x)))


def fftfilt(b,x,nfft):
    fftflops = [18, 59, 138, 303, 660, 1441, 3150, 6875, 14952, 32373, 69762,
                149647, 319644, 680105, 1441974, 3047619, 6422736, 13500637, 28311786,
                59244791, 59244791*2.09]
    nb, _ = np.shape(b)
    nx, mx = np.shape(x)
    n_min = 0
    while 2**n_min < nb-1:
        n_min = n_min+1
    n_temp = np.arange(n_min, 21 + epsc, 1)
    # n = [2 ** x for x in n_temp]
    n = np.power(2,n_temp)
    fftflops = fftflops[n_min-1:21]
    # L = [x -(nb-1) for x in n]
    L = np.subtract(n,nb-1)
    lenL= np.size(L)
    # temp_ind = [np.ceil(nx/ L[x])*fftflops[x] for x in range(lenL)]
    # ind = temp_ind.index(int(np.min(temp_ind)))
    temp_ind0 = np.ceil(np.divide(nx,L))
    temp_ind = np.multiply(temp_ind0,fftflops)
    temp_ind = np.array(temp_ind)
    # ind = temp_ind.index(int(np.min(temp_ind)))
    ind = np.argmin(temp_ind)
    nfft=int(n[ind])
    L=int(L[ind])
    b_tr = np.transpose(b)
    B_tr = fft(b_tr,nfft)
    B = np.transpose(B_tr)
    y = np.zeros([nx, mx])
    istart = 0
    while istart < nx :
        iend = min(istart+L,nx)
        if (iend - istart) == 1 :
            X = x[0][0]*np.ones([nx,mx])
        else :
            xtr = np.transpose(x[istart:iend][:])
            Xtr = fft(xtr,nfft)
            X = np.transpose(Xtr)
        # temp_Y =np.transpose([a * b for a, b in zip(B, X)])
        temp_Y = np.transpose(np.multiply(B,X))
        Ytr = ifft(temp_Y,nfft)
        Y = np.transpose(Ytr)
        yend = np.min([nx, istart + nfft])
        y[istart:yend][:] = y[istart:yend][:] + np.real(Y[0:yend-istart][:])

        istart = istart + L
    # y = np.real(y)
    return y


def cochleagram(r, winLength = 320, winShift=160):
    numChan, sigLength = np.shape(r)
    increment = winLength / winShift
    M = np.floor(sigLength / winShift)
    a = np.zeros([numChan, int(M)])
    rs = np.square(r)
    rsl = np.concatenate((np.zeros([numChan,winLength-winShift]),rs),1)
    for m in range(int(M)):
        temp = rsl[:,m*winShift : m*winShift+winLength]
        a[:, m] = np.sum(temp,1)

    return a


def cochleagram_keep(r, winLength = 320, winShift=160):
    numChan, sigLength = np.shape(r)
    increment = winLength / winShift
    M = np.floor(sigLength / winShift)
    a = np.zeros([numChan, int(M)])
    for m in range(int(M)):
        for i in range(numChan):
            if m < increment:
                a[i,m] = (sum(map(lambda x:x*x,r[i, 0:(m+1)*winShift])))
            else :
                startpoint = (m - increment) * winShift
                a[i, m] = (sum(map(lambda x:x*x,r[i, int(startpoint) :int(startpoint) + winLength])))
    return a


def get_avg( m , v_span, h_span):
    nr,nc = np.shape(m)
    # out = np.zeros([nr+2*h_span,nc+2*h_span])
    fil_size = (2 * v_span + 1) * (2 * h_span + 1)
    meanfil = np.ones([1+2*h_span,1+2*h_span])
    meanfil = np.divide(meanfil,fil_size)

    out = signal.convolve2d(m, meanfil, boundary='fill', fillvalue=0, mode='same')
    return out


def get_avg2( m , v_span, h_span):
    nr,nc = np.shape(m)
    out = np.zeros([nr,nc])
    fil_size = (2 * v_span + 1) * (2 * h_span + 1)
    for i in range(nr):
        row_begin = 0
        row_end = nr
        col_begin = 0
        col_end = nc
        if (i - v_span) >= 0 :
            row_begin = i - v_span
        if (i + v_span + 1) <= nr:
            row_end = i + v_span + 1

        for j in range(nc):
            if (j - h_span) >= 0:
                col_begin = j - h_span
            if (j + h_span + 1) <= nc:
                col_end = j + h_span + 1
            tmp = m[row_begin:row_end, col_begin: col_end]
            out[i, j] = sum(sum(tmp)) / fil_size
    return out


def deltas(x, w=9) :
    nr,nc = np.shape(x)
    if nc ==0 :
        d= x
    else :
        hlen = int(np.floor(w / 2))
        w = 2 * hlen + 1
        win=np.arange(hlen, int(-(hlen+1)), -1)
        temp = x[:, 0]
        fx = np.matlib.repmat(temp.reshape([-1,1]), 1, int(hlen))
        temp = x[:, nc-1]
        ex = np.matlib.repmat(temp.reshape([-1,1]), 1, int(hlen))
        xx = np.concatenate((fx, x, ex),1)
        d = lfilter(win, 1, xx, 1)
        d = d[:,2*hlen:nc+2*hlen]

    return d


# def fftfilt(b, x, *n):
#     """Filter the signal x with the FIR filter described by the
#     coefficients in b using the overlap-add method. If the FFT
#     length n is not specified, it and the overlap-add block length
#     are selected so as to minimize the computational cost of
#     the filtering operation."""
#
#     N_x = len(x)
#     N_b = len(b)
#
#     # Determine the FFT length to use:
#     if len(n):
#
#         # Use the specified FFT length (rounded up to the nearest
#         # power of 2), provided that it is no less than the filter
#         # length:
#         n = n[0]
#         if n != int(n) or n <= 0:
#             raise ValueError('n must be a nonnegative integer')
#         if n < N_b:
#             n = N_b
#         N_fft = 2**nextpow2(n)
#     else:
#
#         if N_x > N_b:
#
#             # When the filter length is smaller than the signal,
#             # choose the FFT length and block size that minimize the
#             # FLOPS cost. Since the cost for a length-N FFT is
#             # (N/2)*log2(N) and the filtering operation of each block
#             # involves 2 FFT operations and N multiplications, the
#             # cost of the overlap-add method for 1 length-N block is
#             # N*(1+log2(N)). For the sake of efficiency, only FFT
#             # lengths that are powers of 2 are considered:
#             N = 2**np.arange(np.ceil(np.log2(N_b)), np.floor(np.log2(N_x)))
#             cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
#             N_fft = N[np.argmin(cost)]
#
#         else:
#
#             # When the filter length is at least as long as the signal,
#             # filter the signal using a single block:
#             N_fft = 2**nextpow2(N_b+N_x-1)
#
#     N_fft = int(N_fft)
#
#     # Compute the block length:
#     L = int(N_fft - N_b + 1)
#
#     # Compute the transform of the filter:
#     H = fft(b, N_fft)
#
#     y = np.zeros(N_x,float)
#     i = 0
#     while i <= N_x:
#         il = min([i+L,N_x])
#         k = min([i+N_fft,N_x])
#         yt = ifft(fft(x[i:il],N_fft)*H,N_fft) # Overlap..
#         y[i:k] = y[i:k] + yt[:k-i]            # and add
#         i += L
#     return y
