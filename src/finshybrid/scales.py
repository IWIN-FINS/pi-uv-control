"""Frozen DE51/DE83 scales; preserve the Unity float32 -> float64 round trip."""
import numpy as np
PC, PS, VS, US = [np.asarray(x,dtype=np.float32).astype(np.float64) for x in (
 [0,-1.2,0,0,0,0], [.6,.2,.35,.2,.4,.2], [.15,.08,.15,.25,.4,.25], [15,12,15,2,2,2])]
DT=.1
BASIC=np.array([2,4,5,6,7,8,9,10,11,12])
CENTER=np.array([0.,-.08,0.])
H=8
