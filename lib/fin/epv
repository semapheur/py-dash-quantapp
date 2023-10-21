import numpy as np
from numba import jit

@jit(nopython=True)
def epvMonteCarlo(nSim, params, opmDst, waccDst):
  '''
  Parameters:
  0: Revenue (rev)
  1: Operating margin (opm)
  2: Tax rate (taxRate)
  3: Depreciation and amortization (da)
  4: Selling, general and administration expenses (sgaEx)
  5: Research and development expenses (rdaEx)
  6: Maintenance capital expenditure (mxCapEx)
  '''
  
  sim = np.zeros(nSim)
      
  for i in range(nSim):
      
    # Normalized earning (growth adjustments)
    opm = params[1] + opmDst[i]
    ne = params[0] * (opm + params[4] + params[5])
    
    # Adjusted earnings
    ae = ne * (1 - params[2]) + (params[3] - params[6]) * params[0]
    
    # Earnings power
    epv = ae / waccDst[i] 
    
    sim[i] = epv
      
  return sim