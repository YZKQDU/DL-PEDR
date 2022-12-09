
from CSI import *
import os  
import pandas as pd
from CSI import *
import numpy as np
#  

list1=['honor1','fast1', 'tenda1', 'fast2','honor4','fast3','fast4', '3601','3602','3603', '3604','H3C1','H3C2','HK1','HK2',
        'honor2','honor3','mercury1', 'mercury2', 'mercury3','mercury4','tenda2', 'tenda3', 
        'tenda4','tp1','tp2','tplink1', 'tplink2','tplink3', 'tplink4']
for i in list1:
    txt_file_path=r'D:\\dataset\\data0\\'+i+'.dat'
    path=r'D:\\dataset\\data0_003\\'+i+'.csv'
    csi = CSI(txt_file_path, 0.0024).get_data()
    csi=csi['fplist']
    csi= np.array(csi)
    print(csi.shape)
    csi = pd.DataFrame(csi)
    csi.to_csv(path,index=False, header=False)#mode='a',0
