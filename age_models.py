def CF_age_model(Ψ, u_Ψ, S, u_S, C0, u_C0, T0_str, core_data):
    
    ###################################### IMPORTS ###########################################
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    
    #################################### FORMAT INPUTS #######################################
    #constant=pd.read_csv('constants.csv', index_col=0, header=None).T
    #constant=constant.drop(1)
    #float(constant['HL'])
    
    #################    DEFINE GLOBAL CONSTANTS (DDEP, 2010)   ##############################
    HL         = 22.23           #yr    (half life of 210Pb)
    u_HL       = 0.12            #yr    (uncertainty of half life of 210Pb)
    λ          = 0.03118         #yr^-1 (decay constant of 210Pb)
    u_λ        = 0.00017         #yr^-1 (uncertainty of half life of 210Pb)
    
    #######################################   DATES   ########################################
    #conversion of string date to decimal year CE
    def year_fraction(date):
        start = datetime.date(date.year, 1, 1).toordinal()
        year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
        return date.year + float(date.toordinal() - start) / year_length
    T0 = year_fraction(datetime.datetime.strptime(T0_str, '%m/%d/%Y')) #y     (Sampling date AD) NOTE: typo in publication, not sure if it's 10/11/2004 or 11/10/2004

    ######################################   DEPTH   #######################################
    #☑ Depth below seafloor of the midpoint of the sediment layer
    core_data['Z_i mdpt, cm']        = [(core_data['Z(i) upper, cm'][i]+core_data['Z(i) lower, cm'][i])/2 for i in range(len(core_data))]                       
    #☑ Thickness of the sediment layer
    core_data['ΔZ_i, cm']            = [(core_data['Z(i) lower, cm'][i]-core_data['Z(i) upper, cm'][i]) for i in range(len(core_data))]                        

    #######################################   MASS   ########################################
    #☑ Aerial dry mass of section i
    core_data['ΔM_i, kg']             = core_data['ΔM_i, g'] /1000
    #☑ Aerial dry mass, pp.191
    core_data['ΔM_i/S, g cm^-2']      = core_data['ΔM_i, g']/S                                                                                                   
    #☑ uncertainty of Aerial dry mass, pp.191
    core_data['u_ΔM_i/S, g cm^-2']    = core_data['ΔM_i/S, g cm^-2']*(u_S/S)  
    #☑ Mass depth at layer above/below i (g cm^-2)
    core_data['M(i) upper, g cm^-2']  = [np.sum(core_data['ΔM_i/S, g cm^-2'][0:i])   for i in range(len(core_data))]
    core_data['M(i) lower, g cm^-2']  = [np.sum(core_data['ΔM_i/S, g cm^-2'][0:i+1]) for i in range(len(core_data))]   

    #☑ Uncertainty of the mass depth at layer above/below i (g cm^-2)
    core_data['u_M(i) upper, g cm^-2']= core_data['M(i) upper, g cm^-2']*(u_S/S)  
    core_data['u_M(i) lower, g cm^-2']= core_data['M(i) lower, g cm^-2']*(u_S/S)                                                                                
    #☑ Mass depth at the midpoint estimated by mean (g cm^-2).
    core_data['M_i mdpt, g cm^-2']    = [(core_data['M(i) upper, g cm^-2'][i]+core_data['M(i) lower, g cm^-2'][i])/2 for i in range(len(core_data))]             

    #☑ Uncertainty of the mass depth at the mean (g cm^-2)
    core_data['u_M_i mdpt, g cm^-2']  =  ( (core_data['u_M(i) upper, g cm^-2'])**2 + 
                                           (core_data['u_M(i) lower, g cm^-2'])**2
                                         )**(1/2)

    #######################################   CONCENTRATION   ########################################
    #☑ average sections to determine C at layer i. Independently determine C(0) and add it to index 0.
    core_data['C(i) upper, Bq kg^-1']    = core_data['C_i, Bq kg^-1'].rolling(window=2).mean()
    core_data['C(i) upper, Bq kg^-1'][0] = C0 #add C(0) to index 0.
    #☑ sqrt the sum of squares of u_c_i.
    core_data['u_C(i) upper, Bq kg^-1']   = ((core_data['u_C_i, Bq kg^-1']**2).rolling(window=2).sum()**(1/2)/2)
    core_data['u_C(i) upper, Bq kg^-1'][0]= u_C0

    #######################################   ACTIVITY   ############################################
    #☑
    core_data['ΔA_i_midpt, Bq m^-2'] = core_data['C_i, Bq kg^-1'] * core_data['ΔM_i, kg']  /   (S/10000)   
    #☑
    core_data['u_ΔA_i, Bq m^-2']     = core_data['ΔA_i_midpt, Bq m^-2'] * (
                                            (core_data['u_C_i, Bq kg^-1']/core_data['C_i, Bq kg^-1'])**2 +
                                            (core_data['u_ΔM_i/S, g cm^-2']/core_data['ΔM_i/S, g cm^-2'])**2
                                            )**(1/2)
    #☑ Cumulative activity depth upper boundary (Bq m^-2)
    core_data['A(i)_upper, Bq m^-2']                 = np.cumsum(core_data['ΔA_i_midpt, Bq m^-2'][::-1])[::-1]                             
    #☑ Cumulative activity depth lower boundary (Bq m^-2)
    core_data['A(i)_lower, Bq m^-2']                 = [core_data['A(i)_upper, Bq m^-2'][i]-core_data['ΔA_i_midpt, Bq m^-2'][i] for i in range(len(core_data))] 
    #☑ 
    core_data['u_A(i)_upper, Bq m^-2']               = np.sqrt(np.cumsum(core_data['u_ΔA_i, Bq m^-2'][::-1]**2)[::-1])        



    ######################################      CF AGES, RATES      ##################################
    #Age of the sediment horizon at the upper boundary of the layer
    core_data['t(i)_upper, ybp']                     = (1/λ)*np.log(core_data['A(i)_upper, Bq m^-2'][0] / core_data['A(i)_upper, Bq m^-2'])
    core_data['t(i)_upper, yCE']                     = T0 - core_data['t(i)_upper, ybp']



    #Uncertainty of the age of the sediment horizon at the upper boundary of the layer
    core_data['u(t(i))_upper, yr']                   = (1/λ)*np.sqrt(
                                                        (core_data['t(i)_upper, ybp'] * u_λ)**2 +
                                                        (core_data['u_A(i)_upper, Bq m^-2'][0] / core_data['A(i)_upper, Bq m^-2'][0] )**2 +
                                                        (1-(2*core_data['A(i)_upper, Bq m^-2'] / core_data['A(i)_upper, Bq m^-2'][0])) * 
                                                        (core_data['u_A(i)_upper, Bq m^-2']    / core_data['A(i)_upper, Bq m^-2'])**2
                                                       )


    #Mass Accumulation Rate
    core_data['r(i), g cm^-2 yr^-1'] = (λ*core_data['A(i)_upper, Bq m^-2']) / core_data['C(i) upper, Bq kg^-1'] *(1/10)

    core_data['u_r(i), g cm^-2 yr^-1'] = core_data['r(i), g cm^-2 yr^-1'] * np.sqrt(
                                                        (u_λ/λ)**2 + 
                                                        (core_data['u_A(i)_upper, Bq m^-2']/core_data['A(i)_upper, Bq m^-2'])**2 +
                                                        (core_data['u_C(i) upper, Bq kg^-1']/core_data['C(i) upper, Bq kg^-1'])**2
                                                        )
    return core_data