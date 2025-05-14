import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import njit

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['font.size'] = 8

#===============================================================================================================================#
#    ___            _              _        _    ___                                _  _                                        #
#   | _ ) __ _  __ | |_  ___  _ _ (_) __ _ | |  / __| ___  _ __   _ __  _  _  _ _  (_)| |_  _  _                                #
#   | _ \/ _` |/ _||  _|/ -_)| '_|| |/ _` || | | (__ / _ \| '  \ | '  \| || || ' \ | ||  _|| || |                               #
#   |___/\__,_|\__| \__|\___||_|  |_|\__,_||_|  \___|\___/|_|_|_||_|_|_|\_,_||_||_||_| \__| \_, |                               #
#                                                                                            |__/                               #
#    __  __          _       _                                                                                                  #
#   |  \/  | ___  __| | ___ | |                                                                                                 #
#   | |\/| |/ _ \/ _` |/ -_)| |                                                                                                 #
#   |_|  |_|\___/\__,_|\___||_|                                                                                                 #
#                                                                                                                               #
#===============================================================================================================================#                               
# Authors:  @ Marco Lorenzetti                                                                                                  #
#           @ Raffaele Gaudio                                                                                                   #
#                                                                                                                               #
# Structure of the code:                                                                                                        #
#   1. Model Functions                                                                                                          #
#   2. Community Model Class                                                                                                    #
#       2.1. Model Simulation                                                                                                   #
#       2.2. Methods                                                                                                            #
#                                                                                                                               #
# List of methods:                                                                                                              #
#   - Simulate():                           integrates the dynamics of the model using odeint                                   #
#   - ModelSeed():                          sets the seed of the model                                                          #
#   - ReturnData():                         returns the array of the result of the simulation ("species" or "resources")        #
#   - ReturnTable():                        returns a data frame with useful informations ("species" or "resources")            #
#   - ReturnNicheOverlap():                 returns niche overlap                                                               #
#   - ReturnSimpsonDiversity():             returns simpson diversity at stationarity                                           #
#   - ReturnMetabolicProduction():          returns the metabolic production of a given species ("graphical" or "matrix")       #
#   - ReturnTotalPreferenceMatrix():        returns the total preference matrix ("graphical" or "matrix")                       #
#   - ReturnCommunityPreferenceMatrix():    returns the community preference matrix ("graphical" or "matrix")                   #
#   - ReturnSimulationsPlots():             returns simulation plots                                                            #
#   - ReturnRelativeAbundance():            returns a barplot of RSA at stationarity                                            #
#   - StabilityTime():                      returns the mean stability time of the dynamics ("species" or "resources")          #
#                                                                                                                               #
# For more details, read the descriptions of the single function/method.                                                        #
# NB: Whenever changes are made to this file, the kernel of the notebook must be restarted for them to take effect!             #                                                        
#===============================================================================================================================#

#================================================================================================================================
# 1. Model Functions:

def class_partition(s, num_classes, quantities): 

    # function that create partitions in families/tiers
    # inputs:   - s: string; "F" for families, "T" for tiers
    #           - num_class: int; number of partitions
    #           - quantities: list; list of partitions

    tiers = [[s+str(x)]*q for (x,q) in zip(range(num_classes), quantities)]
    t = []
    for element in tiers:
        t.extend(element)
    t = np.array(t)
    return t


def species_table(S, R, families, c_control, num_T, m):

    # function that returns a data frame containing family, prefernece and mortality for each species
    #  inputs:  - S: int; number of species
    #           - R: int; number of resources
    #           - families: list; list of partitioned species into families
    #           - c_control: matrix; matrix computed by "compute_c_class()"
    #           - num_T: int, number of tiers
    #           - m: list; list of mortality computed by "compute_m()"

    a = np.zeros((S,R))
    pref = []
    for i in range(S):
        idx_family = int(families[i][-1])
        c_row = c_control[idx_family,:]
        nums=np.array([x for x in range(num_T)])
        idx_tier = nums[c_row==1].item()
        pref.append('T'+str(idx_tier))  
    species_df = pd.DataFrame({"Family": families, "Preference": pref, "m": m})
    return species_df


@njit
def sigma(x, k, N, Type):

    # function that define the sigma function
    # inputs:   - x: argumet of the function
    #           - k: numeric; parameter of the monod and sigmoid functions
    #           - N: numeric; parameter of sigmoid function
    #           - Type: string; string that select the type of sigma function

    if Type == 'linear':
        return x
    elif Type == 'monod':
        return x / (1 + x/k)
    elif Type == 'sigmoid':
        return np.power(x, N) / (1 + np.power(x/k, N))
    else:
        raise ValueError("Invalid type function. \n Valid types are: \n - linear \n - monod \n - sigmoid" )


@njit
def func_A(S, R, l, w, c, r, m, k, N, Type):

    # auxiliary function used in "CMR_dynamics()"
    # inputs: see "CMR_dynamics()"

    output = np.zeros(S)
    for i in range(S):
        element = 0
        for alpha in range(R):
            element += (1 - l[alpha]) * w[alpha] * sigma(c[i,alpha]*r[alpha], k, N, Type) 
        element -= m[i]
        output[i] = element          
    return output


@njit
def func_B(S, R, n, c, r, k, N, Type):

    # auxiliary function used in "CMR_dynamics()"
    # inputs: see "CMR_dynamics()"

    output = np.zeros(R)
    for alpha in range(R):
        element = 0
        for j in range(S):
            element += n[j] * sigma(c[j,alpha] * r[alpha], k, N, Type)
        output[alpha] = element          
    return output


@njit
def func_C(S, R, n, r, c, D, w, l, k, N, Type):

    # auxiliary function used in "CMR_dynamics()"
    # inputs: see "CMR_dynamics()"

    output = np.zeros(R)
    for alpha in range(R):
        element = 0
        for j in range(S):
            for beta in range(R):
                element += n[j] * sigma(c[j,beta] * r[beta], k, N, Type) * (D[j, alpha, beta] * (w[beta]/w[alpha]) * l[beta])
        output[alpha] = element            
    return output


@njit
def compute_h(r, k_alpha, tau_R):

    # function that computes the h function for the resources
    # inputs:   - r: array; array of the resources
    #           - k_alpha: array; vector that select the resource constantly pumped into the system
    #           - tau_R: numeric; period of the costantly pumped resource

    return k_alpha - pow(tau_R, -1) * r


def compute_c_classes(F, T):

    # function that compute a random matrix that associate resource preference for each family
    # inputs:   - F: int; number of families
    #           - T: int; number of tiers

    c_control = np.zeros((F, T))
    for i in range(F):
        j = np.random.randint(0, T)
        c_control[i,j] = 1
    return c_control


def compute_c_matrix(S, R, families, tiers, class_matrix, mu_c, sigma_c, c_0, c_1, q_A, preference_type): 

    # function that computes the preference matrix
    # inputs:   - S: int; number of species
    #           - R: int; number of resources
    #           - families: array; vector of the partitioned species into familes
    #           - tiers: array; vector of the partitioned resource into tiers
    #           - class_matrix: matrix; the c_control matrix computed in "compute_c_classes()"
    #           - mu_c: float; mean of the gaussian distribution
    #           - sigma_c: float; square root of the variance of the gaussian distribution
    #           - q_A: control parameter of how much more species from a given family prefers a given tier of resouces

    if preference_type == "gaussian":
        M_uniques, M_A = np.unique(tiers, return_counts=True)
        variance = pow(sigma_c,2)/R
        c_control = np.zeros((S,R))
        c_matrix = np.zeros((S,R))
        for i in range(S):
            for alpha in range(R):  
                x = int(tiers[alpha][-1])   
                y = int(families[i][-1])    
                c_control[i,alpha] = class_matrix[y,x]            
                mean1 = mu_c/R * (1 + ((R - M_A[x])/M_A[x]) * q_A)
                mean0 = mu_c/R * (1 - q_A)
                if c_control[i,alpha] == 0:
                    c_matrix[i,alpha] = np.random.normal(loc=mean0, scale=variance)
                else:
                    c_matrix[i,alpha] = np.random.normal(loc=mean1, scale=variance)   
    elif preference_type == "binary":
        M_uniques, M_A = np.unique(tiers, return_counts=True)
        c_control = np.zeros((S,R))
        c_matrix = np.zeros((S,R))
        for i in range(S):
            for alpha in range(R):  
                x = int(tiers[alpha][-1])   
                y = int(families[i][-1])    
                c_control[i,alpha] = class_matrix[y,x]            
                if c_control[i,alpha] == 0:
                    p1 = (mu_c/(R * c_1)) * (1 - q_A)
                    p0 = 1 - p1
                else:
                    p1 = (mu_c/(R * c_1)) * (1 + ((R - M_A[x])/M_A[x]) * q_A)
                    p0 = 1 - p1
                X = np.random.choice([0,1], p = [p0, p1])
                c_matrix[i,alpha] = c_0/R + c_1 * X
    else: ValueError("Invalid type function. \n Valid types are: \n - gaussian \n - binary" )
    return c_matrix


def compute_D(S, R, species_df, tiers, f_c, f_s, d0):  

    # function that computes the metabolic matrix
    # inputs:   - S: int; number of species
    #           - R: int; number of resources
    #           - species_df: data frame of the species computed by "species_table()"
    #           - tiers: array; vector of the partitioned resource into tiers
    #           - f_c: float; fracion of secreted flux that goes to the first tier
    #           - f_s: float; fracion of secreted flux that goes to the second tier
    #           - d_0: float; control parameter of the randomness of the partition

    M = np.zeros((S,R,R))
    D = np.zeros_like(M)
    for i in range(S):
        preference=species_df["Preference"].iloc[i]
        f_0 = 1 - f_c -f_s
        M_c = len(tiers[tiers==preference])
        for alpha in range(R):
            for beta in range(R):
                M_s = len(tiers[tiers==tiers[beta]]) 
                if tiers[alpha]==preference:
                    M[i,alpha,beta]=d0*f_c/M_c
                elif tiers[alpha]==tiers[beta]:
                    M[i,alpha,beta]=d0*f_s/M_s
                else:
                    M[i,alpha,beta]=d0*f_0/(R-M_s-M_c)
        for beta in range(R):
            D[i,:,beta]=np.random.dirichlet(M[i,:,beta])       
    return D


def compute_m(partitions):

    # function that computes the mortality for each family
    # inputs:   - partitions: array; vector of the partitioned species into families

    nums = len(partitions)
    randoms = np.random.normal(loc=1, scale=0.1, size=nums)
    l_tot = []
    for i in range(nums):
        l = [randoms[i] for x in range(partitions[i])]
        l_tot = [*l_tot, *l]
        l_tot = np.array(l_tot)
    return l_tot


@njit
def CRM_dynamics(y, t, S, R, g, m, D, w, c, l, k_alpha, tau_R, k, N, Type):   

    # function that defines the differential equations of the model
    # inputs:   - y: array; array of the species/ resources
    #           - t: array; time of the simulation
    #           - S: int; number of species
    #           - R: int; nuber of resouces
    #           - g: array; 
    #           - m: array: mortality of species
    #           - D: matrix; metabolic matrix
    #           - w: array;
    #           - c: matrix; preference matrix
    #           - l: float; leakage parameter
    #           - k_alpha: int; index of the costantly pumped resource 
    #           - tau_R: numeric; period of degradation of the resources 
    #           - k: numeric; parameter of the sigma function for monod/sigmoid
    #           - N: numeric; parameter of the sigma function for sigmoid
    #           - Type: string; type of sigma function 
    
    n = y[:S] 
    r = y[S:]       
    dn = g * n * func_A(S, R, l, w, c, r, m, k, N, Type=Type)

    B = func_B(S, R, n, c, r, k, N, Type=Type)
    C = func_C(S, R, n, r, c, D, w, l, k, N, Type=Type)
    h = compute_h(r, k_alpha, tau_R)
    dr = h - B + C  
  
    results = np.zeros_like(y)
    results[:S] = dn
    results[S:] = dr
    return results


#================================================================================================================================
# 2. Community Model Class

class CommunityModel:

    # Class that simulates a community CRM for F families of bacteria and T tiers of resources.     
    # input: @ dictionary of parameters                                                             

    def __init__(self, inputs):

        self.S_tot = inputs["S_tot"]                        # int, total number of species in the ecosystem
        self.S = inputs["S"]                                # int; total number of species in the community
        self.R = inputs["R"]                                # int; total number of resouces

        # initial condition parameters:
        self.S0 = inputs["S0"]                              # int or array; initial number of species
        self.R0 = inputs["R0"]                              # int or array; initial number of resouces
        self.s0 = np.ones(self.S) * self.S0
        self.r0 = np.ones(self.R) * self.R0
        self.y0 = np.concatenate([self.s0, self.r0])

        # duration of simulation parameters:
        self.time_period = inputs["time_period"]            # int; time period of the simulation
        self.t_step = inputs["t_step"]                      # float; time step of the simulation 

        # partition parameters:
        self.num_F = inputs["num_F"]                        # int; number of species families
        self.num_T = inputs["num_T"]                        # int; number of resouces tires
        self.partition_F = inputs["partition_F"]            # list of ints; families partitions
        self.partition_T = inputs["partition_T"]            # list of ints; tiers partitions

        # leakage parameter:
        self.leakage = inputs["leakage"]                    # float; leakage percentage
        self.l = np.ones(self.R) * self.leakage

        # sigma function selection parameter:
        self.sigma_type = inputs["sigma_type"]              # string; type of sigma function 

        # sigma function parameters:
        self.k = inputs["k"]                                # numeric; parameter of the monod/sigmoid function
        self.N = inputs["N"]                                # numeric; parameter of the sigmoid function
       
        # metabolic matrix parameters:
        self.f_c = inputs["f_c"]                            # float; fracion of secreted flux that goes to the first tier
        self.f_s = inputs["f_s"]                            # float; fracion of secreted flux that goes to the second tier
        self.d0 = inputs["d0"]                              # float; control parameter of the randomness of the partition

        # h function parametes:
        self.tau_R = inputs["tau_R"]                        # numeric; period of degradation of the resources
        self.k_alpha_index = inputs["k_alpha_index"]        # int; index of the costantly pumped resource 
        self.k_alpha = np.zeros(self.R)
        self.k_alpha[self.k_alpha_index] = inputs["k_0"]

        # preference matrix parametes:
        self.preference_type = inputs["preference_type"]
        self.mu_c = inputs["mu_c"]                          # float; mean of the gaussian distribution
        self.sigma_c = inputs["sigma_c"]                    # float; standard deviation of the gaussian distribution
        self.c_0 = inputs["c_0"]                            # numeric;
        self.c_1 = inputs["c_1"]                            # numeric;
        self.q_A = inputs["q_A"]                            # float; control parameter of how much more species from a given family prefers a given tier of resouces
       
        self.w = np.ones(self.R)                            # numeric; energy density of the resources
        self.g = np.ones(self.S)


    #============================================================================================================================
    # 2.1 Model Simulation:

    def Simulate(self):

        # function that simulate the model

        self.t = np.linspace(0, self.time_period, int(self.time_period/self.t_step))

        self.families = class_partition('F',self.num_F, self.partition_F)     
        self.tiers = class_partition('T', self.num_T, self.partition_T)       

        m = compute_m(self.partition_F)

        c_control = compute_c_classes(self.num_F, self.num_T)

        self.c = compute_c_matrix(self.S_tot, self.R, self.families, self.tiers, c_control, self.mu_c, self.sigma_c, self.c_0, self.c_1, self.q_A, self.preference_type) 

        death_indexes=np.sort(np.random.choice(self.S_tot,self.S_tot-self.S,replace=False))
        survived_idexes=np.arange(self.S_tot)
        survived_idexes=np.delete(survived_idexes,death_indexes)
        
        self.c_effective = np.delete(self.c,death_indexes,axis=0)

        self.species_df = species_table(self.S_tot, self.R, self.families, c_control, self.num_T, m).iloc[survived_idexes].reset_index(drop=True)
        
        m_community= np.array(self.species_df["m"])

        self.D = compute_D(self.S, self.R, self.species_df, self.tiers, self.f_c, self.f_s, self.d0)

        y = odeint(CRM_dynamics, self.y0, self.t, args=(self.S, self.R, self.g, m_community, self.D, self.w, self.c_effective, self.l, self.k_alpha, self.tau_R, self.k, self.N, self.sigma_type),rtol=1e-5)
        
        NS = y[:, :self.S]
        NR = y[:, self.S:]

        NS_plot = NS.copy()
        NS_plot[NS_plot < 1] = 0       

        self.NS = NS
        self.NS_plot = NS_plot
        self.NR = NR
        

    #============================================================================================================================
    # 2.2 Methods:

    def ReturnData(self, Type):

        # function that returns the data of the simulation
        # input:    - Type: string; select the data 

        if Type == "resources":
            return self.NR
        if Type == "species":
            return self.NS
        else:
            raise ValueError("Invalid data required. \n Valid data are: \n - resouces \n - species" )


    def ModelSeed(self, seed):

        # function that fix the seed of the simulation
        # input:    - seed: int; seed of np.random

        np.random.seed(seed)


    def ReturnTables(self, Type):

        # function that returns data frames with in information on resources/species 
        # input:    - Type: string; select the data frame

        if Type == "resources":
            self.resources_df = pd.DataFrame({"Tier": self.tiers})
            return self.resources_df
        if Type == "species":
            return self.species_df
        else:
            raise ValueError("Invalid dataframe required. \n Valid dataframe are: \n - resources \n - species" )
        

    def ReturnNicheOverlap(self): 

        # function that returns niche overlap given the preference matrix

        c = self.c.copy()
        n = c.shape[0] 
        overlaps = []
        for i in range(n):
            for j in range(i+1, n):
                numerator = np.sum(c[i,:] * c[j,:])
                denominator = np.sqrt(np.sum(c[i,:] ** 2)) * np.sqrt(np.sum(c[j,:] ** 2))
                overlap = numerator / denominator
                overlaps.append(overlap)
        mean_overlap = 2 * np.mean(overlaps)
        return mean_overlap
    

    def ReturnSimpsonDiversity(self): 

        # function that returns simpson diversity given the incoming flux
       
        J_in = np.zeros((self.S,self.R))
        for i in range(self.S):
            for alpha in range(self.R):
                J_in[i][alpha] = self.w[alpha] * sigma(self.c[i,alpha]*self.NR[-1][alpha], self.k, self.N, self.sigma_type) 

        J_in_total = np.sum(J_in,axis=1)
        squared_ratio = np.zeros((self.S, self.R))
        for i in range(self.S):
            squared_ratio[i,:] = pow(J_in[i] / J_in_total[i], 2)
        SD = pow(np.sum(squared_ratio,axis=1),-1)
        return SD


    def ReturnMetabolicProduction(self, s, Type):

        # function that return a barplot of the metabolic production for given species
        # input:    - s: int; index of the species
        #           - Type: string; select the visualization


        if Type == "graphical":
            a=self.D[s,:,:]
            fig, ax = plt.subplots(figsize=(10,4))
            ax.set(title="Species {} metabolic production (family: {}, preference: {})".format(int(s),self.families[s],self.species_df["Preference"].iloc[s]),xlabel=r"$\beta$",xticks=range(int(self.D.shape[1])))
            ax.set_ylim((0,1.1))
            base=np.zeros(self.R)
            for alpha in range(int(self.D.shape[1])):
                ax.bar(range(int(self.D.shape[1])),a[alpha,:],bottom=base, label=r"$\alpha$={}".format(alpha));
                base += a[alpha,:]
        elif Type == "matrix":
            return self.D[s,:,:]
        else:
            raise ValueError("Invalid visualization required. \n Valid visualizations are: \n - graphical \n - matrix" )
       


    def ReturnTotalPreferenceMatrix(self, Type):

        # function that returns the imshow of the preference matrix
        # inputs:   - Type: string; select the visualization


        if Type == "graphical":
            plt.figure(figsize=(14,8))
            plt.imshow(self.c,cmap="viridis") 
            plt.colorbar(shrink=0.8)
            plt.title("Total Preference Matrix")
            plt.xlabel("Resources"),plt.ylabel("Species")
            plt.xticks(range(self.R),rotation=90), plt.yticks(range(self.S_tot)); 
        elif Type == "matrix":
            return self.c
        else:
            raise ValueError("Invalid visualization required. \n Valid visualizations are: \n - graphical \n - matrix" )

    def ReturnCommunityPreferenceMatrix(self, Type, figsize=(6, 6), title="Community Preference Matrix", ax=None):

        # function that returns the imshow of the preference matrix
        # inputs:   - Type: string; select the visualization

        if Type == "graphical":
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                ax = ax

            im = ax.imshow(self.c_effective, cmap="viridis")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title)
            ax.set_xlabel("Resources")
            ax.set_ylabel("Species")
            ax.set_xticks(range(self.R))
            ax.set_yticks(range(self.S))
            ax.set_xticklabels(range(self.R), rotation=90)

            if not isinstance(ax, plt.Axes):
                plt.show()
        elif Type == "matrix":
            return self.c_effective
        else:
            raise ValueError("Invalid visualization required. \n Valid visualizations are: \n - graphical \n - matrix")


    def ReturnSimulationPlots(self):

        # function that returns the plots of the simulation

        yS=np.vstack(self.NS_plot.T)
        yR=np.vstack(self.NR.T)

        fig, ax= plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("CRM (leakage={}; $k_0w_0$={})".format(self.l[0],self.k_alpha[self.k_alpha_index]), fontsize=11)
        ax[0,0].set(xlabel="Time", ylabel="$N_i(t)$", title="Populations")
        ax[0,0].grid(alpha=0.5)
        for s in range(self.S):
            ax[0,0].plot(self.t, self.NS_plot[:,s])
        ax[0,0].set_xscale("log")
        ax[0,0].set_yscale("log")
        ax[0,0].set_ylim(bottom=1)
        ax[0,0].set_xlim(left=0.01, right=self.time_period)

        ax[0,1].set(xlabel="Time", ylabel=r"$R_{\alpha}(t)$", title="Resources")
        ax[0,1].grid(alpha=0.5)
        for r in range(self.R):
            ax[0,1].plot(self.t, self.NR[:,r])
        ax[0,1].set_xscale("log")
        ax[0,1].set_yscale("log")
        ax[0,1].set_ylim(bottom=pow(10,-3))
        ax[0,1].set_xlim(left=0.01, right=self.time_period)

        ax[1,0].grid(alpha=0.5)
        ax[1,0].set(xlabel="Time", ylabel=r"$N_{total}(t)$")
        ax[1,0].stackplot(self.t,yS)
        ax[1,0].set_xscale("log")
        ax[1,0].set_yscale("log")
        ax[1,0].set_ylim(bottom=1)
        ax[1,0].set_xlim(left=0.01, right=self.time_period)

        ax[1,1].grid(alpha=0.5)
        ax[1,1].set(xlabel="Time", ylabel=r"$R_{total}(t)$")
        ax[1,1].stackplot(self.t,yR)
        ax[1,1].set_xscale("log")
        ax[1,1].set_yscale("log")
        ax[1,1].set_xlim(left=0.01, right=self.time_period);


    def ReturnAbundance(self):

        # function that returns barplot of RSA at stationarity

        S = self.NS_plot[-1]
        R = self.NR[-1]
        R[R<1e-3]=0

        fig, ax= plt.subplots(2,1,figsize=(12,6))
        fig.suptitle("Abundance at stationarity (leakage={}; $k_0w_0$={})".format(self.l[0],self.k_alpha[self.k_alpha_index]), fontsize=11)

        ax[0].bar(range(len(S)),S) #ax[0].bar(range(len(S)),S/np.sum(S))
        ax[0].set(xlabel="Species", ylabel="Abundance")
        ax[0].set_xticks(range(len(S)))
        ax[0].grid(alpha=0.5)

        ax[1].bar(range(len(R)),R,color="darkorange") #ax[1].bar(range(len(R)),R/np.sum(R),color="darkorange"
        ax[1].set(xlabel="Resources", ylabel="Abundance")
        ax[1].set_xticks(range(len(R)))
        ax[1].grid(alpha=0.5);


    def StabilityTime(self, Type, Plots=False, thr=1e-3):
            
        # function that returns the stability time of the system
        # inputs:   - thr: threshold (default 1-e3)
        # inputs:   - Type: string; select the data ("species" or "resources")

        t_diff = self.t[1:]

        if Type == "species":
            sp_diff = np.diff(self.NS,axis=0)
            sp_diff_tot = np.mean(sp_diff,axis=1)   
            t_star = t_diff[np.abs(sp_diff_tot)<thr][0]

            if Plots:
                fig, ax = plt.subplots(figsize=(7,4))
                ax.set(title="Species Dynamics Velocity (leakage={}; $k_0w_0$={})".format(self.l[0],self.k_alpha[self.k_alpha_index]),xlabel="Time",ylabel="Mean Velocity")
                ax.plot(t_diff,sp_diff_tot)
                ax.scatter(t_star,thr,c="red")
                ax.set_xscale("log")
                ax.grid(alpha=0.5);

            return t_star

        if Type == "resources":
            res_diff = np.diff(self.NR,axis=0)
            res_diff_tot = np.mean(res_diff,axis=1)   
            t_star = t_diff[np.abs(res_diff_tot)<thr][0]

            if Plots:
                fig, ax = plt.subplots(figsize=(7,4))
                ax.set(title="Resources Dynamics Velocity (leakage={}; $k_0w_0$={})".format(self.l[0],self.k_alpha[self.k_alpha_index]),xlabel="Time",ylabel="Mean Velocity")
                ax.plot(t_diff,res_diff_tot)
                ax.scatter(t_star,thr,c="red")
                ax.set_xscale("log")
                ax.grid(alpha=0.5);

            return t_star