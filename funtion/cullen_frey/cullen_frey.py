import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median,mean,stdev
from scipy.stats import kurtosis, skew
from math import sqrt, ceil
np.seterr(under='ignore')
np.seterr(over='ignore')


class cullen_frey():
    """
    Create a skewness-curtosis graph based on Cullen and Frey (1999)
    """
    def __init__(self,data,method='unbiased',discrete=False,boot=None,graph=True):
        """
        Parameters
        data: N x 1 list of sample data
        method: 'unbiased' for unbiased estimated values of statistics or 'sample' for sample values.
        discrete: If True, the distribution is considered as discrete.
        boot: If not None, boot values of skewness and kurtosis are plotted from bootstrap samples of data. 
        boot must be fixed in this case to an integer above 10.
        graph: If False, the skewness-kurtosis graph is not plotted.
        """
        
        self.df = data
        self.method = method
        self.discrete = discrete
        self.boot = boot
        self.graph = graph
        
        if not isinstance(self.df,list) or len(np.shape(self.df)) > 1: 
            raise TypeError('Samples must a list with N x 1 dimensions')
        
        if len(self.df) < 4:
            raise ValueError('The number of samples needs to be greater than 4')
            
        if self.boot is not None:    
            if not isinstance(self.boot,int):
                raise ValueError('boot must be integer')
            
        self.stats()
        
        
    def stats(self):
        """
        Evaluates the min, max, median, mean, skewness and kurtosis of data
        """
        if self.method=='unbiased':        
            self.skewdata = skew(self.df, bias=False)
            self.kurtdata = kurtosis(self.df, bias=False)+3

        elif self.method=='sample':
            self.skewdata = skew(self.df, bias=True)
            self.kurtdata = kurtosis(self.df, bias=True)+3

        
        res = [min(self.df),max(self.df),median(self.df),mean(self.df),stdev(self.df),self.skewdata,self.kurtdata]
    
        #Resumo estatístico
        print(f'min:  {res[0]:.6f}   max: {res[1]:.6f}')
        print(f'median:  {res[2]:.6f}')
        print(f'mean:  {res[3]:.6f}')
        print(f'estimated mean:  {res[4]:.6f}')
        print(f'estimated skewness:  {res[5]:.6f}')
        print(f'estimated kurtosis:  {res[6]:.6f}')

        if self.graph:
            self.cullen_frey_graph()
            
    def cullen_frey_graph(self): 
        """
        If graph = True, draws the skewness-kurtosis plot
        """
         
        if self.boot is not None:
            if self.boot < 10:
                raise ValueError('boot must be greater than 10')

            n = len(self.df)

            nrow = n
            ncol = self.boot
            databoot = np.reshape(np.random.choice(self.df, size=n*self.boot, replace=True),(nrow,ncol)) 

            s2boot = (skew(pd.DataFrame(databoot)))**2
            kurtboot = kurtosis(pd.DataFrame(databoot))+3

            kurtmax = max(10,ceil(max(kurtboot)))
            xmax = max(4,ceil(max(s2boot)))

        else:
            kurtmax = max(10,ceil(self.kurtdata))
            xmax = max(4,ceil(self.skewdata**2))

        ymax = kurtmax-1
        
        # If discrete = False
        if not self.discrete:
            #Beta distribution
            p = np.exp(-100)
            lq = np.arange(-100,100.1,0.1)
            q = np.exp(lq)
            s2a = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
            ya = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
            p = np.exp(100)
            lq = np.arange(-100,100.1,0.1)
            q = np.exp(lq)
            s2b = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
            yb = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
            s2 = [*s2a,*s2b]
            y = [*ya,*yb]

            #Gama distribution
            lshape_gama = np.arange(-100,100,0.1)
            shape_gama = np.exp(lshape_gama)
            s2_gama = 4/shape_gama
            y_gama = kurtmax-(3+6/shape_gama) 

            #Lognormal distribution
            lshape_lnorm = np.arange(-100,100,0.1)
            shape_lnorm = np.exp(lshape_lnorm)
            es2_lnorm = np.exp(shape_lnorm**2, dtype=np.float64)
            s2_lnorm = (es2_lnorm+2)**2*(es2_lnorm-1)
            y_lnorm = kurtmax-(es2_lnorm**4+2*es2_lnorm**3+3*es2_lnorm**2-3)

            plt.figure(figsize=(12,9))
            
            #observations
            obs = plt.scatter(self.skewdata**2,kurtmax-self.kurtdata,s=200, c='blue', 
                              label='Observations',zorder=10)
            #beta
            beta = plt.fill(s2,y,color='lightgrey',alpha=0.6, label='beta', zorder=0)
            #gama
            gama = plt.plot(s2_gama,y_gama, '--', c='k', label='gama')
            #lognormal
            lnormal = plt.plot(s2_lnorm,y_lnorm, c='k', label='lognormal')
            
            if self.boot is not None:
                #bootstrap 
                bootstrap = plt.scatter(s2boot,kurtmax-kurtboot,marker='$\circ$',c='orange',s=50, 
                                        label='Bootstrap values', zorder=5)
                legenda1 = plt.legend(handles=[bootstrap],loc=(xmax*0.1065,ymax*0.101), 
                                      labelspacing=2, frameon=False)
                plt.gca().add_artist(legenda1)
            

            #markers
            normal = plt.scatter(0,kurtmax-3, marker=(8,2,0),s=400,c='k',label='normal',zorder=5)

            uniform = plt.scatter(0,kurtmax-9/5, marker='$\\bigtriangleup$',s=400,c='k',label='uniform',zorder=5)   

            exp_dist = plt.scatter(2**2,kurtmax-9, marker='$\\bigotimes$',s=400,c='k',label='exponential',zorder=5) 

            logistic = plt.scatter(0,kurtmax-4.2, marker='+',s=400,c='k',label='logistic',zorder=5)


            #Adjusting the axis
            yax = [str(kurtmax - i) for i in range(0,ymax+1)]
            plt.xlim(-0.08, xmax+0.4)
            plt.ylim(-1, ymax+0.08)
            plt.yticks(list(range(0,ymax+1)),labels=yax)

            #Adding the labels
            plt.xlabel('square of skewness', fontsize=13)
            plt.ylabel('kurtosis', fontsize=13)
            plt.title('Cullen and Frey graph', fontsize=15) 

            #Adding the legends
            legenda2 = plt.legend(handles=[obs],loc='upper center', labelspacing=1, frameon=False)
            plt.gca().add_artist(legenda2)

            plt.legend(handles=[normal,uniform,exp_dist,logistic,beta[0],lnormal[0],gama[0]], 
                       title='Theoretical distributions',loc='upper right',labelspacing=1.4,frameon=False)

            plt.show()
        
        #If discrete = True
        else:
            # negbin distribution
            p = np.exp(-10)
            lr = np.arange(-100,100,0.1)
            r = np.exp(lr)
            s2a = (2-p)**2/(r*(1-p))
            ya = kurtmax-(3+6/r+p**2/(r*(1-p)))
            p = 1-np.exp(-10)
            lr = np.arange(100,-100,-0.1)
            r = np.exp(lr)
            s2b = (2-p)**2/(r*(1-p))
            yb = kurtmax-(3+6/r+p**2/(r*(1-p)))
            s2_negbin = [*s2a,*s2b]
            y_negbin = [*ya,*yb]
            
            # poisson distribution
            llambda = np.arange(-100,100,0.1)
            lambda_ = np.exp(llambda)
            s2_poisson = 1/lambda_
            y_poisson = kurtmax-(3+1/lambda_)
            
            plt.figure(figsize=(12,9))          
            
            #observations
            obs = plt.scatter(self.skewdata**2,kurtmax-self.kurtdata,s=200, c='blue', 
                              label='Observations',zorder=10)
            
            #negative binomial
            negbin = plt.fill(s2_negbin,y_negbin,color='lightgrey',alpha=0.6, label='negative binomial', zorder=0)
            
            #poisson
            poisson = plt.plot(s2_poisson,y_poisson, '--', c='k', label='poisson')
            
            if self.boot is not None:
                #bootstrap 
                bootstrap = plt.scatter(s2boot,kurtmax-kurtboot,marker='$\circ$',c='orange',s=50, 
                                        label='Bootstrap values', zorder=5)
                legenda2 = plt.legend(handles=[bootstrap],loc=(xmax*0.1065,ymax*0.101), 
                                      labelspacing=2, frameon=False)
                plt.gca().add_artist(legenda2)
            
            #markers
            normal = plt.scatter(0,kurtmax-3, marker=(8,2,0),s=400,c='k',label='normal',zorder=5)
            
            #adjusting the axis
            yax = [str(kurtmax - i) for i in range(0,ymax+1)]
            plt.xlim(-0.08, xmax+0.4)
            plt.ylim(-1, ymax+0.08)
            plt.yticks(list(range(0,ymax+1)),labels=yax)

            #adding the labels
            plt.xlabel('square of skewness', fontsize=13)
            plt.ylabel('kurtosis', fontsize=13)
            plt.title('Cullen and Frey graph', fontsize=15) 
            
            #adding the legends
            legenda1 = plt.legend(handles=[obs],loc='upper center', labelspacing=1, frameon=False)
            plt.gca().add_artist(legenda1)

            plt.legend(handles=[normal,negbin[0],poisson[0]],title='Theoretical distributions',loc='upper right',
                       labelspacing=1.4,frameon=False)
            
            plt.show()
