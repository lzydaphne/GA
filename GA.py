import copy
import random
import sys
import numpy as np

from colorama import init
from termcolor import colored

class GAScheduler:
    def __init__(self, pt, ms):
        init()  # init colorama
        self.__original_stdout = sys.stdout
        self.pt = np.array(pt)
        self.ms = np.array(ms)

        dfshape = self.pt.shape
        self.num_mc = dfshape[1] # number of machines
        self.num_job = dfshape[0] # number of jobs
        self.num_gene = self.num_mc * self.num_job # number of genes in a chromosome

        self.population_size = 50
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        self.mutation_selection_rate = 0.2
        self.num_mutation_jobs = round(self.num_gene * self.mutation_selection_rate)
        self.num_iteration = 200

    # initialize population
    def init_population(self, population_size):
        population_list = []
        for i in range(population_size):
            nxm_random_num=list(np.random.permutation(self.num_gene)) # generate a random permutation of 0 to num_job*num_mc-1
            population_list.append(nxm_random_num) # add to the population_list
            for j in range(self.num_gene):
                population_list[i][j]=population_list[i][j]%self.num_job # convert to job number format, every job appears m times
        return population_list
    
    # two point crossover
    def crossover(self, population_size, population_list):
        parent_list = copy.deepcopy(population_list)
        offspring_list = copy.deepcopy(population_list)
        S = list(np.random.permutation(population_size)) # generate a random sequence to select the parent chromosome to crossover
        
        for m in range(int(population_size/2)):
            crossover_prob=np.random.rand()
            if self.crossover_rate >= crossover_prob:
                parent_1 = population_list[S[2*m]][:]
                parent_2 = population_list[S[2*m+1]][:]
                child_1 = parent_1[:]
                child_2 = parent_2[:]
                cutpoint = list(np.random.choice(self.num_gene, 2, replace=False))
                cutpoint.sort()
            
                child_1[cutpoint[0]:cutpoint[1]]=parent_2[cutpoint[0]:cutpoint[1]]
                child_2[cutpoint[0]:cutpoint[1]]=parent_1[cutpoint[0]:cutpoint[1]]
                offspring_list[S[2*m]]=child_1[:]
                offspring_list[S[2*m+1]]=child_2[:]
        return parent_list, offspring_list
    
    # repairment
    def repairment(self, population_size, offspring_list):
        for m in range(population_size):
            job_count={}
            larger,less=[],[] # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
            for i in range(self.num_job):
                if i in offspring_list[m]:
                    count = offspring_list[m].count(i)
                    pos = offspring_list[m].index(i)
                    job_count[i] = [count,pos] # store the above two values to the job_count dictionary
                else:
                    count = 0
                    job_count[i] = [count,0]
                if count > self.num_mc:
                    larger.append(i)
                elif count < self.num_mc:
                    less.append(i)
                    
            for k in range(len(larger)):
                chg_job = larger[k]
                while job_count[chg_job][0] > self.num_mc:
                    for d in range(len(less)):
                        if job_count[less[d]][0] < self.num_mc:                    
                            offspring_list[m][job_count[chg_job][1]] = less[d]
                            job_count[chg_job][1] = offspring_list[m].index(chg_job)
                            job_count[chg_job][0] = job_count[chg_job][0] - 1
                            job_count[less[d]][0] = job_count[less[d]][0] + 1                    
                        if job_count[chg_job][0] == self.num_mc:
                            break
        return offspring_list
    
    # mutation
    def mutate(self, offspring_list):
        for m in range(len(offspring_list)):
            self.mutation_prob = np.random.rand()
            if self.mutation_rate >= self.mutation_prob:
                m_chg = list(np.random.choice(self.num_gene, self.num_mutation_jobs, replace=False)) # chooses the position to mutation
                t_value_last = offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
                for i in range(self.num_mutation_jobs-1):
                    offspring_list[m][m_chg[i]] = offspring_list[m][m_chg[i+1]] # displacement
                
                offspring_list[m][m_chg[self.num_mutation_jobs-1]] = t_value_last # move the value of the first mutation position to the last mutation position
        return offspring_list
    
    # fitness calculation (makespan)
    def fitness(self, population_size, parent_list, offspring_list):
        total_chromosome = copy.deepcopy(parent_list)+copy.deepcopy(offspring_list) # parent and offspring chromosomes combination
        chrom_fitness,chrom_fit = [], []
        total_fitness=0
        for m in range(population_size * 2):
            j_keys = [j for j in range(self.num_job)]
            key_count = {key : 0 for key in j_keys}
            j_count = {key : 0 for key in j_keys}
            m_keys = [j+1 for j in range(self.num_mc)]
            m_count = {key : 0 for key in m_keys}
            
            for i in total_chromosome[m]:
                gen_t = int(self.pt[i][key_count[i]])
                gen_m = int(self.ms[i][key_count[i]])
                j_count[i] = j_count[i] + gen_t
                m_count[gen_m] = m_count[gen_m] + gen_t
                
                if m_count[gen_m] < j_count[i]:
                    m_count[gen_m] = j_count[i]
                elif m_count[gen_m] > j_count[i]:
                    j_count[i] = m_count[gen_m]
                
                key_count[i] = key_count[i]+1
        
            makespan = max(j_count.values())
            chrom_fitness.append(1/makespan)
            chrom_fit.append(makespan)
            total_fitness = total_fitness + chrom_fitness[m]
        return total_chromosome, chrom_fitness, total_fitness, chrom_fit

    def select(self, population_size, population_list, total_chromosome, chrom_fitness, total_fitness):
        pk,qk=[], []
        
        for i in range(population_size * 2):
            pk.append(chrom_fitness[i] / total_fitness)
        for i in range(population_size * 2):
            cumulative=0
            for j in range(0,i+1):
                cumulative=cumulative + pk[j]
            qk.append(cumulative)
        
        selection_rand=[np.random.rand() for i in range(population_size)]
        
        for i in range(population_size):
            if selection_rand[i]<=qk[0]:
                population_list[i] = copy.deepcopy(total_chromosome[0])
            else:
                for j in range(0,population_size*2-1):
                    if selection_rand[i]>qk[j] and selection_rand[i]<=qk[j+1]:
                        population_list[i] = copy.deepcopy(total_chromosome[j+1])
                        break
        return population_list

    RUN_LABLE = "[scheduler]"
    RUN_LABLE_COLOR = "blue"

    # called in main to run GA
    def run_genetic(self, population_size=10, num_iteration=100, verbose=False):
        assert population_size > 0, num_iteration > 0

        # don't print if verbose is set to false
        if not verbose:
            sys.stdout = None

        print(colored(self.RUN_LABLE, self.RUN_LABLE_COLOR), "initializing population...")
        population_list = self.init_population(population_size)
        best_list,best_obj = [],[]
        makespan_record = []
        Tbest = 999999999999999

        for current_generation in range(num_iteration):
            Tbest_now = 99999999999
            # crossover
            parent_list, offspring_list = self.crossover(population_size, population_list)
            # repairment
            offspring_list = self.repairment(population_size, offspring_list)
            # mutatuon
            offspring_list = self.mutate(offspring_list)
            # fitness calculation
            total_chromosome, chrom_fitness, total_fitness, chrom_fit = self.fitness(population_size, parent_list, offspring_list)
            # selection
            self.select(population_size, population_list, total_chromosome, chrom_fitness, total_fitness)

            for i in range(population_size * 2):
                if chrom_fit[i] < Tbest_now:
                    Tbest_now = chrom_fit[i]
                    sequence_now = copy.deepcopy(total_chromosome[i])
            if Tbest_now <= Tbest:
                Tbest = Tbest_now
                sequence_best = copy.deepcopy(sequence_now)
            
            makespan_record.append(Tbest)
        
        return sequence_best, Tbest
    
    def draw_Gnatt(self, sequence_best):
        import pandas as pd
        from chart_studio.plotly import plotly as py
        import plotly.figure_factory as ff
        from plotly.offline import plot
        import datetime

        m_keys = [j+1 for j in range(self.num_mc)]
        j_keys = [j for j in range(self.num_job)]
        key_count = {key:0 for key in j_keys}
        j_count = {key:0 for key in j_keys}
        m_count = {key:0 for key in m_keys}
        j_record = {}
        for i in sequence_best:
            gen_t = int(self.pt[i][key_count[i]])
            gen_m = int(self.ms[i][key_count[i]])
            j_count[i] = j_count[i] + gen_t
            m_count[gen_m] = m_count[gen_m] + gen_t
            
            if m_count[gen_m] < j_count[i]:
                m_count[gen_m] = j_count[i]
            elif m_count[gen_m] > j_count[i]:
                j_count[i]=m_count[gen_m]
            
            start_time=str(datetime.timedelta(seconds=j_count[i] - int(self.pt[i][key_count[i]]))) # convert seconds to hours, minutes and seconds
            end_time=str(datetime.timedelta(seconds=j_count[i]))
                
            j_record[(i,gen_m)] = [start_time,end_time]
            
            key_count[i] = key_count[i]+1
                

        df = []
        for m in m_keys:
            for j in j_keys:
                df.append(dict(Task='Machine %s'%(m), Start='2018-07-14 %s'%(str(j_record[(j,m)][0])), Finish='2018-07-14 %s'%(str(j_record[(j,m)][1])),Resource='Job %s'%(j+1)))
            
        fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
        plot(fig, filename='GA_job_shop_scheduling.html')