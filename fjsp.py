DATASET_DIR = r"C:\Users\dtaku\Documents\GA-FJSP"

import pandas as pd
import os
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import math
import sys
import matplotlib.pyplot as plt

# le o arquivo excel contendo o dataset (mudar sheet name para mudar input)
SHEET_NAME = 2 # de 0 a 7
LOT_NUMBER = 1 # 1, 2 ou 3
raw_df = pd.read_excel(os.path.join(DATASET_DIR,'Dataset.xlsx'), sheet_name=SHEET_NAME)

# dataframe contendo todos os jobs
jobs_df = raw_df[['lot', 'operation', 'machine', 'proc-time']]
jobs_df = jobs_df.dropna()
jobs_df[['lot', 'operation', 'machine']] = jobs_df[['lot', 'operation', 'machine']].astype(int)
jobs_df.head()

# alguns parametros dos jobs a serem realizados
LOTS = jobs_df['lot'].nunique()
OPERATIONS = jobs_df['operation'].nunique()
MACHINES = jobs_df['machine'].nunique()
JOBS = len(jobs_df.index)

print('Jobs: ' + str(JOBS))
print('Lots: ' + str(LOTS))
print('Operations: ' + str(OPERATIONS))
print('Machines: ' + str(MACHINES))

# dataframe com tamanho de lotes
lots_size_df = raw_df[['lot.1', 'lotSize_1', 'lotSize_2', 'lotSize_3']]
lots_size_df = lots_size_df.dropna()
lots_size_df = lots_size_df.rename(columns={'lot.1':'lot'})
lots_size_df[['lot', 'lotSize_1', 'lotSize_2', 'lotSize_3']] = lots_size_df[['lot', 'lotSize_1', 'lotSize_2', 'lotSize_3']].astype(int)
lots_size_df.head()

# responsavel por consertar individuo
# 1. retira operacoes repetidas, dando prioridade a ordem de aparicao no cromossomo
# 2. garante que operacoes obedecam os pre requisitos, dando prioridade a ordem de aparicao no cromossomo
def fix_individual(individual):

  # cria um dataframe para representacao do individuo original
  individual_df = pd.DataFrame(columns=['lot','operation','machine'])

  # preenche o dataframe do individuo original
  for i in individual:
    lot = jobs_df.loc[i, 'lot']
    operation = jobs_df.loc[i, 'operation']
    machine = jobs_df.loc[i, 'machine']

    # checa se operacao ja esta no dataframe do individuo
    is_already = not individual_df[(individual_df['lot'] == lot) & (individual_df['operation'] == operation)].empty
    if(is_already):
      continue

    individual_df.loc[i, ['lot', 'operation', 'machine']] = lot, operation, machine
  individual_df = (individual_df.reset_index()).drop('index', axis=1)
  
  # cria um dataframe para representacao do individuo consertado
  fixed_df = pd.DataFrame(columns=['lot','operation','machine'])

  # conserta o individuo
  for i in individual_df.index:

    if(not (i in individual_df.index)):
      continue

    lot = individual_df.loc[i, 'lot']
    operation = individual_df.loc[i, 'operation']
    machine = individual_df.loc[i, 'machine']

    # checa se a operacao deste lote ja esta no dataframe
    is_already = not fixed_df[(fixed_df['lot'] == lot) & (fixed_df['operation'] == operation) & (fixed_df['machine'] == machine)].empty
    if(is_already):
      continue

    # verificar se esta operacao pode ser realizada
    prev_lot_op = fixed_df.loc[fixed_df['lot'] == lot]['operation'].max()

    if(math.isnan(prev_lot_op)):
      prev_lot_op = 0

    # operacao nao pode ser realizada, pre-requisitos tem que ser realizados antes
    if(operation - prev_lot_op != 1):

      # procura os pre-requisitos
      lot_req_op = (individual_df.loc[(individual_df['lot'] == lot) & 
                                      (individual_df['operation'] < operation)]).sort_values(by='operation', ascending=True)
          
      indexes = lot_req_op.index.values
    
      # retira os pre-requisitos do dataframe do individuo original
      individual_df = individual_df.drop(indexes)

      # adiciona os pre-requisitos no dataframe consertado
      fixed_df = (fixed_df.append(lot_req_op, ignore_index=True))

    # adiciona a operacao no dataframe do individuo consertado
    fixed_lst = [[lot, operation, machine]]
    df = pd.DataFrame(fixed_lst, columns = ['lot', 'operation', 'machine'])
    fixed_df = fixed_df.append(df, ignore_index=True)

    # retira a operacao do dataframe do individuo
    individual_df = individual_df.drop(i)
    
  # preenche o individuo consertado de acordo com o dataframe
  fixed_individual = []
  for i in fixed_df.index:    
    lot = fixed_df.loc[i, 'lot']
    operation = fixed_df.loc[i, 'operation']
    machine = fixed_df.loc[i, 'machine']
    
    index = jobs_df.loc[(jobs_df['lot'] == lot) & 
                        (jobs_df['operation'] == operation) & 
                        (jobs_df['machine'] == machine)].index.values[0]

    fixed_individual.append(index)
  
  return fixed_individual

#individual
########################################
# N indexes that are mapped to jobs_df #
########################################

def decode(individual):

  # acerta a ordem dos indices de acordo com ordem das operacoes
  individual_fixed = fix_individual(individual)

  # cria dataframe para indicar alocacao das operacoes nas maquinas
  schedule_df = pd.DataFrame(columns=['lot','operation','machine','start','finish'])

  # preenche todo dataframe com as informacoes do individuo
  for i in individual_fixed:

    lot = jobs_df.loc[i, 'lot']
    operation = jobs_df.loc[i, 'operation']
    machine = jobs_df.loc[i, 'machine']

    proc_time = jobs_df.loc[(jobs_df['lot'] == lot) & 
                            (jobs_df['operation'] == operation) & 
                            (jobs_df['machine'] == machine)].reset_index().loc[0, 'proc-time']

    lot_size = lots_size_df.loc[(lots_size_df['lot'] == lot)].reset_index().loc[0, 'lotSize_{}'.format(LOT_NUMBER)]

    # verifica qual o final da ultima operacao da maquina
    last_finish_machine = schedule_df.loc[schedule_df['machine'] == machine]['finish'].max()
    if(math.isnan(last_finish_machine)):
      last_finish_machine = 0

    # verifica quando foi o final da ultima operacao do lote
    last_finish_prev_op = schedule_df.loc[schedule_df['lot'] == lot]['finish'].max()
    if(math.isnan(last_finish_prev_op)):
      last_finish_prev_op = 0

    # a operacao vai iniciar apos a maquina ficar livre e a operacao anterior do lote terminar
    start = max(last_finish_machine, last_finish_prev_op)

    # adiciona operacao no dataframe
    #schedule_df.loc[j, 'lot'] = lot
    #schedule_df.loc[j, 'operation'] = operation
    #schedule_df.loc[j, 'machine'] = machine
    #schedule_df.loc[j, 'start'] = start
    #schedule_df.loc[j, 'finish'] = start + proc_time*lot_size

    schedule_lst = [[lot, operation, machine, start, start+proc_time*lot_size]]
    df = pd.DataFrame(schedule_lst, columns = ['lot', 'operation', 'machine', 'start', 'finish'])
    schedule_df = schedule_df.append(df, ignore_index=True)
    
  return schedule_df

def gantt(individual):
  schedule_df = decode(individual)
  makespan = objective_function(individual)[0]

  # Declaring a figure "gnt" 
  fig, gnt = plt.subplots(figsize=(20,5)) 
    
  # Setting Y-axis limits
  step = 20 
  y_lim = MACHINES*step
  gnt.set_ylim(0, y_lim)

  # Setting X-axis limits 
  gnt.set_xlim(0, makespan) 
    
  # Setting labels for x-axis and y-axis 
  gnt.set_xlabel('Time') 
  gnt.set_ylabel('Machine') 
    
  y_ticks = [y for y in range(int(step/2), y_lim, step)] 
  gnt.set_yticks(y_ticks) 

  # Labelling tickes of y-axis
  y_labels =  ['M'+str(y+1) for y in range(MACHINES)]
  gnt.set_yticklabels(y_labels) 
    
  # Setting graph attribute 
  # gnt.grid(True) 

  color_map = {1: 'tab:blue', 
              2: 'tab:orange', 
              3: 'tab:green', 
              4: 'tab:red', 
              5: 'tab:purple',
              6: 'tab:brown', 
              7: 'tab:pink', 
              8: 'tab:gray', 
              9: 'tab:olive', 
              10: 'tab:cyan',
              11: 'b', 
              12: 'g'}

  for op in schedule_df.values:
    lot = op[0]
    operation = op[1]
    machine = op[2]
    start = op[3]
    finish = op[4]

    color = color_map[lot]
    y = y_ticks[machine-1]-int(step/2)

    gnt.broken_barh([(start, finish-start)], (y, step), facecolors =(color), edgecolors=('white'))
    label='O'+str(lot)+','+str(operation)
    gnt.text(x=start + (finish-start)/2, 
                      y=y+int(step/2),
                      s=label, 
                      ha='center', 
                      va='center',
                      color='white',
                    )

# Função Objetivo igual a ultima finalizacao na programacao
def objective_function(individual):
  schedule_df = decode(individual)
  makespan = schedule_df['finish'].max()
  return (makespan),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))        # função objetivo: nome, tipo(f.o.), peso de cada objetivo (no caso só um objetivo)
creator.create("Individual", list,  fitness=creator.FitnessMin)   # indivíduo

toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(JOBS), JOBS)

# Inicializador de indivíduo e população
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                  # lista de indivíduos

# Inicializador de operadores
toolbox.register("evaluate", objective_function)                              # função objetivo
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.05)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=50)                            # inicialização da pop
hof = tools.HallOfFame(1)                                 # melhor indivíduo
stats = tools.Statistics(lambda ind: ind.fitness.values)  # estatísticas
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

gen, min, avg = log.select('gen', 'min', 'avg')
plt.plot(gen, min)
plt.plot(gen, avg)
plt.xlabel('generation')
plt.legend(['minimum makespan', 'average makespan'])

# Melhor solução
print("Melhor Indivíduo:")
print(hof[0])
print(decode(hof[0]))
print(decode(hof[0])['finish'].max())

# Melhor resultado da função objetivo
print("Melhor Resultado da Função Objetivo:")
print(objective_function(hof[0])[0])
gantt(hof[0])
