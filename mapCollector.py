import re
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pyminesweeper import MinesweeperMap

def genMap(g_size):
    MineMap = MinesweeperMap(size=g_size)
    map = MineMap.export_map()
    map_info = re.split(pattern="\s", string=map.splitlines()[0])
    map_lines = map.splitlines()[1:]
    map_matrix = []
    for line in map_lines:
        line = re.sub(pattern="[xX]", repl="-1", string=line)
        line = re.split(pattern="\s", string=line)
        line.remove('')
        line = [int(x) for x in line]
        map_matrix += [line]
    map_matrix = np.array(map_matrix)
    unique, counts = np.unique(map_matrix, return_counts=True)
    val_counts = {x:y for x,y in zip(unique,counts)}
    val_counts.pop(-1)
    return map_matrix, val_counts

if __name__ == "__main__":
    grid_sizes = [8,15]
    n_neighbours = 8
    n_grids = 30000

    for grid_size in grid_sizes:
        print("Working on grid size %d * %d" %(grid_size,grid_size))
        sample_space = {x:[] for x in range(n_neighbours+1)}
        bomb_pos = []; opt_pos = []
        for i in range(n_grids):
            random_map, random_stats = genMap(grid_size)
            bomb_pos += [list(zip(*np.where(random_map == -1)))]
            opt_pos += [list(zip(*np.where(random_map == 0)))]
            for k in sample_space:
                if k in random_stats:
                    sample_space[k] = sample_space[k] + [random_stats[k]]
                else:
                    sample_space[k] = sample_space[k] + [0]
                    
        sample_space = pd.DataFrame(sample_space)

        bomb_pos = np.array(bomb_pos)
        n_bombs = bomb_pos.shape[1]
        bomb_pos = bomb_pos.reshape((bomb_pos.shape[0]*bomb_pos.shape[1],2))
        unique_cords, counts = np.unique(bomb_pos, axis=0, return_counts=True)
        bomb_pos_matrix = np.zeros([grid_size,grid_size])
        for i,j in zip(unique_cords,counts):
            bomb_pos_matrix[i[0],i[1]] = j
        
        opt_pos = np.array(sum(opt_pos, []))
        unique_cords, counts = np.unique(opt_pos, axis=0, return_counts=True)
        opt_pos_matrix = np.zeros([grid_size,grid_size])
        for i,j in zip(unique_cords,counts):
            opt_pos_matrix[i[0],i[1]] = j
        
        plt.figure(figsize=(10,10))
        for i in range(n_neighbours+1):
            plt.subplot(3, 3, i+1)
            sb.histplot(data=sample_space, x=i, kde=True, bins=20)  # Use sb.histplot for newer versions
            plt.title(i)
        plt.tight_layout()
        plt.savefig('hist_%d_%d.pdf' %(grid_size,n_bombs))
        plt.close()

        plt.figure(figsize=(11,9))
        sb.heatmap(bomb_pos_matrix, cmap='BrBG', annot=False)
        plt.savefig('bomb_heatmap_%d_%d.pdf' %(grid_size,n_bombs))
        plt.close()

        plt.figure(figsize=(11,9))
        sb.heatmap(opt_pos_matrix, cmap='BrBG', annot=False)
        plt.savefig('opt_heatmap_%d_%d.pdf' %(grid_size,n_bombs))
        plt.close()

