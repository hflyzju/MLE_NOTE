#coding:utf-8

from os import curdir
import random
import numpy as np
import math

class Cluster:
    def distance(self, x, y, m):
        # 两个项链的欧式距离
        dis = 0.0
        for i in range(m):
            dis += (x[i] - y[i]) ** 2
        return math.sqrt(dis)


    def find_max_dis_index_from_center(self, center_list, data, visited):
        N, m = data.shape
        # 计算其与当前S中所有质心的距离的最大值D
        max_index = -1
        max_dis = 0
        for i in range(N):
            if not visited[i]:
                cur_dis = max([self.distance(x, data[i], m) for x in center_list])
                if cur_dis > max_dis:
                    max_dis = cur_dis
                    max_index = i
        return max_index

        
    def select_k_center(self, data, k):
        N, m = data.shape
        first = random.randint(0,N)
        visited = [False] * N
        visited[first] = True
        center_list = [data[first]]
        # 计算其与当前S中所有质心的距离的最大值D
        for i in range(1, k):
            max_dis_index = self.find_max_dis_index_from_center(center_list, data, visited)
            center_list.append(max_dis_index)
        return center_list
        
    def get_center(self, x, center_list, m):
        dis_list = [self.distance(x, y, m) for y in center_list]
        min_index = -1
        min_dis = float('inf')
        for i in range(len(dis_list)):
            if dis_list[i] < min_dis:
                min_dis = dis_list[i]
                min_index = i
        return min_index

    
    def fit(self, data, k):
        N, m = data.shape
        center_list = self.select_k_center(data, k)   
        data_cluster = [0] * N
        while True:
            # 对每个样本点：
            # 计算其与S中各个质心的距离
            # 将样本点分配至最近质心所属的类
            new_data_cluster = [0] * N
            for i in range(N):
                new_center_index = self.get_center(data[i], center_list, m)
                new_data_cluster[i] = new_center_index
            # 对某个类：
            # 将质心设置为该类所有样本点的均值
            # 在S中更新对应的类的质心
            center_sum = dict() # 累积和
            center_n = dict() # 累积个数
            for i in range(N):
                index = new_data_cluster[i]
                if index not in center_sum:
                    center_sum[index] = np.zeros(m)
                if index not in center_n:
                    center_n[index] = 0
                center_sum[index] += data[i]
                center_n[index] += 1

            new_center_list = []
            for i in range(k):
                new_center_index.append(center_sum[i] / float(center_n[i]))

            dis = 0
            for center_cur, center_pre in zip(new_center_list, center_list):
                dis += self.distance(center_cur, center_pre)
            
            center_list = new_center_list
            if dis == 0:
                break
            
            data_cluster = new_data_cluster
        return data_cluster
            



cluster = Cluster()

data = np.random.rand(40,50)
print(cluster.fit(data, 3))
