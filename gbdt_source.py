import copy
import random
import pandas as pd
import numpy as np

# 数中存储每一子节点的分裂方式，叶子节点中存储该节点的分数
class Tree_node():
	def __init__(self):
		self.split_feature = None
		self.split_value = None
		self.left_child = None
		self.right_child = None
		self.leaf_node = None

	def calc_predict_value(self,data_set):
		if self.leaf_node is not None:
			return(self.leaf_node)
		if data_set[self.split_feature] <= self.split_value:
			return(self.left_child.calc_predict_value(data_set))
		else:
			return(self.right_child.calc_predict_value(data_set))

	def describe_struct(self):
		if self.leaf_node is not None:
			return(self.leaf_node)
		left_info = self.left_child.describe_struct()
		right_info = self.right_child.describe_struct()
		tree_struct = {"split_feature":str(self.split_feature),
                       "split_value":str(self.split_value),
                       "left_child":str(self.left_child.describe_struct()),
                       "right_child":str(self.right_child.describe_struct()),
#                       "leaf_node":str(self.leaf_node)
                      }
		return(tree_struct)


#regression tree split by friedman_mse
class BaseCARTree():
    def __init__(self, max_depth=2, min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 colsample_bytree=1.0, random_state=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.f_m = None

    def fit(self,instances,targets):
        instances_cp = copy.deepcopy(instances)
        targets_cp = copy.deepcopy(targets)

        random.seed(self.random_state)
        if self.subsample<1.0:
            sub_index = random.sample(range(len(instances_cp)),int(self.subsample*len(instances_cp)))
            instances_cp = instances_cp.reindex(index = sub_index).reset_index(drop=True)
            targets_cp = targets[sub_index].reset_index(drop=True)
        if self.colsample_bytree<1.0:
            col_index = random.sample(range(len(instances_cp.columns)),int(self.colsample_bytree*len(instances_cp.columns)))
            instances_cp = instances_cp.reindex(columns=col_index)

        self.tree = self._fit(instances_cp,targets_cp,depth = 0)
        self.f_m = instances.apply(lambda x:self.tree.calc_predict_value(x),axis=1)

    def _fit(self,instances,targets,depth):
        tree = Tree_node()
        if len(targets.unique())==1 or len(instances)<=self.min_samples_split:
            tree.leaf_node = self.calc_leaf_value(targets)
            return(tree)
        if depth<self.max_depth:
            print(("depth is "+str(depth)).center(20,"-"))
            split_feature,split_value = self.choose_best_feature(instances,targets)
            left_ins,right_ins,left_target,right_target = \
            self.split_data(instances,targets,split_feature,split_value)
            if len(left_target)*len(right_target)==0:
                tree.leaf_node = self.calc_leaf_value(targets)
                return(tree)
            
            tree.split_feature = split_feature
            tree.split_value = split_value
            tree.left_child = self._fit(left_ins,left_target,depth+1)
            tree.right_child = self._fit(right_ins,right_target,depth+1)
            return(tree)
        else:
#             tree = Tree_node()
            tree.left_child = None
            tree.right_child = None
            tree.leaf_node = self.calc_leaf_value(targets)
            return(tree)

    def choose_best_feature(self,instances,targets):
        best_feature = ''
        best_split = ''
        best_mse = 0
#         cut_dict = {}
        for col in instances.columns:
            cut_points = list(set(instances[col].tolist()))
            cut_points.sort()
            cut_points = [(cut_points[i]+cut_points[i+1])/2 for i in range(len(cut_points)-1)]
#             mse_list = []
            for i in cut_points:
    #             print(col+str(i))
                left_dt,right_dt,left_target,right_target = self.split_data(instances,targets,col,i)
                fridmen_mse = self.calc_friedman_mse(left_target,right_target)
                if (fridmen_mse>=best_mse):
                    best_feature = col
                    best_split = i
                    best_mse = fridmen_mse
#                 mse_list.append(fridmen_mse)
#             cut_dict[col] = [cut_points[mse_list.index(min(mse_list))],mse_list[mse_list.index(min(mse_list))]]
#             print(col+' feature fridmen_mse is '+str(round(min(mse_list),8))+', cut point is '+str(cut_points[mse_list.index(min(mse_list))]))
#         feature_id = ([cut_dict[i][1] for i in cut_dict]).index(min([cut_dict[i][1] for i in cut_dict]))
#         feature = list(cut_dict.keys())[feature_id]
#         f_cut = cut_dict[feature][0]
        print(best_feature+' '+str(best_split)+' friedman_mse is '+str(best_mse))
        return best_feature, best_split

    def calc_friedman_mse(self,left_targets,right_targets):
        if (len(left_targets)<2) or (len(right_targets)<2):
            return(0)
        left_mean,right_mean = 1.0 * sum(left_targets) / len(left_targets),1.0 * sum(right_targets) / len(right_targets)
        diff = left_mean - right_mean
        friedman_mse = diff*diff*len(left_targets)*len(right_targets)/sum([len(left_targets),len(right_targets)])
        return friedman_mse

    def split_data(self,dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def calc_leaf_value(self,targets):
        # 计算叶子节点值,Algorithm 5,line 5
        sum1 = sum(targets)/len(targets)
        sum2 = sum([abs(y_i) * (2 - abs(y_i)) for y_i in targets])
#         sum2 = 1
        return 1.0 * sum1 / sum2
    
    def print_tree(self):
        return self.tree.describe_tree()

    def predict(self, dataset):
        res = []
        for j in dataset.iterrows():
            res.append(self.tree.calc_predict_value(j[1]))
        return np.array(res)



class GBDT():
    def __init__(self,n_estimators=2, max_depth=2, learning_rate=0.1, min_samples_split=2,
                 min_samples_leaf=1, subsample=1.0, colsample_bytree=1.0,random_state=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.f_0 = None
        self.trees = dict()
    
    def fit(self,instances,targets):
        targets = targets.to_frame(name='label')
        targets['label'] = targets['label'].apply(lambda y: 1 if y == 1 else -1)
        if targets['label'].unique().__len__() != 2:
            raise ValueError("There must be two class for targets!")
        if len([x for x in instances.columns if instances[x].dtype in ['int32', 'float32', 'int64', 'float64']]) \
                != len(instances.columns):
            raise ValueError("The features dtype must be int or float!")
        
        # 计算f_0,Algorithm 5,line 1
        mean = 1.0 * sum(targets['label']) / len(targets['label'])
        self.f_0 = 0.5 * np.log((1 + mean) / (1 - mean))
        targets['f_m'] = self.f_0
        
        for i_learner in range(self.n_estimators):
            print(str(i_learner).center(100,"="))
            targets['label_neg_gratitude'] = targets.apply(lambda x:2*x['label']/(1+np.exp(2*x['label']*x['f_m'])),axis=1)
            tree = BaseCARTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                               min_samples_leaf=self.min_samples_leaf, subsample=self.subsample,
                               colsample_bytree=self.colsample_bytree, random_state=self.random_state)
            tree.fit(instances,targets['label_neg_gratitude'])
            self.trees[i_learner] = tree
            targets['f_m'] = targets['f_m'] + self.learning_rate * tree.predict(instances)
    
    def predict_proba(self,instances):
        res = []
        for i in range(len(instances)):
            f_value = self.f_0
            for stage, tree in self.trees.items():
                f_value = f_value + self.learning_rate * tree.predict(instances.reindex(index=[i]))
            p_0 = 1.0 / (1 + np.exp(2*f_value))
            res.append(p_0)
        return np.array(res)

    
    def predict(self,instances):
        res = []
        for i in self.predict_proba(instances):
            res.append(int(i[0]<i[1]))
        return(np.array(res))