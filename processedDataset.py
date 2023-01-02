from torch_geometric.data import Dataset, Data
import os, csv, torch
from tqdm import tqdm
import pandas as pd
import ast
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler


not_unique_names = []

class ProcessedDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(ProcessedDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, match in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            print("starts processing... this is the match")
            print(match)
            scaler = StandardScaler()
            try:
                # Get node features & normalize it
                node_feats = self._get_node_features(match)
                node_feats = scaler.fit_transform(pd.DataFrame(node_feats))
                # Get edge features(adjacent, front-back, opponent etc.)
                #edge_feats = self._get_edge_features([match['home_formation'], match['visiting_formation']])
                # Get adjacency info(whether or not there's an edge)
                edge_index = self._get_adjacency_info(match['team1_shape'], match['team2_shape'])
                # Get labels info
                label = self._get_labels(match['result'])

                # Create data object
                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            #edge_attr=edge_feats,
                            y=label
                            #match=match #########################
                            ) 
            except Exception as msg:
                print("ERROR HAPPENED: DATA", index, " ABANDONED, PLS CHECK ERROR MSG BELOW FOR MORE INFORMATION")
                print(msg)
            
            else:
                # Save datasets
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                     f'data_test_{index}.pt'))#######################
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                     f'data_{index}.pt')) #########################
    def _get_most_recent_player_stat(self, name, date):
        """
        Returns the most recent player stat for a given player
        """
        # Get player stats
        DATA_PATH = "pre-processed/player_stats.csv"
        data_ = pd.read_csv(DATA_PATH)

        # Get player stats for a given player
        player_stats = data_[data_['name'] == name]
        
        # Check if that player is in player stats
        if player_stats.empty == True:                     ###########################
            print("############## ERROR: ", name, " is not in player stats, pls go fetch the datas ##############")
            raise SystemExit("Stop right there!")
            
        # Check if ID is unique
        elif player_stats['ID'].nunique() != 1:              #########################
            print(name, " is not unique, pls check which is right")

        # Get the most recent date
        filtered_df = player_stats.loc[(data_['stat_date'] <= date)]
        most_recent_date = filtered_df['stat_date'].max()
        if most_recent_date is np.nan:
            # Get the least recent player stat
            most_recent_date = player_stats['stat_date'].min()
            
            print("there's no most recent date for ", name, " in time ", date, "getting ", most_recent_date, " instead")
        else:
            print("the most recent date for ", name, " in time ", date, "is: ", most_recent_date)

        # Get the most recent player stat
        most_recent_player_stat = player_stats[player_stats['stat_date'] == most_recent_date]

        return most_recent_player_stat

    def _get_node_features(self, match):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]

        # features:
        position: x, y
        general: height, weight, preferred_foot, age
        ball skills: ball_control, dribbling
        defense: marking, slide_tackle, stand_tackle
        mental: aggression, reactions, att_position, interceptions, vision, composure
        passing: crossing, short_pass, long_pass
        physical: acceleration, stamina, strength, balance, sprint_speed, agility, jumping
        shooting: heading, shot_power, finishing, long_shots, curve, fk_acc, penalties, volleys
        goalkeeper: gk_positioning, gk_diving, gk_handling, gk_kicking, gk_reflexes
        """
        # team2 assigned backwards bcs node index starts from the last player to the goalkeper
        total_player_names = list(match[['team1_player1_name', 'team1_player2_name', 'team1_player3_name', 'team1_player4_name', 'team1_player5_name', 'team1_player6_name', 'team1_player7_name', 'team1_player8_name', 'team1_player9_name', 'team1_player10_name', 'team1_player11_name', 
                                         'team2_player11_name', 'team2_player10_name', 'team2_player9_name', 'team2_player8_name', 'team2_player7_name', 'team2_player6_name', 'team2_player5_name', 'team2_player4_name', 'team2_player3_name', 'team2_player2_name', 'team2_player1_name']]
                                 )
        
        # match['team1_positions_x'] has the dtype str, ast.literal_eval converts it back to list
        total_player_x = ast.literal_eval(match['team1_positions_x']) + ast.literal_eval(match['team2_positions_x'])[::-1]
        total_player_y = ast.literal_eval(match['team1_positions_y']) + ast.literal_eval(match['team2_positions_y'])[::-1]
        
        #print("total player names: ", total_player_names)
        #print("total player x: ", total_player_x)
        #print("total player y: ", total_player_y)
        
        all_node_feats = []

        for index, player in enumerate(total_player_names):
            # Get player stats
            try:
                player_stats = self._get_most_recent_player_stat(player, match['timestamp']).iloc[0]
                
            except Exception as msg:
                print("error happend when fetching player #######", player)
                print(msg)

            node_feats = []
            # Feature 1~2: x, y
            node_feats.append(total_player_x[index])
            node_feats.append(total_player_y[index])
            # Feature 3~6: height, weight, preferred_foot, age -> General
            node_feats.append(player_stats['height'])
            node_feats.append(player_stats['weight'])                
            if player_stats['preferred_foot'] == 'Right':
                node_feats.append(1)
            else:
                node_feats.append(0)           
            node_feats.append(player_stats['age'])
            # Feature 7~8: ball_control, dribbling -> Ball skills
            node_feats.append(player_stats['ball_control'])
            node_feats.append(player_stats['dribbling'])
            # Feature 9~11: marking, slide_tackle, stand_tackle -> Defense
            node_feats.append(player_stats['marking'])
            node_feats.append(player_stats['slide_tackle'])
            node_feats.append(player_stats['stand_tackle'])
            # Feature 12~17: aggression, reactions, att_position, interceptions, vision, composure -> Mental
            node_feats.append(player_stats['aggression'])
            node_feats.append(player_stats['reactions'])
            node_feats.append(player_stats['att_position'])
            node_feats.append(player_stats['interceptions'])
            node_feats.append(player_stats['vision'])
            node_feats.append(player_stats['composure'])
            # Feature 18~20: crossing, short_pass, long_pass -> Passing
            node_feats.append(player_stats['crossing'])
            node_feats.append(player_stats['short_pass'])
            node_feats.append(player_stats['long_pass'])
            # Feature 21~27: acceleration, stamina, strength, balance, sprint_speed, agility, jumping -> Physical
            node_feats.append(player_stats['acceleration'])
            node_feats.append(player_stats['stamina'])
            node_feats.append(player_stats['strength'])
            node_feats.append(player_stats['balance'])
            node_feats.append(player_stats['sprint_speed'])
            node_feats.append(player_stats['agility'])
            node_feats.append(player_stats['jumping'])
            # Feature 28~35: heading, shot_power, finishing, long_shots, curve, fk_acc, penalties, volleys -> Shooting
            node_feats.append(player_stats['heading'])
            node_feats.append(player_stats['shot_power'])
            node_feats.append(player_stats['finishing'])
            node_feats.append(player_stats['long_shots'])
            node_feats.append(player_stats['curve'])
            node_feats.append(player_stats['fk_acc'])
            node_feats.append(player_stats['penalties'])
            node_feats.append(player_stats['volleys'])
            # Feature 36~40: gk_positioning, gk_diving, gk_handling, gk_kicking, gk_reflexes -> Goalkeeper
            node_feats.append(player_stats['gk_positioning'])
            node_feats.append(player_stats['gk_diving'])
            node_feats.append(player_stats['gk_handling'])
            node_feats.append(player_stats['gk_kicking'])
            node_feats.append(player_stats['gk_reflexes'])

            # Append node features to matrix
            all_node_feats.append(node_feats)
            #print("node feats for player", index, ": ", node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

#     def _get_edge_features(self, home_formation, visiting_formation):
#         """ 
#         This will return a matrix / 2d array of the shape
#         [Number of edges, Edge Feature size]

#         The edge features should be based on the relative position of each player
#         0: side-by-side
#         1: front-to-back
#         2: goalkeeper-to-defender
#         -1: between opponents

#         """

#         # Split the formations
#         home_formation_split = home_formation.split('-')
#         home_formation_len = len(home_formation_split)
#         visiting_formation_split = visiting_formation.split('-')
#         visiting_formation_len = len(visiting_formation_split)
#         total_formation = "1-" + home_formation + "-" + visiting_formation[::-1] + "-1"
#         total_formation_split = total_formation.split('-')
#         total_formation_len = len(total_formation_split)

#         all_edge_feats = []
#         # add goalkeeper-to-defender
#         for _ in range(int(total_formation_split[1])):
#             edge_feats = []
#             edge_feats.append(2)
#             all_edge_feats += [edge_feats, edge_feats]

#         # home
#         for i, s in enumerate(home_formation_split):
#             # add side-by-sde
#             for times in range(int(s)-1):
#                 edge_feats = []
#                 edge_feats.append(0)
#                 all_edge_feats += [edge_feats, edge_feats]
#             # add front-to-back
#             if i < home_formation_len-1:
#                 for times in range(int(home_formation_split[i]) * int(home_formation_split[i+1])):
#                     edge_feats = []
#                     edge_feats.append(1)
#                     all_edge_feats += [edge_feats, edge_feats]

#         # add between opponents
#         for times in range(int(home_formation_split[-1]) * int(visiting_formation_split[-1])):
#             edge_feats = []
#             edge_feats.append(-1)
#             all_edge_feats += [edge_feats, edge_feats]

#         # visiting
#         reversed = visiting_formation_split[::-1]
#         for i, s in enumerate(reversed):
#             # add side-by-sde
#             for times in range(int(s)-1):
#                 edge_feats = []
#                 edge_feats.append(0)
#                 all_edge_feats += [edge_feats, edge_feats]
#             # add front-to-back
#             if i < visiting_formation_len-1:
#                 for times in range(int(reversed[i]) * int(reversed[i+1])):
#                     edge_feats = []
#                     edge_feats.append(1)
#                     all_edge_feats += [edge_feats, edge_feats]
#         # add goalkeeper-to-defender
#         for _ in range(int(total_formation_split[-2])):
#             edge_feats = []
#             edge_feats.append(2)
#             all_edge_feats += [edge_feats, edge_feats]

#         all_edge_feats = np.asarray(all_edge_feats)
#         return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, home_formation, visiting_formation):
        """
        The return should be in COO format.
        Make sure that the order of the indices
        matches the order of the edge features!
        """
        
        # print("original home formation: ", home_formation)
        # print("original visiting formation: ", visiting_formation)
        
        # Remove the _L or _R in formations
        home_formation = home_formation.replace('_L', '')
        visiting_formation = visiting_formation.replace('_R', '')
        
        # Split the formations
        home_formation_split = home_formation.split('-')
        visiting_formation_split = visiting_formation.split('-')
        
        # Drop alphabets in formation & split
        if home_formation_split[-1].isalpha():
            home_formation = home_formation[:-2]
            home_formation_split = home_formation_split[:-1]
        if visiting_formation_split[-1].isalpha():
            visiting_formation = visiting_formation[:-2]
            visiting_formation_split = visiting_formation_split[:-1]
        # print("home formation split after removing & dropping: ", home_formation_split)
        # print("visiting formation split after removing & dropping: ", visiting_formation_split)
        
        home_formation_len = len(home_formation_split)
        visiting_formation_len = len(visiting_formation_split)
        
        sum_formation = home_formation + "-" + visiting_formation[::-1]
        sum_formation_split = sum_formation.split('-')

        # Get the level lists( = the index of nodes for each graph)
        # i.e. for 4-2-3-1 vs 4-3-3 -> [[0], [1, 2, 3, 4], [5, 6], [7, 8, 9], [10], [11, 12, 13], [14, 15, 16], [17, 18, 19, 20], [21]]
        level_lists = [[0]]
        start = 1
        for index, s in enumerate(sum_formation_split):
            adding_list = list(range(start, start + int(s)))
            level_lists.append(adding_list)
            start += int(s)
        level_lists.append([start])
        
        edge_indices = []
        for i, _ in enumerate(level_lists[:-1]):
            # side-by-side
            for node in level_lists[i][:-1]:
                #print(node, ",", node+1)
                edge_indices += [[node, node+1], [node+1, node]]
            # front-to-back
            for r in itertools.product(level_lists[i], level_lists[i+1]): 
                #print (r[0], ",",  r[1])
                edge_indices += [[r[0], r[1]], [r[1], r[0]]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        #print("this is edge indices: ", edge_indices)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data

