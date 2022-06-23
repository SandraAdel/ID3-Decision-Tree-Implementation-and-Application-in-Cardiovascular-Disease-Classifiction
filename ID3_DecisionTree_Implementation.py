
# NOTE: DOCUMENTATION PROVIDED IN COMMENTS


# Imports
import math
import time
import copy
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# User-defined class for a node in ID3 Decision Tree, suitable for its characteristics
class ID3DecisionTreeNode:

    def __init__(self, key, type):

        # Contains either a feature (Root or Internal Node) or a decision (Leaf)
        self.Key = key
        # Whether it is a 'Feature Node' or a 'Decision Node'
        self.Type = type
        # For Feature Nodes, contains its feature values, for dictionary keys and nodes which these feature values lead to, for dictionary values
        self.Children = dict()

    # Setter for key of node
    def SetKey(self, new_key):
        self.Key = new_key

    # Setter for type of node
    def SetType(self, new_type):
        self.Type = new_type

    # Setter for children of node
    def SetChildren(self, new_children):
        self.Children = new_children

    # Getter for key of node
    def GetKey(self):
        return self.Key

    # Getter for type of node
    def GetType(self):
        return self.Type

    # Getter for childen of node
    def GetChildren(self):
        return self.Children


# User-defined class for ID3 Decision Tree implementation
class ID3DecisionTree:

    def __init__(self):

        # Root of ID3 Decision Tree to be accessed from
        self.Root = None
        # Feature names for dictionary keys and list of feature values for dictionary values
        self.Features = dict()
        # List of classifications/decisions available in data
        self.TargetLabels = None

    # Helper Function for extracting feature names, feature values and target labels of dataset before fitting
    def DataExtraction(self, features, target_labels):

        # Extracting feature names
        feature_names_list = features.columns.values.tolist()

        # Extracting feature values
        for feature_name in feature_names_list:
            feature_values = features[feature_name].unique().tolist()
            feature_values.sort()
            self.Features[feature_name] = feature_values

        # Extracting target labels
        self.TargetLabels = target_labels.unique().tolist()
        self.TargetLabels.sort()

    # Helper Function for calculating Information Gain of features and selecting best feature to split data upon
    def SelectBestFeature(self, dataset, featuresToCalculateIGFor):

        # Calculating Entropy of whole dataset
        probabilities_of_target_labels = list()
        for target_label in self.TargetLabels:
            probabilities_of_target_labels.append( len(dataset[dataset['target'] == target_label]) / dataset.shape[0] )
        entropy_of_dataset = CalculateEntropy(probabilities_of_target_labels)

        # Initailizaing feature of maximum Information Gain for comparison
        feature_of_maximum_information_gain = { 'Feature Name': None, 'Information Gain': 0 }
        maximum_information_gain_feature_value_entropies = dict()

        # Looping through features for calculation of Information Gain
        for feature in featuresToCalculateIGFor:
            weighted_sum_of_feature_value_entropies = 0
            feature_value_entropies = dict()

            # Looping through feature values (if avaialble in dataset) to calculate weighted sum of entropies for
            for feature_value in self.Features[feature]:
                probabilities_of_target_labels_holding_this_feature_value = list()
                entropy_of_feature_value = None
                if len(dataset[dataset[feature] == feature_value]) != 0:

                    # Looping through target labels for each feature value to calculate its entropy
                    for target_label in self.TargetLabels:
                            probabilities_of_target_labels_holding_this_feature_value.append( len(dataset[ (dataset['target'] == target_label) & (dataset[feature] == feature_value) ]) / len(dataset[dataset[feature] == feature_value]) )

                    # Entropy Calculation and its addition to the weighted sum of the feature
                    entropy_of_feature_value = CalculateEntropy(probabilities_of_target_labels_holding_this_feature_value)
                    weighted_sum_of_feature_value_entropies += round( ( len(dataset[dataset[feature] == feature_value]) / dataset.shape[0] ) * entropy_of_feature_value, 3)
                    feature_value_entropies[feature_value] = entropy_of_feature_value

            # Information Gain Calculation and comparison
            information_gain = round(entropy_of_dataset - weighted_sum_of_feature_value_entropies, 3)
            if information_gain >= feature_of_maximum_information_gain['Information Gain']:
                feature_of_maximum_information_gain['Feature Name'], feature_of_maximum_information_gain['Information Gain'] = feature, information_gain
                maximum_information_gain_feature_value_entropies = feature_value_entropies
        
        # Returning feature of highest Information Gain to split upon, its entropies (so that the feature value of zero entropy would lead to a decision and that of non-zero entropy
        # would probably lead to another feature), and probabilities of target labels in sub-dataset in case all the features were put as nodes in the branch and a decision has not
        # yet been reached. Therefore, a decision will be made based on the target label of highest probability in sub-dataset
        return feature_of_maximum_information_gain['Feature Name'], maximum_information_gain_feature_value_entropies, probabilities_of_target_labels

    # Main Function for Decision Tree Training/Fitting
    def fit(self, features, target_labels):

        # Dataset characteristics are extracted
        self.DataExtraction(features, target_labels)

        # Features and target labels are combined 
        dataset = features.copy(deep=True)
        dataset["target"] = target_labels

        # Initialization of root node, which is surely a 'Feature Node'
        root_node = ID3DecisionTreeNode(None, 'Feature Node')
        self.Root = root_node

        # Calling the recursion function of tree building
        feature_names = features.columns.values.tolist()
        self.BuildTree(dataset, feature_names, root_node)

    # Helper Function for Tree Building: (Recursion Function)
    def BuildTree(self, dataset, features_left, node):

        # Best feature to split upon is returned, along with its entropies (Feature values of zero entropies will lead to decision nodes, while those of non-zero entropies
        # will likely lead to feature nodes) and probabilities of target labels in dataset in case the decision will be made out of them, if we ran out of features in this branch
        feature_name, feature_value_entropies, probabilities_of_target_labels = self.SelectBestFeature(dataset, features_left)
        
        # If we run out of features to calculate Information Gain for in this branch, then current node will be a 'Decision Node'
        # and decision will be made using probabilities of target labels in the sub-dataset
        # Otherwise, we set this 'Feature Node' with the feature of highest Information Gain returned from SelectBestFeature function
        if feature_name == None:
            highest_target_label_probability = max(probabilities_of_target_labels)
            index_of_highest_target_label_probability = probabilities_of_target_labels.index(highest_target_label_probability)
            node.SetKey(self.TargetLabels[index_of_highest_target_label_probability])
            node.SetType('Decision Node')
        else:
            node.SetKey(feature_name)
        
        # In normal case, a 'Decision Node' is instantiated for feature value of zero entropy, holding appropriate decision as Key
        # and a 'Feature Node' is instantiated for feature value of non-zero entropy, and its key feature will be filled-in later on
        for (feature_value, entropy) in feature_value_entropies.items():
            if entropy == 0:
                classification = dataset.loc[dataset[feature_name] == feature_value, 'target'].iloc[0]
                new_child_node = ID3DecisionTreeNode(classification, 'Decision Node')
            else:
                new_child_node = ID3DecisionTreeNode(None, 'Feature Node')
            new_children = node.GetChildren(); new_children[feature_value] = new_child_node
            node.SetChildren( new_children )

        # After setting the current node's children nodes according to its feature values, recursion stops at the occurence of decicion node (Base Case) (End of branch)
        # While for feature node, iteself, the new sub-dataset and new features to calculate Information Gain for, are passed into the recursion function,
        # to set it with the appropriate feature
        for (feature_value, child_node) in node.GetChildren().items():
            if child_node.GetType() == 'Decision Node':
                pass
            elif child_node.GetType() == 'Feature Node':
                new_sub_dataset = dataset.copy(deep=True)
                new_sub_dataset = new_sub_dataset[new_sub_dataset[feature_name] == feature_value]
                new_features_left = copy.deepcopy(features_left) 
                new_features_left.remove(feature_name)
                self.BuildTree(new_sub_dataset, new_features_left, child_node)

    # Main Prediction Function for Decision Tree
    def predict(self, features):

        # Extracting feature names for appropriate (Feature - Feature Value) comparison
        feature_names = list(self.Features.keys())
        predictions = list()

        # For each record, a recursion fucntion for traversing tree is called until a 'Decision Node' is reached
        for record in features:
            classification = self.TraverseTree(self.Root, record, feature_names)
            predictions.append(classification)
        return predictions

    # Helper Function for Tree Traversal until a 'Decision Node' is reached with the record's feature values: (Recursion Function)
    def TraverseTree(self, node, features, feature_names):
        
        # Normal Base Case is to reach a 'Decision Node' and its key is returned
        if node.GetType() == 'Decision Node':
            return node.GetKey()
        
        # If a 'Decsion Node' was not reached, we get next branch/subtree to traverse through (Recall Recusrion Function) until we reach one
        else:
            feature_to_check = node.GetKey()
            index_of_feature_to_check = feature_names.index(feature_to_check)
            next_node_to_check = None
            for (feature_value, child_node) in node.GetChildren().items():
                if features[index_of_feature_to_check] == feature_value:
                    next_node_to_check = child_node

            # If according to feature value in record, a branch for this feature value was not found in the built decision tree
            # (due to to customization to data provided in training: this feature value was not found in the remaining sub-dataset
            # to base a decison or to build a branch upon, so a branch was not built for it), a random target label (Uniform Probabilities)
            # will be assigned to this record (Inevitable percentage of error due to Tree's customization to training dataset)
            # Otherwise, we traverse to next 'Feature Node' to reach a 'Decision Node'
            if next_node_to_check == None:
                return random.choice(self.TargetLabels)
            else:
                return self.TraverseTree(next_node_to_check, features, feature_names)

# Global Function for Entropy Calculation
# Global since it does not deal with any of the class attributes
def CalculateEntropy(probabilities_list):
    entropy = 0
    for probability in probabilities_list:
        if probability != 0:
            entropy += ( -probability * math.log2(probability) )
    return round(entropy, 3)
