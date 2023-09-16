from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpContinuous, LpStatus, PULP_CBC_CMD
import matplotlib.pyplot as plt
import numpy as np

class LeaderAssignmentModel:
    """ Represents the leader assignment optimization model.

    Attributes:
    - leader_names (list): List of leader names.
    - groups (list): List of group names.
    - group_bounds (dict): Dictionary of group names to their lower and upper bounds.
    - alpha (float): The balance parameter.
    - named_experience (dict): Dictionary of leader names to their experience levels.
    - named_commitment (dict): Dictionary of leader names to their commitment levels.
    - named_preferences (dict): Dictionary of leader names to their preferences towards each group.
    - ratings_named (dict): Dictionary of leader names to their ratings of other leaders.
    - named_family_constraints (dict): Dictionary of leader names to the groups they cannot be assigned to due to family constraints.
    - model (LpProblem): The optimization model.
    - assignments (dict): Dictionary of (leader, group) to assignment value.
    - leader_scores_result (dict): Dictionary of leader names to their satisfaction scores.
    - group_scores_result (dict): Dictionary of group names to their average satisfaction scores.    
    """
    def __init__(self, leader_names, groups, group_bounds, alpha, named_experience, named_commitment, named_preferences, ratings_named, named_family_constraints):
        self.leader_names = leader_names
        self.N = len(leader_names)
        self.groups = groups
        self.group_bounds = group_bounds
        self.alpha = alpha
        self.named_experience = named_experience
        self.experience = [named_experience[name] for name in leader_names]
        self.named_commitment = named_commitment
        self.commitment = [named_commitment[name] for name in leader_names]
        self.named_preferences = named_preferences
        self.preferences = self.transform_preferences_to_model_format(leader_names, named_preferences)
        self.ratings_named = ratings_named
        self.ratings = self.named_ratings_to_indexed_ratings(ratings_named, leader_names)
        self.named_family_constraints = named_family_constraints
        self.family_constraints = self.named_family_constraints_to_dict(named_family_constraints, leader_names)

        self.model = LpProblem(name="leader-assignment", sense=LpMaximize)
        self.x = None
        self.z = None
        self.y = None
        self.c = None
        self.w = None

        self.assignments = None
        self.leader_scores_result = None
        self.group_scores_result = None
        self.status = None
        self.objective_value = None

    def solve(self):
        N = len(self.leader_names)
        # Transform named data into indexed data
        preferences = self.preferences
        ratings = self.ratings
        family_constraints = self.family_constraints
        experience = self.experience
        commitment = self.commitment

        # Initialize variables
        x = LpVariable.dicts("x", ((i, g) for i in range(N) for g in self.groups), 0, 1, LpBinary)  # Assignment variables
        z = LpVariable.dicts("z", ((i, j, g) for i in range(N) for j in range(N) for g in self.groups if i != j), 0, None, LpContinuous)  # Compatibility variables
        y = LpVariable.dicts("y", ((i, j, g) for i in range(N) for j in range(N) for g in self.groups if i != j), 0, 1, LpBinary)  # Auxiliary variables
        c = LpVariable.dicts("c", (i for i in range(N)), 0, 1, LpBinary)  # Commitment
        w = LpVariable.dicts("w", ((i, g) for i in range(N) for g in self.groups), 0, 1, LpBinary)  # Auxiliary variables for commitment

        # Big-M constant
        M = 1000

        # Normalize Group Preference Score and Co-leaders Compatibility Score
        normalized_group_preference = lpSum(preferences[i, g] * x[i, g] for i in range(N) for g in self.groups) / (5 * N) * 100
        normalized_compatibility = lpSum(z[i, j, g] for i in range(N) for j in range(N) for g in self.groups if i != j) / (2 * N * (N-1) * 5) * 100

        # Define Objective Function
        self.model += self.alpha * normalized_group_preference + (1 - self.alpha) * normalized_compatibility, "Objective"

        # Add family in group constraints
        for i, group_list in family_constraints.items():
            for g in group_list:
                self.model += x[i, g] == 0

        # One group per leader
        for i in range(N):
            self.model += lpSum(x[i, g] for g in self.groups) == 1, f"One_Group_{i}"

        # Group size within limits
        for g in self.groups:
            Lg, Ug = self.group_bounds[g]
            self.model += lpSum(x[i, g] for i in range(N)) >= Lg, f"Group_Size_Lower_{g}"
            self.model += lpSum(x[i, g] for i in range(N)) <= Ug, f"Group_Size_Upper_{g}"

        # Experience level restrictions
        for i in range(N):
            if experience[i] == 1:
                self.model += lpSum(x[i, g] for g in ['Pios', 'Clan']) == 0, f"First_Year_{i}"
            elif experience[i] == 2:
                self.model += lpSum(x[i, g] for g in ['Clan']) == 0, f"Second_Year_{i}"

        # Auxiliary variable (y) definition
        for i in range(N):
            for j in range(N):
                if i != j:
                    for g in self.groups:
                        self.model += y[i, j, g] >= x[i, g] + x[j, g] - 1, f"Auxiliary_Lower_{i}_{j}_{g}"
                        self.model += y[i, j, g] <= x[i, g], f"Auxiliary_Upper1_{i}_{j}_{g}"
                        self.model += y[i, j, g] <= x[j, g], f"Auxiliary_Upper2_{i}_{j}_{g}"

        # Compatibility variable (z) definition
        for i in range(N):
            for j in range(N):
                if i != j:
                    for g in self.groups:
                        self.model += z[i, j, g] >= ratings[i, j] - M * (1 - y[i, j, g]), f"Compatibility_Lower1_{i}_{j}_{g}"
                        self.model += z[i, j, g] <= ratings[i, j], f"Compatibility_Upper1_{i}_{j}_{g}"

        # Set commitment levels
        for i in range(N):
            self.model += c[i] == commitment[i], f"Set_Commitment_{i}"

        # Define auxiliary variables for commitment (w)
        for i in range(N):
            for g in self.groups:
                self.model += w[i, g] >= c[i] + x[i, g] - 1, f"Define_w_Lower_{i}_{g}"
                self.model += w[i, g] <= c[i], f"Define_w_Upper1_{i}_{g}"
                self.model += w[i, g] <= x[i, g], f"Define_w_Upper2_{i}_{g}"

        # At least two leaders in each group should have 100% commitment
        for g in self.groups:
            self.model += lpSum(w[i, g] for i in range(N)) >= 2, f"Commitment_{g}"

        # Solve the model
        self.model.solve(PULP_CBC_CMD(msg=1))

        # Save the status
        self.status = LpStatus[self.model.status]
        self.objective_value = self.model.objective.value()

        # Save variables
        self.x = x
        self.z = z
        self.y = y
        self.c = c
        self.w = w

        # Save the assignments
        self.assignments = {(i, g): x[i, g].varValue for i in range(N) for g in self.groups}

        # Save the leader and group scores
        self.leader_scores_result = self.calculate_leader_scores()
        self.group_scores_result = self.calculate_group_scores()

    def calculate_leader_scores(self):
        """
        Calculate the satisfaction scores of leaders based on group and compatibility.

        Parameters:
        None

        Returns:
        - leader_scores (dict): Dictionary of leader names to their satisfaction scores.
        """
        leader_scores = {}
        for i in range(self.N):
            # Determine the group leader i is assigned to
            assigned_group = next(g for g in self.groups if self.x[i, g].varValue == 1)

            # Determine the size of the group the leader i is assigned to
            assigned_group_size = sum(self.x[j, assigned_group].varValue for j in range(self.N))

            # Preference score for the assigned group
            group_score = sum(self.preferences[i, g] * self.x[i, g].varValue for g in self.groups)

            # Compatibility score with co-leaders in the assigned group
            compatibility_score = sum(self.z[i, j, g].varValue for j in range(self.N) for g in self.groups if i != j and self.x[i, g].varValue > 0 and self.x[j, g].varValue > 0)

            min_possible_score = self.alpha + (1-self.alpha) * 0
            max_possible_score = self.alpha * 5 + (1-self.alpha) * 2 * (assigned_group_size - 1)

            # The final score is a weighted sum of group_score and compatibility_score
            raw_score = self.alpha * group_score + (1 - self.alpha) * compatibility_score

            # Normalize the score to the 0-10 range
            normalized_score = ((raw_score - min_possible_score) / (max_possible_score - min_possible_score)) * 10

            leader_scores[self.leader_names[i]] = normalized_score

        return leader_scores

    def calculate_group_scores(self):
        """
        Compute the average satisfaction scores for each group based on the scores of the assigned leaders.

        Returns:
        - group_scores (dict): Dictionary of group names to their average satisfaction scores.
        """
        group_scores = {}
        for g in self.groups:
            assigned_leaders = [leader for leader in self.leader_names if self.x[self.leader_names.index(leader), g].varValue == 1]
            total_score = sum(self.leader_scores_result[leader] for leader in assigned_leaders)

            # Calculate average score for the group
            group_scores[g] = total_score / len(assigned_leaders) if assigned_leaders else 0

        return group_scores

    def visualize_assignments(self):
        """
        Visualize the leader assignments to groups as a heatmap.

        Parameters:
        - assignments (dict): The assignment dictionary mapping (leader, group) to assignment value.

        Returns:
        - Figure: matplotlib figure of the heatmap.
        """
        assignments = {(i, g): self.x[i, g].varValue for i in range(self.N) for g in self.groups}
        # Extract unique leaders
        leaders = self.leader_names

        # Create a 2D array to hold the assignment values
        assignment_matrix = np.zeros((len(leaders), len(self.groups)))

        # Populate the 2D array with assignment values
        for (i, g), value in assignments.items():
            j = self.groups.index(g)
            assignment_matrix[i, j] = value

        # Create a heatmap to visualize the assignments
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(assignment_matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'Leader Assignments to Groups (alpha {self.alpha})')
        ax.set_xlabel('Groups')
        ax.set_ylabel('Leaders')
        ax.set_xticks(np.arange(len(self.groups)))
        ax.set_xticklabels(self.groups, rotation=45)
        ax.set_yticks(np.arange(len(leaders)))
        ax.set_yticklabels(leaders)

        # Add horizontal grid lines
        ax.grid(which='both', axis='y', linestyle='--', linewidth=0.7, color='white')

        # Add colorbar if non-binary assignments
        if any(0 < a < 1 for a in assignments.values()):
            plt.colorbar(label='Assignment Value', ax=ax)

        # Return the figure
        return plt.gcf()

    def visualize_leader_satisfaction(self):
        """
        Plot the satisfaction scores of leaders in a horizontal bar chart.

        Parameters:
        - leader_scores (dict): Leader scores to plot.

        Returns:
        - Figure: matplotlib figure of the plotted scores.
        """
        # Sort leaders by their scores
        sorted_leader_scores = sorted(self.leader_scores_result.items(), key=lambda x: x[1], reverse=False)
        leaders_sorted = [item[0] for item in sorted_leader_scores]
        scores_sorted = [item[1] for item in sorted_leader_scores]

        # Make coloring from 0 to 10
        colors = plt.cm.viridis(np.array(scores_sorted) / 10)

        #norm = plt.Normalize(min(scores_sorted), max(scores_sorted))
        #colors = plt.cm.viridis(norm(scores_sorted))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(leaders_sorted, scores_sorted, color=colors)
        ax.set_xlabel('Satisfaction Score')
        ax.set_ylabel('Leader')
        ax.set_title('Leader Satisfaction Scores')
        ax.grid(axis='x')
        ax.set_xlim(0, 10)

        # Return the figure
        return plt.gcf()

    def visualize_group_satisfaction(self):
        """
        Plot the average satisfaction scores of groups in a horizontal bar chart.

        Parameters:
        - group_scores (dict): Group scores to plot.

        Returns:
        - Figure: matplotlib figure of the plotted scores.
        """
        sorted_groups = sorted(self.group_scores_result.keys(), key=lambda g: self.group_scores_result[g], reverse=True)
        scores = [self.group_scores_result[g] for g in sorted_groups]

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(sorted_groups, scores, color=plt.cm.viridis(np.array(scores) / 10))
        ax.set_xlabel('Average Satisfaction Score')
        ax.set_title('Average Group Satisfaction Scores')
        ax.grid(axis='x')
        ax.set_xlim(0, 10)

        # Return the figure
        return plt.gcf()

    def transform_preferences_to_model_format(self, leader_names, named_preferences):
        """
        Transform the named preferences dictionary to a format compatible with the optimization model.

        Parameters:
        - leader_names: List of leader names in the order they appear in the model.
        - named_preferences: Dictionary containing the preferences of each leader towards each group.

        Returns:
        - A dictionary compatible with the optimization model.
        """
        model_preferences = {}
        
        for idx, name in enumerate(leader_names):
            leader_pref = named_preferences.get(name, {})
            for g_idx, group in enumerate(self.groups):
                model_preferences[(idx, group)] = leader_pref.get(group, 0)

        return model_preferences


    def named_ratings_to_indexed_ratings(self, named_ratings, leader_names):
        """
        Convert named ratings to indexed ratings.

        Parameters:
        - named_ratings: Ratings dictionary with leader names.
        - leader_names: List of leader names in the order they appear in the model.

        Returns:
        - A dictionary with indexed ratings.
        """
        indexed_ratings = {}
        for i, leader_i in enumerate(leader_names):
            for j, leader_j in enumerate(leader_names):
                if i != j:
                    indexed_ratings[(i, j)] = named_ratings.get(leader_i, {}).get(leader_j, 0)
        return indexed_ratings

    def named_family_constraints_to_dict(self, named_family_constraints, leader_names):
        """
        Convert named family constraints to indexed format.

        Parameters:
        - named_family_constraints: Dictionary with family constraints using leader names.
        - leader_names: List of leader names in the order they appear in the model.

        Returns:
        - A dictionary with indexed family constraints.
        """
        family_constraints = {}
        for name, groups in named_family_constraints.items():
            if name in leader_names:
                idx = leader_names.index(name)
                family_constraints[idx] = groups
        return family_constraints


