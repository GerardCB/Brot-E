import streamlit as st
from model_optimization import LeaderAssignmentModel

def main():
    st.title('Brot-E App')

    # Input for leader names
    st.markdown("### Leaders")
    leader_names = st.text_area("Enter all candidate leader names (comma separated):").split(',')    
    leader_names = [name.strip().replace("'",'') for name in leader_names]
    st.write(f"Total number of candidate leaders: {len(leader_names)}")
    
    # Ensure st.session_state.completed_leaders exists and is a set
    if 'completed_leaders' not in st.session_state:
        st.session_state.completed_leaders = set()
    else:
        # Prune old leader names from st.session_state.completed_leaders
        st.session_state.completed_leaders = st.session_state.completed_leaders.intersection(leader_names)

    # Input for groups
    st.markdown("### Groups")
    groups_input = st.text_area("Enter all groups (comma separated):", value='Follets, Llops, Raiers, Pios, Clan').split(',')
    groups_input = [group.strip().replace("'",'') for group in groups_input]

    # Group bounds based on the entered groups
    group_bounds = {}
    for group in groups_input:
        bounds = st.slider(
            f"{group} Bounds:",
            value=(5, 9),
            min_value=1,
            max_value=15,
        )
        group_bounds[group] = bounds

    # Input for alpha: 0 = Maximize Leader Satisfaction, 1 = Maximize Group Satisfaction
    st.markdown("### Alpha")
    
    # Explain the alpha parameter
    st.markdown("""
    Alpha is a parameter that controls the tradeoff between maximizing leader satisfaction and maximizing group satisfaction.
    - Alpha = 0: Maximize leader satisfaction
    - Alpha = 1: Maximize group satisfaction
    - Alpha = 0.5: Maximize the average of leader and group satisfaction on equal terms
    """)
    alpha = st.slider("Alpha:", 0.0, 1.0, 0.33, format="%f")

    # Leader Data Inputs
    st.markdown("### Leader Data Inputs")
    named_experience = {}
    named_commitment = {}
    named_preferences = {}
    ratings_named = {}
    named_family_constraints = {}

    if 'completed_leaders' not in st.session_state:
        st.session_state.completed_leaders = set()

    for leader in leader_names:
        # Check if leader has already been completed
        status_hint = "✅" if leader in st.session_state.completed_leaders else "❌"

        with st.expander(f"{leader} {status_hint}", expanded=False):
            st.subheader(f"Data for {leader}")

            # Experience: integer value, starting at 1
            st.markdown("### Experience and Commitment")
            exp = st.number_input(f"Experience for {leader} (in years):", 1, 10, 2)
            named_experience[leader] = exp

            # Commitment: 0 = Low, 100 = High; Add % symbol to slider
            com = st.slider(f"Commitment for {leader}:", 0, 100, 75, format="%d%%")
            
            named_commitment[leader] = 1 if com >= 75 else 0

            st.markdown("---")

            # Family Constraints Section
            st.markdown("### Group Constraints")
            family_groups = st.multiselect(f"Groups {leader} can't be assigned to:", groups_input)
            named_family_constraints[leader] = family_groups

            st.markdown("---")

            # Preferences Section
            st.markdown("### Preferences")
            st.write(f"Rate your preference for each group (1 to 5):")
            prefs = {}
            for group in groups_input:
                preference = st.slider(f"Preference for {group}:", 1, 5, 3, key=f"pref_{leader}_{group}")
                prefs[group] = preference
            named_preferences[leader] = prefs

            st.markdown("---")

            # Willingness to Work with Other Leaders Section
            st.markdown("### Willingness to Work with Other Leaders")
            st.write(f"Rate your willingness to work with each other leader (0 to 2):")
            other_leaders = [l for l in leader_names if l != leader]
            ratings = {}
            for other_leader in other_leaders:
                rating = st.slider(f"Rating for {other_leader}:", 0, 2, 1, key=f"rating_{leader}_{other_leader}")
                ratings[other_leader] = rating
            ratings_named[leader] = ratings

            if st.button(f"Save", key=f"save_{leader}"):
                st.session_state.completed_leaders.add(leader)
                st.experimental_rerun()
        
    # Calculate the fraction of leaders who have completed their inputs
    completed_fraction = len(st.session_state.completed_leaders) / len(leader_names)

    # Display the progress bar
    st.progress(completed_fraction)

    # Display the number of leaders who have completed their inputs
    st.write(f"{len(st.session_state.completed_leaders)}/{len(leader_names)} leaders have completed their inputs.")

    # Add a section break
    st.markdown("---")

    st.markdown("""
<style>
button:last-child {
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

    if st.button('Run Assignment'):
        model = LeaderAssignmentModel(
            leader_names, groups_input, group_bounds, alpha, named_experience, named_commitment, named_preferences, ratings_named, named_family_constraints
        )
        model.solve()

        # Display sucess baloons
        st.balloons()

        # Display Solver status
        st.subheader("Solver Status")
        # Format the status string and objective value
        st.write(f"Status: {model.status}")
        st.write(f"Objective Value: {round(model.objective_value, 2)} / 100")
        
        # Display Results
        st.subheader("Assignment Results")
        st.pyplot(model.visualize_assignments())
        
        st.subheader("Leader Satisfaction Scores")
        st.pyplot(model.visualize_leader_satisfaction()) 

        st.subheader("Group Satisfaction Scores")
        st.pyplot(model.visualize_group_satisfaction())


if __name__ == '__main__':
    main()

