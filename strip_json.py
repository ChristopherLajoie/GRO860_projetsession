import json

# Define file names
input_filename = 'reward_logs/rewards_20251104_112516.json'
output_filename = 'reward_logs/rewards_filtered.json'

try:
    # Step 1: Load the original JSON data from the file
    print(f"Opening file: {input_filename}")
    with open(input_filename, 'r') as f:
        original_data = json.load(f)
    print("File loaded successfully.")

    # Step 2: Create a new data structure for the filtered results
    if not isinstance(original_data, dict):
        print("Error: Original data is not a dictionary.")
    else:
        filtered_data = original_data.copy()
        print("Copied top-level structure.")
        
        # List to hold the processed episodes
        processed_episodes = []

        # Step 3: Get the 'episodes' list
        original_episodes_list = original_data.get('episodes')
        if not isinstance(original_episodes_list, list):
            print("Error: 'episodes' key not found or is not a list.")
        else:
            # Get the first 20 episodes
            episodes_to_process = original_episodes_list[:20]
            print(f"Processing the first {len(episodes_to_process)} episodes...")

            # Step 4: Process each of these episodes
            for i, episode in enumerate(episodes_to_process):
                if not isinstance(episode, dict):
                    print(f"Warning: Episode {i} is not a dictionary, skipping.")
                    continue
                    
                # Create a copy of the episode to modify
                new_episode = episode.copy()
                
                original_steps = episode.get('steps', [])
                if not isinstance(original_steps, list):
                    print(f"Warning: 'steps' in episode {i} is not a list, skipping steps.")
                    new_episode['steps'] = []
                    processed_episodes.append(new_episode)
                    continue

                filtered_steps = []

                # Step 5: Filter the steps: keep only every 10th step (0, 10, 20, ...)
                for step in original_steps:
                    if isinstance(step, dict) and 'step' in step and isinstance(step['step'], int) and step['step'] % 10 == 0:
                        filtered_steps.append(step)
                
                # Step 6: Replace the original 'steps' with the filtered list
                new_episode['steps'] = filtered_steps
                
                # Add the modified episode to our new list
                processed_episodes.append(new_episode)
            
            # Step 7: Replace the 'episodes' list in our new data structure
            filtered_data['episodes'] = processed_episodes
            print("Step filtering complete.")

            # Step 8: Save the new filtered data to a new JSON file
            print(f"Saving filtered data to {output_filename}...")
            with open(output_filename, 'w') as f:
                json.dump(filtered_data, f, indent=2)

            print(f"\nSuccessfully processed the file.")
            print(f"Kept the first {len(filtered_data['episodes'])} episodes.")
            print(f"Filtered steps to keep only every 10th step.")
            print(f"The smaller file has been saved as: {output_filename}")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode the JSON from '{input_filename}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()