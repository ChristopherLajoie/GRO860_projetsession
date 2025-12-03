
import os
import csv
import argparse
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def export_to_csv(logdir, output_file, keywords, fill=False, interpolate=False):
    # Find all tfevents files
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if "tfevents" in file:
                event_files.append(os.path.join(root, file))

    if not event_files:
        print(f"No tfevents files found in {logdir}")
        return

    print(f"Found {len(event_files)} event files. Processing...")

    # Dictionary to store data: {tag: [(step, value), ...]}
    data = defaultdict(list)

    for ef in event_files:
        try:
            ea = EventAccumulator(ef)
            ea.Reload()
            tags = ea.Tags()['scalars']
            
            for tag in tags:
                # Filter by keywords
                if any(k in tag for k in keywords):
                    events = ea.Scalars(tag)
                    for e in events:
                        data[tag].append((e.step, e.value))
        except Exception as e:
            print(f"Could not process {ef}: {e}")

    # Convert to DataFrame for easier handling
    # We want a unified CSV where index is Step.
    # Since different runs might have different steps, we'll merge them.
    
    df_final = pd.DataFrame()
    
    for tag, values in data.items():
        # Sort by step
        values.sort(key=lambda x: x[0])
        steps = [v[0] for v in values]
        vals = [v[1] for v in values]
        
        # Create a temp DF
        temp_df = pd.DataFrame(vals, index=steps, columns=[tag])
        
        # Merge into final DF
        if df_final.empty:
            df_final = temp_df
        else:
            # Join on index (Step)
            # We use outer join to keep all steps, but this might result in sparse data
            # if runs are not aligned. 
            # For a single run export, this is fine.
            # If exporting multiple runs, we might want to group by run.
            # But let's assume the user points to a specific run folder (e.g. PPO_23).
            df_final = df_final.join(temp_df, how='outer')

    if df_final.empty:
        print("No data found matching keywords.")
        return

    df_final.sort_index(inplace=True)
    
    if interpolate:
        print("Linearly interpolating missing values...")
        df_final.interpolate(method='linear', inplace=True)
    elif fill:
        print("Forward filling missing values...")
        df_final.fillna(method='ffill', inplace=True)
    
    df_final.index.name = 'Step'
    
    print(f"Saving to {output_file}...")
    df_final.to_csv(output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TensorBoard scalars to CSV")
    parser.add_argument("--logdir", required=True, help="Path to the specific run directory (e.g. runs/unified/PPO_23)")
    parser.add_argument("--output", default="training_data.csv", help="Output CSV file name")
    parser.add_argument("--keywords", nargs="+", default=["mean_reward", "stage", "learning_rate"], help="Keywords to filter tags")
    parser.add_argument("--fill", action="store_true", help="Forward fill missing values (step function)")
    parser.add_argument("--interpolate", action="store_true", help="Linearly interpolate missing values (smooth lines)")
    args = parser.parse_args()

    export_to_csv(args.logdir, args.output, args.keywords, args.fill, args.interpolate)
