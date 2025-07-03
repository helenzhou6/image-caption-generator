import pickle

# Path to your pickle file
pickle_file = "nutrition_dataset.pkl"

# Load the dataset
with open(pickle_file, "rb") as f:
    dataset = pickle.load(f)

# Print first 5 entries
for i, entry in enumerate(dataset[:5]):
    print(f"Entry {i+1}:")
    print(f"Image path: {entry['image_path']}")
    print(f"Summary: {entry['summary']}")
    print(f"Ingredients: {entry['ingredients']}")
    print(f"Should eat: {entry['should_eat']}")
    print("-" * 40)
