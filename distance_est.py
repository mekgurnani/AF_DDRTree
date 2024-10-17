import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

orig_matrix = np.load('../external_val/orig_matrix.npy')
pred_matrix = np.load('../external_val/pred_matrix.npy')


# Calculate Euclidean distance between two points
def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Find the closest point in orig_matrix for a given row in pred_matrix
def find_closest_point(row, orig_matrix):
    distances = [calculate_euclidean_distance(row, orig_row) for orig_row in orig_matrix]
    closest_index = np.argmin(distances)
    return orig_matrix[closest_index]

# Process a batch of rows in pred_matrix concurrently
def process_batch(start_index, end_index, batch_number):
    for i in range(start_index, end_index):

        current_row = pred_matrix_test[i]
        closest_point = find_closest_point(current_row, orig_matrix_test)
        new_pred_matrix[i] = closest_point

    print(f"Batch {batch_number} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

orig_matrix_test = orig_matrix
pred_matrix_test = pred_matrix

# Create a new matrix for the updated predictions
new_pred_matrix = np.zeros_like(pred_matrix_test)

batch_size = 1000

start_time = time.time()

with ThreadPoolExecutor() as executor:
    for batch_number, start_index in enumerate(range(0, pred_matrix_test.shape[0], batch_size)):
        end_index = min(start_index + batch_size, pred_matrix_test.shape[0])
        executor.submit(process_batch, start_index, end_index, batch_number)


end_time = time.time()

print("Original Matrix:")
print(orig_matrix_test.shape)

print("\nPredicted Matrix:")
print(pred_matrix_test.shape)

print("\nNew Predicted Matrix:")
print(new_pred_matrix.shape)

elapsed_time_minutes = (end_time - start_time) / 60
print("\nElapsed Time:", elapsed_time_minutes, "minutes")

# Save the new_pred_matrix
np.save('../new_pred_matrix_xgboost_ukb.npy', new_pred_matrix)
